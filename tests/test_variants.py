"""Thesis classifier variants: every config builds and runs; SegMamba adapter; LGG/HGG; peritumoral ring."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from glioma_mtl.config import load_config
from glioma_mtl.data.metadata import TASKS
from glioma_mtl.engine import add_derived_lgg_hgg
from glioma_mtl.experiment import ClassificationData, build_classifier
from glioma_mtl.explain import grad_cam
from glioma_mtl.models import MultiTaskGliomaNet

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("cls_*.yaml"))
ROI = 64


class _StubEncoder(nn.Module):
    def __init__(self, feat):
        super().__init__()
        chans = [1, *feat]
        self.stages = nn.ModuleList([nn.Conv3d(a, b, 3, stride=2, padding=1) for a, b in zip(chans[:-1], chans[1:])])

    def forward(self, x):
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return tuple(outs)


class StubSegMamba(nn.Module):
    """Same module layout as ge-xing/SegMamba (vit -> 4 stages, encoder5 bottleneck, out), tiny and CPU-only."""

    def __init__(self, feat=(48, 96, 192, 384), hidden=768):
        super().__init__()
        self.vit = _StubEncoder(feat)
        self.encoder5 = nn.Conv3d(feat[3], hidden, 1)
        self.decoder1 = nn.Conv3d(1, feat[0], 3, padding=1)
        self.out = nn.Conv3d(feat[0], 1, 1)

    def forward(self, x):
        h = self.encoder5(self.vit(x)[3])
        return self.out(self.decoder1(x) * (1 + h.mean()))


def _data(cfg):
    tasks = {t: len(TASKS[t][1]) for t in cfg["tasks"]}
    tab = {}
    if (cfg.get("clinical") or {}).get("groups"):
        tab["clinical"] = 5
    if "radiomics" in cfg and any(k in cfg["name"] for k in ("radiomics", "trinity", "segmamba")):
        tab["radiomics"] = 10
    return ClassificationData(records={}, tasks=tasks, tabular_dims=tab, class_weights={}, featurizers={})


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_classifier_config_builds_and_runs(path, monkeypatch):
    monkeypatch.setattr("glioma_mtl.models.segmentation.build_segmentation_model",
                        lambda cfg, *a, **k: StubSegMamba(cfg.get("feat_size", (48, 96, 192, 384)),
                                                          cfg.get("hidden_size", 768)))
    cfg = load_config(path, ["model.feature_size=12", "model.use_checkpoint=false"])
    data = _data(cfg)
    model = build_classifier(cfg, data).eval()
    tab = {n: torch.randn(1, d) for n, d in data.tabular_dims.items()}
    with torch.no_grad():
        out = model(torch.randn(1, 1, ROI, ROI, ROI), tab)
    assert set(out["logits"]) == set(cfg["tasks"])
    assert model.type_embed.shape[1] == len(model.token_names)


def test_segmamba_classifier_tokens_heads_and_gradcam(monkeypatch):
    monkeypatch.setattr("glioma_mtl.models.segmentation.build_segmentation_model", lambda *a, **k: StubSegMamba())
    model = MultiTaskGliomaNet({"grade": 3, "lgg_hgg": 2}, {"clinical": 4}, backbone="segmamba",
                               encoder_stages=[4], use_decoder_token=False, mask_mode="hard", peritumoral=True).eval()
    assert model.token_names == ["image_stage4", "image_stage4_peritumoral", "clinical"]
    x, tab = torch.randn(1, 1, ROI, ROI, ROI), {"clinical": torch.randn(1, 4)}
    with torch.no_grad():
        out = model(x, tab)
    assert out["logits"]["lgg_hgg"].shape == (1, 2) and len(out["aux"]["grade"]) == 2
    cam, probs, _ = grad_cam(model, x, tab, "grade", "stage4")
    assert cam.shape == (ROI, ROI, ROI) and np.isclose(cam.max(), 1.0)


def test_segmamba_adapter_rejects_unexpected_layout(monkeypatch):
    monkeypatch.setattr("glioma_mtl.models.segmentation.build_segmentation_model", lambda *a, **k: nn.Conv3d(1, 1, 1))
    with pytest.raises(RuntimeError, match="no \\['vit', 'encoder5', 'out'\\]"):
        MultiTaskGliomaNet({"grade": 3}, backbone="segmamba")


def test_segmamba_channel_mismatch_fails_loudly(monkeypatch):
    monkeypatch.setattr("glioma_mtl.models.segmentation.build_segmentation_model",
                        lambda *a, **k: StubSegMamba(hidden=512))   # model expects 768 from its config
    model = MultiTaskGliomaNet({"grade": 3}, backbone="segmamba", encoder_stages=[4], use_decoder_token=False).eval()
    with pytest.raises(RuntimeError, match="512 channels, projection expects 768"):
        model(torch.randn(1, 1, ROI, ROI, ROI))


def test_deep_features_from_a_segmamba_backbone():
    from glioma_mtl.features import extract_deep_features

    model = StubSegMamba(feat=(4, 8, 16, 32), hidden=64)
    label = torch.zeros(2, 1, ROI, ROI, ROI)
    label[:, :, 20:40, 20:40, 20:40] = 1
    loader = [{"image": torch.randn(2, 1, ROI, ROI, ROI), "label": label,
               "patient_id": ["UCSF-PDGM-0001", "UCSF-PDGM-0002"]}]
    df = extract_deep_features(model, loader, torch.device("cpu"), stages=(2, 3, 4), mask_source="label")
    assert df.shape == (2, 16 + 32 + 64 + 4)        # stages 2, 3, the bottleneck and the decoder
    assert np.isfinite(df.values).all() and df.std().gt(0).any()


def test_peritumoral_ring_excludes_core():
    model = MultiTaskGliomaNet({"grade": 3}, feature_size=12, use_checkpoint=False, peritumoral=True,
                               peritumoral_dilation=2)
    mask = torch.zeros(1, 1, 16, 16, 16)
    mask[:, :, 6:10, 6:10, 6:10] = 1
    ring = model.peritumoral_ring(mask)
    assert ring[mask > 0].sum() == 0 and ring.sum() == (8 ** 3 - 4 ** 3)


def test_lgg_hgg_labels_and_derived_probabilities():
    df = pd.DataFrame({"patient_id": ["a", "b", "c"], "y_grade": [0, 2, -1],
                       "p_grade_0": [0.6, 0.1, 0.3], "p_grade_1": [0.3, 0.2, 0.3], "p_grade_2": [0.1, 0.7, 0.4]})
    out = add_derived_lgg_hgg(df)
    assert out["y_lgg_hgg"].tolist() == [0, 1, -1]
    assert np.allclose(out[["p_lgg_hgg_0", "p_lgg_hgg_1"]].sum(1), 1)
    assert np.allclose(out["p_lgg_hgg_1"], df["p_grade_2"])
    own_head = out.assign(p_lgg_hgg_1=0.5)
    assert add_derived_lgg_hgg(own_head) is own_head          # a dedicated head is never overwritten
