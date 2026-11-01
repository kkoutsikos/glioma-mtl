import numpy as np
import pytest
import torch

from glioma_mtl import plotting
from glioma_mtl.explain import attention_keys, cam_layers, fusion_attention, grad_cam, token_occlusion
from glioma_mtl.models import MultiTaskGliomaNet

FS, ROI = 12, 64
TASKS = {"grade": 3, "diagnosis": 4}


def _model(**kw):
    torch.manual_seed(0)
    return MultiTaskGliomaNet(TASKS, {"clinical": 5}, feature_size=FS, use_checkpoint=False, **kw).eval()


def _inputs():
    torch.manual_seed(1)
    return torch.randn(1, 1, ROI, ROI, ROI), {"clinical": torch.randn(1, 5)}


@pytest.mark.parametrize("fusion", ["transformer", "cross_attention"])
def test_attention_recomputation_matches_model(fusion):
    model = _model(fusion=fusion)
    image, tab = _inputs()
    with torch.no_grad():
        enc = model.encode(image, tab)
        ctx = model.context(enc["image_tokens"], enc["tab_tokens"])
        maps, final = fusion_attention(model, ctx)
        _, fused = model.fuse(ctx)
    assert torch.allclose(final, fused, atol=1e-5)          # the recomputation is the real forward pass
    assert len(maps) == len(model.fuser.layers)
    assert maps[0].shape == (1, len(TASKS), len(attention_keys(model)))
    assert torch.allclose(maps[0].sum(-1), torch.ones(1, len(TASKS)), atol=1e-5)


def test_concat_has_no_attention_but_occlusion_works():
    model = _model(fusion="concat")
    image, tab = _inputs()
    with torch.no_grad():
        enc = model.encode(image, tab)
        assert fusion_attention(model, model.context(enc["image_tokens"], enc["tab_tokens"])) == (None, None)
        base, dropped = token_occlusion(model, enc)
        full = torch.softmax(model(image, tab)["logits"]["grade"], 1)
    assert torch.allclose(base["grade"], full, atol=1e-5)
    assert dropped["diagnosis"].shape == (1, len(model.token_names), 4)
    assert model.token_names == ["image_stage4", "image_decoder", "clinical"]


def test_grad_cam_shape_and_range():
    model = _model(mask_mode="soft")
    image, tab = _inputs()
    for layer in cam_layers(model):
        cam, probs, cls = grad_cam(model, image, tab, "grade", layer)
        assert cam.shape == (ROI, ROI, ROI) and cam.min() >= 0 and np.isclose(cam.max(), 1.0)
        assert np.isclose(probs.sum(), 1.0) and 0 <= cls < 3
    assert not model.freeze_backbone          # Grad-CAM's temporary unfreeze is undone


def test_grad_cam_works_on_a_frozen_backbone():
    model = _model(fusion="concat", mask_mode="none", freeze_backbone=True, use_decoder_token=False)
    image, tab = _inputs()
    cam, _, _ = grad_cam(model, image, tab, "grade", "stage4")
    assert np.isclose(cam.max(), 1.0)
    assert all(not p.requires_grad and p.grad is None for p in model.backbone.parameters())  # weights untouched


def test_grad_cam_refuses_maps_the_classifier_does_not_use():
    model = _model(mask_mode="none", encoder_stages=[4], use_decoder_token=False)
    image, tab = _inputs()
    with pytest.raises(RuntimeError, match="no gradient"):
        grad_cam(model, image, tab, "grade", "stage1")


def test_plots_write_files(tmp_path):
    rng = np.random.default_rng(0)
    vol = rng.random((32, 32, 32))
    gt = np.zeros_like(vol, dtype=bool)
    gt[10:20, 10:20, 10:20] = True
    paths = [
        plotting.plot_cam(vol, vol, gt, gt, tmp_path / "cam.png", "cam"),
        plotting.plot_confusion(np.array([[3, 1], [0, 4]]), ["mutant", "wildtype"], tmp_path / "cm.png", "cm"),
        plotting.plot_roc(np.array([0, 1, 1, 0, 1]), rng.dirichlet([1, 1], 5), ["a", "b"], tmp_path / "roc.png", "roc"),
        plotting.plot_bars(["a", "b"], [0.2, -0.1], [0.05, 0.02], tmp_path / "bars.png", "bars", "x"),
        plotting.plot_heatmap(rng.random((2, 3)), ["r1", "r2"], ["a", "b", "c"], tmp_path / "hm.png", "hm"),
        plotting.plot_box({"m1": rng.random(10), "m2": rng.random(10)}, tmp_path / "box.png", "Dice", "box"),
        plotting.plot_seg_cases([{"case": "p1", "image": vol, "gt": gt, "preds": {"m1": gt}, "dice": {"m1": 1.0}}],
                                tmp_path / "cases.png", "cases"),
    ]
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)
