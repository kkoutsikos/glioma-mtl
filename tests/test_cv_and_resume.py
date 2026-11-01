import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

from glioma_mtl.engine import train_segmentation
from glioma_mtl.engine.common import AmpStepper, BestTracker, make_optimizer, make_scheduler, save_last, try_resume
from glioma_mtl.features import extract_deep_features
from glioma_mtl.models import build_swin_unetr
from glioma_mtl.models.segmentation import build_segmentation_model
from glioma_mtl.radiomics import TopK
from glioma_mtl.utils import CSVLogger

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_topk_survives_dropped_columns_and_picks_informative():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 20)
    X = np.c_[rng.normal(size=(40, 3)), y + rng.normal(scale=0.1, size=40), np.ones(40)]  # col 3 informative, 4 constant
    pipe = Pipeline([("var", VarianceThreshold()), ("top", TopK(k=30))]).fit(X, y)
    assert pipe.transform(X).shape == (40, 4)                      # k clamped to the 4 non-constant columns
    assert TopK(k=1).fit(X, y).support_.tolist() == [3]


def test_holm_correction():
    holm = _load_script("aggregate_results").holm
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_csv_logger_keeps_columns_that_appear_later_and_across_resumes(tmp_path):
    path = tmp_path / "history.csv"
    log = CSVLogger(path)
    log.log({"epoch": 1, "train_loss": 0.9})                       # no validation in epoch 1
    log.log({"epoch": 2, "train_loss": 0.8, "val_dice": 0.5})
    CSVLogger(path).log({"epoch": 3, "train_loss": 0.7, "val_dice": 0.6, "gpu": "x"})   # resumed run
    df = pd.read_csv(path)
    assert df.columns.tolist() == ["epoch", "train_loss", "val_dice", "gpu"]
    assert df["val_dice"].tolist()[1:] == [0.5, 0.6] and np.isnan(df["val_dice"][0])
    assert df["train_loss"].tolist() == [0.9, 0.8, 0.7]


def test_segmentation_history_has_validation_and_cost_columns(tmp_path):
    model = build_segmentation_model({"name": "unet", "channels": [4, 8], "strides": [2], "num_res_units": 1})
    label = torch.zeros(1, 1, 32, 32, 32)
    label[..., 8:20, 8:20, 8:20] = 1
    batch = {"image": torch.randn(1, 1, 32, 32, 32), "label": label, "patient_id": ["p1"]}
    cfg = {"train": {"epochs": 3, "lr": 1e-3, "val_every": 2, "amp": False}, "data": {"roi_size": [32, 32, 32]}}
    train_segmentation(model, [batch], [batch], cfg, tmp_path, torch.device("cpu"))
    df = pd.read_csv(tmp_path / "history.csv")
    assert df["val_dice"].notna().tolist() == [False, True, True]   # epochs 2 and 3 (last) validate
    assert (df["epoch_seconds"] > 0).all() and (df["gpu"] == "cpu").all()
    info = json.loads((tmp_path / "model_info.json").read_text())
    assert info["params"] == sum(p.numel() for p in model.parameters()) == info["trainable_params"]


def test_segmentation_table_reports_cost(tmp_path):
    agg = _load_script("aggregate_results")
    rng = np.random.default_rng(0)
    for name, dice, params in (("seg_a", 0.9, 2e6), ("seg_b", 0.8, 5e6)):
        run = tmp_path / f"{name}_s42"
        run.mkdir()
        pd.DataFrame({"patient_id": [f"p{i}" for i in range(12)], "dice": dice + rng.normal(0, 0.01, 12),
                      "hd95": 5.0 + rng.normal(0, 0.5, 12), "nsd": 0.9, "iou": 0.8, "sensitivity": 0.9, "precision": 0.9,
                      "seconds": 1.5}).to_csv(run / "per_case_segmentation_test.csv", index=False)
        (run / "model_info.json").write_text(json.dumps({"params": params, "trainable_params": params, "gpu": "A100"}))
        pd.DataFrame({"epoch": [1, 2], "epoch_seconds": [1800, 1800], "peak_mem_gb": [7.0, 7.5],
                      "gpu": "A100"}).to_csv(run / "history.csv", index=False)
    table, _ = agg.segmentation_table({"segmentation": {"models": {"seg_a": {}, "seg_b": {}}, "seeds": [42]}}, tmp_path)
    row = table.set_index("model").loc["seg_b"]
    assert (row["params_M"], row["train_h"], row["peak_mem_gb"], row["infer_s_per_case"], row["gpu"]) == \
        ("5.0", "1.0", "7.5", "1.50", "A100")
    assert table.set_index("model").loc["seg_a", "p_dice_vs_best"] == "best"


def test_resume_restores_training_state(tmp_path):
    def setup():
        model = torch.nn.Linear(3, 1)
        opt = make_optimizer(model.parameters(), {"lr": 0.1})
        sched = make_scheduler(opt, {"warmup_epochs": 0}, epochs=10)
        return model, opt, sched, AmpStepper(opt, model.parameters(), torch.device("cpu")), BestTracker("max")

    model, opt, sched, stepper, tracker = setup()
    for epoch in range(3):
        model(torch.randn(4, 3)).sum().backward()
        stepper.step()
        sched.step()
        tracker.update(float(epoch), epoch)
    save_last(tmp_path / "last.pt", model, opt, sched, stepper, tracker, epoch=2)

    model2, opt2, sched2, stepper2, tracker2 = setup()
    assert try_resume(tmp_path / "last.pt", model2, opt2, sched2, stepper2, tracker2) == 3
    assert torch.equal(model.weight, model2.weight)
    assert sched2.last_epoch == sched.last_epoch and tracker2.best_epoch == 2
    assert try_resume(tmp_path / "missing.pt", *setup()) == 0

    other = torch.nn.Linear(3, 2)                     # e.g. a run started before a head changed shape
    other_opt = make_optimizer(other.parameters(), {"lr": 0.1})
    with pytest.raises(RuntimeError, match="Delete its run folder"):
        try_resume(tmp_path / "last.pt", other, other_opt, make_scheduler(other_opt, {"warmup_epochs": 0}, epochs=10),
                   AmpStepper(other_opt, other.parameters(), torch.device("cpu")), BestTracker("max"))


def test_deep_features_use_mask_and_one_row_per_patient():
    model = build_swin_unetr(1, 1, feature_size=12, use_checkpoint=False)
    image = torch.randn(2, 1, 64, 64, 64)
    label = torch.zeros(2, 1, 64, 64, 64)
    label[:, :, 20:40, 20:40, 20:40] = 1
    loader = [{"image": image, "label": label, "patient_id": ["UCSF-PDGM-0001", "UCSF-PDGM-0002"]}]
    oracle = extract_deep_features(model, loader, torch.device("cpu"), stages=(2, 4), mask_source="label")
    pred = extract_deep_features(model, loader, torch.device("cpu"), stages=(2, 4), mask_source="pred")
    assert list(oracle.index) == ["UCSF-PDGM-0001", "UCSF-PDGM-0002"]
    assert oracle.shape[1] == 12 * 4 + 12 * 16 + 12          # stage 2 + stage 4 + decoder
    assert not np.allclose(oracle.values, pred.values)        # pooling region matters
