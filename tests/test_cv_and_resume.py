import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

from glioma_mtl.engine import train_classifier, train_segmentation
from glioma_mtl.engine.common import AmpStepper, BestTracker, make_optimizer, make_scheduler, save_last, try_resume
from glioma_mtl.features import extract_deep_features
from glioma_mtl.losses import MultiTaskLoss
from glioma_mtl.models import MultiTaskGliomaNet, build_swin_unetr
from glioma_mtl.models.segmentation import build_segmentation_model
from glioma_mtl.radiomics import TopK
from glioma_mtl.utils import CSVLogger, read_history, save_checkpoint

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


def test_classifier_trains_and_logs_cost(tmp_path):
    model = MultiTaskGliomaNet({"grade": 3}, feature_size=12, use_checkpoint=False)
    label = torch.zeros(2, 1, 64, 64, 64)
    label[..., 20:40, 20:40, 20:40] = 1
    batch = {"image": torch.randn(2, 1, 64, 64, 64), "label": label, "patient_id": ["p1", "p2"],
             "y_grade": torch.tensor([0, 2])}
    cfg = {"train": {"epochs": 2, "lr": 1e-3, "amp": False}, "data": {}}
    train_classifier(model, [batch], [batch], MultiTaskLoss({"grade": 1.0}), cfg, tmp_path, torch.device("cpu"))
    df = pd.read_csv(tmp_path / "history.csv")
    assert df["epoch"].tolist() == [1, 2] and (df["epoch_seconds"] > 0).all() and df["val_loss"].notna().all()
    assert (tmp_path / "best.pt").exists() and (tmp_path / "model_info.json").exists()


def test_save_checkpoint_survives_a_crash_mid_write(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    model = torch.nn.Linear(3, 1)
    save_checkpoint(path, model, epoch=1)
    save_checkpoint(path, model, epoch=2)                          # overwrites in place
    assert torch.load(path, weights_only=False)["epoch"] == 2 and not path.with_name("last.pt.tmp").exists()

    def dies_mid_write(state, f):
        Path(f).write_bytes(b"PK\x03\x04 truncated")
        raise RuntimeError("session died")
    monkeypatch.setattr(torch, "save", dies_mid_write)
    with pytest.raises(RuntimeError, match="session died"):
        save_checkpoint(path, model, epoch=3)
    assert torch.load(path, weights_only=False)["epoch"] == 2      # the previous checkpoint is intact


def test_resume_from_a_damaged_checkpoint_says_what_to_do(tmp_path):
    (tmp_path / "last.pt").write_bytes(b"PK\x03\x04 truncated")
    model = torch.nn.Linear(3, 1)
    opt = make_optimizer(model.parameters(), {"lr": 0.1})
    with pytest.raises(RuntimeError, match="incomplete or damaged.*Delete its run folder"):
        try_resume(tmp_path / "last.pt", model, opt, make_scheduler(opt, {"warmup_epochs": 0}, epochs=10),
                   AmpStepper(opt, model.parameters(), torch.device("cpu")), BestTracker("max"))


def test_history_and_cost_of_a_run_started_before_cost_logging(tmp_path):
    """Epochs 1-3 ran with the old logger (no val_dice, no timing), then the run resumed with the new code."""
    pd.DataFrame({"epoch": [1, 2, 3, 4, 5, 6], "lr": 1e-3, "train_loss": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                  "epoch_seconds": [None, None, None, 95.0, 60.0, 95.0], "gpu": [None] * 3 + ["A100"] * 3,
                  "val_dice": [None, None, None, 0.6, None, 0.7]}).to_csv(tmp_path / "history.csv", index=False)
    (tmp_path / "log.txt").write_text("\n".join([
        "2026-11-01 10:00:00,000 INFO Run seg_x -> runs/seg_x on cuda",
        "2026-11-01 10:01:00,000 INFO epoch=1 | lr=0.001 | train_loss=0.9",
        "2026-11-01 10:02:20,000 INFO epoch 2: new best val Dice 0.5000",
        "2026-11-01 10:02:30,000 INFO epoch=2 | lr=0.001 | train_loss=0.8 | val_dice=0.5",
        "2026-11-01 10:03:30,000 INFO epoch=3 | lr=0.001 | train_loss=0.7",
        "2026-11-01 11:00:00,000 INFO Run seg_x -> runs/seg_x on cuda",
        "2026-11-01 11:00:05,000 INFO Resumed from runs/seg_x/last.pt: continuing with epoch 4",
        "2026-11-01 11:01:40,000 INFO epoch=4 | lr=0.001 | train_loss=0.6 | val_dice=0.6 | epoch_seconds=95 | gpu=A100",
    ]) + "\n", encoding="utf-8")
    h = read_history(tmp_path)
    assert h["epoch"].tolist() == [1, 2, 3, 4, 5, 6]
    assert h["val_dice"].tolist()[1] == 0.5 and h["train_loss"].tolist()[3] == 0.6   # CSV wins where both exist
    assert h["log_seconds"].isna().tolist() == [True, False, False, True, True, True]
    assert h["log_seconds"].tolist()[1:3] == [90.0, 60.0]

    agg = _load_script("aggregate_results")
    unet = {"name": "unet", "channels": [4, 8], "strides": [2], "num_res_units": 1}
    (tmp_path / "config.yaml").write_text(json.dumps({"model": unet, "data": {}}), encoding="utf-8")
    cost = agg.run_cost(tmp_path)
    # epoch 1 (first of its session, no validation) takes the mean of the other non-validation epochs
    assert cost["train_h"] * 3600 == pytest.approx(60 + 90 + 60 + 95 + 60 + 95) and cost["train_h_estimated"]
    assert cost["params_M"] * 1e6 == pytest.approx(sum(p.numel() for p in build_segmentation_model(unet).parameters()))


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
