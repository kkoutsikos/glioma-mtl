import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from glioma_mtl.engine import predict, score, train_classifier, train_segmentation
from glioma_mtl.engine.classification import FLIPS
from glioma_mtl.engine.common import AmpStepper, BestTracker, make_optimizer, make_scheduler, save_last, try_resume
from glioma_mtl.features import extract_deep_features
from glioma_mtl.losses import MultiTaskLoss
from glioma_mtl.models import MultiTaskGliomaNet, build_swin_unetr
from glioma_mtl.models.segmentation import build_segmentation_model
from glioma_mtl.radiomics import RadiomicsFeaturizer, RFETopK, TopK, make_selector
from glioma_mtl.utils import CSVLogger, read_history, save_checkpoint

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module   # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def test_topk_survives_dropped_columns_and_picks_informative():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 20)
    X = np.c_[rng.normal(size=(40, 3)), y + rng.normal(scale=0.1, size=40), np.ones(40)]  # col 3 informative, 4 constant
    pipe = Pipeline([("var", VarianceThreshold()), ("top", TopK(k=30))]).fit(X, y)
    assert pipe.transform(X).shape == (40, 4)                      # k clamped to the 4 non-constant columns
    assert TopK(k=1).fit(X, y).support_.tolist() == [3]


def test_rfe_keeps_informative_radiomics_on_training_labels_only(tmp_path):
    from joblib import Memory

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 20)
    X = np.c_[rng.normal(size=(60, 20)), y + rng.normal(scale=0.3, size=60), (y == 2) + rng.normal(scale=0.3, size=60)]
    rfe = RFETopK(k=2).fit(X, y)
    assert rfe.support_.tolist() == [20, 21] and rfe.transform(X).shape == (60, 2)
    assert RFETopK(k=50).fit(X[:, :5], y).support_.tolist() == [0, 1, 2, 3, 4]   # k clamped
    assert isinstance(make_selector("rfe", 3), RFETopK) and type(make_selector("anova", 3)) is TopK

    cols = [f"original_f{i}" for i in range(X.shape[1])]
    feat = RadiomicsFeaturizer(k=2, method="rfe").fit(pd.DataFrame(X, columns=cols), y)
    assert feat.dim == 2

    kfold = _load_script("kfold_cv")
    df = pd.DataFrame(X, columns=cols)
    pipe = kfold.build_pipeline([], cols, [], 2, 32, LogisticRegression(), "rfe", Memory(tmp_path, verbose=0))
    pipe.fit(df, y)
    assert list(pipe[:-1].get_feature_names_out()) == ["rad__original_f20", "rad__original_f21"]


def test_kfold_passes_sample_weights_under_either_scikit_learn_name():
    kfold = _load_script("kfold_cv")
    old = lambda estimator, X, y=None, *, fit_params=None: None   # scikit-learn < 1.4
    new = lambda estimator, X, y=None, *, params=None: None       # scikit-learn >= 1.4 (only name from 1.6)
    assert kfold.fit_params_name(old) == "fit_params" and kfold.fit_params_name(new) == "params"


def test_bootstrap_metrics_do_not_warn_about_classes_missing_from_a_resample(recwarn):
    from glioma_mtl.evaluation import classification_metrics

    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(12, int), np.ones(12, int), 2]                # one patient of a rare class
    prob = rng.dirichlet(np.ones(3), size=len(y))
    table, _ = classification_metrics(y, prob, ["a", "b", "c"])
    assert table.set_index("metric").loc["balanced_accuracy", "ci_low"] <= table.set_index("metric").loc[
        "balanced_accuracy", "ci_high"]
    assert not [w for w in recwarn if "y_pred contains classes not in y_true" in str(w.message)]


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

    # A sensitivity run added after the main results is shown and tested, but never becomes the reference,
    # even when it scores highest.
    run = tmp_path / "seg_c_s42"
    run.mkdir()
    pd.DataFrame({"patient_id": [f"p{i}" for i in range(12)], "dice": 0.95 + rng.normal(0, 0.01, 12),
                  "hd95": 4.0, "nsd": 0.9, "iou": 0.9, "sensitivity": 0.9, "precision": 0.9,
                  "seconds": 1.5}).to_csv(run / "per_case_segmentation_test.csv", index=False)
    models = {"seg_a": {}, "seg_b": {}, "seg_c": {"sensitivity": True}}
    table, _ = agg.segmentation_table({"segmentation": {"models": models, "seeds": [42]}}, tmp_path)
    table = table.set_index("model")
    assert table.loc["seg_a", "p_dice_vs_best"] == "best"
    assert float(table.loc["seg_c (sensitivity)", "p_dice_vs_best"]) < 0.05


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


def _cls_batch():
    label = torch.zeros(2, 1, 64, 64, 64)
    label[..., 20:40, 20:40, 20:40] = 1
    return {"image": torch.randn(2, 1, 64, 64, 64), "label": label, "patient_id": ["p1", "p2"],
            "y_grade": torch.tensor([0, 2])}


def test_checkpoint_score_is_classification_only_and_task_weighted():
    df = pd.DataFrame({"patient_id": ["a", "b"], "y_grade": [0, 2], "p_grade_0": [0.8, 0.1], "p_grade_1": [0.1, 0.1],
                       "p_grade_2": [0.1, 0.8], "y_diagnosis": [1, -1], "p_diagnosis_0": [0.5, 0.5],
                       "p_diagnosis_1": [0.5, 0.5]})
    s = score(df, ["grade", "diagnosis"], {"grade": 1.0, "diagnosis": 0.5})
    assert np.isclose(s["grade_nll"], -np.log(0.8))
    assert np.isclose(s["diagnosis_nll"], -np.log(0.5))                  # the missing label is skipped
    assert np.isclose(s["cls_nll"], (-np.log(0.8) - 0.5 * np.log(0.5)) / 1.5)


def test_classifier_picks_its_checkpoint_on_classification_nll(tmp_path):
    model = MultiTaskGliomaNet({"grade": 3}, feature_size=12, use_checkpoint=False)
    cfg = {"train": {"epochs": 2, "lr": 1e-3, "amp": False}, "data": {}, "selection": {"metric": "cls_nll"}}
    train_classifier(model, [_cls_batch()], [_cls_batch()], MultiTaskLoss({"grade": 1.0}), cfg, tmp_path,
                     torch.device("cpu"))
    df = pd.read_csv(tmp_path / "history.csv")
    best = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert best["metrics"]["epoch"] == df.loc[df["val_cls_nll"].idxmin(), "epoch"]   # lower is better


def test_flip_averaging_is_the_mean_over_all_flips():
    torch.manual_seed(0)
    model = MultiTaskGliomaNet({"grade": 3}, feature_size=12, use_checkpoint=False).eval()
    batch = _cls_batch()
    tta, _ = predict(model, [batch], torch.device("cpu"), tta=True)
    plain, _ = predict(model, [batch], torch.device("cpu"))
    with torch.no_grad():
        mean = sum(torch.softmax(model(batch["image"].flip(d) if d else batch["image"])["logits"]["grade"], 1)
                   for d in FLIPS) / len(FLIPS)
    cols = ["p_grade_0", "p_grade_1", "p_grade_2"]
    assert len(FLIPS) == 8 and np.allclose(tta[cols].to_numpy(), mean.numpy(), atol=1e-5)
    assert not np.allclose(tta[cols].to_numpy(), plain[cols].to_numpy())


def test_backbone_stays_frozen_for_the_first_epochs(tmp_path):
    torch.manual_seed(0)
    model = MultiTaskGliomaNet({"grade": 3}, feature_size=12, use_checkpoint=False)
    cfg = {"train": {"epochs": 1, "lr": 1e-3, "lr_backbone": 1e-3, "amp": False, "warmup_epochs": 0,
                     "freeze_backbone_epochs": 1}, "data": {}}
    backbone0 = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    heads0 = [p.detach().clone() for p in model.head_parameters()]
    train = lambda: train_classifier(model, [_cls_batch()], [_cls_batch()], MultiTaskLoss({"grade": 1.0}), cfg,
                                     tmp_path, torch.device("cpu"))
    train()
    assert all(torch.equal(v, model.backbone.state_dict()[k]) for k, v in backbone0.items())
    assert any(not torch.equal(a, b) for a, b in zip(heads0, model.head_parameters()))
    cfg["train"]["epochs"] = 2                                    # resumes at epoch 2, where the backbone trains
    train()
    assert any(not torch.equal(v, model.backbone.state_dict()[k]) for k, v in backbone0.items())
    assert model.freeze_backbone is False


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


def test_radiomics_leave_an_empty_predicted_mask_missing(tmp_path):
    import SimpleITK as sitk

    from glioma_mtl.radiomics import extract_radiomics

    image = sitk.GetImageFromArray(np.random.default_rng(0).normal(100, 20, (24, 24, 24)).astype(np.float32))
    sitk.WriteImage(image, str(tmp_path / "t2.nii.gz"))
    tumour = np.zeros((24, 24, 24), np.uint8)
    tumour[6:18, 6:18, 6:18] = 1
    for name, arr in (("tumour", tumour), ("empty", np.zeros_like(tumour))):
        mask = sitk.GetImageFromArray(arr)
        mask.CopyInformation(image)
        sitk.WriteImage(mask, str(tmp_path / f"{name}.nii.gz"))
    df = extract_radiomics([("UCSF-PDGM-0001", tmp_path / "t2.nii.gz", tmp_path / "tumour.nii.gz"),
                            ("UCSF-PDGM-0002", tmp_path / "t2.nii.gz", tmp_path / "empty.nii.gz")],
                           feature_classes=["firstorder"])
    assert len(df) == 2 and df.shape[1] > 10
    assert df.iloc[0].notna().all() and df.iloc[1].isna().all()


def test_masks_from_before_the_cleanup_are_predicted_again(tmp_path):
    """Markers written before the largest-component cleanup must not count as done (stale masks on Drive)."""
    import yaml

    runner = _load_script("run_paper")
    plan = yaml.safe_load((runner.REPO / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    steps = {s.id: s for s in runner.build_steps(plan, [f"output_root={tmp_path.as_posix()}"])[0]}
    for part in ("train", "val", "test"):
        assert "--largest-component" in steps[f"predict:{part}"].args
    for step_id in ("predict:test", "features:deep", "features:radiomics"):
        step = steps[step_id]
        assert step.version.startswith(runner.MASKS_VERSION)
        marker = tmp_path / "marker.done"
        marker.write_text("Mon Nov  2 16:35:49 2026 (1.2 min)\n")
        assert not runner.is_done(marker, step)
        marker.write_text(f"Mon Nov  2 18:00:00 2026 (1.2 min)\nversion: {step.version}\n")
        assert runner.is_done(marker, step)
    old = tmp_path / "seg.done"
    old.write_text("Mon Nov  2 12:00:00 2026 (170.0 min)\n")
    assert runner.is_done(old, steps["seg_train:seg_swin_unetr_ssl_s42"])   # unversioned steps are unaffected


def test_switching_the_backbone_redoes_everything_downstream(tmp_path):
    """Masks, features and classifiers made with another backbone are redone; their run folders are set aside."""
    import yaml

    runner = _load_script("run_paper")
    plan = yaml.safe_load((runner.REPO / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    root_set = [f"output_root={tmp_path.as_posix()}"]
    new = {s.id: s for s in runner.build_steps(plan, root_set)[0]}
    plan["classification"]["backbone"] = "seg_swin_unetr_ssl"
    old = {s.id: s for s in runner.build_steps(plan, root_set)[0]}
    marker = tmp_path / "marker.done"
    for step_id in ("predict:val", "features:deep", "features:radiomics", "kfold",
                    "cls_train:cls_image_only_s42", "cls_eval:cls_image_only_s42", "explain:cls_image_only_s42"):
        marker.write_text(f"Mon Nov  2 18:00:00 2026 (1.2 min)\nversion: {old[step_id].version}\n")
        assert runner.is_done(marker, old[step_id]) and not runner.is_done(marker, new[step_id]), step_id
    assert "seg_segmamba_lr5e5_s42/best.pt" in " ".join(new["cls_train:cls_image_only_s42"].args)

    step = new["cls_train:cls_image_only_s42"]
    for tag in (None, old[step.id].version):          # folders from before tagging, and from the old backbone
        step.run_dir.mkdir()
        (step.run_dir / "last.pt").write_text("SwinUNETR classifier")
        if tag:
            (step.run_dir / ".step_version").write_text(tag + "\n")
        runner.set_aside_stale_run(step, tmp_path, dry_run=False)
        assert not step.run_dir.exists()
    assert len(list((tmp_path / ".stale").glob("cls_image_only_s42-*/last.pt"))) == 2
    step.run_dir.mkdir()                               # an interrupted run of this version resumes
    (step.run_dir / ".step_version").write_text(step.version + "\n")
    runner.set_aside_stale_run(step, tmp_path, dry_run=False)
    assert step.run_dir.exists()


def test_a_new_evaluation_version_re_evaluates_without_retraining(tmp_path):
    """Classifier test metrics made by an older evaluation are redone; training and its run folder are kept."""
    import yaml

    runner = _load_script("run_paper")
    plan = yaml.safe_load((runner.REPO / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    steps = {s.id: s for s in runner.build_steps(plan, [f"output_root={tmp_path.as_posix()}"])[0]}
    train, evaluate = steps["cls_train:cls_image_only_s42"], steps["cls_eval:cls_image_only_s42"]
    assert evaluate.version == f"{train.version}, eval {runner.EVAL_VERSION}" and evaluate.run_dir is None
    marker = tmp_path / "marker.done"
    marker.write_text(f"Mon Nov  2 18:00:00 2026 (1.2 min)\nversion: {train.version}\n")   # written before EVAL_VERSION
    assert runner.is_done(marker, train) and not runner.is_done(marker, evaluate)
    train.run_dir.mkdir()
    (train.run_dir / ".step_version").write_text(train.version + "\n")
    runner.set_aside_stale_run(train, tmp_path, dry_run=False)
    assert train.run_dir.exists()


def test_oracle_stage_rescores_the_trained_models_with_ground_truth_masks(tmp_path):
    """The oracle stage re-evaluates every classifier and refits late fusion on GT-mask features, retraining nothing."""
    import yaml

    runner = _load_script("run_paper")
    plan = yaml.safe_load((runner.REPO / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    steps = runner.build_steps(plan, [f"output_root={tmp_path.as_posix()}"])[0]
    by_id = {s.id: s for s in steps}
    oracle = [s for s in steps if s.stage == "oracle"]
    assert all(s.optional and s.run_dir is None for s in oracle)
    order = [s.id for s in steps]
    assert order.index("kfold") < order.index(oracle[0].id) < order.index("aggregate")

    plain, gt = by_id["cls_eval:cls_image_clinical_radiomics_s42"], by_id["cls_oracle:cls_image_clinical_radiomics_s42"]
    assert gt.version == plain.version and "data.crop_mask_key=label" in gt.args
    assert gt.args[gt.args.index("--out-dir") + 1].endswith("cls_image_clinical_radiomics_s42/oracle")
    assert any(a.endswith("radiomics_label/radiomics_label.csv") for a in gt.args)
    assert "oracle_features:radiomics" in gt.needs and "cls_train:cls_image_clinical_radiomics_s42" in gt.needs
    image_only = by_id["cls_oracle:cls_image_only_s42"]
    assert not any("radiomics" in a for a in image_only.args) and image_only.needs == ["cls_train:cls_image_only_s42"]

    kfold = by_id["kfold_oracle"]
    assert {"oracle_features:radiomics", "oracle_features:deep"} == set(kfold.needs)
    assert kfold.args[kfold.args.index("--out-dir") + 1].endswith("kfold_cv_oracle")
    assert f"cv.feature_sets=[{', '.join(runner.ORACLE_FEATURE_SETS)}]" in kfold.args
    assert by_id["kfold"].version.endswith(runner.KFOLD_VERSION)      # reruns once, to save the fold scores
    assert "--mask-source" in by_id["oracle_features:deep"].args

    plan["oracle"] = False
    assert not [s for s in runner.build_steps(plan, [f"output_root={tmp_path.as_posix()}"])[0] if s.stage == "oracle"]

    curves = by_id["learning_curves"]
    assert curves.stage == "kfold" and curves.optional and "--learning-curve" in curves.args
    assert curves.needs == by_id["kfold"].needs and curves.version.endswith(runner.LEARNING_CURVE_VERSION)
    plan["kfold"]["learning_curve"] = False
    assert "learning_curves" not in {s.id for s in runner.build_steps(plan, [f"output_root={tmp_path.as_posix()}"])[0]}


def test_kfold_saves_every_fold_score_on_folds_shared_by_all_feature_sets(tmp_path, metadata_csv, monkeypatch):
    kfold = _load_script("kfold_cv")
    meta = pd.read_csv(metadata_csv)
    cohort = tmp_path / "cohort.csv"
    meta[["ID"]].to_csv(cohort, index=False)
    out = tmp_path / "cv"
    config = str(SCRIPTS.parent / "configs" / "kfold_cv.yaml")
    overrides = [f"data.metadata_csv={metadata_csv.as_posix()}", f"data.cohort_csv={cohort.as_posix()}",
                 f"data.splits_file={(tmp_path / 'none.json').as_posix()}", "tasks=[grade]", "cv.folds=2",
                 "cv.repeats=2", "cv.models=[logreg]", "cv.feature_sets=[age+sex, clinical]"]
    monkeypatch.setattr(sys, "argv", ["kfold_cv.py", "--config", config, "--out-dir", str(out), "--set", *overrides])
    kfold.main()
    folds = pd.read_csv(out / "cv_folds.csv")
    summary = pd.read_csv(out / "cv_results.csv")
    assert set(folds["features"]) == {"-", "age+sex", "clinical"} and sorted(folds["fold"].unique()) == [0, 1, 2, 3]
    auc = folds[folds["metric"] == "macro_auc"]
    sizes = auc.groupby("fold")[["n_train", "n_test"]].nunique()
    assert (sizes == 1).all().all() and (auc["n_train"] + auc["n_test"] == len(meta)).all()   # same folds everywhere
    lr = auc[(auc["features"] == "clinical") & (auc["model"] == "logreg")]["value"]
    row = summary[(summary["features"] == "clinical") & (summary["metric"] == "macro_auc")].iloc[0]
    assert np.isclose(lr.mean(), row["mean"])

    oof = pd.read_csv(out / "oof_predictions.csv")                   # one out-of-fold row per patient and model
    assert oof.groupby(["features", "model"]).size().eq(len(meta)).all()
    assert np.allclose(oof[["p_0", "p_1", "p_2"]].sum(1), 1.0)

    curve_dir = tmp_path / "curves"
    monkeypatch.setattr(sys, "argv", [
        "kfold_cv.py", "--config", config, "--learning-curve", "--out-dir", str(curve_dir), "--set", *overrides,
        "learning_curve.fractions=[0.5, 1.0]", "learning_curve.repeats=1", "learning_curve.models=[logreg]",
        "learning_curve.feature_sets=[age+sex, clinical]"])
    kfold.main()
    curves = pd.read_csv(curve_dir / "learning_curves.csv")
    assert set(curves["fraction"]) == {0.5, 1.0} and curves["macro_auc"].notna().all()
    sizes = curves.groupby("fraction")["n_train"].mean()
    assert sizes[0.5] < sizes[1.0] == len(meta) / 2                  # 2 folds: half the cohort trains
    assert curves.groupby(["fraction", "fold"])["n_train"].nunique().eq(1).all()   # same subsample for every set


def _write_predictions(path: Path, y: np.ndarray, prob: np.ndarray, task: str = "diagnosis") -> None:
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"patient_id": [f"P{i}" for i in range(len(y))], f"y_{task}": y,
                       **{f"p_{task}_{k}": prob[:, k] for k in range(prob.shape[1])}})
    df.to_csv(path / "predictions_test.csv", index=False)


def test_aggregate_tests_the_prespecified_comparisons_and_the_oracle(tmp_path):
    agg = _load_script("aggregate_results")
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2, 3], 10)
    sharp = np.exp(np.eye(4)[y] * 2.5 + rng.normal(0, 1, (40, 4)))
    blurred = np.exp(np.eye(4)[y] * 0.5 + rng.normal(0, 1, (40, 4)))
    probs = {"cls_image_only": blurred, "cls_image_clinical": sharp}
    for exp, p in probs.items():
        _write_predictions(tmp_path / f"{exp}_s42", y, p / p.sum(1, keepdims=True))
    _write_predictions(tmp_path / "cls_image_only_s42" / "oracle", y, sharp / sharp.sum(1, keepdims=True))
    plan = {"classification": {"seeds": [42], "experiments": {e: {} for e in probs}}}

    tests = agg.classification_tests(plan, tmp_path)
    assert len(tests) == 1 and tests.iloc[0]["comparison"] == "clinical token"   # the only pair that exists
    assert float(tests.iloc[0]["p"]) < 0.05 and tests.iloc[0]["test"] == "paired bootstrap"
    oracle = agg.classification_oracle_table(plan, tmp_path)
    assert list(oracle["experiment"]) == ["cls_image_only"]
    assert float(oracle.iloc[0]["auc_gt_masks"]) > float(oracle.iloc[0]["auc_predicted_masks"])

    rows = []
    for out, shift in (("kfold_cv", 0.0), ("kfold_cv_oracle", 0.03)):
        for fs, base in (("clinical", 0.70), ("clinical+radiomics", 0.80)):
            rows += [{"out": out, "task": "grade", "features": fs, "model": "xgboost", "metric": "macro_auc", "fold": i,
                      "value": base + shift + rng.normal(0, 0.02), "n_train": 128, "n_test": 32} for i in range(50)]
    for out, g in pd.DataFrame(rows).groupby("out"):
        (tmp_path / out).mkdir()
        g.drop(columns="out").to_csv(tmp_path / out / "cv_folds.csv", index=False)
    kt = agg.kfold_tests(tmp_path)
    assert list(kt["comparison"]) == ["radiomics added to clinical"] and float(kt.iloc[0]["p"]) < 0.001
    ko = agg.kfold_oracle_table(tmp_path)
    assert set(ko["features"]) == {"clinical", "clinical+radiomics"} and (ko["delta_auc"].str.startswith("+")).all()


def test_aggregate_volume_calibration_curves_and_radiomics_stability(tmp_path):
    agg = _load_script("aggregate_results")
    rng = np.random.default_rng(1)
    volume = np.geomspace(1, 200, 30)
    for model, noise in (("seg_a", 0.02), ("seg_b", 0.1)):
        run = tmp_path / f"{model}_s42"
        run.mkdir()
        pd.DataFrame({"patient_id": [f"P{i}" for i in range(30)], "gt_volume_ml": volume,
                      "dice": np.clip(0.5 + 0.2 * np.log10(volume) + rng.normal(0, noise, 30), 0, 1)}).to_csv(
            run / "per_case_segmentation_test.csv", index=False)
    plan = {"segmentation": {"seeds": [42], "models": {"seg_a": {}, "seg_b": {}}},
            "classification": {"seeds": [42], "experiments": {}}}
    table, edges = agg.segmentation_volume_table(plan, tmp_path)
    assert list(table["model"]) == ["seg_a", "seg_b"] and (table["spearman_rho"].astype(float) > 0.5).all()
    assert edges[0] < edges[1] and float(table.iloc[0]["median_dice_small"]) < float(table.iloc[0]["median_dice_large"])

    (tmp_path / "kfold_cv").mkdir()
    y = np.repeat([0, 1], 20)
    pd.DataFrame({"task": "lgg_hgg", "features": "age+sex", "model": "xgboost", "patient_id": range(40), "y": y,
                  "p_0": 1 - y * 0.8 - 0.1, "p_1": y * 0.8 + 0.1}).to_csv(tmp_path / "kfold_cv" / "oof_predictions.csv",
                                                                         index=False)
    e2e, late = agg.calibration_tables(plan, tmp_path)
    assert e2e is None and np.isclose(float(late.iloc[0]["brier"]), 2 * 0.1 ** 2)   # two-class Brier, doubled

    (tmp_path / "learning_curves").mkdir()
    pd.DataFrame([{"task": "grade", "features": "radiomics", "model": "xgboost", "fraction": f, "n_train": 100 * f,
                   "fold": k, "macro_auc": 0.6 + 0.1 * f} for f in (0.5, 1.0) for k in range(3)]).to_csv(
        tmp_path / "learning_curves" / "learning_curves.csv", index=False)
    curves = agg.learning_curve_table(tmp_path)
    assert list(curves.columns[-2:]) == ["50%", "100%"] and curves.iloc[0]["100%"].startswith("0.700")

    ids = [f"UCSF-PDGM-{i:04d}" for i in range(1, 21)]
    gt = pd.DataFrame({"patient_id": ids, "original_shape_Volume": rng.uniform(10, 100, 20),
                       "original_glcm_Contrast": rng.uniform(0, 1, 20)})
    pred = gt.assign(original_shape_Volume=gt["original_shape_Volume"] * 1.01,       # robust
                     original_glcm_Contrast=rng.uniform(0, 1, 20))                   # unrelated
    for folder, df, name in (("radiomics", pred, "radiomics_pred.csv"), ("radiomics_label", gt, "radiomics_label.csv")):
        (tmp_path / folder).mkdir()
        df.to_csv(tmp_path / folder / name, index=False)
    (tmp_path / "splits_seed42.json").write_text(json.dumps({"classification": {"train": ids[:5], "val": ids[5:12],
                                                                                 "test": ids[12:]}}))
    per_feature, summary, patients = agg.radiomics_stability(tmp_path)
    assert patients.startswith("classification validation") and set(per_feature["n"]) == {15}
    icc = per_feature.set_index("feature")["icc"]
    assert icc["original_shape_Volume"] > 0.95 and icc["original_glcm_Contrast"] < 0.5
    assert set(summary["feature_class"]) == {"shape", "glcm"}


def test_the_gradcam_panel_is_its_own_explain_step(tmp_path):
    """The paper's Grad-CAM figure runs after a finished explain step without redoing the per-patient maps."""
    import yaml

    runner = _load_script("run_paper")
    plan = yaml.safe_load((runner.REPO / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    steps = {s.id: s for s in runner.build_steps(plan, [f"output_root={tmp_path.as_posix()}"])[0]}
    panel, explain = steps["gradcam_panel:cls_image_only_s42"], steps["explain:cls_image_only_s42"]
    assert panel.stage == explain.stage == "explain" and panel.args == [*explain.args, "--panel-only"]
    assert panel.version == f"{explain.version}, panel {runner.PANEL_VERSION}"   # reruns alone when the panel changes
    assert panel.needs == explain.needs and panel.run_dir is None
