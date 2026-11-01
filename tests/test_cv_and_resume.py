import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

from glioma_mtl.engine.common import AmpStepper, BestTracker, make_optimizer, make_scheduler, save_last, try_resume
from glioma_mtl.features import extract_deep_features
from glioma_mtl.models import build_swin_unetr
from glioma_mtl.radiomics import TopK

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
