import numpy as np
import pytest
import torch

from glioma_mtl.config import apply_overrides
from glioma_mtl.data.transforms import CropAroundMaskd
from glioma_mtl.evaluation import bootstrap_ci, classification_metrics
from glioma_mtl.utils import load_weights, save_checkpoint


def test_crop_around_mask_contains_tumour_near_border():
    image = np.zeros((1, 100, 100, 80), dtype=np.float32)
    mask = np.zeros_like(image)
    mask[0, 90:98, 5:12, 70:78] = 1                    # tumour in a corner
    out = CropAroundMaskd(["image", "mask"], "mask", (64, 64, 64))({"image": image, "mask": mask})
    assert out["mask"].shape[1:] == (64, 64, 64)
    assert float(out["mask"].sum()) == float(mask.sum())   # whole tumour kept


def test_crop_around_empty_mask_uses_centre():
    data = {"image": np.arange(1000.0).reshape(1, 10, 10, 10), "mask": np.zeros((1, 10, 10, 10))}
    out = CropAroundMaskd(["image", "mask"], "mask", (4, 4, 4))(data)
    assert out["image"].shape[1:] == (4, 4, 4)


def test_load_weights_fails_loudly_on_mismatch(tmp_path):
    src = torch.nn.Linear(4, 2)
    save_checkpoint(tmp_path / "ok.pt", src, epoch=1)
    dst = torch.nn.Linear(4, 2)
    assert load_weights(dst, tmp_path / "ok.pt") == 1.0
    assert torch.equal(dst.weight, src.weight)
    torch.save({"model_state": {"other.weight": torch.zeros(1)}}, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError, match="0/2"):
        load_weights(torch.nn.Linear(4, 2), tmp_path / "bad.pt")
    torch.save(src.state_dict(), tmp_path / "bare.pt")    # bare state dict (e.g. thesis encoder weights)
    assert load_weights(torch.nn.Linear(4, 2), tmp_path / "bare.pt") == 1.0


def test_overrides_parse_yaml_values():
    cfg = apply_overrides({"train": {"epochs": 1}}, ["train.epochs=5", "model.stages=[2, 3]", "train.lr=1e-4"])
    assert cfg["train"]["epochs"] == 5 and cfg["model"]["stages"] == [2, 3] and cfg["train"]["lr"] == pytest.approx(1e-4)


def test_bootstrap_and_classification_metrics():
    point, low, high = bootstrap_ci(np.linspace(0.8, 0.9, 50))
    assert low < point < high
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 40)
    prob = rng.dirichlet(np.ones(3), 40)
    prob[np.arange(40), y] += 1.0
    table, cm = classification_metrics(y, prob / prob.sum(1, keepdims=True), ["2", "3", "4"])
    acc = table.set_index("metric").loc["accuracy"]
    assert acc["value"] == 1.0 and cm.trace() == 40
