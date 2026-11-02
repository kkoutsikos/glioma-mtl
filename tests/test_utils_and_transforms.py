import numpy as np
import pytest
import torch

from glioma_mtl.config import apply_overrides
from glioma_mtl.data.transforms import CropAroundMaskd, ssl_transforms
from glioma_mtl.evaluation import (bootstrap_ci, classification_metrics, evaluate_segmentation,
                                  keep_largest_component)
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


def test_ssl_rotation_view_keeps_orientation(tmp_path):
    """Two in-plane flips equal a 180° rotation, so the rotation view must never be flipped (thesis bug)."""
    nib = pytest.importorskip("nibabel")
    g = np.meshgrid(*[np.linspace(0, 1, 40)] * 3, indexing="ij")
    volume = (1 + g[0] + 2 * g[1] + 3 * g[2]).astype(np.float32)   # asymmetric along every axis
    nib.save(nib.Nifti1Image(volume, np.eye(4)), tmp_path / "x.nii.gz")
    tf = ssl_transforms({"spacing": [1.0, 1.0, 1.0], "roi_size": [32, 32, 32], "intensity": {"kind": "zscore"}})
    flipped_con = 0
    for seed in range(20):
        tf.set_random_state(seed)
        out = tf({"image": str(tmp_path / "x.nii.gz")})
        base = np.asarray(out["image"]).ravel()
        assert np.corrcoef(base, np.asarray(out["view_rot"]).ravel())[0, 1] > 0.95
        flipped_con += np.corrcoef(base, np.asarray(out["view_con1"]).ravel())[0, 1] < 0
    assert flipped_con > 0   # the contrastive views are still flipped


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


def test_keep_largest_component_drops_stray_blobs():
    mask = np.zeros((40, 40, 40), dtype=np.uint8)
    mask[5:15, 5:15, 5:15] = 1          # tumour
    mask[30:33, 30:33, 30:33] = 1       # false-positive blob on the far side
    mask[15, 15, 15] = 1                # touches the tumour diagonally: same component (26-connectivity)
    kept, n = keep_largest_component(mask)
    assert n == 2 and kept.dtype == mask.dtype
    assert kept.sum() == 10 ** 3 + 1 and kept[30:33, 30:33, 30:33].sum() == 0
    empty = np.zeros_like(mask)
    assert keep_largest_component(empty)[1] == 0 and not keep_largest_component(empty)[0].any()


def test_cleaned_predictions_are_what_is_scored_and_saved():
    class Fixed(torch.nn.Module):     # predicts the tumour plus a distant blob, whatever the input
        def forward(self, x):
            logits = torch.full_like(x, -10.0)
            logits[..., 4:12, 4:12, 4:12] = 10.0
            logits[..., 26:29, 26:29, 26:29] = 10.0
            return logits

    label = torch.zeros(1, 1, 32, 32, 32)
    label[..., 4:12, 4:12, 4:12] = 1
    batch = {"image": torch.zeros(1, 1, 32, 32, 32), "label": label, "patient_id": ["UCSF-PDGM-0001"]}
    saved = []
    raw = evaluate_segmentation(Fixed(), [batch], (32, 32, 32), torch.device("cpu"), amp=False)
    clean = evaluate_segmentation(Fixed(), [batch], (32, 32, 32), torch.device("cpu"), amp=False,
                                  on_prediction=lambda b, pred: saved.append(pred), largest_component=True)
    assert raw.loc[0, "dice"] < 1.0 and "components" not in raw
    assert clean.loc[0, "dice"] == pytest.approx(1.0) and clean.loc[0, "hd95"] == 0.0
    assert clean.loc[0, "components"] == 2 and clean.loc[0, "removed_ml"] == pytest.approx(27 / 1000)
    assert torch.equal(saved[0], label)
