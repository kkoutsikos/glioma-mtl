"""CPU forward/backward checks on small inputs (feature_size 12, 64^3 crops)."""
import pytest
import torch
from monai.networks.nets import SwinUNETR

from glioma_mtl.losses import MultiTaskLoss, SSLLoss
from glioma_mtl.models import (HybridSSLPretrainer, MultiTaskGliomaNet, build_segmentation_model,
                               masked_mean, swin_stage_channels, swin_unetr_forward)

FS, ROI = 12, 64


def test_swin_forward_matches_monai():
    torch.manual_seed(0)
    model = SwinUNETR(in_channels=1, out_channels=1, feature_size=FS).eval()
    x = torch.randn(1, 1, ROI, ROI, ROI)
    with torch.no_grad():
        out = swin_unetr_forward(model, x)
        ref = model(x)
    assert torch.allclose(out.logits, ref, atol=1e-5)
    assert [h.shape[1] for h in out.hidden] == swin_stage_channels(FS)
    assert out.decoder.shape[1] == FS


def test_masked_mean_uses_mask_and_handles_empty():
    feat = torch.zeros(2, 3, 4, 4, 4)
    feat[:, :, :2] = 10.0
    mask = torch.zeros(2, 1, 8, 8, 8)
    mask[0, :, :4] = 1.0                      # sample 0: tumour where feat == 10; sample 1: empty mask
    pooled = masked_mean(feat, mask)
    assert torch.allclose(pooled[0], torch.full((3,), 10.0))
    assert torch.allclose(pooled[1], torch.full((3,), 5.0))   # falls back to the global mean


@pytest.mark.parametrize("fusion", ["transformer", "cross_attention", "concat"])
@pytest.mark.parametrize("stages,mask_mode", [([4], "soft"), ([1, 2, 3], "hard"), ([2, 3, 4], "none")])
def test_multitask_net_forward_backward(fusion, stages, mask_mode):
    tasks = {"grade": 3, "diagnosis": 4}
    model = MultiTaskGliomaNet(tasks, {"clinical": 5}, feature_size=FS, use_checkpoint=False,
                               encoder_stages=stages, mask_mode=mask_mode, fusion=fusion, modality_dropout=0.5)
    x = torch.randn(2, 1, ROI, ROI, ROI)
    out = model(x, {"clinical": torch.randn(2, 5)})
    assert out["logits"]["grade"].shape == (2, 3) and out["logits"]["diagnosis"].shape == (2, 4)
    assert len(out["aux"]["grade"]) == len(stages) + 1

    loss_fn = MultiTaskLoss({"grade": 1.0, "diagnosis": 0.5}, supcon_weight=0.1)
    targets = {"grade": torch.tensor([0, 2]), "diagnosis": torch.tensor([1, -1])}
    loss, parts = loss_fn(out, targets, (torch.rand(2, 1, ROI, ROI, ROI) > 0.7).float())
    loss.backward()
    assert torch.isfinite(loss) and "seg" in parts
    for proj in model.image_proj:                         # every image token receives gradient
        assert proj[0].weight.grad is not None and proj[0].weight.grad.abs().sum() > 0


def test_image_only_and_frozen_backbone():
    model = MultiTaskGliomaNet({"idh": 2}, None, feature_size=FS, use_checkpoint=False, freeze_backbone=True)
    model.train()
    assert not model.backbone.training
    out = model(torch.randn(1, 1, ROI, ROI, ROI))
    out["logits"]["idh"].sum().backward()
    assert all(p.grad is None for p in model.backbone.parameters())


def test_ssl_pretrainer_patch_mask_and_loss():
    model = HybridSSLPretrainer(feature_size=FS, mask_patch_size=16, use_checkpoint=False)
    x = torch.randn(2, 1, ROI, ROI, ROI)
    masked, mask = model.random_mask(x)
    assert torch.equal(masked * mask, torch.zeros_like(x))
    assert mask.unique().tolist() in ([0.0, 1.0], [0.0], [1.0])
    x_rot, k = model.rotate(x)
    out = model(masked, x, x.flip(-1), x_rot)
    assert out["recon"].shape == out["light_recon"].shape == x.shape
    loss, logs = SSLLoss({"mae": 1, "rotation": 0.5, "contrastive": 1, "light": 0.1})(out, x, mask, k)
    loss.backward()
    assert {"mae", "rotation", "contrastive", "light", "rot_acc"} <= logs.keys()


@pytest.mark.parametrize("name", ["swin_unetr", "segresnet", "unet"])
def test_segmentation_factory(name):
    cfg = {"name": name, "feature_size": FS, "use_checkpoint": False, "channels": [8, 16, 32, 64], "strides": [2, 2, 2]}
    model = build_segmentation_model(cfg)
    assert model(torch.randn(1, 1, ROI, ROI, ROI)).shape == (1, 1, ROI, ROI, ROI)
