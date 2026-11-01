"""MONAI transform pipelines for SSL, segmentation and classification.

Differences from the thesis notebooks (all configurable):

* ``intensity.kind: zscore`` (per-volume, brain voxels only) by default instead of a fixed [0, 3000]
  window, because raw MRI intensities are not comparable across scans. Use ``kind: range`` to reproduce.
* masks are binarised (``> 0``), so the original multi-label UCSF-PDGM masks also work;
* classification crops are **tumour-centred** for training (``pos=1, neg=0``) and evaluation
  (:class:`CropAroundMaskd`) instead of 50% background crops / a fixed centre crop.
"""
from __future__ import annotations

from typing import Hashable, Mapping, Sequence

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import (
    AsDiscreted,
    Compose,
    CopyItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandSpatialCropd,
    RandStdShiftIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SpatialCrop,
    SpatialPadd,
    Spacingd,
)

IMAGE, LABEL, PRED = "image", "label", "pred_mask"


def _intensity(cfg: Mapping, keys: KeysCollection = IMAGE) -> list:
    kind = cfg.get("kind", "zscore")
    if kind == "zscore":
        return [NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True)]
    if kind == "range":
        return [ScaleIntensityRanged(keys=keys, a_min=cfg.get("a_min", 0), a_max=cfg.get("a_max", 3000),
                                     b_min=0.0, b_max=1.0, clip=True)]
    if kind == "percentile":
        return [ScaleIntensityRangePercentilesd(keys=keys, lower=cfg.get("lower", 0.5),
                                                upper=cfg.get("upper", 99.5), b_min=0.0, b_max=1.0,
                                                clip=True, channel_wise=True)]
    raise ValueError(f"Unknown intensity.kind {kind!r}")


def _load(image_keys: Sequence[str], mask_keys: Sequence[str], spacing: Sequence[float]) -> list:
    keys = list(image_keys) + list(mask_keys)
    modes = ["bilinear"] * len(image_keys) + ["nearest"] * len(mask_keys)
    ops = [
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS", labels=None),
        Spacingd(keys=keys, pixdim=tuple(spacing), mode=modes),
    ]
    if mask_keys:
        ops.append(AsDiscreted(keys=list(mask_keys), threshold=0.5))  # whole-tumour mask {0, 1}
    return ops


def _spatial_aug(keys: Sequence[str], modes: Sequence[str], cfg: Mapping) -> list:
    return [
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandAffined(keys=keys, prob=cfg.get("affine_prob", 0.2), rotate_range=(0.05,) * 3,
                    scale_range=(0.1,) * 3, mode=list(modes), padding_mode="zeros"),
    ]


def _intensity_aug(cfg: Mapping) -> list:
    return [
        RandGaussianNoised(keys=IMAGE, prob=0.2, mean=0.0, std=0.01),
        RandStdShiftIntensityd(keys=IMAGE, factors=0.1, prob=0.3),
        RandScaleIntensityd(keys=IMAGE, factors=0.1, prob=0.3),
        RandBiasFieldd(keys=IMAGE, prob=cfg.get("bias_field_prob", 0.3)),
    ]


# --------------------------------------------------------------------------------------------- SSL
SSL_VIEWS = ("view_rot", "view_con1", "view_con2")


def ssl_transforms(cfg: Mapping) -> Compose:
    roi = tuple(cfg["roi_size"])
    return Compose([
        *_load([IMAGE], [], cfg["spacing"]),
        # Crop to the head on raw intensities (> 0) *before* z-scoring makes half the brain negative.
        CropForegroundd(keys=IMAGE, source_key=IMAGE, allow_smaller=True),
        *_intensity(cfg["intensity"]),
        SpatialPadd(keys=IMAGE, spatial_size=roi),
        RandSpatialCropd(keys=IMAGE, roi_size=roi, random_size=False),
        CopyItemsd(keys=IMAGE, times=3, names=list(SSL_VIEWS)),
        # Flips only on the contrastive views. Flipping both in-plane axes of the rotation view equals a 180°
        # rotation, so the rotation label would be ambiguous (k vs k+2): the thesis' rotation task sat at chance.
        RandFlipd(keys=["view_con1", "view_con2"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandGaussianNoised(keys=list(SSL_VIEWS), prob=0.3, mean=0.0, std=0.02),
        RandAdjustContrastd(keys=list(SSL_VIEWS), prob=0.3, gamma=(0.7, 1.3)),
        # Only the contrastive views get small random rotations; the rotation-prediction view must not.
        RandAffined(keys="view_con1", prob=0.5, rotate_range=(0.1,) * 3, scale_range=(0.1,) * 3, mode="bilinear"),
        RandAffined(keys="view_con2", prob=0.5, rotate_range=(0.1,) * 3, scale_range=(0.1,) * 3, mode="bilinear"),
        EnsureTyped(keys=list(SSL_VIEWS), dtype=torch.float32, track_meta=False),
    ])


# ------------------------------------------------------------------------------------ segmentation
def seg_train_transforms(cfg: Mapping) -> Compose:
    roi = tuple(cfg["roi_size"])
    keys = [IMAGE, LABEL]
    return Compose([
        *_load([IMAGE], [LABEL], cfg["spacing"]),
        *_intensity(cfg["intensity"]),
        SpatialPadd(keys=keys, spatial_size=roi),
        RandCropByPosNegLabeld(keys=keys, label_key=LABEL, spatial_size=roi, pos=1.0, neg=1.0,
                               num_samples=cfg.get("num_samples", 2), image_key=IMAGE, image_threshold=0.0),
        *_spatial_aug(keys, ["bilinear", "nearest"], cfg.get("augment", {})),
        *_intensity_aug(cfg.get("augment", {})),
        EnsureTyped(keys=keys, dtype=torch.float32, track_meta=False),
    ])


def seg_eval_transforms(cfg: Mapping, with_label: bool = True) -> Compose:
    masks = [LABEL] if with_label else []
    # Keep meta data here so predictions can be saved back in the original space.
    return Compose([
        *_load([IMAGE], masks, cfg["spacing"]),
        *_intensity(cfg["intensity"]),
        EnsureTyped(keys=[IMAGE, *masks], dtype=torch.float32),
    ])


# ---------------------------------------------------------------------------------- classification
class CropAroundMaskd(MapTransform):
    """Fixed-size crop centred on the bounding box of ``mask_key``; pads when the volume is smaller.

    The window is shifted to stay inside the volume, so the tumour is never cut by the border unless it
    is larger than ``roi_size``. An empty mask falls back to the volume centre.
    """

    def __init__(self, keys: KeysCollection, mask_key: Hashable, roi_size: Sequence[int],
                 allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.roi = np.asarray(roi_size, dtype=int)

    def window(self, mask: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        m = np.asarray(mask.detach().cpu() if torch.is_tensor(mask) else mask)[0] > 0
        shape = np.asarray(m.shape)
        if m.any():
            idx = np.argwhere(m)
            center = (idx.min(0) + idx.max(0)) // 2
        else:
            center = shape // 2
        start = np.clip(center - self.roi // 2, 0, np.maximum(shape - self.roi, 0))
        return start, start + self.roi

    def __call__(self, data: Mapping) -> dict:
        d = dict(data)
        start, end = self.window(d[self.mask_key])
        cropper = SpatialCrop(roi_start=start.tolist(), roi_end=end.tolist())
        for key in self.key_iterator(d):
            d[key] = cropper(d[key])
        return d


def cls_train_transforms(cfg: Mapping) -> Compose:
    roi = tuple(cfg["roi_size"])
    keys = [IMAGE, LABEL]
    crop = cfg.get("crop", {})
    return Compose([
        *_load([IMAGE], [LABEL], cfg["spacing"]),
        *_intensity(cfg["intensity"]),
        SpatialPadd(keys=keys, spatial_size=roi),
        RandCropByPosNegLabeld(keys=keys, label_key=LABEL, spatial_size=roi, pos=crop.get("pos", 1.0),
                               neg=crop.get("neg", 0.0), num_samples=cfg.get("num_samples", 2),
                               image_key=IMAGE, image_threshold=0.0),
        *_spatial_aug(keys, ["bilinear", "nearest"], cfg.get("augment", {})),
        *_intensity_aug(cfg.get("augment", {})),
        EnsureTyped(keys=keys, dtype=torch.float32, track_meta=False),
    ])


def cls_eval_transforms(cfg: Mapping, crop_mask_key: str = LABEL) -> Compose:
    """``crop_mask_key='label'`` = oracle localisation (GT mask); ``'pred_mask'`` = predicted mask."""
    roi = tuple(cfg["roi_size"])
    masks = [LABEL] if crop_mask_key == LABEL else [LABEL, PRED]
    keys = [IMAGE, *masks]
    return Compose([
        *_load([IMAGE], masks, cfg["spacing"]),
        *_intensity(cfg["intensity"]),
        SpatialPadd(keys=keys, spatial_size=roi),
        CropAroundMaskd(keys=keys, mask_key=crop_mask_key, roi_size=roi),
        EnsureTyped(keys=keys, dtype=torch.float32, track_meta=False),
    ])
