"""Dataset / DataLoader construction with optional caching."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset, list_data_collate


def make_dataset(records: Sequence[dict], transform, cache: str = "none", cache_dir: str | Path | None = None,
                 name: str = "data"):
    """``cache``: ``none`` | ``memory`` (CacheDataset) | ``disk`` (PersistentDataset under cache_dir/name).

    Disk caches are keyed by the record contents, so a changed transform needs a fresh ``cache_dir``.
    """
    if cache == "none":
        return Dataset(records, transform)
    if cache == "memory":
        return CacheDataset(records, transform, cache_rate=1.0, num_workers=None)
    if cache == "disk":
        if cache_dir is None:
            raise ValueError("data.cache=disk needs data.cache_dir")
        return PersistentDataset(records, transform, cache_dir=Path(cache_dir) / name)
    raise ValueError(f"Unknown cache mode {cache!r}")


def make_loader(dataset, cfg: Mapping, train: bool, batch_size: int | None = None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size or (cfg["batch_size"] if train else 1),
        shuffle=train,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=list_data_collate,  # flattens the num_samples crops of RandCropByPosNegLabeld
        drop_last=train and cfg.get("drop_last", False),
    )
