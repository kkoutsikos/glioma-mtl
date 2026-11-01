"""Seeding, devices, logging and checkpoint I/O."""
from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

LOG = logging.getLogger("glioma_mtl")


def setup_logging(out_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(out_dir) / "log.txt", encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers, force=True)
    return LOG


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        from monai.utils import set_determinism
        set_determinism(seed=seed)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def save_checkpoint(path: str | Path, model: torch.nn.Module, **extra: Any) -> None:
    """Save ``model_state`` plus any extra objects (optimizer, scheduler, scaler, epoch, metrics, config).

    Objects with a ``state_dict`` method are stored via their state dict.
    """
    state: dict[str, Any] = {"model_state": model.state_dict()}
    for key, value in extra.items():
        state[key] = value.state_dict() if hasattr(value, "state_dict") else value
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, Mapping) and len(obj) > 0 and all(torch.is_tensor(v) for v in obj.values())


def load_weights(module: torch.nn.Module, path: str | Path, key: str | None = "model_state",
                 strip_prefix: str | None = None, min_fraction: float = 0.9) -> float:
    """Load matching weights into ``module`` and fail loudly if too few match.

    Handles both full checkpoints (``{key: state_dict, ...}``) and bare state dicts. Keys are matched
    by name *and* shape. Raises ``RuntimeError`` when less than ``min_fraction`` of the module's
    parameters are found, which is the failure mode that ``strict=False`` hides silently.

    Returns the fraction of target keys that were loaded.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if key and isinstance(ckpt, Mapping) and key in ckpt:
        state = ckpt[key]
    elif _looks_like_state_dict(ckpt):
        state = ckpt
    else:
        found = list(ckpt.keys()) if isinstance(ckpt, Mapping) else type(ckpt).__name__
        raise RuntimeError(f"{path}: no '{key}' entry and not a bare state dict (found: {found})")

    if strip_prefix:
        state = {k[len(strip_prefix):]: v for k, v in state.items() if k.startswith(strip_prefix)}

    target = module.state_dict()
    matched = {k: v for k, v in state.items() if k in target and target[k].shape == v.shape}
    fraction = len(matched) / max(len(target), 1)
    if fraction < min_fraction:
        sample_missing = [k for k in target if k not in matched][:5]
        raise RuntimeError(
            f"Only {len(matched)}/{len(target)} ({fraction:.0%}) parameters of "
            f"{module.__class__.__name__} found in {path}. First missing: {sample_missing}")
    module.load_state_dict(matched, strict=False)
    LOG.info("Loaded %d/%d tensors (%.0f%%) into %s from %s", len(matched), len(target),
             100 * fraction, module.__class__.__name__, path)
    return fraction


def format_row(row: Mapping[str, Any]) -> str:
    """Compact one-line rendering of an epoch row (4 significant digits, so small LRs stay visible)."""
    return " | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())


class CSVLogger:
    """Append one row per epoch. Columns are the union of all rows logged so far (also across resumes).

    A key that first appears in a later row (e.g. ``val_dice`` when validation runs every few epochs)
    adds a column and the file is rewritten, so nothing is dropped and old rows get an empty cell.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> tuple[list[str], list[dict]]:
        if not self.path.exists():
            return [], []
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader.fieldnames or []), list(reader)

    def log(self, row: Mapping[str, Any]) -> None:
        fields, rows = self._read()
        new = [k for k in row if k not in fields]
        if new or not fields:
            fields += new
            rows.append(dict(row))
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        else:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
