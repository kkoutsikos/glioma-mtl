"""YAML configuration with single inheritance and command-line overrides.

A config file may contain ``inherit: other.yaml`` (path relative to the file). Values are deep-merged,
the child wins. Overrides use dotted keys, e.g. ``--set train.epochs=10 model.name=segresnet``; the value
is parsed as YAML, so ``true``, ``1e-4`` and ``[2, 3, 4]`` work.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def deep_update(base: dict, other: dict) -> dict:
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _load_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = deep_update(_load_file(path.parent / parent), cfg)
    return cfg


def _parse_value(raw: str) -> Any:
    value = yaml.safe_load(raw)
    if isinstance(value, str):  # YAML 1.1 reads "1e-4" (no dot) as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.strip().split(".")
        for name in parents:
            node = node.setdefault(name, {})
        node[leaf] = _parse_value(raw)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    return apply_overrides(_load_file(Path(path)), overrides)


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for name in dotted.split("."):
        if not isinstance(node, dict) or name not in node:
            return default
        node = node[name]
    return node


def base_argparser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config values, e.g. --set train.epochs=5 data.cache=none")
    return parser
