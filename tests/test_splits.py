import pytest

from glioma_mtl.data.metadata import load_metadata
from glioma_mtl.data.splits import load_splits, make_splits, save_splits, validate_splits


def _splits(meta, seed=42):
    cohort = list(meta.index[:60])
    return make_splits(meta, cohort, list(meta.index), seed=seed)


def test_shared_split_invariants(metadata_csv):
    meta = load_metadata(metadata_csv)
    s = _splits(meta)
    cls, seg = s["classification"], s["segmentation"]
    assert sum(len(v) for v in cls.values()) == 60
    assert set(cls["test"]) <= set(seg["test"])
    assert not set(cls["test"]) & (set(s["ssl"]["train"]) | set(s["ssl"]["val"]))
    assert "test" not in s["ssl"]


def test_split_is_deterministic_and_stratified(metadata_csv):
    meta = load_metadata(metadata_csv)
    assert _splits(meta, 1) == _splits(meta, 1)
    assert _splits(meta, 1) != _splits(meta, 2)
    test_grades = set(meta.loc[_splits(meta)["classification"]["test"], "Grade"])
    assert test_grades == {2, 3, 4}


def test_validate_rejects_leaky_split(metadata_csv, tmp_path):
    meta = load_metadata(metadata_csv)
    s = _splits(meta)
    leaked = s["classification"]["test"][0]
    s["segmentation"]["train"].append(leaked)
    with pytest.raises(ValueError):
        validate_splits(s)


def test_roundtrip(metadata_csv, tmp_path):
    meta = load_metadata(metadata_csv)
    s = _splits(meta)
    save_splits(s, tmp_path / "s.json")
    assert load_splits(tmp_path / "s.json") == s
