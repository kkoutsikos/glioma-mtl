from __future__ import annotations

import torch  # noqa: F401  (load before pandas, see glioma_mtl/__init__.py)
import numpy as np
import pandas as pd
import pytest

DIAG = ["Astrocytoma, IDH-mutant", "Astrocytoma, IDH-wildtype", "Glioblastoma, IDH-wildtype",
        "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted"]


def synthetic_metadata(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Raw-format UCSF-PDGM metadata with plausible grade/diagnosis/IDH combinations."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        kind = i % 4
        grade, diag, idh, codel = [(2, DIAG[0], "IDH1 p.R132H", "intact"), (3, DIAG[1], "wildtype", "intact"),
                                   (4, DIAG[2], "wildtype", "unknown"), (2, DIAG[3], "IDH1 p.R132C", "co-deletion")][kind]
        rows.append({
            "ID": f"UCSF-PDGM-{i + 4:03d}", "Sex": rng.choice(["M", "F"]), "Age at MRI": int(rng.integers(20, 80)),
            "WHO CNS Grade": grade, "Final pathologic diagnosis (WHO 2021)": diag,
            "MGMT status": rng.choice(["positive", "negative", "unknown"]),
            "MGMT index": rng.choice(["unknown", "3", "10"]), "1p/19q": codel, "IDH": idh,
            "1-dead 0-alive": 0, "OS": 100.0, "EOR": rng.choice(["GTR", "STR", "biopsy"]),
            "Biopsy prior to imaging": rng.choice(["Yes", "No"]),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def metadata_csv(tmp_path):
    path = tmp_path / "meta.csv"
    synthetic_metadata().to_csv(path, index=False)
    return path
