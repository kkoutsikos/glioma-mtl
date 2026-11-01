import numpy as np
import pytest

from glioma_mtl.data.metadata import (ClinicalFeaturizer, check_leakage, class_weights, load_metadata,
                                      normalize_patient_id)


@pytest.mark.parametrize("raw", ["UCSF-PDGM-004", "UCSF-PDGM-0004", "UCSF-PDGM-0004_T2.nii.gz",
                                 "ucsf_pdgm_4_tumor_segmentation.nii.gz", "4"])
def test_normalize_patient_id(raw):
    assert normalize_patient_id(raw) == "UCSF-PDGM-0004"


def test_load_metadata_labels(metadata_csv):
    meta = load_metadata(metadata_csv)
    assert meta.index[0] == "UCSF-PDGM-0004"
    assert set(meta["grade_label"]) == {0, 1, 2}
    assert set(meta["IDH_Status"].dropna()) == {"mutant", "wildtype"}
    assert meta["MGMT_Index"].isna().any()          # 'unknown' became NaN, not a string


def test_followup_scans_are_dropped(tmp_path):
    import pandas as pd
    from conftest import synthetic_metadata

    df = synthetic_metadata(8)
    fu = df.iloc[[2]].assign(ID=df.iloc[2]["ID"].replace("PDGM-", "PDGM-0") + "_FU016d")
    path = tmp_path / "meta_fu.csv"
    pd.concat([df, fu]).to_csv(path, index=False)
    assert len(load_metadata(path)) == 8
    with pytest.raises(ValueError, match="Duplicate"):
        load_metadata(path, include_followup=True)


def test_leakage_guard():
    check_leakage(["grade"], ["demographics", "history"])
    with pytest.raises(ValueError, match="molecular"):
        check_leakage(["diagnosis"], ["demographics", "molecular"])
    check_leakage(["diagnosis"], ["molecular"], allow_leakage=True)


def test_featurizer_fits_on_train_only(metadata_csv):
    meta = load_metadata(metadata_csv)
    train, test = meta.iloc[:80], meta.iloc[80:]
    feat = ClinicalFeaturizer(["demographics"]).fit(train)
    x_train, x_test = feat.transform(train), feat.transform(test)
    assert x_train.shape[1] == x_test.shape[1] == feat.dim
    assert np.isfinite(x_test).all()
    age_col = feat.feature_names.index("num__Age")
    assert abs(x_train[:, age_col].mean()) < 1e-5            # standardised with train statistics
    assert abs(x_test[:, age_col].mean()) > 1e-5             # ... not refit on test


def test_class_weights_ignore_missing():
    w = class_weights([0, 0, 1, -1, 2, 2, 2], 3)
    assert w.shape == (3,) and w[0] > w[2]
