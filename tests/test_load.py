"""Integration checks on the real data files (skipped if the data is absent)."""
import pytest

from src.config import path_for
from src.data.load import CLAIM_FIELDS, RAW_COLUMNS, TARGET, load_raw, load_splits
from src.data.validation import run_rules

pytestmark = pytest.mark.skipif(not path_for("raw").exists(), reason="data files not present")


@pytest.fixture(scope="module")
def splits():
    return load_splits()


def test_raw_shape_and_columns():
    raw = load_raw()
    assert raw.shape == (15420, 33)
    assert list(raw.columns) == RAW_COLUMNS


def test_split_sizes(splits):
    assert len(splits["real_train"]) == 10793
    assert len(splits["synthetic"]) == 289207
    assert len(splits["real_validation"]) == 2313
    assert splits["real_train"][TARGET].sum() == 646


def test_policynumber_separates_origin(splits):
    assert splits["real_train"]["PolicyNumber"].max() <= 15420
    assert splits["synthetic"]["PolicyNumber"].min() == 15421


def test_no_nulls_and_no_blocking_domain_errors(splits):
    for df in splits.values():
        assert not df[CLAIM_FIELDS].isna().any().any()
        res = run_rules(df)
        assert res["V01_REQUIRED_FIELDS"].all()
        assert res["V02_DOMAIN_VALUES"].all()
        assert res["V04_AGE_HOLDER_BIN"].all()


def test_raw_invalid_row_is_caught():
    raw = load_raw()
    res = run_rules(raw)
    bad = raw.loc[~res["V02_DOMAIN_VALUES"], "PolicyNumber"].tolist()
    assert bad == [1517]
