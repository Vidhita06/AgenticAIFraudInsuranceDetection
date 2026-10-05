import pandas as pd
import pytest

from src.config import path_for
from src.ml.data import CLAIM_FIELDS, RAW_COLUMNS, TARGET, load_raw, load_splits
from src.validation.consistency_rules import CHECKS as CONSISTENCY_CHECKS
from src.validation.consistency_rules import expected_holder_bin
from src.validation.validators import (
    CHECKS, RULES, has_blocking_failure, load_policy_rules, run_rules, validate_claim,
)

VALID_CLAIM = {
    "Month": "Oct", "WeekOfMonth": 2, "DayOfWeek": "Friday", "Make": "Honda",
    "AccidentArea": "Urban", "DayOfWeekClaimed": "Monday", "MonthClaimed": "Oct",
    "WeekOfMonthClaimed": 3, "Sex": "Male", "MaritalStatus": "Married", "Age": 47,
    "Fault": "Policy Holder", "PolicyType": "Sedan - Collision", "VehicleCategory": "Sedan",
    "VehiclePrice": "20000 to 29000", "RepNumber": 7, "Deductible": 400,
    "DriverRating": 3, "Days_Policy_Accident": "more than 30",
    "Days_Policy_Claim": "more than 30", "PastNumberOfClaims": "1", "AgeOfVehicle": "7 years",
    "AgeOfPolicyHolder": "41 to 50", "PoliceReportFiled": "No", "WitnessPresent": "No",
    "AgentType": "External", "NumberOfSuppliments": "none", "AddressChange_Claim": "no change",
    "NumberOfCars": "1 vehicle", "Year": 1994, "BasePolicy": "Collision",
}


def failed(claim):
    return {r["rule_id"] for r in validate_claim(claim) if not r["passed"]}


def test_valid_claim_passes_all_rules():
    results = validate_claim(VALID_CLAIM)
    assert len(results) == len(RULES)
    assert all(r["passed"] for r in results)
    assert not has_blocking_failure(results)
    for r in results:
        assert set(r) == {"rule_id", "passed", "severity", "message"}
        assert r["severity"] in {"blocking", "warning", "info"}


def test_missing_field_is_blocking():
    claim = {k: v for k, v in VALID_CLAIM.items() if k != "Fault"}
    results = validate_claim(claim)
    assert "V01_REQUIRED_FIELDS" in failed(claim)
    assert has_blocking_failure(results)
    msg = next(r["message"] for r in results if r["rule_id"] == "V01_REQUIRED_FIELDS")
    assert "Fault" in msg


@pytest.mark.parametrize("field,value", [
    ("MonthClaimed", "0"), ("DayOfWeekClaimed", "0"), ("Make", "Tesla"),
    ("Deductible", 250), ("Age", 7),
])
def test_out_of_domain_values(field, value):
    assert "V02_DOMAIN_VALUES" in failed({**VALID_CLAIM, field: value})


def test_age_zero_is_missing_and_blocking():
    claim = {**VALID_CLAIM, "Age": 0, "AgeOfPolicyHolder": "16 to 17"}
    assert failed(claim) == {"V03_AGE_MISSING"}
    assert has_blocking_failure(validate_claim(claim))


def test_age_holder_bin_uses_empirical_mapping():
    # Age 30 maps to "31 to 35" in the reference data (shifted bands).
    assert "V04_AGE_HOLDER_BIN" not in failed({**VALID_CLAIM, "Age": 30, "AgeOfPolicyHolder": "31 to 35"})
    assert "V04_AGE_HOLDER_BIN" in failed({**VALID_CLAIM, "Age": 30, "AgeOfPolicyHolder": "26 to 30"})
    bins = expected_holder_bin(pd.Series([0, 16, 18, 21, 26, 35, 36, 46, 56, 66, 80]))
    assert bins.tolist() == ["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35",
                             "31 to 35", "36 to 40", "41 to 50", "51 to 65", "over 65", "over 65"]


def test_policytype_alias_is_info_and_other_mismatch_is_warning():
    alias = {**VALID_CLAIM, "PolicyType": "Sedan - Liability", "VehicleCategory": "Sport",
             "BasePolicy": "Liability"}
    assert failed(alias) == {"V05B_POLICYTYPE_KNOWN_ALIAS"}
    other = {**VALID_CLAIM, "PolicyType": "Utility - Collision"}
    assert failed(other) == {"V05_POLICYTYPE_CONSISTENT"}


def test_claim_before_accident_same_month():
    claim = {**VALID_CLAIM, "WeekOfMonth": 4, "WeekOfMonthClaimed": 2}
    assert "V06_CLAIM_AFTER_ACCIDENT" in failed(claim)


def test_year_wraparound_is_not_a_violation():
    claim = {**VALID_CLAIM, "Month": "Dec", "WeekOfMonth": 5, "MonthClaimed": "Jan",
             "WeekOfMonthClaimed": 1}
    assert failed(claim) == set()


def test_long_reporting_delay():
    claim = {**VALID_CLAIM, "Month": "Jan", "MonthClaimed": "Aug"}
    assert failed(claim) == {"V07_REPORTING_DELAY"}


@pytest.mark.parametrize("acc,clm", [("none", "more than 30"), ("1 to 7", "8 to 15")])
def test_early_policy_incident(acc, clm):
    claim = {**VALID_CLAIM, "Days_Policy_Accident": acc, "Days_Policy_Claim": clm}
    assert "V08_EARLY_POLICY_INCIDENT" in failed(claim)


def test_policy_days_order():
    claim = {**VALID_CLAIM, "Days_Policy_Accident": "more than 30", "Days_Policy_Claim": "15 to 30"}
    assert "V09_POLICY_DAYS_ORDER" in failed(claim)


def test_policy_days_vs_lag():
    claim = {**VALID_CLAIM, "Days_Policy_Accident": "8 to 15", "Days_Policy_Claim": "15 to 30",
             "Month": "Jan", "MonthClaimed": "Apr"}
    assert "V10_POLICY_DAYS_VS_LAG" in failed(claim)


def test_young_holder_rules():
    young = {**VALID_CLAIM, "Age": 18, "AgeOfPolicyHolder": "21 to 25"}
    assert "V11_VEHICLE_VS_DRIVER_AGE" in failed(young)  # 7-year-old vehicle, 2 driving years
    assert "V11_VEHICLE_VS_DRIVER_AGE" not in failed({**young, "AgeOfVehicle": "new"})
    assert "V12_YOUNG_HOLDER_HISTORY" in failed({**young, "PastNumberOfClaims": "more than 4"})


def test_run_rules_matches_validate_claim():
    claims = [VALID_CLAIM, {**VALID_CLAIM, "Age": 0, "AgeOfPolicyHolder": "16 to 17"},
              {**VALID_CLAIM, "Days_Policy_Accident": "none"}]
    frame = run_rules(pd.DataFrame(claims))
    for i, claim in enumerate(claims):
        single = {r["rule_id"]: r["passed"] for r in validate_claim(claim)}
        assert frame.iloc[i].to_dict() == single


def test_all_claim_fields_covered():
    assert set(VALID_CLAIM) == set(CLAIM_FIELDS)


def test_policy_rules_yaml_matches_code():
    spec = load_policy_rules()
    ids = [r["id"] for r in spec["rules"] if r["type"] in ("validation", "consistency")]
    assert ids == [r.rule_id for r in RULES]
    assert set(ids) == set(CHECKS)
    for r in spec["rules"]:
        expected = "consistency" if r["id"] in CONSISTENCY_CHECKS else "validation"
        assert r["type"] == expected, r["id"]


# ---------------------------------------------------------------------------
# Data-loading checks on the real files (skipped if the data is absent)
# ---------------------------------------------------------------------------
needs_data = pytest.mark.skipif(not path_for("raw").exists(), reason="data files not present")


@pytest.fixture(scope="module")
def splits():
    return load_splits()

@needs_data
def test_raw_shape_and_columns():
    raw = load_raw()
    assert raw.shape == (15420, 33)
    assert list(raw.columns) == RAW_COLUMNS


@needs_data
def test_split_sizes(splits):
    assert len(splits["real_train"]) == 10793
    assert len(splits["synthetic"]) == 289207
    assert len(splits["real_validation"]) == 2313
    assert splits["real_train"][TARGET].sum() == 646


@needs_data
def test_policynumber_separates_origin(splits):
    assert splits["real_train"]["PolicyNumber"].max() <= 15420
    assert splits["synthetic"]["PolicyNumber"].min() == 15421


@needs_data
def test_no_nulls_and_no_blocking_domain_errors(splits):
    for df in splits.values():
        assert not df[CLAIM_FIELDS].isna().any().any()
        res = run_rules(df)
        assert res["V01_REQUIRED_FIELDS"].all()
        assert res["V02_DOMAIN_VALUES"].all()
        assert res["V04_AGE_HOLDER_BIN"].all()


@needs_data
def test_raw_invalid_row_is_caught():
    raw = load_raw()
    res = run_rules(raw)
    bad = raw.loc[~res["V02_DOMAIN_VALUES"], "PolicyNumber"].tolist()
    assert bad == [1517]


# ---------------------------------------------------------------------------
# Red flags and engineered features
# ---------------------------------------------------------------------------
from src.validation.feature_engineering import add_engineered_features, age_band, claim_lag_weeks  # noqa: E402
from src.validation.validators import evaluate_red_flags, load_red_flags, red_flag_frame  # noqa: E402


def test_red_flags_load_and_have_valid_columns():
    flags = load_red_flags()
    assert len(flags) >= 10
    assert len({f.rule_id for f in flags}) == len(flags)
    assert all(f.rule_id.startswith("RF") for f in flags)


def test_red_flag_fires_on_matching_claim():
    claim = {**VALID_CLAIM, "Fault": "Third Party", "Deductible": 500}
    fired = {r["rule_id"] for r in evaluate_red_flags(claim)}
    assert {"RF03_DEDUCTIBLE_500", "RF04_THIRD_PARTY_DEDUCTIBLE_500"} <= fired
    for r in evaluate_red_flags(claim):
        assert set(r) == {"rule_id", "severity", "condition", "message", "stats"}


def test_red_flags_quiet_on_low_risk_claim():
    claim = {**VALID_CLAIM, "Fault": "Third Party", "BasePolicy": "Liability",
             "VehicleCategory": "Sport", "PolicyType": "Sedan - Liability"}
    assert evaluate_red_flags(claim) == []


def test_red_flag_frame_matches_single_claim():
    claims = [VALID_CLAIM, {**VALID_CLAIM, "Age": 0, "AgeOfPolicyHolder": "16 to 17",
                            "BasePolicy": "All Perils", "PolicyType": "Sedan - All Perils"}]
    frame = red_flag_frame(pd.DataFrame(claims))
    for i, claim in enumerate(claims):
        assert set(frame.columns[frame.iloc[i]]) == {r["rule_id"] for r in evaluate_red_flags(claim)}
    assert frame.loc[1, "RF08_MISSING_AGE_ALL_PERILS"]


def test_engineered_features():
    df = pd.DataFrame([VALID_CLAIM,
                       {**VALID_CLAIM, "Age": 0, "Month": "Dec", "WeekOfMonth": 5, "MonthClaimed": "Jan",
                        "WeekOfMonthClaimed": 1, "DayOfWeek": "Sunday", "Days_Policy_Accident": "none",
                        "VehiclePrice": "more than 69000"}])
    out = add_engineered_features(df)
    assert out["age_missing"].tolist() == [0, 1]
    assert out["age_band"].tolist() == ["41-50", "missing"]
    assert out["claim_lag_months"].tolist() == [0, 1]
    assert out["claim_lag_weeks"].iloc[0] == 1
    assert 0 < out["claim_lag_weeks"].iloc[1] < 1          # Dec week 5 -> Jan week 1
    assert out["accident_weekend"].tolist() == [0, 1]
    assert out["early_policy_incident"].tolist() == [0, 1]
    assert out["price_extreme"].tolist() == [0, 1]
    assert len(df.columns) < len(out.columns)                  # input not modified
    assert age_band(pd.Series([16, 20, 21, 66])).tolist() == ["16-20", "16-20", "21-25", "66+"]
