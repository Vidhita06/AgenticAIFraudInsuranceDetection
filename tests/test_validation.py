import pandas as pd
import pytest

from src.data.load import CLAIM_FIELDS
from src.data.validation import (
    RULES, expected_holder_bin, has_blocking_failure, run_rules, validate_claim,
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
