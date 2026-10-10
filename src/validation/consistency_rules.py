"""Timeline and logic consistency checks (V04-V12).

Each check takes the claims DataFrame and the rule's `params` from
config/policy_rules.yaml and returns a boolean Series (True = passed).
Rule ids, severities and explanation text live in the YAML; the rule engine
is in src/validation/validators.py.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import pandas as pd

from src.ml.data import MONTHS

AGE_MISSING_SENTINEL = 0

# Deterministic Age -> AgeOfPolicyHolder mapping found in the original Kaggle data
# (holds for 100% of real and synthetic rows). The bin labels do NOT literally
# contain Age (e.g. Age 26-35 -> "31 to 35"); Age = 0 always maps to "16 to 17".
AGE_TO_HOLDER_BIN: list[tuple[int, str]] = [
    (17, "18 to 20"), (20, "21 to 25"), (25, "26 to 30"), (35, "31 to 35"),
    (45, "36 to 40"), (55, "41 to 50"), (65, "51 to 65"), (10_000, "over 65"),
]

# Order of the Days_Policy_* bins, used for ordering checks.
POLICY_DAYS_ORDER = {"none": 0, "1 to 7": 1, "8 to 15": 2, "15 to 30": 3, "more than 30": 4}
# Lower bound in years of each AgeOfVehicle bin.
VEHICLE_AGE_YEARS = {
    "new": 0, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5, "6 years": 6,
    "7 years": 7, "more than 7": 8,
}
_MONTH_INDEX = {m: i for i, m in enumerate(MONTHS)}

Check = Callable[[pd.DataFrame, Mapping[str, Any]], pd.Series]


def expected_holder_bin(age: pd.Series) -> pd.Series:
    out = pd.Series("over 65", index=age.index, dtype="string")
    for upper, label in reversed(AGE_TO_HOLDER_BIN):
        out[age <= upper] = label
    out[age == AGE_MISSING_SENTINEL] = "16 to 17"
    out[age.isna()] = pd.NA
    return out


def month_lag(df: pd.DataFrame) -> pd.Series:
    """Months from accident to claim, assuming the claim follows the accident (0-11)."""
    acc = df["Month"].map(_MONTH_INDEX)
    clm = df["MonthClaimed"].map(_MONTH_INDEX)
    return (clm - acc) % 12


def _valid_age(df: pd.DataFrame) -> pd.Series:
    age = pd.to_numeric(df["Age"], errors="coerce")
    return age.notna() & (age != AGE_MISSING_SENTINEL)


def _passed(bad: pd.Series) -> pd.Series:
    return ~bad.fillna(False).astype(bool)


def age_holder_bin(df, params):
    expected = expected_holder_bin(pd.to_numeric(df["Age"], errors="coerce"))
    return (expected == df["AgeOfPolicyHolder"]).fillna(True).astype(bool)


def _policytype_mismatch(df):
    combined = df["VehicleCategory"] + " - " + df["BasePolicy"]
    return (combined != df["PolicyType"]).fillna(False).astype(bool)


def _is_known_policytype_alias(df, params):
    alias = params["alias"]
    return (
        (df["PolicyType"] == alias["PolicyType"])
        & (df["VehicleCategory"] == alias["VehicleCategory"])
        & (df["BasePolicy"] == alias["BasePolicy"])
    ).fillna(False).astype(bool)


def policytype_consistent(df, params):
    # Mismatches other than the known alias are genuine inconsistencies.
    return ~(_policytype_mismatch(df) & ~_is_known_policytype_alias(df, params))


def policytype_alias(df, params):
    return ~_is_known_policytype_alias(df, params)


def claim_after_accident(df, params):
    # Only detectable when both dates fall in the same month: a claimed week earlier
    # than the accident week means the claim precedes the accident (or lags ~12 months).
    return _passed((month_lag(df) == 0) & (df["WeekOfMonthClaimed"] < df["WeekOfMonth"]))


def reporting_delay(df, params):
    return _passed(month_lag(df) >= params["min_months"])


def early_policy_incident(df, params):
    return _passed(df["Days_Policy_Accident"].isin(params["accident_bins"])
                   | df["Days_Policy_Claim"].isin(params["claim_bins"]))


def policy_days_order(df, params):
    acc = df["Days_Policy_Accident"].map(POLICY_DAYS_ORDER)
    clm = df["Days_Policy_Claim"].map(POLICY_DAYS_ORDER)
    return _passed(clm < acc)


def policy_days_vs_lag(df, params):
    # A claim filed within 30 days of policy start cannot be 2+ months after the accident.
    within_30 = df["Days_Policy_Claim"].isin(params["claim_bins_within_30_days"])
    return _passed(within_30 & (month_lag(df) >= params["min_months"]))


def vehicle_vs_driver_age(df, params):
    age = pd.to_numeric(df["Age"], errors="coerce")
    veh = df["AgeOfVehicle"].map(VEHICLE_AGE_YEARS)
    driving_years = age - params["min_driving_age"]
    return _passed(_valid_age(df) & (age <= params["max_age"]) & (veh > driving_years))


def young_holder_history(df, params):
    age = pd.to_numeric(df["Age"], errors="coerce")
    young = _valid_age(df) & (age <= params["max_age"])
    heavy = (df["PastNumberOfClaims"].isin(params["past_claims"])
             | df["NumberOfCars"].isin(params["number_of_cars"]))
    return _passed(young & heavy)


CHECKS: dict[str, Check] = {
    "V04_AGE_HOLDER_BIN": age_holder_bin,
    "V05_POLICYTYPE_CONSISTENT": policytype_consistent,
    "V05B_POLICYTYPE_KNOWN_ALIAS": policytype_alias,
    "V06_CLAIM_AFTER_ACCIDENT": claim_after_accident,
    "V07_REPORTING_DELAY": reporting_delay,
    "V08_EARLY_POLICY_INCIDENT": early_policy_incident,
    "V09_POLICY_DAYS_ORDER": policy_days_order,
    "V10_POLICY_DAYS_VS_LAG": policy_days_vs_lag,
    "V11_VEHICLE_VS_DRIVER_AGE": vehicle_vs_driver_age,
    "V12_YOUNG_HOLDER_HISTORY": young_holder_history,
}
