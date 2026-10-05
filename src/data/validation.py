"""Claim validation rules (synopsis Phase 2: business-rule checks).

Every rule is vectorised over a DataFrame of claims and returns a boolean Series
(True = passed). `validate_claim` applies the same rules to one claim and returns
`[{rule_id, passed, severity, message}]`, which becomes the policy-check tool for
the agent. Rules flag problems; they never modify or "fix" the data.

Severity levels:
    blocking: the claim cannot be assessed reliably (candidate for "Request More Information").
    warning:  inconsistent or suspicious fields an adjuster should look at.
    info:     unusual but plausible context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from src.data.load import CATEGORIES, CLAIM_FIELDS, INTEGER_DOMAINS, MONTHS

SEVERITIES = ("blocking", "warning", "info")

AGE_MISSING_SENTINEL = 0
AGE_MIN, AGE_MAX = 16, 100

# Deterministic Age -> AgeOfPolicyHolder mapping found in the original Kaggle data
# (holds for 100% of real and synthetic rows). The bin labels do NOT literally
# contain Age (e.g. Age 26-35 -> "31 to 35"); Age = 0 always maps to "16 to 17".
AGE_TO_HOLDER_BIN: list[tuple[int, str]] = [
    (17, "18 to 20"), (20, "21 to 25"), (25, "26 to 30"), (35, "31 to 35"),
    (45, "36 to 40"), (55, "41 to 50"), (65, "51 to 65"), (10_000, "over 65"),
]

# Lower bound in days of each Days_Policy_* bin, used for ordering checks.
POLICY_DAYS_ORDER = {"none": 0, "1 to 7": 1, "8 to 15": 2, "15 to 30": 3, "more than 30": 4}
# Lower bound in years of each AgeOfVehicle bin.
VEHICLE_AGE_YEARS = {
    "new": 0, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5, "6 years": 6,
    "7 years": 7, "more than 7": 8,
}
MIN_DRIVING_AGE = 16
# Every PolicyType mismatch in the original data (~32% of rows) is this single pattern:
# PolicyType "Sedan - Liability" recorded for a Sport vehicle on a Liability base policy.
KNOWN_POLICYTYPE_ALIAS = ("Sedan - Liability", "Sport", "Liability")
LONG_DELAY_MONTHS = 6
_MONTH_INDEX = {m: i for i, m in enumerate(MONTHS)}


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


# ---------------------------------------------------------------------------
# Rules. Each check takes the claims DataFrame and returns True where passed.
# ---------------------------------------------------------------------------

def _required_fields(df):
    return df[CLAIM_FIELDS].notna().all(axis=1)


def _domain_values(df):
    ok = pd.Series(True, index=df.index)
    for col, allowed in CATEGORIES.items():
        ok &= df[col].isna() | df[col].isin(allowed)
    for col, allowed in INTEGER_DOMAINS.items():
        if col not in df or allowed is None:
            continue
        ok &= df[col].isna() | df[col].isin(allowed)
    age = pd.to_numeric(df["Age"], errors="coerce")
    ok &= age.isna() | (age == AGE_MISSING_SENTINEL) | age.between(AGE_MIN, AGE_MAX)
    return ok


def _age_present(df):
    return df["Age"].isna() | (df["Age"] != AGE_MISSING_SENTINEL)


def _age_holder_bin(df):
    expected = expected_holder_bin(pd.to_numeric(df["Age"], errors="coerce"))
    return (expected == df["AgeOfPolicyHolder"]).fillna(True).astype(bool)


def _policytype_mismatch(df):
    combined = df["VehicleCategory"] + " - " + df["BasePolicy"]
    return (combined != df["PolicyType"]).fillna(False).astype(bool)


def _is_known_policytype_alias(df):
    return (
        (df["PolicyType"] == KNOWN_POLICYTYPE_ALIAS[0])
        & (df["VehicleCategory"] == KNOWN_POLICYTYPE_ALIAS[1])
        & (df["BasePolicy"] == KNOWN_POLICYTYPE_ALIAS[2])
    ).fillna(False).astype(bool)


def _policytype_consistent(df):
    # Mismatches other than the known alias are genuine inconsistencies.
    return ~(_policytype_mismatch(df) & ~_is_known_policytype_alias(df))


def _policytype_alias(df):
    return ~_is_known_policytype_alias(df)


def _claim_after_accident(df):
    # Only detectable when both dates fall in the same month: a claimed week earlier
    # than the accident week means the claim precedes the accident (or lags ~12 months).
    lag = month_lag(df)
    bad = (lag == 0) & (df["WeekOfMonthClaimed"] < df["WeekOfMonth"])
    return ~bad.fillna(False).astype(bool)


def _reporting_delay(df):
    return ~(month_lag(df) >= LONG_DELAY_MONTHS).fillna(False).astype(bool)


def _early_policy_incident(df):
    bad = df["Days_Policy_Accident"].isin(["none", "1 to 7"]) | (df["Days_Policy_Claim"] == "none")
    return ~bad.fillna(False).astype(bool)


def _policy_days_order(df):
    acc = df["Days_Policy_Accident"].map(POLICY_DAYS_ORDER)
    clm = df["Days_Policy_Claim"].map(POLICY_DAYS_ORDER)
    return ~(clm < acc).fillna(False).astype(bool)


def _policy_days_vs_lag(df):
    # Claim filed within 30 days of policy start cannot be >= 2 months after the accident.
    within_30 = df["Days_Policy_Claim"].isin(["none", "8 to 15", "15 to 30"])
    return ~(within_30 & (month_lag(df) >= 2)).fillna(False).astype(bool)


def _vehicle_vs_driver_age(df):
    age = pd.to_numeric(df["Age"], errors="coerce")
    veh = df["AgeOfVehicle"].map(VEHICLE_AGE_YEARS)
    driving_years = age - MIN_DRIVING_AGE
    bad = _valid_age(df) & (age < 21) & (veh > driving_years)
    return ~bad.fillna(False).astype(bool)


def _young_holder_history(df):
    age = pd.to_numeric(df["Age"], errors="coerce")
    young = _valid_age(df) & (age < 21)
    heavy = (df["PastNumberOfClaims"] == "more than 4") | df["NumberOfCars"].isin(
        ["5 to 8", "more than 8"]
    )
    return ~(young & heavy).fillna(False).astype(bool)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    severity: str
    description: str
    check: Callable[[pd.DataFrame], pd.Series]
    message: Callable[[Mapping[str, Any]], str]


def _fmt_domain(c: Mapping[str, Any]) -> str:
    bad = []
    for col, allowed in CATEGORIES.items():
        v = c.get(col)
        if v is not None and not pd.isna(v) and v not in allowed:
            bad.append(f"{col}={v!r}")
    for col, allowed in INTEGER_DOMAINS.items():
        v = c.get(col)
        if allowed is not None and v is not None and not pd.isna(v) and v not in allowed:
            bad.append(f"{col}={v!r}")
    age = c.get("Age")
    if age is not None and not pd.isna(age) and age != 0 and not AGE_MIN <= age <= AGE_MAX:
        bad.append(f"Age={age!r}")
    return "Values outside the known category sets: " + ", ".join(bad)


RULES: list[Rule] = [
    Rule("V01_REQUIRED_FIELDS", "blocking", "All claim fields are present.",
         _required_fields,
         lambda c: "Missing fields: " + ", ".join(
             f for f in CLAIM_FIELDS if c.get(f) is None or pd.isna(c.get(f)))),
    Rule("V02_DOMAIN_VALUES", "blocking", "Every field takes a known value.",
         _domain_values, _fmt_domain),
    Rule("V03_AGE_MISSING", "blocking", "Age is recorded (0 is the missing-age sentinel).",
         _age_present,
         lambda c: "Age is 0 (missing-age sentinel); policyholder age must be obtained."),
    Rule("V04_AGE_HOLDER_BIN", "warning",
         "AgeOfPolicyHolder matches the band implied by Age in the reference data.",
         _age_holder_bin,
         lambda c: f"Age {c.get('Age')} implies AgeOfPolicyHolder "
                   f"'{expected_holder_bin(pd.Series([c.get('Age')])).iloc[0]}', "
                   f"got '{c.get('AgeOfPolicyHolder')}'."),
    Rule("V05_POLICYTYPE_CONSISTENT", "warning",
         "PolicyType equals VehicleCategory + ' - ' + BasePolicy (or the known alias).",
         _policytype_consistent,
         lambda c: f"PolicyType '{c.get('PolicyType')}' disagrees with VehicleCategory "
                   f"'{c.get('VehicleCategory')}' and BasePolicy '{c.get('BasePolicy')}'."),
    Rule("V05B_POLICYTYPE_KNOWN_ALIAS", "info",
         "PolicyType is not the known 'Sedan - Liability' alias for a Sport/Liability claim.",
         _policytype_alias,
         lambda c: "PolicyType 'Sedan - Liability' recorded for a Sport vehicle on a Liability "
                   "policy: a known labelling quirk of the source data, not a claim error."),
    Rule("V06_CLAIM_AFTER_ACCIDENT", "warning",
         "Claim is not dated before the accident (same-month week check).",
         _claim_after_accident,
         lambda c: f"Claimed in week {c.get('WeekOfMonthClaimed')} of {c.get('MonthClaimed')} but "
                   f"accident in week {c.get('WeekOfMonth')} of {c.get('Month')}: claim precedes "
                   "the accident, or was filed ~12 months later."),
    Rule("V07_REPORTING_DELAY", "info",
         f"Claim filed less than {LONG_DELAY_MONTHS} months after the accident.",
         _reporting_delay,
         lambda c: f"Accident in {c.get('Month')}, claim in {c.get('MonthClaimed')}: reporting "
                   f"delay of {LONG_DELAY_MONTHS}+ months."),
    Rule("V08_EARLY_POLICY_INCIDENT", "warning",
         "Incident and claim are not at the very start of the coverage period.",
         _early_policy_incident,
         lambda c: f"Days_Policy_Accident='{c.get('Days_Policy_Accident')}', "
                   f"Days_Policy_Claim='{c.get('Days_Policy_Claim')}': incident at or before "
                   "the start of the policy; check the coverage period."),
    Rule("V09_POLICY_DAYS_ORDER", "warning",
         "Days_Policy_Claim is not shorter than Days_Policy_Accident.",
         _policy_days_order,
         lambda c: f"Days_Policy_Claim '{c.get('Days_Policy_Claim')}' is shorter than "
                   f"Days_Policy_Accident '{c.get('Days_Policy_Accident')}'."),
    Rule("V10_POLICY_DAYS_VS_LAG", "warning",
         "A claim filed within 30 days of policy start is not 2+ months after the accident.",
         _policy_days_vs_lag,
         lambda c: f"Claim filed '{c.get('Days_Policy_Claim')}' days into the policy, yet "
                   f"{c.get('Month')} -> {c.get('MonthClaimed')} implies a lag of 2+ months."),
    Rule("V11_VEHICLE_VS_DRIVER_AGE", "info",
         "For holders under 21, the vehicle is not older than their driving years.",
         _vehicle_vs_driver_age,
         lambda c: f"Holder aged {c.get('Age')} with a vehicle aged '{c.get('AgeOfVehicle')}': "
                   "older than the holder's driving years (possible, e.g. used car)."),
    Rule("V12_YOUNG_HOLDER_HISTORY", "info",
         "Holders under 21 do not report >4 past claims or 5+ cars.",
         _young_holder_history,
         lambda c: f"Holder aged {c.get('Age')} with PastNumberOfClaims "
                   f"'{c.get('PastNumberOfClaims')}' and NumberOfCars '{c.get('NumberOfCars')}'."),
]
RULES_BY_ID = {r.rule_id: r for r in RULES}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CLAIM_FIELDS if c not in df.columns]
    if missing:
        df = df.assign(**{c: pd.NA for c in missing})
    return df


def run_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame (rows x rule_id), True = passed."""
    df = _prepare(df)
    return pd.DataFrame({r.rule_id: r.check(df).astype(bool) for r in RULES}, index=df.index)


def validate_claim(claim: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Apply every rule to a single claim (a mapping of raw field -> value)."""
    row = pd.DataFrame([{f: claim.get(f) for f in CLAIM_FIELDS}])
    for col in INTEGER_DOMAINS:
        if col in row:
            row[col] = pd.to_numeric(row[col], errors="coerce")
    results = run_rules(row).iloc[0]
    out = []
    for rule in RULES:
        passed = bool(results[rule.rule_id])
        out.append({
            "rule_id": rule.rule_id,
            "passed": passed,
            "severity": rule.severity,
            "message": rule.description if passed else rule.message(claim),
        })
    return out


def has_blocking_failure(results: list[dict[str, Any]]) -> bool:
    return any(not r["passed"] and r["severity"] == "blocking" for r in results)


def rule_hit_summary(df: pd.DataFrame, target: str = "FraudFound_P") -> pd.DataFrame:
    """Per-rule violation counts and fraud rate among violators vs passers."""
    passed = run_rules(df)
    rows = []
    for rule in RULES:
        hit = ~passed[rule.rule_id]
        rec = {
            "rule_id": rule.rule_id,
            "severity": rule.severity,
            "n": len(df),
            "violations": int(hit.sum()),
            "violation_pct": 100 * hit.mean() if len(df) else np.nan,
        }
        if target in df:
            y = df[target]
            rec["fraud_rate_violators_pct"] = 100 * y[hit].mean() if hit.any() else np.nan
            rec["fraud_rate_passers_pct"] = 100 * y[~hit].mean() if (~hit).any() else np.nan
            rec["fraud_violators"] = int(y[hit].sum())
        rows.append(rec)
    return pd.DataFrame(rows)
