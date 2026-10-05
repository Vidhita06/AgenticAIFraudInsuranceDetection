"""Derived claim features. The single source of truth for feature engineering:
EDA, training (src/ml/preprocess.py) and the agent tools all call these functions.

All features are row-wise functions of the raw claim fields, so they need no
fitting and cannot leak information across splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.consistency_rules import AGE_MISSING_SENTINEL, month_lag

WEEKS_PER_MONTH = 52 / 12
AGE_BAND_EDGES = [0, 20, 25, 30, 35, 40, 50, 65, 200]
AGE_BAND_LABELS = ["16-20", "21-25", "26-30", "31-35", "36-40", "41-50", "51-65", "66+"]
AGE_BAND_ORDER = ["missing"] + AGE_BAND_LABELS
WEEKEND = ("Saturday", "Sunday")


def age_band(age: pd.Series) -> pd.Series:
    """Policyholder age band from Age, with 'missing' for the Age = 0 sentinel."""
    age = pd.to_numeric(age, errors="coerce")
    band = pd.cut(age, AGE_BAND_EDGES, labels=AGE_BAND_LABELS, right=True).astype("string")
    return band.mask(age == AGE_MISSING_SENTINEL, "missing")


def claim_lag_weeks(df: pd.DataFrame) -> pd.Series:
    """Approximate weeks from accident to claim (month lag assumes year wrap-around).
    Negative values mean the claimed week precedes the accident week (rule V06)."""
    weeks = month_lag(df) * WEEKS_PER_MONTH + (df["WeekOfMonthClaimed"] - df["WeekOfMonth"])
    return weeks.astype(float)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the engineered columns appended."""
    out = df.copy()
    age = pd.to_numeric(out["Age"], errors="coerce")
    out["age_missing"] = (age == AGE_MISSING_SENTINEL).astype(int)
    out["age_band"] = age_band(age)
    out["claim_lag_months"] = month_lag(out).astype(float)
    out["claim_lag_weeks"] = claim_lag_weeks(out)
    out["accident_weekend"] = out["DayOfWeek"].isin(WEEKEND).astype(int)
    out["claim_weekend"] = out["DayOfWeekClaimed"].isin(WEEKEND).astype(int)
    out["early_policy_incident"] = (
        out["Days_Policy_Accident"].isin(["none", "1 to 7"]) | (out["Days_Policy_Claim"] == "none")
    ).astype(int)
    out["price_extreme"] = out["VehiclePrice"].isin(["less than 20000", "more than 69000"]).astype(int)
    out["policyholder_fault"] = (out["Fault"] == "Policy Holder").astype(int)
    out["liability_only"] = (out["BasePolicy"] == "Liability").astype(int)
    return out


ENGINEERED_FEATURES = [
    "age_missing", "age_band", "claim_lag_months", "claim_lag_weeks", "accident_weekend",
    "claim_weekend", "early_policy_incident", "price_extreme", "policyholder_fault",
    "liability_only",
]


def lag_bucket(lag_months: pd.Series) -> pd.Series:
    """Coarse claim-lag buckets for reporting: same month / 1 month / 2+ months."""
    return pd.Series(np.select([lag_months == 0, lag_months == 1, lag_months >= 2],
                               ["same month", "1 month", "2+ months"], default="unknown"),
                     index=lag_months.index, dtype="string")
