"""Loading the raw and processed claim files with explicit dtypes.

Input files are never modified; derived data goes to data/interim or data/features.
Categorical columns are read as plain strings so out-of-domain values (e.g. the
'0' sentinel in DayOfWeekClaimed) survive loading and can be caught by validation.
"""
from __future__ import annotations

import pandas as pd

from src.config import path_for

TARGET = "FraudFound_P"
ID_COL = "PolicyNumber"
ORIGIN_COL = "is_synthetic"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Allowed values per categorical column. Range columns are listed in natural order.
CATEGORIES: dict[str, list[str]] = {
    "Month": MONTHS,
    "DayOfWeek": WEEKDAYS,
    "Make": [
        "Accura", "BMW", "Chevrolet", "Dodge", "Ferrari", "Ford", "Honda", "Jaguar", "Lexus",
        "Mazda", "Mecedes", "Mercury", "Nisson", "Pontiac", "Porche", "Saab", "Saturn",
        "Toyota", "VW",
    ],
    "AccidentArea": ["Rural", "Urban"],
    "DayOfWeekClaimed": WEEKDAYS,
    "MonthClaimed": MONTHS,
    "Sex": ["Female", "Male"],
    "MaritalStatus": ["Divorced", "Married", "Single", "Widow"],
    "Fault": ["Policy Holder", "Third Party"],
    "PolicyType": [
        f"{v} - {b}"
        for v in ["Sedan", "Sport", "Utility"]
        for b in ["All Perils", "Collision", "Liability"]
    ],
    "VehicleCategory": ["Sedan", "Sport", "Utility"],
    "VehiclePrice": [
        "less than 20000", "20000 to 29000", "30000 to 39000", "40000 to 59000",
        "60000 to 69000", "more than 69000",
    ],
    "Days_Policy_Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days_Policy_Claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    "AgeOfVehicle": [
        "new", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7",
    ],
    "AgeOfPolicyHolder": [
        "16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35", "36 to 40", "41 to 50",
        "51 to 65", "over 65",
    ],
    "PoliceReportFiled": ["No", "Yes"],
    "WitnessPresent": ["No", "Yes"],
    "AgentType": ["External", "Internal"],
    "NumberOfSuppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "AddressChange_Claim": [
        "under 6 months", "1 year", "2 to 3 years", "4 to 8 years", "no change",
    ],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
    "BasePolicy": ["All Perils", "Collision", "Liability"],
}

# Columns whose category list above is a natural order (used for ordinal encoding).
# AddressChange_Claim is ordered by time since the address change ("no change" = never).
ORDINAL_COLUMNS = [
    "VehiclePrice", "Days_Policy_Accident", "Days_Policy_Claim", "PastNumberOfClaims",
    "AgeOfVehicle", "AgeOfPolicyHolder", "NumberOfSuppliments", "AddressChange_Claim",
    "NumberOfCars",
]

# Numeric columns and their allowed domains (None = open-ended).
INTEGER_DOMAINS: dict[str, set[int] | None] = {
    "WeekOfMonth": {1, 2, 3, 4, 5},
    "WeekOfMonthClaimed": {1, 2, 3, 4, 5},
    "Age": None,  # 0 = missing sentinel, otherwise 16..80 observed
    "FraudFound_P": {0, 1},
    "PolicyNumber": None,
    "RepNumber": set(range(1, 17)),
    "Deductible": {300, 400, 500, 700},
    "DriverRating": {1, 2, 3, 4},
    "Year": {1994, 1995, 1996},
}

RAW_COLUMNS = [
    "Month", "WeekOfMonth", "DayOfWeek", "Make", "AccidentArea", "DayOfWeekClaimed",
    "MonthClaimed", "WeekOfMonthClaimed", "Sex", "MaritalStatus", "Age", "Fault",
    "PolicyType", "VehicleCategory", "VehiclePrice", "FraudFound_P", "PolicyNumber",
    "RepNumber", "Deductible", "DriverRating", "Days_Policy_Accident", "Days_Policy_Claim",
    "PastNumberOfClaims", "AgeOfVehicle", "AgeOfPolicyHolder", "PoliceReportFiled",
    "WitnessPresent", "AgentType", "NumberOfSuppliments", "AddressChange_Claim",
    "NumberOfCars", "Year", "BasePolicy",
]
CATEGORICAL_COLUMNS = [c for c in RAW_COLUMNS if c in CATEGORIES]
INTEGER_COLUMNS = [c for c in RAW_COLUMNS if c in INTEGER_DOMAINS]
# Claim fields (everything except the label and the identifier).
CLAIM_FIELDS = [c for c in RAW_COLUMNS if c not in (TARGET, ID_COL)]

DTYPES: dict[str, str] = {c: "string" for c in CATEGORICAL_COLUMNS}
DTYPES.update({c: "int64" for c in INTEGER_COLUMNS})


def _read(path, extra_dtypes: dict[str, str] | None = None) -> pd.DataFrame:
    dtypes = dict(DTYPES)
    if extra_dtypes:
        dtypes.update(extra_dtypes)
    # utf-8-sig strips the BOM present in the raw Kaggle file (harmless otherwise).
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=dtypes)
    expected = RAW_COLUMNS + list(extra_dtypes or {})
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")
    return df


def load_raw() -> pd.DataFrame:
    """Original Kaggle data (15,420 x 33)."""
    return _read(path_for("raw"))


def load_augmented_train() -> pd.DataFrame:
    """Real-train + synthetic rows (300,000 x 34, with is_synthetic)."""
    return _read(path_for("augmented_train"), {ORIGIN_COL: "int8"})


def load_real_train() -> pd.DataFrame:
    df = load_augmented_train()
    return df[df[ORIGIN_COL] == 0].drop(columns=ORIGIN_COL).reset_index(drop=True)


def load_synthetic_train() -> pd.DataFrame:
    df = load_augmented_train()
    return df[df[ORIGIN_COL] == 1].drop(columns=ORIGIN_COL).reset_index(drop=True)


def load_real_validation() -> pd.DataFrame:
    return _read(path_for("real_validation"))


def load_real_test() -> pd.DataFrame:
    """Held-out test set. Only to be opened for the final evaluation (Phase 6)
    and for Phase 1 integrity checks that never look at model performance."""
    return _read(path_for("real_test"))


def load_splits(include_test: bool = False) -> dict[str, pd.DataFrame]:
    """Named splits: real_train, synthetic, real_validation (and real_test if asked)."""
    aug = load_augmented_train()
    splits = {
        "real_train": aug[aug[ORIGIN_COL] == 0].drop(columns=ORIGIN_COL).reset_index(drop=True),
        "synthetic": aug[aug[ORIGIN_COL] == 1].drop(columns=ORIGIN_COL).reset_index(drop=True),
        "real_validation": load_real_validation(),
    }
    if include_test:
        splits["real_test"] = load_real_test()
    return splits
