"""Preprocessing: the single source of truth for turning raw claims into model inputs.

    raw claim DataFrame
      -> ClaimFeatureBuilder   (stateless: engineered features, red-flag indicators,
                                consistency-rule count, ordinal codes; snapshots the
                                red-flag definitions at fit time)
      -> ColumnTransformer     (fitted on training data only)

`build_preprocessor(kind)` returns the sklearn Pipeline for one model family:
    "linear"    one-hot nominal, sin/cos cyclic, scaled ordinal/numeric  (LogReg)
    "tree"      one-hot nominal + raw ordinal codes/numeric              (RF, XGBoost, LightGBM)
    "catboost"  builder output only; nominal columns stay strings        (CatBoost native)
    "embedding" integer codes for categorical columns first, then scaled numeric (MLP)

The notebooks and the agent tools import this module; they never re-implement it, so a
pickled preprocessor references the same src.* classes on the B200 and locally.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import load_config
from src.ml.data import CATEGORIES, MONTHS, WEEKDAYS
from src.validation.feature_engineering import AGE_BAND_ORDER, add_engineered_features
from src.validation.validators import RULES, condition_mask, load_red_flags, run_rules

CYCLE_ORDER = {"Month": MONTHS, "DayOfWeek": WEEKDAYS}
ORDINAL_ORDER = {**{c: v for c, v in CATEGORIES.items()}, "age_band": AGE_BAND_ORDER}
CONSISTENCY_RULE_IDS = [r.rule_id for r in RULES if r.rule_type == "consistency"
                        and r.rule_id != "V05B_POLICYTYPE_KNOWN_ALIAS"]


def feature_config() -> dict[str, Any]:
    return load_config()["features"]


class ClaimFeatureBuilder(BaseEstimator, TransformerMixin):
    """Raw claims -> DataFrame of model-ready columns (no learned state except the
    red-flag snapshot, which is taken from config/policy_rules.yaml at fit time)."""

    def __init__(self, nominal=None, cyclic=None, ordinal=None, numeric=None, binary=None,
                 red_flags: bool = True, consistency_count: bool = True):
        self.nominal = nominal
        self.cyclic = cyclic
        self.ordinal = ordinal
        self.numeric = numeric
        self.binary = binary
        self.red_flags = red_flags
        self.consistency_count = consistency_count

    def fit(self, X: pd.DataFrame, y=None):
        cfg = feature_config()
        self.nominal_ = list(self.nominal if self.nominal is not None else cfg["nominal"])
        self.cyclic_ = list(self.cyclic if self.cyclic is not None else cfg["cyclic"])
        self.ordinal_ = list(self.ordinal if self.ordinal is not None else cfg["ordinal"])
        self.numeric_ = list(self.numeric if self.numeric is not None else cfg["numeric"])
        self.binary_ = list(self.binary if self.binary is not None else cfg["binary"])
        self.red_flag_defs_ = ([{"id": f.rule_id, "when": dict(f.when),
                                 "weight": 2 if f.severity == "warning" else 1}
                                for f in load_red_flags()] if self.red_flags else [])
        self.rf_columns_ = [f"rf_{d['id'].split('_')[0].lower()}" for d in self.red_flag_defs_]
        extra = list(self.rf_columns_)
        if self.red_flags:
            extra.append("red_flag_score")
        if self.consistency_count:
            extra.append("consistency_failures")
        self.flag_columns_ = self.binary_ + extra
        self.feature_names_out_ = (self.nominal_ + self.cyclic_ + self.ordinal_ + self.numeric_
                                   + self.flag_columns_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = add_engineered_features(pd.DataFrame(X).reset_index(drop=True))
        out = pd.DataFrame(index=df.index)
        for c in self.nominal_ + self.cyclic_:
            out[c] = df[c].astype(object).where(df[c].notna(), "missing").astype(str)
        for c in self.ordinal_:
            order = {v: i for i, v in enumerate(ORDINAL_ORDER[c])}
            out[c] = df[c].map(order).astype(float).fillna(-1.0)
        for c in self.numeric_:
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        for c in self.binary_:
            out[c] = df[c].astype(float)
        if self.red_flags:
            score = np.zeros(len(df))
            for d, col in zip(self.red_flag_defs_, self.rf_columns_):
                hit = condition_mask(df, d["when"]).to_numpy().astype(float)
                out[col] = hit
                score += d["weight"] * hit
            out["red_flag_score"] = score
        if self.consistency_count:
            res = run_rules(df)
            out["consistency_failures"] = (~res[CONSISTENCY_RULE_IDS]).sum(axis=1).astype(float)
        return out[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


class CyclicEncoder(BaseEstimator, TransformerMixin):
    """Month / weekday names -> sin and cos of their position in the cycle."""

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        parts = []
        for c in self.columns_:
            order = CYCLE_ORDER[c]
            angle = 2 * np.pi * X[c].map({v: i for i, v in enumerate(order)}).astype(float) / len(order)
            parts += [np.sin(angle).fillna(0.0), np.cos(angle).fillna(0.0)]
        return np.column_stack(parts) if parts else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_{t}" for c in self.columns_ for t in ("sin", "cos")], dtype=object)


class ColumnsExcept:
    """Picklable ColumnTransformer selector: every column not in `known`."""

    def __init__(self, known):
        self.known = list(known)

    def __call__(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in self.known]


def _onehot(min_freq: int) -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_freq,
                         sparse_output=False, dtype=np.float32)


def build_preprocessor(kind: str = "tree", **builder_kwargs) -> Pipeline:
    cfg = feature_config()
    builder = ClaimFeatureBuilder(red_flags=cfg.get("red_flags", True),
                                  consistency_count=cfg.get("consistency_count", True),
                                  **builder_kwargs)
    nominal, cyclic = cfg["nominal"], cfg["cyclic"]
    ordinal, numeric, binary = cfg["ordinal"], cfg["numeric"], cfg["binary"]
    min_freq = cfg.get("rare_category_min_frequency", 20)

    # Flag columns are only known after the builder is fitted, so select them by exclusion.
    rest = ColumnsExcept

    if kind == "catboost":
        return Pipeline([("features", builder)])
    if kind == "linear":
        ct = ColumnTransformer([
            ("nominal", _onehot(min_freq), nominal),
            ("cyclic", CyclicEncoder(), cyclic),
            ("scaled", StandardScaler(), ordinal + numeric),
            ("flags", "passthrough", rest(nominal + cyclic + ordinal + numeric)),
        ], verbose_feature_names_out=False, sparse_threshold=0)
    elif kind == "tree":
        ct = ColumnTransformer([
            ("nominal", _onehot(min_freq), nominal + cyclic),
            ("rest", "passthrough", rest(nominal + cyclic)),
        ], verbose_feature_names_out=False, sparse_threshold=0)
    elif kind == "embedding":
        ct = ColumnTransformer([
            ("codes", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                     encoded_missing_value=-1, min_frequency=min_freq,
                                     dtype=np.float32), nominal + cyclic),
            ("scaled", StandardScaler(), rest(nominal + cyclic)),
        ], verbose_feature_names_out=False, sparse_threshold=0)
    else:
        raise ValueError(f"unknown preprocessor kind {kind!r}")
    return Pipeline([("features", builder), ("encode", ct)])


def feature_names(pre: Pipeline) -> list[str]:
    return [str(n) for n in pre.get_feature_names_out()]


def embedding_layout(pre: Pipeline) -> tuple[int, list[int]]:
    """(number of leading categorical-code columns, cardinality of each) for the MLP.
    Codes run 0..card-1 plus -1 for unknown, so the MLP shifts them by +1."""
    enc = pre.named_steps["encode"].named_transformers_["codes"]
    cards = []
    for i, cats in enumerate(enc.categories_):
        n = len(cats)
        infreq = getattr(enc, "infrequent_categories_", None)
        if infreq is not None and infreq[i] is not None:
            n = n - len(infreq[i]) + 1
        cards.append(n + 1)   # +1 for the unknown/missing slot
    return len(cards), cards


def catboost_cat_features(pre: Pipeline) -> list[str]:
    b = pre.named_steps["features"]
    return b.nominal_ + b.cyclic_
