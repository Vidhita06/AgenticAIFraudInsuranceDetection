"""SHAP explanations for the fraud model.

`ModelExplainer` explains the *uncalibrated* base model in log-odds (calibration is a
monotone map, so the ranking of contributions is unchanged). Tree models use exact
TreeSHAP (CatBoost via its native ShapValues); the MLP uses a permutation explainer
over a small background sample.

models/shap_explainer.pkl stores only the explainer's metadata and background sample
(no model, no SHAP internals), so it unpickles across shap versions. If it is missing
or fails to load, `load_explainer` builds a fresh explainer from the model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.ml.preprocess import catboost_cat_features, feature_names


def base_models(model) -> list:
    """Unwrap CalibratedModel / SeedEnsemble down to the fitted estimators."""
    m = getattr(model, "base", model)
    return list(m.models) if type(m).__name__ == "SeedEnsemble" else [m]


class ModelExplainer:
    def __init__(self, preprocessor, model, background: np.ndarray | pd.DataFrame | None = None):
        self.feature_names = feature_names(preprocessor)
        self.kind = type(base_models(model)[0]).__name__
        self.cat_features = (catboost_cat_features(preprocessor)
                             if self.kind == "CatBoostClassifier" else None)
        self.background = background
        self.attach(model)

    def attach(self, model) -> "ModelExplainer":
        self._models = base_models(model)
        self._tree = None
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_models", None)
        state.pop("_tree", None)
        return state

    def _values_one(self, m, X) -> np.ndarray:
        import shap

        if self.kind == "CatBoostClassifier":
            from catboost import Pool
            sv = m.get_feature_importance(Pool(X, cat_features=self.cat_features), type="ShapValues")
            return np.asarray(sv)[:, :-1]
        if self.kind in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier"):
            ex = shap.TreeExplainer(m)
            sv = ex.shap_values(np.asarray(X, dtype=float))
            if isinstance(sv, list):          # older shap: one array per class
                sv = sv[1]
            sv = np.asarray(sv)
            return sv[..., 1] if sv.ndim == 3 else sv
        if self.kind == "LogisticRegression":
            Xa = np.asarray(X, dtype=float)
            bg = np.asarray(self.background, dtype=float).mean(axis=0) if self.background is not None else 0
            return (Xa - bg) * m.coef_[0]
        # MLP or anything else: model-agnostic permutation SHAP on the logit.
        bg = np.asarray(self.background if self.background is not None else X[:100], dtype=float)

        def f(z):
            p = np.clip(m.predict_proba(z)[:, 1], 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))

        ex = shap.PermutationExplainer(f, shap.maskers.Independent(bg[:100]))
        return np.asarray(ex(np.asarray(X, dtype=float), max_evals=2 * X.shape[1] + 1).values)

    def shap_values(self, X) -> np.ndarray:
        return np.mean([self._values_one(m, X) for m in self._models], axis=0)


def build_explainer(preprocessor, model, background_raw: pd.DataFrame | None = None,
                    n_background: int = 200, seed: int = 42) -> ModelExplainer:
    bg = None
    if background_raw is not None:
        sample = background_raw.sample(min(n_background, len(background_raw)), random_state=seed)
        bg = preprocessor.transform(sample)
    return ModelExplainer(preprocessor, model, bg)


def load_explainer(artifacts) -> ModelExplainer:
    import joblib

    path = artifacts.directory / "shap_explainer.pkl"
    try:
        ex = joblib.load(path)
        return ex.attach(artifacts.model)
    except Exception:  # noqa: BLE001  (missing file or version mismatch)
        return ModelExplainer(artifacts.preprocessor, artifacts.model)


# Plain-English names for model features.
PRETTY = {
    "Fault": "fault", "BasePolicy": "base policy", "VehicleCategory": "vehicle category",
    "AgentType": "agent type", "AccidentArea": "accident area", "Make": "vehicle make",
    "PoliceReportFiled": "police report filed", "WitnessPresent": "witness present",
    "Month": "accident month", "DayOfWeek": "accident weekday", "VehiclePrice": "vehicle price",
    "Days_Policy_Accident": "days from policy start to accident",
    "Days_Policy_Claim": "days from policy start to claim",
    "PastNumberOfClaims": "past claims", "AgeOfVehicle": "vehicle age",
    "NumberOfSuppliments": "number of supplements", "AddressChange_Claim": "address change before claim",
    "NumberOfCars": "number of cars", "age_band": "policyholder age band", "Deductible": "deductible",
    "DriverRating": "driver rating", "Year": "accident year", "WeekOfMonth": "accident week of month",
    "claim_lag_weeks": "weeks from accident to claim", "claim_lag_months": "months from accident to claim",
    "age_missing": "age missing", "early_policy_incident": "incident at policy start",
    "price_extreme": "extreme vehicle price", "policyholder_fault": "policyholder at fault",
    "liability_only": "liability-only policy", "accident_weekend": "weekend accident",
    "claim_weekend": "claim filed at weekend", "red_flag_score": "red-flag score",
    "consistency_failures": "failed consistency checks",
}


def describe_feature(name: str, raw_claim: dict[str, Any] | None = None) -> tuple[str, Any]:
    """Model feature name -> (readable label, the claim's raw value)."""
    raw_claim = raw_claim or {}
    for base in sorted(PRETTY, key=len, reverse=True):
        if name == base:
            return PRETTY[base], raw_claim.get(base)
        if name.startswith(base + "_") and base not in ("claim_lag", "age"):
            level = name[len(base) + 1:]
            if level in ("sin", "cos"):
                return PRETTY[base], raw_claim.get(base)
            return f"{PRETTY[base]} = {level}", raw_claim.get(base)
    if name.startswith("rf_"):
        return f"red flag {name[3:].upper()}", None
    return name, raw_claim.get(name)


def top_contributions(shap_row: np.ndarray, feature_vals: np.ndarray, names: list[str],
                      raw_claim: dict[str, Any] | None = None, top_k: int = 5) -> list[dict[str, Any]]:
    """Top-k features by |SHAP|, merged per readable feature (one-hot levels collapse)."""
    agg: dict[str, dict[str, Any]] = {}
    for name, sv, fv in zip(names, shap_row, feature_vals):
        label, raw = describe_feature(name, raw_claim)
        key = label.split(" = ")[0]
        e = agg.setdefault(key, {"feature": key, "value": raw if raw is not None else fv,
                                 "shap_contribution": 0.0})
        e["shap_contribution"] += float(sv)
    out = sorted(agg.values(), key=lambda e: abs(e["shap_contribution"]), reverse=True)[:top_k]
    for e in out:
        direction = "raises" if e["shap_contribution"] > 0 else "lowers"
        e["value"] = e["value"].item() if hasattr(e["value"], "item") else e["value"]
        e["shap_contribution"] = round(e["shap_contribution"], 4)
        e["reason"] = f"{e['feature']} = {e['value']} {direction} the fraud score"
    return out
