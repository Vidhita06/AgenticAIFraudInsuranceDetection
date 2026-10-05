"""CPU inference from the saved artifacts in models/:
preprocessor.joblib (fitted src.ml.preprocess pipeline), fraud_model.joblib (calibrated
model, switched to CPU), model_card.json (thresholds, risk bands, versions).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.config import path_for
from src.ml.data import CLAIM_FIELDS, INTEGER_DOMAINS

RISK_BANDS = ("low", "medium", "high")


@dataclass
class Artifacts:
    preprocessor: Any
    model: Any
    card: dict[str, Any]
    directory: Path

    @property
    def version(self) -> str:
        return str(self.card.get("model_version", "unknown"))

    @property
    def bands(self) -> dict[str, float]:
        return self.card["risk_bands"]


@lru_cache(maxsize=2)
def load_artifacts(models_dir: str | None = None) -> Artifacts:
    import joblib

    d = Path(models_dir) if models_dir else path_for("models_dir")
    missing = [f for f in ("preprocessor.joblib", "fraud_model.joblib", "model_card.json")
               if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts in {d}: {missing}. Train with "
                                "notebooks/02_model_training.ipynb or scripts/train_model.py.")
    card = json.loads((d / "model_card.json").read_text(encoding="utf-8"))
    return Artifacts(joblib.load(d / "preprocessor.joblib"), joblib.load(d / "fraud_model.joblib"),
                     card, d)


def claims_frame(claims: pd.DataFrame | Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Raw claim(s) -> DataFrame with every claim field, integer fields coerced."""
    if isinstance(claims, pd.DataFrame):
        df = claims.copy()
    elif isinstance(claims, Mapping):
        df = pd.DataFrame([dict(claims)])
    else:
        df = pd.DataFrame([dict(c) for c in claims])
    for col in CLAIM_FIELDS:
        if col not in df:
            df[col] = pd.NA
    for col in INTEGER_DOMAINS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def risk_band(p: float, bands: Mapping[str, float]) -> str:
    if p >= bands["high"]:
        return "high"
    if p >= bands["medium"]:
        return "medium"
    return "low"


def predict_proba(claims, artifacts: Artifacts | None = None) -> np.ndarray:
    a = artifacts or load_artifacts()
    X = a.preprocessor.transform(claims_frame(claims))
    return a.model.predict_proba(X)[:, 1]


def score_claims(claims, artifacts: Artifacts | None = None) -> list[dict[str, Any]]:
    a = artifacts or load_artifacts()
    p = predict_proba(claims, a)
    return [{"fraud_probability": float(round(v, 6)), "risk_band": risk_band(v, a.bands),
             "thresholds": dict(a.bands), "model_version": a.version} for v in p]
