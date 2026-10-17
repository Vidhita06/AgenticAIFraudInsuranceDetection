"""Tool 2, SHAP Explanation: which features pushed this claim's fraud score up or down.
SHAP explains the model's behaviour; it does not show that the claimant committed fraud."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from src.ml.explain import base_models, load_explainer, top_contributions
from src.tools._common import jsonable, require_valid_claim, safe_tool
from src.tools.fraud_score_tool import model_frame

LIMITATIONS = [
    "SHAP values explain the uncalibrated ensemble in log-odds; calibration is monotone, so "
    "the ranking and direction of contributions also hold for the calibrated probability.",
    "Contributions are associations learned from 1994-1996 claims, not causes, and not "
    "evidence that this claimant committed fraud.",
    "One-hot levels are summed into one value per readable feature.",
]


@lru_cache(maxsize=4)
def _base_value(version: str) -> float | None:
    """Mean model output (log-odds) over the training background, averaged across members."""
    import shap
    from src.ml.predict import load_artifacts
    try:
        vals = []
        for m in base_models(load_artifacts().model):
            ev = np.atleast_1d(np.asarray(shap.TreeExplainer(m).expected_value, dtype=float))
            vals.append(float(ev[-1]))
        return round(float(np.mean(vals)), 4)
    except Exception:  # noqa: BLE001  (non-tree models: no closed-form base value)
        return None


@safe_tool
def explain_claim(claim: Mapping[str, Any], top_k: int = 5,
                  fraud_score: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the top-k features by |SHAP| with their observed values, signed contributions,
    direction, and the positive / negative factor lists."""
    top_k = int(max(1, min(int(top_k), 15)))
    if fraud_score is not None and not fraud_score.get("ok", False):
        raise RuntimeError("fraud score unavailable, so there is no prediction to explain")
    n = require_valid_claim(claim)
    record = n.claim.to_record()
    art, X = model_frame(record)
    ex = load_explainer(art)
    sv = np.asarray(ex.shap_values(X))[0]
    xrow = X.iloc[0].to_numpy() if hasattr(X, "iloc") else np.asarray(X)[0]
    factors = top_contributions(sv, xrow, ex.feature_names, record, top_k=top_k)
    for f in factors:
        f["shap_value"] = f.pop("shap_contribution")
        f["direction"] = "increases fraud score" if f["shap_value"] > 0 else "decreases fraud score"
    return jsonable({
        "claim_id": n.claim_id, "status": "completed",
        "model_version": art.version, "explained_class": "fraud (class 1)",
        "scale": "log-odds (uncalibrated ensemble)", "base_value": _base_value(art.version),
        "total_shap": round(float(sv.sum()), 4),
        "top_factors": factors,
        "positive_factors": [f["feature"] for f in factors if f["shap_value"] > 0],
        "negative_factors": [f["feature"] for f in factors if f["shap_value"] < 0],
        "limitations": LIMITATIONS,
    })
