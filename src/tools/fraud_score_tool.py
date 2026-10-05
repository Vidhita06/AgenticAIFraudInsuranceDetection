"""Fraud score: calibrated probability, risk band and the top SHAP factors in plain English."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.ml.explain import load_explainer, top_contributions
from src.ml.predict import load_artifacts, risk_band
from src.tools._common import jsonable, require_valid_claim, safe_tool


@safe_tool
def score_claim(claim: Mapping[str, Any], top_k: int = 5) -> dict[str, Any]:
    """Return {fraud_probability, risk_band, thresholds, top_factors, model_version}."""
    n = require_valid_claim(claim)
    record = n.claim.to_record()
    art = load_artifacts()
    X = art.preprocessor.transform(_frame(record))
    p = float(art.model.predict_proba(X)[:, 1][0])
    factors: list[dict] = []
    try:
        ex = load_explainer(art)
        sv = ex.shap_values(X)
        factors = top_contributions(np.asarray(sv)[0], np.asarray(X)[0] if not hasattr(X, "iloc") else X.iloc[0].to_numpy(),
                                    ex.feature_names, record, top_k=top_k)
    except Exception as e:  # noqa: BLE001  (explanations are best-effort)
        factors = [{"feature": "explanation unavailable", "value": None, "shap_contribution": 0.0,
                    "reason": f"{type(e).__name__}"}]
    return jsonable({
        "claim_id": n.claim_id, "fraud_probability": round(p, 4),
        "risk_band": risk_band(p, art.bands), "thresholds": dict(art.bands),
        "base_rate": 0.06, "top_factors": factors, "model_version": art.version,
        "placeholder_model": bool(art.card.get("placeholder", False)),
    })


def _frame(record):
    from src.ml.predict import claims_frame
    return claims_frame(record)
