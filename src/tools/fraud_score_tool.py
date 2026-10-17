"""Tool 1, Fraud Score: calibrated fraud probability, predicted class, risk band and model
metadata from the trained model, plus the EDA red flags as rule-based risk indicators.
The score guides the investigation; it never establishes fraud on its own."""
from __future__ import annotations

from typing import Any, Mapping

from src.ml.predict import load_artifacts, risk_band
from src.tools._common import jsonable, require_valid_claim, safe_tool
from src.validation.validators import evaluate_red_flags


def model_frame(record: Mapping[str, Any], art=None):
    """(artifacts, model-ready feature matrix) for one validated claim record."""
    from src.ml.predict import claims_frame
    art = art or load_artifacts()
    return art, art.preprocessor.transform(claims_frame(dict(record)))


def risk_indicators(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    flags = evaluate_red_flags(record)
    for f in flags:
        tr = f["stats"].get("real_train", {})
        f["support"] = (f"{tr.get('fraud_rate', 0):.1%} fraud in {tr.get('n')} real training claims "
                        f"(lift {tr.get('lift')})") if tr else None
        f.pop("stats", None)
    return flags


@safe_tool
def score_claim(claim: Mapping[str, Any]) -> dict[str, Any]:
    """Return {status, predicted_class, fraud_probability, risk_band, decision_threshold,
    thresholds, risk_indicators, model}."""
    n = require_valid_claim(claim)
    record = n.claim.to_record()
    art, X = model_frame(record)
    p = float(art.model.predict_proba(X)[:, 1][0])
    threshold = float(art.bands["high"])
    return jsonable({
        "claim_id": n.claim_id, "status": "completed",
        "predicted_class": "fraud" if p >= threshold else "not_fraud",
        "fraud_probability": round(p, 4), "risk_band": risk_band(p, art.bands),
        "decision_threshold": round(threshold, 4), "thresholds": dict(art.bands), "base_rate": 0.06,
        "risk_indicators": risk_indicators(record),
        "model": {"version": art.version, "family": art.card.get("family"),
                  "seed_members": art.card.get("n_seed_members"), "calibrated": True, "placeholder": bool(art.card.get("placeholder", False))},
        "model_version": art.version,
        "note": "The predicted class uses the 10:1 cost threshold (the high-band cut-off). "
                "A score supports investigation; it is not proof of fraud.",
    })
