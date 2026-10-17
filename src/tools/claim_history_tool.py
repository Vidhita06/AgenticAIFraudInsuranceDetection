"""Tool 6, Claim History: previous claims of the *same* policy or verified customer.

This differs from similar claims (other entities). It reads only an authorised history
source configured as `claim_history.source` in config/settings.yaml: a CSV or JSONL file
with columns policy_id (or customer_id), claim_id, claim_date (YYYY-MM-DD), incident_type,
status and outcome.

The project data has no such source: PolicyNumber is unique across all 15,420 claims, so no
policy has a second claim to link. The tool then reports `unavailable` instead of "no
previous claims". The claim's own PastNumberOfClaims / AddressChange_Claim are summary
fields recorded on this claim, not previous claim records, and are labelled that way.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.tools._common import jsonable, require_valid_claim, safe_tool
from src.tools.policy_retrieval_tool import accident_month

SUMMARY_FIELDS = ["PastNumberOfClaims", "AddressChange_Claim", "NumberOfCars"]


def history_source() -> Optional[Path]:
    rel = (load_config().get("claim_history") or {}).get("source")
    return (PROJECT_ROOT / rel) if rel else None


def _load(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    return pd.read_csv(path, dtype=str)


def simulated_history(record: Mapping[str, Any], claim_id: str, years: Optional[int]) -> Optional[dict[str, Any]]:
    from src.simulated import DATA_ORIGIN
    from src.simulated.dates import accident_date
    from src.simulated.db import get_db
    db = get_db()
    pol = db.policy(record.get("PolicyNumber"))
    if pol is None:
        return None
    acc = accident_date(record)
    lo = (acc - timedelta(days=int(365.25 * int(years)))).isoformat() if (years and acc) else None
    rows = db.history(pol["customer_id"], lo, acc.isoformat() if acc else None)
    prior = [{k: r[k] for k in ("prior_claim_id", "claim_date", "incident_type", "status")} for r in rows]
    return {"claim_id": claim_id, "entity": {"customer_id": pol["customer_id"], "policy_id": pol["policy_id"]},
            "status": "completed", "history_available": True, "source": DATA_ORIGIN, "data_source": DATA_ORIGIN,
            "previous_claim_count": len(prior), "previous_claims": prior,
            "window": {"from": lo, "to": acc.isoformat() if acc else None, "years": years},
            "recorded_summary": {"fields": {f: record.get(f) for f in SUMMARY_FIELDS},
                                 "note": "Summary fields recorded on this claim (real data)."},
            "note": ("Simulated prior claims: their number follows the claim's real PastNumberOfClaims field; dates, "
                     "types and statuses are random. No investigation outcomes are recorded."),
            "limitation": None if prior else "No simulated prior claims in the window."}


@safe_tool
def get_claim_history(claim: Mapping[str, Any], years: Optional[int] = 5,
                      source: Optional[str | Path] = None, customer_id: Optional[str] = None,
                      simulated: Optional[bool] = None) -> dict[str, Any]:
    """Return previous claims for the same policy (or verified customer) within `years`
    before the accident, with dates, statuses and outcomes, or `unavailable` and why.
    With simulated data enabled (and no explicit source), prior claims come from the
    SIMULATED reference database."""
    from src.simulated.db import enabled
    n = require_valid_claim(claim)
    record = n.claim.to_record()
    if enabled(simulated) and source is None and customer_id is None:
        sim = simulated_history(record, n.claim_id, years)
        if sim is not None:
            return jsonable(sim)
    pn = record.get("PolicyNumber")
    entity = {"customer_id": customer_id} if customer_id else {"policy_id": f"PN-{pn}" if pn is not None else None}
    summary = {"fields": {f: record.get(f) for f in SUMMARY_FIELDS},
               "note": "Summary fields recorded on this claim; they are not individual previous claim records."}
    path = Path(source) if source else history_source()
    base = {"claim_id": n.claim_id, "entity": entity, "recorded_summary": summary}

    reason = None
    if path is None:
        reason = ("No authorised claim-history source is configured (claim_history.source). In the project data "
                  "PolicyNumber is unique per claim, so no linked history exists.")
    elif not path.exists():
        reason = f"The configured claim-history source {path.name} does not exist."
    elif not entity.get("customer_id") and pn is None:
        reason = "The claim has no PolicyNumber and no verified customer id, so history cannot be linked."
    if reason:
        return jsonable({**base, "status": "unavailable", "history_available": False, "previous_claim_count": None,
                         "previous_claims": [], "data_source": str(path.name) if path else None, "limitation": reason})

    df = _load(path)
    key, value = (("customer_id", customer_id) if customer_id else ("policy_id", f"PN-{pn}"))
    if key not in df.columns:
        raise ValueError(f"history source {path.name} has no {key} column")
    own = df[df[key].astype(str).isin({value, value.removeprefix("PN-")})].copy()
    if "claim_id" in own.columns:
        own = own[own["claim_id"].astype(str) != n.claim_id]          # never the claim itself
    window = None
    acc = accident_month(record)
    if years and acc is not None and "claim_date" in own:
        dates = pd.to_datetime(own["claim_date"], errors="coerce").dt.date
        lo = acc - timedelta(days=int(365.25 * int(years)))
        own = own[(dates >= lo) & (dates < acc)]
        window = {"from": lo.isoformat(), "to": acc.isoformat(), "years": int(years)}
    cols = [c for c in ("claim_id", "claim_date", "incident_type", "status", "outcome") if c in own]
    rows = own[cols].sort_values(cols[1] if "claim_date" in cols else cols[0]).to_dict("records") if cols else []
    return jsonable({**base, "status": "completed", "history_available": True,
                     "previous_claim_count": len(rows), "previous_claims": rows, "window": window,
                     "data_source": path.name,
                     "limitation": None if rows else "No previous claims found for this entity in the source."})
