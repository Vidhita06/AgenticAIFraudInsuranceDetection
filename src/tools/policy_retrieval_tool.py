"""Tool 3, Policy Retrieval: the actual policy information that applies to the claim.

Sources, in order:
1. A policy document store (`paths.policy_store_dir`, one `<PolicyNumber>.json` per policy;
   format in docs/agent_restructure_plan.md and tests/test_six_tools.py). None ships with
   the repository and none is generated: policy terms are never invented.
2. The policy attributes recorded on the claim itself (BasePolicy, PolicyType, Deductible,
   AgentType, policy-day bins). These are real data but not a policy document.

Every category the plan asks for that neither source provides is listed in
`missing_information`, so "not available" is never mistaken for "no restriction".
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional

from src.config import PROJECT_ROOT, load_config
from src.tools._common import jsonable, require_valid_claim, safe_tool

RECORDED_FIELDS = ["BasePolicy", "PolicyType", "VehicleCategory", "Deductible", "AgentType",
                   "Days_Policy_Accident", "Days_Policy_Claim"]
CATEGORIES = {
    "version": "policy version",
    "validity": "policy validity dates (effective from / to)",
    "coverage": "coverage details",
    "limits": "coverage limits",
    "exclusions": "exclusions",
    "conditions": "policy conditions",
    "reporting_requirements": "reporting requirements",
    "clauses": "policy clauses",
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def policy_store_dir() -> Optional[Path]:
    rel = load_config()["paths"].get("policy_store_dir")
    return (PROJECT_ROOT / rel) if rel else None


def accident_month(record: Mapping[str, Any]) -> Optional[date]:
    try:
        return date(int(record["Year"]), MONTHS.index(str(record["Month"])) + 1, 1)
    except (KeyError, ValueError, TypeError):
        return None


def _parse_date(s: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(s)) if s else None
    except ValueError:
        return None


def _pick_version(doc: Mapping[str, Any], when: Optional[date]) -> tuple[dict, str]:
    """The policy version in force in the accident month (or the only version)."""
    versions = list(doc.get("versions") or [doc])
    if len(versions) == 1:
        return dict(versions[0]), "only version on file"
    if when is not None:
        for v in versions:
            lo, hi = _parse_date(v.get("effective_from")), _parse_date(v.get("effective_to"))
            if (lo is None or lo <= when) and (hi is None or when <= hi):
                return dict(v), "version in force in the accident month"
    return {}, "no version covers the accident month"


def load_policy_document(policy_number: Any, store: Optional[Path] = None) -> tuple[Optional[dict], str]:
    store = store if store is not None else policy_store_dir()
    if policy_number is None:
        return None, "claim has no PolicyNumber, so no policy document can be looked up"
    if store is None or not store.exists():
        return None, f"no policy document store at {_rel(store) if store else 'paths.policy_store_dir (unset)'}"
    path = store / f"{int(policy_number)}.json"
    if not path.exists():
        return None, f"no policy document for PN-{policy_number} in {path.parent.name}/"
    return json.loads(path.read_text(encoding="utf-8")), _rel(path)


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def simulated_policy(record: Mapping[str, Any], claim_id: str) -> Optional[dict[str, Any]]:
    """Policy record and clauses from the SIMULATED reference database, or None if the
    database has no policy for this claim."""
    from src.simulated import DATA_ORIGIN
    from src.simulated.db import get_db
    db = get_db()
    pol = db.policy(record.get("PolicyNumber"))
    if pol is None:
        return None
    clauses = db.clauses(pol["base_policy"])
    by_cat = lambda cat: [{k: c[k] for k in ("clause_id", "title", "wording")} for c in clauses if c["category"] == cat]
    limits = {k.removeprefix("limit_"): v for k, v in pol.items() if k.startswith("limit_") and v is not None}
    pol_view = {k: v for k, v in pol.items() if k != "data_origin" and not k.startswith("limit_")}
    return {
        "claim_id": claim_id, "policy_id": pol["policy_id"], "status": "found", "document_found": True,
        "source": DATA_ORIGIN, "wording_note": clauses[0]["wording_note"] if clauses else None,
        "accident_period": f"{record.get('Month')} {record.get('Year')}, week {record.get('WeekOfMonth')}",
        "recorded_policy_attributes": {f: record.get(f) for f in RECORDED_FIELDS},
        "simulated_policy": pol_view,
        "applicable_version": {"version": f"simulated term {pol['term_start']} to {pol['term_end']}",
                               "selected_because": "the current annual term containing the derived accident date",
                               "effective_from": pol["term_start"], "effective_to": pol["term_end"]},
        "document": {"coverage": by_cat("coverage"), "conditions": by_cat("condition"),
                     "exclusions": by_cat("exclusion"), "reporting_requirements": by_cat("reporting"),
                     "limits": limits, "limit_clauses": by_cat("limit"), "deductible": pol["deductible"]},
        "missing_information": ["claim amount: not recorded, so the coverage limits cannot be checked"],
        "source_references": [
            {"source": "claim record", "fields": RECORDED_FIELDS, "note": "real policy attributes on the claim"},
            {"source": DATA_ORIGIN, "tables": ["policies", "policy_rules"],
             "note": "simulated dates and limits; illustrative clause wording written by the project team"}],
    }


@safe_tool
def retrieve_policy(claim: Mapping[str, Any], store: Optional[str | Path] = None,
                    simulated: Optional[bool] = None) -> dict[str, Any]:
    """Return the matching policy record, the applicable version, coverage, conditions and
    clauses (when a document exists), source references and the missing categories.
    With simulated data enabled, the policy comes from the SIMULATED reference database."""
    from src.simulated.db import enabled
    n = require_valid_claim(claim)
    record = n.claim.to_record()
    if enabled(simulated):
        sim = simulated_policy(record, n.claim_id)
        if sim is not None:
            return jsonable(sim)
    pn = record.get("PolicyNumber")
    when = accident_month(record)
    doc, where = load_policy_document(pn, Path(store) if store else None)
    recorded = {f: record.get(f) for f in RECORDED_FIELDS}
    refs = [{"source": "claim record", "fields": RECORDED_FIELDS,
             "note": "policy attributes recorded on the claim (Kaggle vehicle-claim data)"}]
    out: dict[str, Any] = {"claim_id": n.claim_id, "policy_id": f"PN-{pn}" if pn is not None else None,
                           "accident_period": f"{record.get('Month')} {record.get('Year')}, week {record.get('WeekOfMonth')}",
                           "recorded_policy_attributes": recorded}
    missing = []
    if doc is None:
        out.update(status="partial", document_found=False, document_lookup=where, applicable_version=None)
        missing = [f"{label}: not available ({where})" for label in CATEGORIES.values()]
    else:
        version, how = _pick_version(doc, when)
        refs.append({"source": "policy document", "path": where, "version": version.get("version")})
        out.update(status="found" if version else "partial", document_found=True, document_lookup=where,
                   applicable_version={"version": version.get("version"), "selected_because": how,
                                       "effective_from": version.get("effective_from"),
                                       "effective_to": version.get("effective_to")},
                   document={k: version.get(k) for k in ("coverage", "limits", "deductible", "exclusions",
                                                         "conditions", "reporting_requirements", "clauses")
                             if version.get(k) is not None})
        present = {"version": version.get("version"),
                   "validity": version.get("effective_from") or version.get("effective_to"),
                   **{k: version.get(k) for k in CATEGORIES if k not in ("version", "validity")}}
        missing = [f"{label}: not in the policy document" for k, label in CATEGORIES.items() if not present.get(k)]
        if not version:
            missing.insert(0, f"applicable policy version: {how}")
    out["missing_information"] = missing
    out["source_references"] = refs
    return jsonable(out)
