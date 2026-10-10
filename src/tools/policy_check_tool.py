"""Policy and history checks: validation + consistency rules and red flags
from config/policy_rules.yaml."""
from __future__ import annotations

from typing import Any, Mapping

from src.ingestion.normalizer import normalize_claim
from src.tools._common import jsonable, safe_tool
from src.validation.validators import evaluate_red_flags, has_blocking_failure, validate_claim


@safe_tool
def check_policy_rules(claim: Mapping[str, Any]) -> dict[str, Any]:
    """Return failed rules (blocking / warning / info), red flags and a blocking summary.
    Works on incomplete claims too: missing or invalid fields are reported, not raised."""
    n = normalize_claim(claim)
    record = n.record
    results = validate_claim(record)
    failed = [r for r in results if not r["passed"]]
    flags = evaluate_red_flags(record) if n.ok else []
    for f in flags:
        tr = f["stats"].get("real_train", {})
        f["support"] = (f"{tr.get('fraud_rate', 0):.1%} fraud in {tr.get('n')} real training claims "
                        f"(lift {tr.get('lift')})") if tr else None
        f.pop("stats", None)
    return jsonable({
        "claim_id": n.claim_id,
        "blocking": has_blocking_failure(results),
        "failed_rules": failed,
        "passed_rule_count": len(results) - len(failed),
        "red_flags": flags,
        "schema_issues": n.issues,
    })
