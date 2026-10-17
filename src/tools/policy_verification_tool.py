"""Tool 4, Policy Verification: check the claim facts against the applicable policy.

Every check returns PASS (the facts satisfy the condition), FAIL (the facts conflict with
it) or UNKNOWN (there is not enough information to judge). Checks that need policy terms
(coverage conditions, exclusions, limits, reporting requirements) are UNKNOWN unless an
actual policy document was retrieved (Tool 3); they are never filled in with assumed terms.
A FAIL is not evidence of fraud: it may need clarification or a separate coverage decision.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional

from src.tools._common import jsonable, require_valid_claim, safe_tool
from src.tools.policy_retrieval_tool import accident_month, retrieve_policy
from src.validation.consistency_rules import _MONTH_INDEX
from src.validation.validators import validate_claim

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"


def _check(check_id, category, result, rule, reason, evidence=None, clause=None, source="claim record"):
    return {"check_id": check_id, "category": category, "result": result, "rule": rule,
            "clause_reference": clause, "reason": reason, "evidence": evidence or {}, "source": source}


def _month_end(d: date) -> date:
    nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return date.fromordinal(nxt.toordinal() - 1)


def _parse(s) -> Optional[date]:
    try:
        return date.fromisoformat(str(s)) if s else None
    except ValueError:
        return None


def _matches(record: Mapping[str, Any], when: Mapping[str, Any]) -> Optional[bool]:
    """True if every field takes one of the listed values; None if a field is absent."""
    for field, allowed in (when or {}).items():
        if record.get(field) is None:
            return None
        allowed = allowed if isinstance(allowed, (list, tuple)) else [allowed]
        if str(record[field]) not in {str(a) for a in allowed}:
            return False
    return True


@safe_tool
def verify_policy(claim: Mapping[str, Any], policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return {overall_result, counts, checks[...], failed_checks, unknown_checks}."""
    n = require_valid_claim(claim)
    record = n.claim.to_record()
    policy = dict(policy) if policy is not None else retrieve_policy(claim)
    if not policy.get("ok", False):
        raise RuntimeError(f"policy retrieval failed: {policy.get('error')}")
    doc = policy.get("document") if policy.get("document_found") else None
    ver = policy.get("applicable_version") or {}
    doc_src = f"policy document {policy.get('policy_id')} v{ver.get('version')}" if doc is not None else None
    rules = {r["rule_id"]: r for r in validate_claim(record)}
    checks = []

    # PV01 policy identified
    if doc is not None:
        checks.append(_check("PV01_POLICY_IDENTIFIED", "applicability", PASS, "An applicable policy record exists.",
                             f"Policy document found ({ver.get('selected_because')}).",
                             {"policy_id": policy.get("policy_id"), "version": ver.get("version")}, source=doc_src))
    else:
        checks.append(_check("PV01_POLICY_IDENTIFIED", "applicability", UNKNOWN, "An applicable policy record exists.",
                             "No policy document is available; only the policy attributes recorded on the claim.",
                             {"lookup": policy.get("document_lookup")}))

    # PV02 policy type consistent with vehicle category and base policy (rule V05)
    v05, v05b = rules["V05_POLICYTYPE_CONSISTENT"], rules["V05B_POLICYTYPE_KNOWN_ALIAS"]
    ev = {f: record[f] for f in ("PolicyType", "VehicleCategory", "BasePolicy")}
    if not v05["passed"]:
        checks.append(_check("PV02_POLICY_TYPE_CONSISTENT", "applicability", FAIL, v05["rule_id"], v05["message"], ev))
    else:
        note = " (known 'Sedan - Liability' labelling alias of the source data)" if not v05b["passed"] else ""
        checks.append(_check("PV02_POLICY_TYPE_CONSISTENT", "applicability", PASS, v05["rule_id"],
                             "PolicyType matches VehicleCategory and BasePolicy" + note + ".", ev))

    # PV03 incident inside the coverage period
    when = accident_month(record)
    lo, hi = _parse(ver.get("effective_from")), _parse(ver.get("effective_to"))
    ev = {"accident_period": policy.get("accident_period"), "Days_Policy_Accident": record["Days_Policy_Accident"]}
    if doc is not None and when is not None and (lo or hi):
        ev.update(effective_from=ver.get("effective_from"), effective_to=ver.get("effective_to"))
        start, end = when, _month_end(when)
        if (lo is None or lo <= start) and (hi is None or end <= hi):
            res, why = PASS, "The accident month lies inside the policy's validity dates."
        elif (lo is not None and end < lo) or (hi is not None and start > hi):
            res, why = FAIL, "The accident month lies outside the policy's validity dates."
        else:
            res, why = UNKNOWN, "The accident month straddles a validity boundary; the exact accident date is not recorded."
        checks.append(_check("PV03_INCIDENT_IN_COVERAGE_PERIOD", "effective_date", res,
                             "The incident occurred while the policy was in force.", why, ev, source=doc_src))
    elif record["Days_Policy_Accident"] == "none":
        checks.append(_check("PV03_INCIDENT_IN_COVERAGE_PERIOD", "effective_date", UNKNOWN,
                             "The incident occurred while the policy was in force.",
                             "Days from policy start to accident is recorded as 'none'; without exact dates it "
                             "cannot be confirmed that the accident fell inside the coverage period.", ev))
    else:
        checks.append(_check("PV03_INCIDENT_IN_COVERAGE_PERIOD", "effective_date", PASS,
                             "The incident occurred while the policy was in force.",
                             f"The accident is recorded {record['Days_Policy_Accident']} days after the policy start.", ev))

    # PV04 claim timeline (rules V06, V09, V10)
    bad = [rules[r] for r in ("V06_CLAIM_AFTER_ACCIDENT", "V09_POLICY_DAYS_ORDER", "V10_POLICY_DAYS_VS_LAG")
           if not rules[r]["passed"]]
    ev = {f: record[f] for f in ("Month", "WeekOfMonth", "MonthClaimed", "WeekOfMonthClaimed",
                                 "Days_Policy_Accident", "Days_Policy_Claim")}
    checks.append(_check("PV04_CLAIM_TIMELINE", "reporting", FAIL if bad else PASS,
                         "V06_CLAIM_AFTER_ACCIDENT, V09_POLICY_DAYS_ORDER, V10_POLICY_DAYS_VS_LAG",
                         " ".join(f"{r['rule_id']}: {r['message']}" for r in bad) if bad
                         else "The claim follows the accident and the policy-day fields are consistent.", ev))

    # PV05 deductible matches the policy
    if doc is not None and doc.get("deductible") is not None:
        ok = int(doc["deductible"]) == int(record["Deductible"])
        checks.append(_check("PV05_DEDUCTIBLE_MATCHES_POLICY", "coverage_limits", PASS if ok else FAIL,
                             "The claim's deductible equals the policy deductible.",
                             "Deductibles match." if ok else "The claim's deductible differs from the policy's.",
                             {"claim": record["Deductible"], "policy": doc["deductible"]}, source=doc_src))
    else:
        checks.append(_check("PV05_DEDUCTIBLE_MATCHES_POLICY", "coverage_limits", UNKNOWN,
                             "The claim's deductible equals the policy deductible.",
                             "No policy document states the deductible; the claim records "
                             f"{record['Deductible']}.", {"claim": record["Deductible"]}))

    # PV06 coverage conditions, PV07 exclusions (machine-checkable terms from the document)
    for check_id, key, category in (("PV06_COVERAGE_CONDITIONS", "conditions", "coverage_conditions"),
                                    ("PV07_EXCLUSIONS", "exclusions", "exclusions")):
        items = (doc or {}).get(key) or []
        if not items:
            checks.append(_check(check_id, category, UNKNOWN, f"The claim satisfies the policy {key}.",
                                 f"No policy {key} are available to check against.", {}))
            continue
        hits, unknown = [], []
        for it in items:
            m = _matches(record, it.get("requires") if key == "conditions" else it.get("when"))
            if m is None:
                unknown.append(it.get("id"))
            elif (key == "conditions" and not m) or (key == "exclusions" and m):
                hits.append(it)
        res = FAIL if hits else (UNKNOWN if unknown else PASS)
        why = ("; ".join(f"{h.get('id')}: {h.get('text', h.get('clause'))}" for h in hits) if hits else
               f"Claim fields needed for {', '.join(map(str, unknown))} are not recorded." if unknown else
               f"All {len(items)} {key} checked.")
        checks.append(_check(check_id, category, res, f"The claim satisfies the policy {key}.", why,
                             {"checked": [i.get("id") for i in items]},
                             clause=", ".join(str(h.get("clause")) for h in hits) or None, source=doc_src))

    # PV08 coverage limits: the claim amount is not part of the data
    checks.append(_check("PV08_COVERAGE_LIMITS", "coverage_limits", UNKNOWN,
                         "The claimed amount is within the coverage limits.",
                         "The claim amount is not recorded, so limits cannot be checked.",
                         {"policy_limits": (doc or {}).get("limits")}, source=doc_src or "claim record"))

    # PV09 reporting requirement
    lag = (_MONTH_INDEX[record["MonthClaimed"]] - _MONTH_INDEX[record["Month"]]) % 12
    req = (doc or {}).get("reporting_requirements") or {}
    if req.get("max_months_after_accident") is not None:
        ok = lag <= int(req["max_months_after_accident"])
        checks.append(_check("PV09_REPORTING_REQUIREMENT", "reporting", PASS if ok else FAIL,
                             "The claim was reported within the policy's deadline.",
                             f"Reported {lag} month(s) after the accident; the policy allows "
                             f"{req['max_months_after_accident']}.", {"lag_months": lag},
                             clause=req.get("clause"), source=doc_src))
    else:
        checks.append(_check("PV09_REPORTING_REQUIREMENT", "reporting", UNKNOWN,
                             "The claim was reported within the policy's deadline.",
                             f"No reporting requirement is available; the claim was reported {lag} month(s) "
                             "after the accident.", {"lag_months": lag}))

    counts = {r: sum(c["result"] == r for c in checks) for r in (PASS, FAIL, UNKNOWN)}
    overall = FAIL if counts[FAIL] else (UNKNOWN if counts[UNKNOWN] else PASS)
    return jsonable({
        "claim_id": n.claim_id, "status": "completed", "policy_id": policy.get("policy_id"),
        "policy_source": doc_src or "claim-recorded policy attributes only",
        "overall_result": overall, "counts": counts, "checks": checks,
        "failed_checks": [c["check_id"] for c in checks if c["result"] == FAIL],
        "unknown_checks": [c["check_id"] for c in checks if c["result"] == UNKNOWN],
        "note": "A FAIL may need clarification or a coverage decision; it is not by itself evidence of fraud.",
    })
