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


CLAIM_RECORD_CHECKS = ("PV02_POLICY_TYPE_CONSISTENT", "PV04_CLAIM_TIMELINE")


def _check(check_id, category, result, rule, reason, evidence=None, clause=None, source="claim record"):
    """`level`: claim_record = the claim's own data is contradictory (needs more information);
    policy_condition = the claim conflicts with (or cannot be checked against) policy terms."""
    level = "claim_record" if check_id in CLAIM_RECORD_CHECKS else "policy_condition"
    return {"check_id": check_id, "category": category, "level": level, "result": result, "rule": rule,
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


def _claim_record_checks(record, rules) -> list[dict]:
    """PV02 and PV04: contradictions inside the claim record itself (no policy terms needed)."""
    checks = []
    v05, v05b = rules["V05_POLICYTYPE_CONSISTENT"], rules["V05B_POLICYTYPE_KNOWN_ALIAS"]
    ev = {f: record[f] for f in ("PolicyType", "VehicleCategory", "BasePolicy")}
    if not v05["passed"]:
        checks.append(_check("PV02_POLICY_TYPE_CONSISTENT", "applicability", FAIL, v05["rule_id"], v05["message"], ev))
    else:
        note = " (known 'Sedan - Liability' labelling alias of the source data)" if not v05b["passed"] else ""
        checks.append(_check("PV02_POLICY_TYPE_CONSISTENT", "applicability", PASS, v05["rule_id"],
                             "PolicyType matches VehicleCategory and BasePolicy" + note + ".", ev))
    bad = [rules[r] for r in ("V06_CLAIM_AFTER_ACCIDENT", "V09_POLICY_DAYS_ORDER", "V10_POLICY_DAYS_VS_LAG")
           if not rules[r]["passed"]]
    ev = {f: record[f] for f in ("Month", "WeekOfMonth", "MonthClaimed", "WeekOfMonthClaimed",
                                 "Days_Policy_Accident", "Days_Policy_Claim")}
    checks.append(_check("PV04_CLAIM_TIMELINE", "reporting", FAIL if bad else PASS,
                         "V06_CLAIM_AFTER_ACCIDENT, V09_POLICY_DAYS_ORDER, V10_POLICY_DAYS_VS_LAG",
                         " ".join(f"{r['rule_id']}: {r['message']}" for r in bad) if bad
                         else "The claim follows the accident and the policy-day fields are consistent.", ev))
    return checks


def _simulated_checks(record, policy) -> list[dict]:
    """One check per SIMULATED clause that has a test (illustrative wording)."""
    from src.simulated import DATA_ORIGIN
    from src.simulated.dates import accident_date, reporting_lag_days
    from src.simulated.db import get_db
    pol = policy["simulated_policy"]
    acc = accident_date(record)
    checks = [_check("PV01_POLICY_IDENTIFIED", "applicability", PASS, "An applicable policy record exists.",
                     f"Simulated policy {pol['policy_id']} found; current term {pol['term_start']} to {pol['term_end']}.",
                     {"policy_id": pol["policy_id"]}, source=DATA_ORIGIN)]
    for c in get_db().clauses(pol["base_policy"]):
        t = c.get("test")
        if not t:
            continue                                   # wording only (coverage description)
        kind, ev = t["type"], {}
        if kind == "term_contains_accident":
            ev = {"accident_date_derived": acc.isoformat() if acc else None, "term_start": pol["term_start"],
                  "term_end": pol["term_end"], "dates_consistent": pol["dates_consistent"]}
            if acc is None or not pol["term_start"]:
                res, why = UNKNOWN, "The accident date cannot be derived from the claim fields."
            elif pol["term_start"] <= acc.isoformat() <= pol["term_end"]:
                res, why = PASS, "The derived accident date lies inside the simulated policy term."
            else:
                res, why = FAIL, "The derived accident date lies outside the simulated policy term."
        elif kind == "vehicle_category_matches":
            ok = pol["vehicle_category"] == record["VehicleCategory"]
            ev = {"claim": record["VehicleCategory"], "policy": pol["vehicle_category"]}
            res, why = (PASS, "The claimed vehicle category matches the insured vehicle.") if ok else \
                       (FAIL, "The claimed vehicle category differs from the insured vehicle.")
        elif kind == "deductible_matches":
            ok = int(pol["deductible"]) == int(record["Deductible"])
            ev = {"claim": record["Deductible"], "policy": pol["deductible"]}
            res, why = (PASS, "Deductibles match (the simulated schedule copies the claim's deductible).") if ok else \
                       (FAIL, "The claim's deductible differs from the scheduled deductible.")
        elif kind == "reporting_within_days":
            lag = reporting_lag_days(record)
            ev = {"reporting_lag_days": lag, "deadline_days": t["days"],
                  "lag_basis": "smallest lag consistent with the recorded accident and claim weeks"}
            if lag is None:
                res, why = UNKNOWN, "The reporting lag cannot be derived from the claim fields."
            elif lag <= int(t["days"]):
                res, why = PASS, f"Reported about {lag} days after the accident (deadline {t['days']} days)."
            else:
                res, why = FAIL, f"Reported at least {lag} days after the accident (deadline {t['days']} days)."
        elif kind == "field_not_in":
            val = record.get(t["field"])
            ev = {t["field"]: val}
            ok = str(val) not in {str(x) for x in t["values"]}
            res, why = (PASS, f"{t['field']} is recorded ({val}).") if ok else \
                       (FAIL, f"{t['field']} = {val} does not meet the condition.")
        else:
            res, why = UNKNOWN, t.get("reason", "Cannot be checked with the recorded fields.")
        checks.append(_check(c["clause_id"], c["category"], res, c["title"], why, ev, clause=c["clause_id"],
                             source=DATA_ORIGIN))
    return checks


def _summary(n, policy, checks, source, extra=None) -> dict:
    counts = {r: sum(c["result"] == r for c in checks) for r in (PASS, FAIL, UNKNOWN)}
    overall = FAIL if counts[FAIL] else (UNKNOWN if counts[UNKNOWN] else PASS)
    fails = [c for c in checks if c["result"] == FAIL]
    return jsonable({
        "claim_id": n.claim_id, "status": "completed", "policy_id": policy.get("policy_id"),
        "policy_source": source, **(extra or {}),
        "overall_result": overall, "counts": counts, "checks": checks,
        "failed_checks": [c["check_id"] for c in fails],
        "claim_record_failures": [c["check_id"] for c in fails if c["level"] == "claim_record"],
        "policy_condition_failures": [c["check_id"] for c in fails if c["level"] == "policy_condition"],
        "unknown_checks": [c["check_id"] for c in checks if c["result"] == UNKNOWN],
        "note": "A FAIL may need clarification or a coverage decision; it is not by itself evidence of fraud.",
    })


@safe_tool
def verify_policy(claim: Mapping[str, Any], policy: Mapping[str, Any] | None = None,
                  simulated: Optional[bool] = None) -> dict[str, Any]:
    """Return {overall_result, counts, checks[...], failed_checks, claim_record_failures,
    policy_condition_failures, unknown_checks}."""
    from src.simulated import DATA_ORIGIN
    n = require_valid_claim(claim)
    record = n.claim.to_record()
    policy = dict(policy) if policy is not None else retrieve_policy(claim, simulated=simulated)
    if not policy.get("ok", False):
        raise RuntimeError(f"policy retrieval failed: {policy.get('error')}")
    if policy.get("source") == DATA_ORIGIN and policy.get("simulated_policy"):
        rules = {r["rule_id"]: r for r in validate_claim(record)}
        checks = _claim_record_checks(record, rules) + _simulated_checks(record, policy)
        checks.sort(key=lambda c: (not c["check_id"].startswith("PV"), c["check_id"]))
        return _summary(n, policy, checks, f"{DATA_ORIGIN} (policy {policy.get('policy_id')})",
                        {"source": DATA_ORIGIN, "wording_note": policy.get("wording_note")})
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

    claim_checks = _claim_record_checks(record, rules)
    checks.append(claim_checks[0])

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

    checks.append(claim_checks[1])

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

    return _summary(n, policy, checks, doc_src or "claim-recorded policy attributes only")
