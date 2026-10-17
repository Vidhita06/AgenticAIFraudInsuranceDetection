"""Evidence aggregation (plan §10) and reason codes (plan §12), all deterministic.

The aggregated record keeps three things apart that a reviewer must not confuse:
a negative finding (e.g. 0 of 5 similar claims were fraudulent), missing information
(e.g. no policy document, no linked claim history) and a failed lookup (tool error).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from src.tools import SOURCES

ALL_SOURCES = ("validation",) + SOURCES
DONE = ("completed", "partial", "unavailable")
REASON_CODES = (
    "BLOCKING_VALIDATION", "CLAIM_RECORD_CONTRADICTION", "HIGH_MODEL_RISK", "MEDIUM_MODEL_RISK", "LOW_MODEL_RISK",
    "MODEL_SCORE_UNAVAILABLE", "RED_FLAGS_PRESENT", "POLICY_CONDITION_FAILED", "POLICY_TERMS_UNVERIFIED",
    "SIMILAR_CLAIMS_ELEVATED_FRAUD", "SIMILAR_CLAIMS_NO_FRAUD", "STRONG_SHAP_DRIVER", "CLAIM_HISTORY_UNAVAILABLE", "CONFLICTING_EVIDENCE",
    "TOOL_FAILURE", "INCOMPLETE_INVESTIGATION", "SIMULATED_DATA_USED", "NO_MATERIAL_CONCERNS",
)
CLAIM_RECORD_CHECKS = ("PV02_POLICY_TYPE_CONSISTENT", "PV04_CLAIM_TIMELINE")
SIMULATED = "simulated reference data"


def pv_failures(pv: Mapping[str, Any] | None) -> tuple[list[str], list[str]]:
    """(claim-record failures, policy-condition failures) of a policy-verification output."""
    pv = pv or {}
    if not pv.get("ok"):
        return [], []
    if "claim_record_failures" in pv:
        return list(pv["claim_record_failures"]), list(pv.get("policy_condition_failures", []))
    failed = list(pv.get("failed_checks") or [])
    return [f for f in failed if f in CLAIM_RECORD_CHECKS], [f for f in failed if f not in CLAIM_RECORD_CHECKS]


def claim_level_grounds(tool_results: Mapping[str, Any]) -> list[str]:
    """Reasons that justify REQUEST_MORE_INFORMATION: the claim's own data is missing
    (blocking validation) or contradictory (a claim-record check FAILed)."""
    v = tool_results.get("validation") or {}
    out = [r["rule_id"] for r in v.get("failed_rules", []) if r.get("severity") == "blocking"]
    if v.get("blocking") and not out:
        out.append("schema")
    return out + pv_failures(tool_results.get("policy_verification"))[0]


def uses_simulated(output: Mapping[str, Any] | None) -> bool:
    return bool(output) and output.get("ok", False) and output.get("source") == SIMULATED


def _agent_setting(key: str, default: Any) -> Any:
    from src.config import load_config
    return (load_config().get("agent") or {}).get(key, default)


def concrete_factor_codes() -> frozenset[str]:
    """Reason codes that count as a concrete, claim-specific factor for a medium-band fraud flag."""
    return frozenset(_agent_setting("medium_band_concrete_factors", [
        "RED_FLAGS_PRESENT", "POLICY_CONDITION_FAILED", "SIMILAR_CLAIMS_ELEVATED_FRAUD",
        "STRONG_SHAP_DRIVER", "CONFLICTING_EVIDENCE"]))


def strong_shap_drivers(shap: Mapping[str, Any] | None, threshold: Optional[float] = None) -> list[str]:
    """Features other than the red flags whose SHAP contribution is at least `threshold`
    log-odds. The red-flag score drives every medium/high score, so it is not counted here
    (warning-level red flags have their own code)."""
    shap = shap or {}
    if not shap.get("ok"):
        return []
    t = float(_agent_setting("strong_shap_log_odds", 0.5) if threshold is None else threshold)
    return [f["feature"] for f in shap.get("top_factors", [])
            if (f.get("shap_value") or 0) >= t and not str(f["feature"]).startswith(("red flag", "red-flag"))]


def similar_fraud_elevated(sim: Mapping[str, Any] | None) -> bool:
    sim = sim or {}
    return bool(sim.get("ok") and sim.get("neighbour_frauds", 0) >= 2
                and (sim.get("neighbour_fraud_rate") or 0) >= 2 * (sim.get("historical_base_rate") or 1))


def fraud_evidence(tool_results: Mapping[str, Any]) -> bool:
    """A fraud-risk signal beyond the score band: high band, a warning-level red flag, fraud
    among the similar claims, or a strong (non-red-flag) SHAP driver. A medium band alone is
    not one: on real validation its fraud rate is 8.8%, about 1.5x the 6% base rate."""
    fs = tool_results.get("fraud_score") or {}
    if fs.get("ok") and (fs.get("risk_band") == "high" or _warning_indicators(fs)):
        return True
    return similar_fraud_elevated(tool_results.get("similar_claims")) or \
        (fs.get("risk_band") == "medium" and bool(strong_shap_drivers(tool_results.get("shap_explanation"))))


def failed_required(tool_results: Mapping[str, Any], required: Iterable[str]) -> list[str]:
    return [s for s in required if tool_status(tool_results.get(s)) == "failed"]


def flag_category(tool_results: Mapping[str, Any], required: Iterable[str] = ()) -> str:
    """Why an LLM FLAG_FOR_INVESTIGATION was raised. Coverage problems and system failures are
    kept apart from fraud risk so they never count as "fraud caught"."""
    fs = tool_results.get("fraud_score") or {}
    if fs.get("ok") and fs.get("risk_band") == "high":
        return "FRAUD_RISK"
    if not fs.get("ok"):
        return "SYSTEM_FAILURE"
    if fraud_evidence(tool_results):
        return "FRAUD_RISK"
    if pv_failures(tool_results.get("policy_verification"))[1]:
        return "POLICY_CONDITION"
    if failed_required(tool_results, required):
        return "SYSTEM_FAILURE"
    return "FRAUD_RISK"


def tool_status(output: Optional[Mapping[str, Any]]) -> str:
    if output is None:
        return "not_run"
    if not output.get("ok", False):
        return "failed"
    status = str(output.get("status") or "completed")
    return {"found": "completed"}.get(status, status)


def _warning_indicators(fs: Mapping[str, Any]) -> list[str]:
    return [f["rule_id"] for f in fs.get("risk_indicators") or [] if f.get("severity") == "warning"]


def find_conflicts(tr: Mapping[str, Any]) -> list[str]:
    fs, sim, pv = (tr.get(k) or {} for k in ("fraud_score", "similar_claims", "policy_verification"))
    band = fs.get("risk_band") if fs.get("ok") else None
    out = []
    if band and sim.get("ok") and (sim.get("k") or 0) >= 5:
        rate, base = sim.get("neighbour_fraud_rate") or 0.0, sim.get("historical_base_rate") or 0.0
        if band == "high" and rate <= base:
            out.append(f"The model places the claim in the high risk band, but only {sim['neighbour_frauds']} of "
                       f"{sim['k']} most similar past claims were fraudulent.")
        if band == "low" and rate >= 0.4:
            out.append(f"The model places the claim in the low risk band, but {sim['neighbour_frauds']} of "
                       f"{sim['k']} most similar past claims were fraudulent.")
    warn = _warning_indicators(fs) if fs.get("ok") else []
    if band == "low" and warn:
        out.append("The model places the claim in the low risk band, but warning-level red flags fired: "
                   + ", ".join(warn) + ".")
    policy_fails = pv_failures(pv)[1]
    if band in ("low", "medium") and policy_fails:
        out.append(f"The model risk band is {band}, but policy verification failed: "
                   + ", ".join(policy_fails) + ".")
    return out


def _missing(tr: Mapping[str, Any], statuses: Mapping[str, str], required: Iterable[str]) -> list[str]:
    out: list[str] = []
    v = tr.get("validation") or {}
    for r in v.get("failed_rules", []):
        if r.get("severity") == "blocking":
            out.append(f"Validation {r['rule_id']}: {r['message']}")
    out += [f"Claim field: {i}" for i in v.get("schema_issues", [])]
    for src in required:
        if statuses.get(src) == "not_run":
            out.append(f"{src} was not run")
    for src, st in statuses.items():
        if st == "failed":
            out.append(f"{src} failed: {(tr.get(src) or {}).get('error', '')[:160]}")
    pr = tr.get("policy_retrieval") or {}
    if pr.get("ok"):
        if not pr.get("document_found"):
            out.append("Policy document unavailable: no policy version, validity dates, coverage details, limits, "
                       "exclusions, conditions, reporting requirements or clauses are on file "
                       f"({pr.get('document_lookup')}).")
        else:
            out += [f"Policy {m}" for m in pr.get("missing_information", [])]
    pv = tr.get("policy_verification") or {}
    if pv.get("ok") and pv.get("unknown_checks"):
        out.append("Policy checks that could not be verified (UNKNOWN): " + ", ".join(pv["unknown_checks"]) + ".")
        for c in pv.get("checks", []):
            if c["result"] == "UNKNOWN" and c["check_id"] == "PV03_INCIDENT_IN_COVERAGE_PERIOD":
                out.append(f"{c['check_id']}: {c['reason']}")
    ch = tr.get("claim_history") or {}
    if ch.get("ok") and ch.get("status") == "unavailable":
        out.append(f"Claim history unavailable: {ch.get('limitation')}")
    return out


def aggregate_evidence(investigation_id: str, claim_id: str, tool_results: Mapping[str, Any],
                       tool_origin: Mapping[str, str], required: Iterable[str],
                       max_steps_reached: bool = False) -> dict[str, Any]:
    tr = tool_results
    required = tuple(required)
    statuses = {s: tool_status(tr.get(s)) for s in ALL_SOURCES}
    fs, shap, pr, pv, sim, ch, v = (tr.get(k) or {} for k in ("fraud_score", "shap_explanation", "policy_retrieval",
                                                               "policy_verification", "similar_claims",
                                                               "claim_history", "validation"))
    findings = {
        "fraud_score": {k: fs.get(k) for k in ("fraud_probability", "risk_band", "predicted_class", "decision_threshold")}
                       | {"risk_indicators": [f"{f['rule_id']} ({f['severity']})" for f in fs.get("risk_indicators") or []]}
                       if fs.get("ok") else None,
        "shap_explanation": {"top_factors": [{k: f.get(k) for k in ("feature", "value", "shap_value")}
                                             for f in shap.get("top_factors", [])],
                             "positive_factors": shap.get("positive_factors"),
                             "negative_factors": shap.get("negative_factors")} if shap.get("ok") else None,
        "policy_retrieval": {"policy_id": pr.get("policy_id"), "document_found": pr.get("document_found"),
                             "status": pr.get("status")} if pr.get("ok") else None,
        "policy_verification": {"overall_result": pv.get("overall_result"), "counts": pv.get("counts"),
                                "failed": [{"check_id": c["check_id"], "reason": c["reason"]}
                                           for c in pv.get("checks", []) if c["result"] == "FAIL"],
                                "unknown": pv.get("unknown_checks"), "policy_source": pv.get("policy_source")}
                               if pv.get("ok") else None,
        "similar_claims": {k: sim.get(k) for k in ("k", "neighbour_frauds", "neighbour_fraud_rate",
                                                   "historical_base_rate", "key_differences")}
                          | {"claim_ids": [n.get("claim_id") or f"PN-{n.get('policy_number')}" for n in sim.get("neighbours", [])],
                             "source": (sim.get("provenance") or {}).get("record_type")} if sim.get("ok") else None,
        "claim_history": {k: ch.get(k) for k in ("status", "history_available", "previous_claim_count",
                                                 "previous_claims", "limitation", "data_source")}
                         if ch.get("ok") else None,
    }
    warnings = [f"{r['rule_id']}: {r['message']}" for r in v.get("failed_rules", []) if r.get("severity") == "warning"]
    if (fs.get("model") or {}).get("placeholder"):
        warnings.append("Scores come from a placeholder model, not the final trained model.")
    if max_steps_reached:
        warnings.append("The orchestrator reached its step limit before finishing.")
    done = [s for s in required if statuses.get(s) in DONE]
    if v.get("blocking"):
        inv_status = "blocked"
    elif len(done) < len(required):
        inv_status = "incomplete"
    else:
        inv_status = "complete"
    return {
        "investigation_id": investigation_id, "claim_id": claim_id,
        "tool_statuses": {s: {"status": statuses[s], "origin": tool_origin.get(s),
                              "detail": (tr.get(s) or {}).get("error") if statuses[s] == "failed" else None,
                              "data_origin": SIMULATED if uses_simulated(tr.get(s)) else None}
                          for s in ALL_SOURCES},
        "simulated_data_used": [s for s in ALL_SOURCES if uses_simulated(tr.get(s))],
        "findings": findings,
        "missing_information": _missing(tr, statuses, required),
        "conflicts": find_conflicts(tr),
        "warnings": warnings,
        "completeness": {"required": list(required), "completed": done,
                         "fraction": round(len(done) / len(required), 3) if required else 1.0},
        "investigation_status": inv_status,
    }


def reason_codes(decision: str, record: Mapping[str, Any], tool_results: Mapping[str, Any]) -> list[str]:
    tr, codes = tool_results, []
    v, fs, pv, sim, ch = (tr.get(k) or {} for k in ("validation", "fraud_score", "policy_verification",
                                                     "similar_claims", "claim_history"))
    if v.get("blocking"):
        codes.append("BLOCKING_VALIDATION")
    if fs.get("ok"):
        codes.append({"high": "HIGH_MODEL_RISK", "medium": "MEDIUM_MODEL_RISK"}.get(fs.get("risk_band"), "LOW_MODEL_RISK"))
        if _warning_indicators(fs):
            codes.append("RED_FLAGS_PRESENT")
    else:
        codes.append("MODEL_SCORE_UNAVAILABLE")
    claim_fails, policy_fails = pv_failures(pv)
    if claim_fails:
        codes.append("CLAIM_RECORD_CONTRADICTION")
    if policy_fails:
        codes.append("POLICY_CONDITION_FAILED")
    if pv.get("ok") and pv.get("unknown_checks"):
        codes.append("POLICY_TERMS_UNVERIFIED")
    if sim.get("ok") and sim.get("k"):
        if similar_fraud_elevated(sim):
            codes.append("SIMILAR_CLAIMS_ELEVATED_FRAUD")
        elif sim.get("neighbour_frauds", 0) == 0:
            codes.append("SIMILAR_CLAIMS_NO_FRAUD")
    # Only above the low band: in a low-band claim one strong driver is outweighed by the rest
    # (e.g. PN-5736: fault = Policy Holder +0.54 log-odds, probability 0.0086).
    if fs.get("ok") and fs.get("risk_band") != "low" and strong_shap_drivers(tr.get("shap_explanation")):
        codes.append("STRONG_SHAP_DRIVER")
    if ch.get("ok") and ch.get("status") == "unavailable":
        codes.append("CLAIM_HISTORY_UNAVAILABLE")
    if record.get("conflicts"):
        codes.append("CONFLICTING_EVIDENCE")
    if any(s["status"] == "failed" for s in (record.get("tool_statuses") or {}).values()):
        codes.append("TOOL_FAILURE")
    if record.get("investigation_status") == "incomplete":
        codes.append("INCOMPLETE_INVESTIGATION")
    if record.get("simulated_data_used"):
        codes.append("SIMULATED_DATA_USED")
    concerns = {"BLOCKING_VALIDATION", "HIGH_MODEL_RISK", "MEDIUM_MODEL_RISK", "RED_FLAGS_PRESENT",
                "CLAIM_RECORD_CONTRADICTION", "POLICY_CONDITION_FAILED", "SIMILAR_CLAIMS_ELEVATED_FRAUD",
                "STRONG_SHAP_DRIVER", "CONFLICTING_EVIDENCE", "TOOL_FAILURE",
                "INCOMPLETE_INVESTIGATION", "MODEL_SCORE_UNAVAILABLE"}
    if decision == "APPROVE" and not concerns & set(codes):
        codes.append("NO_MATERIAL_CONCERNS")
    return codes


def compact_for_prompt(source: str, out: Mapping[str, Any] | None) -> Any:
    """A smaller view of a tool output for the reasoning prompt (every value is copied from the
    full output, so facts quoted from it still pass the grounding check)."""
    if not out or not out.get("ok", False):
        return out
    o = dict(out)
    if source == "validation":
        return {"blocking": o.get("blocking"), "schema_issues": o.get("schema_issues"),
                "failed_rules": [{k: r[k] for k in ("rule_id", "severity", "message")} for r in o.get("failed_rules", [])]}
    if source == "fraud_score":
        return {k: o.get(k) for k in ("fraud_probability", "risk_band", "predicted_class", "decision_threshold",
                                      "base_rate", "model_version")} | {
            "risk_indicators": [{k: f.get(k) for k in ("rule_id", "severity", "condition", "support")}
                                for f in o.get("risk_indicators", [])]}
    if source == "shap_explanation":
        return {"scale": o.get("scale"), "base_value": o.get("base_value"),
                "top_factors": [{k: f.get(k) for k in ("feature", "value", "shap_value", "direction")}
                                for f in o.get("top_factors", [])]}
    if source == "policy_retrieval":
        keep = ("status", "source", "wording_note", "policy_id", "document_found", "accident_period",
                "recorded_policy_attributes", "applicable_version", "document")
        return {k: o.get(k) for k in keep if o.get(k) is not None}
    if source == "policy_verification":
        return {"overall_result": o.get("overall_result"), "counts": o.get("counts"),
                "policy_source": o.get("policy_source"), "source": o.get("source"),
                "checks": [{k: c[k] for k in ("check_id", "level", "result", "reason") if k in c}
                           for c in o.get("checks", [])]}
    if source == "similar_claims":
        return {k: o.get(k) for k in ("k", "neighbour_frauds", "neighbour_fraud_rate", "historical_base_rate",
                                      "shared_by_all_matches", "key_differences")} | {
            "neighbours": [{k: n.get(k) for k in ("claim_id", "similarity", "outcome")} for n in o.get("neighbours", [])],
            "source": (o.get("provenance") or {}).get("record_type")}
    if source == "claim_history":
        return {k: o.get(k) for k in ("status", "source", "history_available", "previous_claim_count",
                                      "previous_claims", "note", "limitation", "recorded_summary") if k in o}
    return o
