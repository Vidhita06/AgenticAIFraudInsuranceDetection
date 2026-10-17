"""Guardrails enforced in code (not only in the prompt):

1. Only APPROVE / FLAG_FOR_INVESTIGATION / REQUEST_MORE_INFORMATION; never a denial.
2. REQUEST_MORE_INFORMATION is only for claim-level problems: blocking validation (missing or
   invalid fields) or a contradiction inside the claim record (PV02, PV04). Blocking
   validation forces it; an LLM request for more information without such grounds is replaced
   by the rule-based decision.
3. A `high` risk band can never be APPROVE (it becomes FLAG_FOR_INVESTIGATION). A high score
   means "investigate", never "reject".
4. APPROVE needs a completed fraud score, every required tool to have run, a consistent claim
   record and no policy-condition FAIL. Otherwise: claim-record contradiction ->
   REQUEST_MORE_INFORMATION; missing score or failed tool -> FLAG (SYSTEM_FAILURE); policy
   condition FAIL -> FLAG (POLICY_CONDITION).
4b. Every FLAG carries a category: FRAUD_RISK, POLICY_CONDITION or SYSTEM_FAILURE. Only
   FRAUD_RISK flags count as fraud caught in the evaluation.
4c. A FRAUD_RISK flag needs a risk signal. If every reason code is a low-risk or systemic one
   (LOW_RISK_REASON_CODES: low band, unverifiable policy terms, unavailable history, no fraud
   among similar claims, simulated data), the flag rests only on known data-source gaps and
   is replaced by the rule-based decision (`unsupported_fraud_flag`). In the medium band a
   FRAUD_RISK flag needs at least one concrete factor (`agent.medium_band_concrete_factors`:
   warning red flag, policy-condition FAIL, fraud among similar claims, strong SHAP driver,
   conflicting evidence); a medium band alone is not enough.
5. Every evidence fact must be traceable to the output of the tool it cites; untraceable
   facts are removed and logged. A tool that reported `unavailable` can only be cited for
   that limitation (so missing history is never presented as a clean history).
6. If the LLM fails or returns invalid output twice, a deterministic rule-based decision
   is used (`rule_based_decision`).
7. Reason codes, missing information, conflicts, tool statuses and findings are copied from
   the evidence record (code), never taken from the LLM. Evidence drawn from simulated
   reference data is labelled "(simulated reference data)".
8. Everything is marked `requires_human_review = True`.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from src.agent.evidence import (SIMULATED, aggregate_evidence, claim_level_grounds, concrete_factor_codes,
                                failed_required, flag_category, pv_failures, reason_codes, uses_simulated)
from src.schemas.decision import ALLOWED_DECISIONS, DecisionDraft, Evidence, TriageDecision, normalize_decision

# Reason codes that are no risk signal on their own: a low score, or gaps of the data source that
# apply to every claim (no policy document, no linked history), or simulated data.
LOW_RISK_REASON_CODES = frozenset({"LOW_MODEL_RISK", "POLICY_TERMS_UNVERIFIED", "CLAIM_HISTORY_UNAVAILABLE",
                                   "SIMILAR_CLAIMS_NO_FRAUD", "SIMULATED_DATA_USED", "NO_MATERIAL_CONCERNS"})
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)")
_MISSING_WORDS = re.compile(r"unavailable|not available|no (?:authori[sz]ed|linked)|cannot|could not|unknown|"
                            r"not (?:configured|recorded|on file)|missing", re.I)
_IDS = re.compile(r"\b(?:RF\d{2}|V\d{2}B?)(?:_[A-Z0-9_]+)?\b")


# ---------------------------------------------------------------------------
# Evidence grounding
# ---------------------------------------------------------------------------

def _corpus_numbers(text: str) -> list[float]:
    return [float(m.group(1)) for m in _NUM.finditer(text)]


def _matches(value: float, decimals: int, numbers: list[float]) -> bool:
    for n in numbers:
        if round(n, decimals) == round(value, decimals):
            return True
        if decimals == 0 and abs(n - value) < 1e-9:
            return True
    return False


def is_grounded(fact: str, source_output: Any) -> tuple[bool, str]:
    """A fact is grounded if its source tool ran successfully, every number it states
    appears in that tool's output (a percentage may also appear as a fraction), and every
    rule / red-flag id it cites appears there too."""
    if not isinstance(source_output, Mapping) or not source_output.get("ok", False):
        return False, "source tool did not run or failed"
    if source_output.get("status") == "unavailable" and not _MISSING_WORDS.search(fact):
        return False, "the source reported its data as unavailable; only that limitation can be cited"
    text = json.dumps(source_output, default=str)
    numbers = _corpus_numbers(text)
    for m in _NUM.finditer(fact):
        raw, pct = m.group(1), m.group(2)
        val = float(raw)
        dec = len(raw.split(".")[1]) if "." in raw else 0
        ok = _matches(val, dec, numbers)
        if not ok and pct:
            ok = _matches(val / 100, dec + 2, numbers)
        if not ok and not pct and val <= 1 and dec:
            ok = _matches(val * 100, max(dec - 2, 0), numbers)
        if not ok:
            return False, f"number {raw}{pct} not found in {source_output.get('claim_id', 'tool')} output"
    for rid in _IDS.findall(fact):
        if rid not in text:
            return False, f"id {rid} not found in tool output"
    return True, ""


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def rule_based_decision(claim_id: str, tool_results: Mapping[str, Any], reason: str,
                        required: tuple[str, ...] = ()) -> DecisionDraft:
    """Deterministic decision from tool outputs (used when the LLM is unavailable or fails).

    blocking validation / claim-record contradiction -> REQUEST_MORE_INFORMATION
    no fraud score or a failed required tool          -> FLAG (SYSTEM_FAILURE)
    high risk band                                    -> FLAG (FRAUD_RISK)
    policy-condition FAIL                             -> FLAG (POLICY_CONDITION)
    otherwise                                         -> APPROVE
    """
    v = tool_results.get("validation") or {}
    s = tool_results.get("fraud_score") or {}
    shap = tool_results.get("shap_explanation") or {}
    pv = tool_results.get("policy_verification") or {}
    sim = tool_results.get("similar_claims") or {}
    ch = tool_results.get("claim_history") or {}
    claim_fails, policy_fails = pv_failures(pv)
    evidence: list[Evidence] = []
    if v.get("blocking"):
        failed = [r["rule_id"] for r in v.get("failed_rules", []) if r.get("severity") == "blocking"]
        issues = v.get("schema_issues", [])
        evidence.append(Evidence(source_tool="validation",
                                 fact=f"Blocking validation failures: {', '.join(failed) or 'schema'}"
                                      + (f" ({'; '.join(issues[:3])})" if issues else "")))
        return DecisionDraft(decision="REQUEST_MORE_INFORMATION", evidence=evidence,
                             rationale="The claim cannot be assessed until the missing or invalid information is "
                                       f"provided ({', '.join(failed) or 'schema issues'}). [Rule-based decision: {reason}]")
    band, p = s.get("risk_band"), s.get("fraud_probability")
    if s.get("ok"):
        evidence.append(Evidence(source_tool="fraud_score", fact=f"Fraud probability {p} ({band} risk band)"))
        flags = [f["rule_id"] for f in s.get("risk_indicators", [])]
        if flags:
            evidence.append(Evidence(source_tool="fraud_score", fact="Red flags: " + ", ".join(flags)))
    if shap.get("ok") and shap.get("top_factors"):
        f = shap["top_factors"][0]
        evidence.append(Evidence(source_tool="shap_explanation",
                                 fact=f"Largest SHAP contribution: {f['feature']} ({f['shap_value']})"))
    if pv.get("ok"):
        fails = pv.get("failed_checks") or []
        evidence.append(Evidence(source_tool="policy_verification",
                                 fact=(f"Policy checks failed: {', '.join(fails)}" if fails else
                                       f"Policy verification: {pv['counts']['PASS']} PASS, {pv['counts']['UNKNOWN']} UNKNOWN, 0 FAIL")))
    if sim.get("ok") and sim.get("k"):
        evidence.append(Evidence(source_tool="similar_claims",
                                 fact=f"{sim['neighbour_frauds']} of {sim['k']} most similar past claims were fraudulent"))
    if ch.get("ok") and ch.get("status") == "unavailable":
        evidence.append(Evidence(source_tool="claim_history", fact="Claim history unavailable for this policy"))
    elif ch.get("ok") and ch.get("history_available"):
        evidence.append(Evidence(source_tool="claim_history",
                                 fact=f"{ch.get('previous_claim_count')} previous claims in the look-back window"))
    tool_fails = failed_required(tool_results, required)
    category = None
    if claim_fails:
        decision = "REQUEST_MORE_INFORMATION"
        text = (f"The claim record is internally inconsistent ({', '.join(claim_fails)}); the claimant needs to "
                "clarify it before the claim can be assessed.")
    elif not s.get("ok"):
        decision, category = "FLAG_FOR_INVESTIGATION", "SYSTEM_FAILURE"
        text = "The fraud score could not be computed, so the claim needs manual assessment."
    elif band == "high":
        decision, category = "FLAG_FOR_INVESTIGATION", "FRAUD_RISK"
        text = (f"The fraud model places this claim in the high risk band (probability {p}), which warrants "
                "investigation; the score alone does not establish fraud.")
    elif policy_fails:
        decision, category = "FLAG_FOR_INVESTIGATION", "POLICY_CONDITION"
        text = (f"Policy verification failed ({', '.join(policy_fails)}): a coverage question for an adjuster, "
                f"not a fraud finding. The fraud model places the claim in the {band} risk band (probability {p}).")
    elif tool_fails:
        decision, category = "FLAG_FOR_INVESTIGATION", "SYSTEM_FAILURE"
        text = f"Required checks failed to run ({', '.join(tool_fails)}), so the claim needs manual assessment."
    else:
        decision = "APPROVE"
        text = f"No blocking issues and the fraud model places the claim in the {band} risk band (probability {p})."
    return DecisionDraft(decision=decision, rationale=f"{text} [Rule-based decision: {reason}]", evidence=evidence,
                         flag_category=category)


# ---------------------------------------------------------------------------
# Apply all guardrails
# ---------------------------------------------------------------------------

def _label_simulated(ev: Evidence, output: Mapping[str, Any] | None) -> Evidence:
    if uses_simulated(output) and SIMULATED not in ev.fact.lower():
        suffix = f" ({SIMULATED})"
        return Evidence(source_tool=ev.source_tool, fact=ev.fact[:400 - len(suffix)] + suffix)
    return ev


def apply_guardrails(draft: DecisionDraft | None, claim_id: str, tool_results: Mapping[str, Any],
                     decided_by: str = "llm", fallback_reason: str = "",
                     record: Mapping[str, Any] | None = None, required: tuple[str, ...] = (),
                     simulated_data: bool = False) -> tuple[TriageDecision, list[dict]]:
    """Return the final TriageDecision and a list of intervention records for the audit log.
    `record` is the evidence record from aggregate_evidence (built here if not given);
    `required` lists the evidence sources that must have run before APPROVE."""
    if record is None:
        record = aggregate_evidence("", claim_id, tool_results, {}, required or ("fraud_score",))
    interventions: list[dict] = []
    category = None
    if draft is None:
        draft = rule_based_decision(claim_id, tool_results, fallback_reason or "no valid LLM output", required)
        decided_by = "rule_fallback"
        interventions.append({"guardrail": "fallback", "detail": fallback_reason or "no valid LLM output"})

    decision = normalize_decision(draft.decision)
    if decision not in ALLOWED_DECISIONS:
        interventions.append({"guardrail": "allowed_decisions", "detail": f"{draft.decision!r} is not allowed"})
        draft = rule_based_decision(claim_id, tool_results, f"LLM proposed {draft.decision!r}", required)
        decision, decided_by = draft.decision, "rule_fallback"
    if decided_by == "rule_fallback":
        category = draft.flag_category

    v = tool_results.get("validation") or {}
    s = tool_results.get("fraud_score") or {}
    grounds = claim_level_grounds(tool_results)
    claim_fails, policy_fails = pv_failures(tool_results.get("policy_verification"))
    tool_fails = failed_required(tool_results, required)
    if v.get("blocking") and decision != "REQUEST_MORE_INFORMATION":
        interventions.append({"guardrail": "blocking_validation", "detail": f"{decision} -> REQUEST_MORE_INFORMATION"})
        decision, category = "REQUEST_MORE_INFORMATION", None
    if decision == "REQUEST_MORE_INFORMATION" and not grounds:
        fb = rule_based_decision(claim_id, tool_results, "more information requested without a claim-level reason",
                                 required)
        interventions.append({"guardrail": "request_info_not_claim_level",
                              "detail": f"REQUEST_MORE_INFORMATION -> {fb.decision}"
                                        + (f" ({fb.flag_category})" if fb.flag_category else "")})
        decision, category = fb.decision, fb.flag_category
    if s.get("risk_band") == "high" and decision == "APPROVE":
        interventions.append({"guardrail": "high_risk_not_approve", "detail": "APPROVE -> FLAG_FOR_INVESTIGATION"})
        decision, category = "FLAG_FOR_INVESTIGATION", "FRAUD_RISK"
    if decision == "APPROVE" and not s.get("ok"):
        interventions.append({"guardrail": "approve_needs_score",
                              "detail": "APPROVE -> FLAG_FOR_INVESTIGATION (SYSTEM_FAILURE: no fraud score)"})
        decision, category = "FLAG_FOR_INVESTIGATION", "SYSTEM_FAILURE"
    if decision == "APPROVE" and claim_fails:
        interventions.append({"guardrail": "approve_needs_consistent_claim",
                              "detail": "APPROVE -> REQUEST_MORE_INFORMATION (" + ", ".join(claim_fails) + ")"})
        decision = "REQUEST_MORE_INFORMATION"
    if decision == "APPROVE" and policy_fails:
        interventions.append({"guardrail": "approve_needs_policy_pass",
                              "detail": "APPROVE -> FLAG_FOR_INVESTIGATION (POLICY_CONDITION: "
                                        + ", ".join(policy_fails) + ")"})
        decision, category = "FLAG_FOR_INVESTIGATION", "POLICY_CONDITION"
    if decision == "APPROVE" and tool_fails:
        interventions.append({"guardrail": "approve_needs_tools",
                              "detail": "APPROVE -> FLAG_FOR_INVESTIGATION (SYSTEM_FAILURE: "
                                        + ", ".join(tool_fails) + " failed)"})
        decision, category = "FLAG_FOR_INVESTIGATION", "SYSTEM_FAILURE"
    if decision == "FLAG_FOR_INVESTIGATION" and category is None:
        category = flag_category(tool_results, required)
    if decision == "FLAG_FOR_INVESTIGATION" and category == "FRAUD_RISK":
        codes = set(reason_codes(decision, record, tool_results))
        why = None
        if codes <= LOW_RISK_REASON_CODES:
            why = "reason codes only " + ", ".join(sorted(codes))
        elif s.get("risk_band") == "medium" and not codes & concrete_factor_codes():
            why = "medium band without a concrete factor; reason codes " + ", ".join(sorted(codes))
        if why:
            fb = rule_based_decision(claim_id, tool_results, "fraud flag without a risk signal", required)
            interventions.append({"guardrail": "unsupported_fraud_flag",
                                  "detail": f"FLAG_FOR_INVESTIGATION -> {fb.decision} ({why})"})
            decision, category = fb.decision, fb.flag_category
    if decision != "FLAG_FOR_INVESTIGATION":
        category = None

    kept: list[Evidence] = []
    for ev in draft.evidence:
        ok, why = is_grounded(ev.fact, tool_results.get(ev.source_tool))
        if ok:
            kept.append(_label_simulated(ev, tool_results.get(ev.source_tool)))
        else:
            interventions.append({"guardrail": "evidence_grounding", "detail": f"removed [{ev.source_tool}] "
                                  f"{ev.fact!r}: {why}"})

    notes = [f"{i['guardrail']}: {i['detail']}" for i in interventions]
    adjusting = ("blocking_validation", "request_info_not_claim_level", "high_risk_not_approve", "approve_needs_score",
                 "approve_needs_consistent_claim", "approve_needs_policy_pass", "approve_needs_tools",
                 "unsupported_fraud_flag")
    rationale = draft.rationale
    if any(i["guardrail"] in adjusting for i in interventions):
        rationale += " [Decision adjusted by guardrails: " + "; ".join(
            i["detail"] for i in interventions if i["guardrail"] in adjusting) + "]"
    sim = tool_results.get("similar_claims") or {}
    f = record.get("findings") or {}
    final = TriageDecision(
        claim_id=claim_id, investigation_id=record.get("investigation_id") or None,
        investigation_status=record.get("investigation_status"), decision=decision, flag_category=category,
        simulated_data=bool(simulated_data),
        fraud_probability=s.get("fraud_probability"), risk_band=s.get("risk_band"),
        predicted_class=s.get("predicted_class"),
        reason_codes=reason_codes(decision, record, tool_results),
        rationale=rationale, evidence=kept,
        policy_findings={"retrieval": f.get("policy_retrieval"), "verification": f.get("policy_verification")},
        similar_claim_findings=f.get("similar_claims") or {},
        claim_history_summary=f.get("claim_history") or {},
        missing_information=list(record.get("missing_information", [])),
        conflicting_evidence=list(record.get("conflicts", [])),
        unresolved_questions=[q for q in draft.unresolved_questions if isinstance(q, str)][:8],
        tool_statuses=record.get("tool_statuses", {}),
        validation_issues=[r["message"] for r in v.get("failed_rules", []) if r.get("severity") in ("blocking", "warning")]
                          + list(v.get("schema_issues", [])),
        similar_claim_ids=[n.get("claim_id") or f"PN-{n['policy_number']}" for n in sim.get("neighbours", [])]
                          if sim.get("ok") else [],
        decided_by=decided_by, guardrail_notes=notes, model_version=s.get("model_version"))
    return final, interventions
