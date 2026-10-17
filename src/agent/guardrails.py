"""Guardrails enforced in code (not only in the prompt):

1. Only APPROVE / FLAG_FOR_INVESTIGATION / REQUEST_MORE_INFORMATION; never a denial.
2. Any blocking validation failure forces REQUEST_MORE_INFORMATION.
3. A `high` risk band can never be APPROVE (it becomes FLAG_FOR_INVESTIGATION). A high score
   means "investigate", never "reject".
4. APPROVE needs a completed fraud score and no policy-verification FAIL; otherwise the claim
   needs more information (manual scoring, or clarification of the failed policy check).
5. Every evidence fact must be traceable to the output of the tool it cites; untraceable
   facts are removed and logged. A tool that reported `unavailable` can only be cited for
   that limitation (so missing history is never presented as a clean history).
6. If the LLM fails or returns invalid output twice, a deterministic rule-based decision
   is used (`rule_based_decision`).
7. Reason codes, missing information, conflicts, tool statuses and findings are copied from
   the evidence record (code), never taken from the LLM.
8. Everything is marked `requires_human_review = True`.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from src.agent.evidence import aggregate_evidence, reason_codes
from src.schemas.decision import ALLOWED_DECISIONS, DecisionDraft, Evidence, TriageDecision, normalize_decision

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

def rule_based_decision(claim_id: str, tool_results: Mapping[str, Any], reason: str) -> DecisionDraft:
    """Deterministic decision from tool outputs (used when the LLM is unavailable or fails)."""
    v = tool_results.get("validation") or {}
    s = tool_results.get("fraud_score") or {}
    shap = tool_results.get("shap_explanation") or {}
    pv = tool_results.get("policy_verification") or {}
    sim = tool_results.get("similar_claims") or {}
    ch = tool_results.get("claim_history") or {}
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
    if band == "high":
        decision = "FLAG_FOR_INVESTIGATION"
        text = (f"The fraud model places this claim in the high risk band (probability {p}), which warrants "
                "investigation; the score alone does not establish fraud.")
    elif not s.get("ok"):
        decision = "REQUEST_MORE_INFORMATION"
        text = "The fraud score could not be computed, so the claim needs manual assessment."
    elif pv.get("ok") and pv.get("failed_checks"):
        decision = "REQUEST_MORE_INFORMATION"
        text = (f"Policy verification failed ({', '.join(pv['failed_checks'])}); the claim needs clarification "
                f"before it can be assessed. The fraud model places it in the {band} risk band (probability {p}).")
    else:
        decision = "APPROVE"
        text = f"No blocking issues and the fraud model places the claim in the {band} risk band (probability {p})."
    return DecisionDraft(decision=decision, rationale=f"{text} [Rule-based decision: {reason}]", evidence=evidence)


# ---------------------------------------------------------------------------
# Apply all guardrails
# ---------------------------------------------------------------------------

def apply_guardrails(draft: DecisionDraft | None, claim_id: str, tool_results: Mapping[str, Any],
                     decided_by: str = "llm", fallback_reason: str = "",
                     record: Mapping[str, Any] | None = None) -> tuple[TriageDecision, list[dict]]:
    """Return the final TriageDecision and a list of intervention records for the audit log.
    `record` is the evidence record from aggregate_evidence (built here if not given)."""
    if record is None:
        record = aggregate_evidence("", claim_id, tool_results, {}, ("fraud_score",))
    interventions: list[dict] = []
    if draft is None:
        draft = rule_based_decision(claim_id, tool_results, fallback_reason or "no valid LLM output")
        decided_by = "rule_fallback"
        interventions.append({"guardrail": "fallback", "detail": fallback_reason or "no valid LLM output"})

    decision = normalize_decision(draft.decision)
    if decision not in ALLOWED_DECISIONS:
        interventions.append({"guardrail": "allowed_decisions", "detail": f"{draft.decision!r} is not allowed"})
        fb = rule_based_decision(claim_id, tool_results, f"LLM proposed {draft.decision!r}")
        decision, draft, decided_by = fb.decision, fb, "rule_fallback"

    v = tool_results.get("validation") or {}
    s = tool_results.get("fraud_score") or {}
    pv = tool_results.get("policy_verification") or {}
    if v.get("blocking") and decision != "REQUEST_MORE_INFORMATION":
        interventions.append({"guardrail": "blocking_validation", "detail": f"{decision} -> REQUEST_MORE_INFORMATION"})
        decision = "REQUEST_MORE_INFORMATION"
    if s.get("risk_band") == "high" and decision == "APPROVE":
        interventions.append({"guardrail": "high_risk_not_approve", "detail": "APPROVE -> FLAG_FOR_INVESTIGATION"})
        decision = "FLAG_FOR_INVESTIGATION"
    if decision == "APPROVE" and not s.get("ok"):
        interventions.append({"guardrail": "approve_needs_score",
                              "detail": "APPROVE -> REQUEST_MORE_INFORMATION (no fraud score)"})
        decision = "REQUEST_MORE_INFORMATION"
    if decision == "APPROVE" and pv.get("ok") and pv.get("failed_checks"):
        interventions.append({"guardrail": "approve_needs_policy_pass",
                              "detail": "APPROVE -> REQUEST_MORE_INFORMATION (policy check FAIL: "
                                        + ", ".join(pv["failed_checks"]) + ")"})
        decision = "REQUEST_MORE_INFORMATION"

    kept: list[Evidence] = []
    for ev in draft.evidence:
        ok, why = is_grounded(ev.fact, tool_results.get(ev.source_tool))
        if ok:
            kept.append(ev)
        else:
            interventions.append({"guardrail": "evidence_grounding", "detail": f"removed [{ev.source_tool}] "
                                  f"{ev.fact!r}: {why}"})

    notes = [f"{i['guardrail']}: {i['detail']}" for i in interventions]
    adjusting = ("blocking_validation", "high_risk_not_approve", "approve_needs_score", "approve_needs_policy_pass")
    rationale = draft.rationale
    if any(i["guardrail"] in adjusting for i in interventions):
        rationale += " [Decision adjusted by guardrails: " + "; ".join(
            i["detail"] for i in interventions if i["guardrail"] in adjusting) + "]"
    sim = tool_results.get("similar_claims") or {}
    f = record.get("findings") or {}
    final = TriageDecision(
        claim_id=claim_id, investigation_id=record.get("investigation_id") or None,
        investigation_status=record.get("investigation_status"), decision=decision,
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
