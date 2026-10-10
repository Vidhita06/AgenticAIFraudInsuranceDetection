"""Guardrails enforced in code (not only in the prompt):

1. Only APPROVE / FLAG_FOR_INVESTIGATION / REQUEST_MORE_INFO; never a denial.
2. Any blocking validation failure forces REQUEST_MORE_INFO.
3. A `high` risk band can never be APPROVE (it becomes FLAG_FOR_INVESTIGATION).
4. Every evidence fact must be traceable to a tool output; untraceable facts are removed
   and logged.
5. If the LLM fails or returns invalid output twice, a deterministic rule-based decision
   is used (`rule_based_decision`).
6. Everything is marked `requires_human_review = True`.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from src.schemas.decision import ALLOWED_DECISIONS, DecisionDraft, Evidence, TriageDecision

_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)")
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
    pc = tool_results.get("policy_check") or {}
    sim = tool_results.get("similar_claims") or {}
    evidence: list[Evidence] = []
    if v.get("blocking"):
        failed = [r["rule_id"] for r in v.get("failed_rules", []) if r.get("severity") == "blocking"]
        issues = v.get("schema_issues", [])
        evidence.append(Evidence(source_tool="validation",
                                 fact=f"Blocking validation failures: {', '.join(failed) or 'schema'}"
                                      + (f" ({'; '.join(issues[:3])})" if issues else "")))
        decision = "REQUEST_MORE_INFO"
        text = ("The claim cannot be assessed until the missing or invalid information is provided "
                f"({', '.join(failed) or 'schema issues'}).")
    else:
        band = s.get("risk_band")
        p = s.get("fraud_probability")
        if s.get("ok"):
            evidence.append(Evidence(source_tool="fraud_score",
                                     fact=f"Fraud probability {p} ({band} risk band)"))
        flags = [f["rule_id"] for f in pc.get("red_flags", [])] if pc.get("ok") else []
        if flags:
            evidence.append(Evidence(source_tool="policy_check", fact="Red flags: " + ", ".join(flags)))
        if sim.get("ok") and sim.get("k"):
            evidence.append(Evidence(source_tool="similar_claims",
                                     fact=f"{sim['neighbour_frauds']} of {sim['k']} most similar past claims were fraudulent"))
        if band == "high":
            decision = "FLAG_FOR_INVESTIGATION"
            text = f"The fraud model places this claim in the high risk band (probability {p})."
        elif not s.get("ok"):
            decision = "REQUEST_MORE_INFO"
            text = "The fraud score could not be computed, so the claim needs manual assessment."
        else:
            decision = "APPROVE"
            text = f"No blocking issues and the fraud model places the claim in the {band} risk band (probability {p})."
    return DecisionDraft(decision=decision, rationale=f"{text} [Rule-based decision: {reason}]",
                         evidence=evidence)


# ---------------------------------------------------------------------------
# Apply all guardrails
# ---------------------------------------------------------------------------

def apply_guardrails(draft: DecisionDraft | None, claim_id: str, tool_results: Mapping[str, Any],
                     decided_by: str = "llm", fallback_reason: str = "") -> tuple[TriageDecision, list[dict]]:
    """Return the final TriageDecision and a list of intervention records for the audit log."""
    interventions: list[dict] = []
    if draft is None:
        draft = rule_based_decision(claim_id, tool_results, fallback_reason or "no valid LLM output")
        decided_by = "rule_fallback"
        interventions.append({"guardrail": "fallback", "detail": fallback_reason or "no valid LLM output"})

    decision = str(draft.decision).strip().upper().replace(" ", "_")
    if decision not in ALLOWED_DECISIONS:
        interventions.append({"guardrail": "allowed_decisions", "detail": f"{draft.decision!r} is not allowed"})
        fb = rule_based_decision(claim_id, tool_results, f"LLM proposed {draft.decision!r}")
        decision, draft, decided_by = fb.decision, fb, "rule_fallback"

    v = tool_results.get("validation") or {}
    s = tool_results.get("fraud_score") or {}
    if v.get("blocking") and decision != "REQUEST_MORE_INFO":
        interventions.append({"guardrail": "blocking_validation", "detail": f"{decision} -> REQUEST_MORE_INFO"})
        decision = "REQUEST_MORE_INFO"
    if s.get("risk_band") == "high" and decision == "APPROVE":
        interventions.append({"guardrail": "high_risk_not_approve", "detail": "APPROVE -> FLAG_FOR_INVESTIGATION"})
        decision = "FLAG_FOR_INVESTIGATION"

    kept: list[Evidence] = []
    for ev in draft.evidence:
        ok, why = is_grounded(ev.fact, tool_results.get(ev.source_tool))
        if ok:
            kept.append(ev)
        else:
            interventions.append({"guardrail": "evidence_grounding", "detail": f"removed [{ev.source_tool}] "
                                  f"{ev.fact!r}: {why}"})

    notes = [f"{i['guardrail']}: {i['detail']}" for i in interventions]
    rationale = draft.rationale
    if any(i["guardrail"] in ("blocking_validation", "high_risk_not_approve") for i in interventions):
        rationale += " [Decision adjusted by guardrails: " + "; ".join(
            i["detail"] for i in interventions if i["guardrail"] in ("blocking_validation", "high_risk_not_approve")) + "]"
    sim = tool_results.get("similar_claims") or {}
    final = TriageDecision(
        claim_id=claim_id, decision=decision,
        fraud_probability=s.get("fraud_probability"), risk_band=s.get("risk_band"),
        rationale=rationale, evidence=kept,
        validation_issues=[r["message"] for r in v.get("failed_rules", []) if r.get("severity") in ("blocking", "warning")]
                          + list(v.get("schema_issues", [])),
        similar_claim_ids=[f"PN-{n['policy_number']}" for n in sim.get("neighbours", [])] if sim.get("ok") else [],
        decided_by=decided_by, guardrail_notes=notes, model_version=s.get("model_version"))
    return final, interventions
