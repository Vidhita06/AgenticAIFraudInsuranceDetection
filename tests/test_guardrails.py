"""Every guardrail, applied directly to drafts (no graph, no LLM)."""
from src.agent.guardrails import apply_guardrails, is_grounded, rule_based_decision
from src.schemas.decision import DecisionDraft, Evidence

LOW = {"ok": True, "claim_id": "PN-1", "fraud_probability": 0.0312, "risk_band": "low", "model_version": "m"}
HIGH = {**LOW, "fraud_probability": 0.4123, "risk_band": "high"}
OK_VAL = {"ok": True, "blocking": False, "failed_rules": [], "schema_issues": []}
BLOCK_VAL = {"ok": True, "blocking": True, "schema_issues": [],
             "failed_rules": [{"rule_id": "V03_AGE_MISSING", "severity": "blocking", "message": "Age is 0"}]}
POLICY = {"ok": True, "claim_id": "PN-1", "red_flags": [
    {"rule_id": "RF04_THIRD_PARTY_DEDUCTIBLE_500", "severity": "warning",
     "support": "54.5% fraud in 55 real training claims (lift 9.11)"}]}
SIM = {"ok": True, "k": 5, "neighbour_frauds": 3, "neighbour_fraud_rate": 0.6,
       "neighbours": [{"policy_number": 7, "similarity": 0.91, "fraud_label": 1}]}


def tools(score=LOW, val=OK_VAL, **extra):
    return {"validation": val, "fraud_score": score, **extra}


def draft(decision, *evidence):
    return DecisionDraft(decision=decision, rationale="Because of the evidence listed below.",
                         evidence=[Evidence(source_tool=s, fact=f) for s, f in evidence])


def test_only_allowed_decisions_and_never_deny():
    d, iv = apply_guardrails(draft("DENY"), "PN-1", tools(HIGH))
    assert d.decision == "FLAG_FOR_INVESTIGATION" and d.decided_by == "rule_fallback"
    assert any(i["guardrail"] == "allowed_decisions" for i in iv)
    d, _ = apply_guardrails(draft("reject claim"), "PN-1", tools(LOW))
    assert d.decision == "APPROVE"


def test_lowercase_decision_is_normalised():
    d, iv = apply_guardrails(draft("flag for investigation"), "PN-1", tools(HIGH))
    assert d.decision == "FLAG_FOR_INVESTIGATION" and d.decided_by == "llm" and not iv


def test_blocking_validation_forces_request_more_info():
    for proposed in ("APPROVE", "FLAG_FOR_INVESTIGATION"):
        d, iv = apply_guardrails(draft(proposed), "PN-1", tools(HIGH, BLOCK_VAL))
        assert d.decision == "REQUEST_MORE_INFO"
        assert any(i["guardrail"] == "blocking_validation" for i in iv)
        assert "Age is 0" in d.validation_issues


def test_high_risk_band_cannot_be_approved():
    d, iv = apply_guardrails(draft("APPROVE"), "PN-1", tools(HIGH))
    assert d.decision == "FLAG_FOR_INVESTIGATION"
    assert any(i["guardrail"] == "high_risk_not_approve" for i in iv)
    d, _ = apply_guardrails(draft("APPROVE"), "PN-1", tools(LOW))
    assert d.decision == "APPROVE"


def test_ungrounded_evidence_is_removed():
    d, iv = apply_guardrails(draft("FLAG_FOR_INVESTIGATION",
                                   ("fraud_score", "Fraud probability 0.4123 (high band)"),
                                   ("fraud_score", "Fraud probability 0.99"),
                                   ("policy_check", "RF04_THIRD_PARTY_DEDUCTIBLE_500: 54.5% fraud, lift 9.11"),
                                   ("policy_check", "RF99_MADE_UP flag fired"),
                                   ("similar_claims", "3 of 5 similar claims were fraud"),
                                   ("similar_claims", "neighbour fraud rate 60%"),
                                   ("similar_claims", "4 of 5 similar claims were fraud")),
                             "PN-1", tools(HIGH, policy_check=POLICY, similar_claims=SIM))
    kept = [e.fact for e in d.evidence]
    assert kept == ["Fraud probability 0.4123 (high band)",
                    "RF04_THIRD_PARTY_DEDUCTIBLE_500: 54.5% fraud, lift 9.11",
                    "3 of 5 similar claims were fraud", "neighbour fraud rate 60%"]
    assert sum(i["guardrail"] == "evidence_grounding" for i in iv) == 3


def test_evidence_from_a_tool_that_never_ran_is_removed():
    d, iv = apply_guardrails(draft("APPROVE", ("similar_claims", "0 of 5 similar claims were fraud")),
                             "PN-1", tools(LOW))
    assert d.evidence == [] and iv[0]["guardrail"] == "evidence_grounding"


def test_fallback_and_human_review_flag():
    d, iv = apply_guardrails(None, "PN-1", tools(HIGH, policy_check=POLICY), fallback_reason="LLM down")
    assert d.decided_by == "rule_fallback" and d.decision == "FLAG_FOR_INVESTIGATION"
    assert d.requires_human_review is True and iv[0]["guardrail"] == "fallback"
    for e in d.evidence:   # rule-based evidence is grounded by construction
        assert is_grounded(e.fact, tools(HIGH, policy_check=POLICY)[e.source_tool])[0]


def test_rule_based_decision_cases():
    assert rule_based_decision("x", tools(LOW), "r").decision == "APPROVE"
    assert rule_based_decision("x", tools(HIGH), "r").decision == "FLAG_FOR_INVESTIGATION"
    assert rule_based_decision("x", tools(HIGH, BLOCK_VAL), "r").decision == "REQUEST_MORE_INFO"
    assert rule_based_decision("x", tools({"ok": False, "error": "e"}), "r").decision == "REQUEST_MORE_INFO"
