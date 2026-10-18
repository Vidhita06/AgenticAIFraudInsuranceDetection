"""Every guardrail, applied directly to drafts (no graph, no LLM)."""
from src.agent.guardrails import apply_guardrails, is_grounded, rule_based_decision
from src.schemas.decision import DecisionDraft, Evidence

LOW = {"ok": True, "claim_id": "PN-1", "fraud_probability": 0.0312, "risk_band": "low", "model_version": "m"}
HIGH = {**LOW, "fraud_probability": 0.4123, "risk_band": "high"}
OK_VAL = {"ok": True, "blocking": False, "failed_rules": [], "schema_issues": []}
BLOCK_VAL = {"ok": True, "blocking": True, "schema_issues": [],
             "failed_rules": [{"rule_id": "V03_AGE_MISSING", "severity": "blocking", "message": "Age is 0"}]}
FLAGS = [{"rule_id": "RF04_THIRD_PARTY_DEDUCTIBLE_500", "severity": "warning",
          "support": "54.5% fraud in 55 real training claims (lift 9.11)"}]
HIGH_FLAGS = {**HIGH, "risk_indicators": FLAGS}
PV_FAIL = {"ok": True, "status": "completed", "overall_result": "FAIL", "counts": {"PASS": 7, "FAIL": 1, "UNKNOWN": 1},
           "failed_checks": ["PV04_CLAIM_TIMELINE"], "unknown_checks": ["PV08_COVERAGE_LIMITS"],
           "checks": [{"check_id": "PV04_CLAIM_TIMELINE", "result": "FAIL", "reason": "claim precedes accident"}]}
PV_POLICY_FAIL = {**PV_FAIL, "failed_checks": ["SIM-ALL-04"], "claim_record_failures": [],
                  "policy_condition_failures": ["SIM-ALL-04"], "source": "simulated reference data",
                  "checks": [{"check_id": "SIM-ALL-04", "level": "policy_condition", "result": "FAIL",
                              "reason": "Reported at least 75 days after the accident (deadline 60 days)."}]}
PV_OK = {**PV_FAIL, "overall_result": "UNKNOWN", "counts": {"PASS": 8, "FAIL": 0, "UNKNOWN": 1}, "failed_checks": [],
         "checks": []}
SIM = {"ok": True, "k": 5, "neighbour_frauds": 3, "neighbour_fraud_rate": 0.6, "historical_base_rate": 0.0599,
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


def test_blocking_validation_forces_request_more_information():
    for proposed in ("APPROVE", "FLAG_FOR_INVESTIGATION"):
        d, iv = apply_guardrails(draft(proposed), "PN-1", tools(HIGH, BLOCK_VAL))
        assert d.decision == "REQUEST_MORE_INFORMATION"
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
                                   ("fraud_score", "RF04_THIRD_PARTY_DEDUCTIBLE_500: 54.5% fraud, lift 9.11"),
                                   ("fraud_score", "RF99_MADE_UP flag fired"),
                                   ("similar_claims", "3 of 5 similar claims were fraud"),
                                   ("similar_claims", "neighbour fraud rate 60%"),
                                   ("similar_claims", "4 of 5 similar claims were fraud")),
                             "PN-1", tools(HIGH_FLAGS, similar_claims=SIM))
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
    d, iv = apply_guardrails(None, "PN-1", tools(HIGH_FLAGS, policy_verification=PV_OK), fallback_reason="LLM down")
    assert d.decided_by == "rule_fallback" and d.decision == "FLAG_FOR_INVESTIGATION"
    assert d.requires_human_review is True and iv[0]["guardrail"] == "fallback"
    for e in d.evidence:   # rule-based evidence is grounded by construction
        assert is_grounded(e.fact, tools(HIGH_FLAGS, policy_verification=PV_OK)[e.source_tool])[0]


def test_rule_based_decision_cases():
    assert rule_based_decision("x", tools(LOW), "r").decision == "APPROVE"
    assert rule_based_decision("x", tools(HIGH), "r").decision == "FLAG_FOR_INVESTIGATION"
    assert rule_based_decision("x", tools(HIGH, BLOCK_VAL), "r").decision == "REQUEST_MORE_INFORMATION"
    fb = rule_based_decision("x", tools({"ok": False, "error": "e"}), "r")
    assert fb.decision == "FLAG_FOR_INVESTIGATION" and fb.flag_category == "SYSTEM_FAILURE"
    fb = rule_based_decision("x", tools(LOW, policy_verification=PV_POLICY_FAIL), "r")
    assert fb.decision == "FLAG_FOR_INVESTIGATION" and fb.flag_category == "POLICY_CONDITION"
    fb = rule_based_decision("x", tools(LOW, shap_explanation={"ok": False, "error": "x"}), "r",
                             required=("fraud_score", "shap_explanation"))
    assert fb.decision == "FLAG_FOR_INVESTIGATION" and fb.flag_category == "SYSTEM_FAILURE"
    assert rule_based_decision("x", tools(LOW, policy_verification=PV_FAIL), "r").decision == "REQUEST_MORE_INFORMATION"


def test_old_label_is_accepted():
    d, iv = apply_guardrails(draft("REQUEST_MORE_INFO"), "PN-1", tools(LOW, BLOCK_VAL))
    assert d.decision == "REQUEST_MORE_INFORMATION" and not iv


def test_request_info_needs_a_claim_level_reason():
    d, iv = apply_guardrails(draft("REQUEST_MORE_INFORMATION"), "PN-1", tools(HIGH))
    assert d.decision == "FLAG_FOR_INVESTIGATION" and d.flag_category == "FRAUD_RISK"
    assert iv[0]["guardrail"] == "request_info_not_claim_level"
    d, iv = apply_guardrails(draft("REQUEST_MORE_INFORMATION"), "PN-1", tools(LOW))
    assert d.decision == "APPROVE"
    d, iv = apply_guardrails(draft("REQUEST_MORE_INFORMATION"), "PN-1", tools(LOW, policy_verification=PV_FAIL))
    assert d.decision == "REQUEST_MORE_INFORMATION" and not iv          # PV04: the claim contradicts itself


def test_every_flag_has_a_category():
    d, _ = apply_guardrails(draft("FLAG_FOR_INVESTIGATION"), "PN-1", tools(HIGH))
    assert d.flag_category == "FRAUD_RISK"
    d, _ = apply_guardrails(draft("FLAG_FOR_INVESTIGATION"), "PN-1", tools(LOW, policy_verification=PV_POLICY_FAIL))
    assert d.flag_category == "POLICY_CONDITION"
    d, _ = apply_guardrails(draft("FLAG_FOR_INVESTIGATION"), "PN-1", tools({"ok": False, "error": "x"}))
    assert d.flag_category == "SYSTEM_FAILURE"
    d, _ = apply_guardrails(draft("FLAG_FOR_INVESTIGATION"), "PN-1", tools({**LOW, "risk_indicators": FLAGS}))
    assert d.flag_category == "FRAUD_RISK"                               # low band, but a warning red flag
    d, _ = apply_guardrails(draft("APPROVE"), "PN-1", tools(LOW))
    assert d.flag_category is None


def test_simulated_evidence_is_labelled():
    hist = {"ok": True, "status": "completed", "source": "simulated reference data", "history_available": True,
            "previous_claim_count": 3}
    d, _ = apply_guardrails(draft("APPROVE", ("claim_history", "3 previous claims in the window")), "PN-1",
                            tools(LOW, claim_history=hist), simulated_data=True)
    assert d.evidence[0].fact == "3 previous claims in the window (simulated reference data)"
    assert d.simulated_data is True


def test_approve_needs_a_fraud_score():
    d, iv = apply_guardrails(draft("APPROVE"), "PN-1", tools({"ok": False, "error": "model missing"}))
    assert d.decision == "FLAG_FOR_INVESTIGATION" and d.flag_category == "SYSTEM_FAILURE"
    assert [i["guardrail"] for i in iv] == ["approve_needs_score"]
    assert "MODEL_SCORE_UNAVAILABLE" in d.reason_codes


def test_approve_needs_no_policy_fail():
    d, iv = apply_guardrails(draft("APPROVE"), "PN-1", tools(LOW, policy_verification=PV_FAIL))
    assert d.decision == "REQUEST_MORE_INFORMATION"            # PV04 is a claim-record contradiction
    assert iv[0]["guardrail"] == "approve_needs_consistent_claim" and "PV04_CLAIM_TIMELINE" in iv[0]["detail"]
    assert "CLAIM_RECORD_CONTRADICTION" in d.reason_codes
    d, iv = apply_guardrails(draft("APPROVE"), "PN-1", tools(LOW, policy_verification=PV_POLICY_FAIL))
    assert d.decision == "FLAG_FOR_INVESTIGATION" and d.flag_category == "POLICY_CONDITION"
    assert iv[0]["guardrail"] == "approve_needs_policy_pass" and "POLICY_CONDITION_FAILED" in d.reason_codes
    # UNKNOWN checks alone do not block approval
    d, iv = apply_guardrails(draft("APPROVE"), "PN-1", tools(LOW, policy_verification=PV_OK))
    assert d.decision == "APPROVE" and not iv
    assert {"LOW_MODEL_RISK", "POLICY_TERMS_UNVERIFIED", "NO_MATERIAL_CONCERNS"} <= set(d.reason_codes)


def test_new_sources_are_grounded():
    shap = {"ok": True, "top_factors": [{"feature": "fault", "value": "Policy Holder", "shap_value": 0.3476}]}
    d, iv = apply_guardrails(draft("FLAG_FOR_INVESTIGATION",
                                   ("shap_explanation", "fault = Policy Holder raises the score by 0.3476"),
                                   ("shap_explanation", "fault contributes 0.9"),
                                   ("policy_verification", "PV04_CLAIM_TIMELINE failed"),
                                   ("claim_history", "No previous claims")),
                             "PN-1", tools(HIGH, shap_explanation=shap, policy_verification=PV_FAIL))
    assert [e.source_tool for e in d.evidence] == ["shap_explanation", "policy_verification"]
    assert sum(i["guardrail"] == "evidence_grounding" for i in iv) == 2


def test_final_output_fields_come_from_the_evidence_record():
    from src.agent.evidence import aggregate_evidence
    tr = tools(LOW, policy_verification=PV_OK, similar_claims=SIM)
    rec = aggregate_evidence("INV-1", "PN-1", tr, {"fraud_score": "orchestrator"}, ("fraud_score", "policy_verification"))
    d, _ = apply_guardrails(draft("APPROVE"), "PN-1", tr, record=rec)
    assert d.investigation_id == "INV-1" and d.investigation_status == "complete"
    assert d.tool_statuses["claim_history"].status == "not_run"
    assert d.tool_statuses["fraud_score"].origin == "orchestrator"
    # low band but 3 of 5 similar claims were fraud -> conflict, so no "no material concerns"
    assert d.conflicting_evidence and "CONFLICTING_EVIDENCE" in d.reason_codes
    assert "NO_MATERIAL_CONCERNS" not in d.reason_codes
    assert any("PV08_COVERAGE_LIMITS" in m for m in d.missing_information)


def test_unavailable_history_cannot_be_cited_as_a_clean_record():
    hist = {"ok": True, "status": "unavailable", "previous_claim_count": None, "limitation": "no source"}
    d, iv = apply_guardrails(draft("APPROVE", ("claim_history", "No previous claims on this policy"),
                                   ("claim_history", "Claim history is unavailable for this policy")),
                             "PN-1", tools(LOW, claim_history=hist))
    assert [e.fact for e in d.evidence] == ["Claim history is unavailable for this policy"]
    assert "unavailable" in iv[0]["detail"]


PV_UNKNOWN = {"ok": True, "status": "completed", "overall_result": "UNKNOWN", "counts": {"PASS": 3, "FAIL": 0, "UNKNOWN": 6},
              "failed_checks": [], "claim_record_failures": [], "policy_condition_failures": [],
              "unknown_checks": ["PV01_POLICY_IDENTIFIED", "PV08_COVERAGE_LIMITS"], "checks": []}
HIST_UNAVAILABLE = {"ok": True, "status": "unavailable", "history_available": False, "previous_claim_count": None}
SIM_NONE = {"ok": True, "k": 5, "neighbour_frauds": 0, "neighbour_fraud_rate": 0.0, "historical_base_rate": 0.0599,
            "neighbours": []}


def test_fraud_flag_resting_only_on_data_gaps_is_replaced():
    """The pilot case: low band, and the only "concerns" are systemic gaps of the data source."""
    tr = tools(LOW, policy_verification=PV_UNKNOWN, claim_history=HIST_UNAVAILABLE, similar_claims=SIM_NONE)
    d, iv = apply_guardrails(draft("FLAG_FOR_INVESTIGATION"), "PN-1", tr)
    assert d.decision == "APPROVE" and d.flag_category is None
    assert [i["guardrail"] for i in iv] == ["unsupported_fraud_flag"]
    assert "POLICY_TERMS_UNVERIFIED" in iv[0]["detail"] and "unsupported_fraud_flag" in d.guardrail_notes[0]
    # a real risk signal keeps the flag
    for score in (HIGH, {**LOW, "risk_indicators": FLAGS}):
        d, iv = apply_guardrails(draft("FLAG_FOR_INVESTIGATION"), "PN-1", {**tr, "fraud_score": score})
        assert d.decision == "FLAG_FOR_INVESTIGATION" and not iv
    # a strong single SHAP driver does not rescue a low-band flag
    shap = {"ok": True, "top_factors": [{"feature": "fault", "value": "Policy Holder", "shap_value": 0.5433}]}
    d, iv = apply_guardrails(draft("FLAG_FOR_INVESTIGATION"), "PN-1", {**tr, "shap_explanation": shap})
    assert d.decision == "APPROVE" and "STRONG_SHAP_DRIVER" not in d.reason_codes


MEDIUM = {**LOW, "fraud_probability": 0.0612, "risk_band": "medium"}


def test_medium_band_alone_does_not_justify_a_fraud_flag():
    """Pilot --n 10 (qwen2.5:72b): all 4 medium-band claims were flagged on the band alone."""
    tr = tools(MEDIUM, policy_verification=PV_UNKNOWN, claim_history=HIST_UNAVAILABLE, similar_claims=SIM_NONE)
    d, iv = apply_guardrails(draft("FLAG_FOR_INVESTIGATION"), "PN-8162", tr)
    assert d.decision == "APPROVE" and d.flag_category is None
    assert iv[0]["guardrail"] == "unsupported_fraud_flag" and "medium band without a concrete factor" in iv[0]["detail"]
    assert "MEDIUM_MODEL_RISK" in d.reason_codes


def test_medium_band_model_features_are_context_not_triggers():
    """Warning red flags, fraud among similar claims and strong SHAP drivers are model features:
    they stay in the reason codes (context) but no longer keep a medium-band flag."""
    base = tools(MEDIUM, policy_verification=PV_UNKNOWN, claim_history=HIST_UNAVAILABLE, similar_claims=SIM_NONE)
    shap = {"ok": True, "top_factors": [{"feature": "accident month", "value": "Mar", "shap_value": 0.61},
                                        {"feature": "red-flag score", "value": 2.0, "shap_value": 1.9}]}
    elevated = {**SIM_NONE, "neighbour_frauds": 3, "neighbour_fraud_rate": 0.6}
    cases = {"RED_FLAGS_PRESENT": {"fraud_score": {**MEDIUM, "risk_indicators": FLAGS}},
             "SIMILAR_CLAIMS_ELEVATED_FRAUD": {"similar_claims": elevated},
             "STRONG_SHAP_DRIVER": {"shap_explanation": shap}}
    for code, extra in cases.items():
        d, iv = apply_guardrails(draft("FLAG_FOR_INVESTIGATION"), "PN-1", {**base, **extra})
        assert d.decision == "APPROVE" and iv[0]["guardrail"] == "unsupported_fraud_flag", code
        assert code in d.reason_codes, code                  # still reported as context


def test_medium_band_flags_need_information_the_model_does_not_see():
    base = tools(MEDIUM, policy_verification=PV_UNKNOWN, claim_history=HIST_UNAVAILABLE, similar_claims=SIM_NONE)
    # policy-condition FAIL: kept, as a coverage question (POLICY_CONDITION), even with a red flag
    tr = {**base, "policy_verification": PV_POLICY_FAIL, "fraud_score": {**MEDIUM, "risk_indicators": FLAGS}}
    d, iv = apply_guardrails(draft("FLAG_FOR_INVESTIGATION"), "PN-1", tr)
    assert d.decision == "FLAG_FOR_INVESTIGATION" and d.flag_category == "POLICY_CONDITION" and not iv
    # history source lists more prior claims than the claim reports: FRAUD_RISK
    hist = {"ok": True, "status": "completed", "history_available": True, "previous_claim_count": 3,
            "recorded_summary": {"fields": {"PastNumberOfClaims": "none"}}}
    d, iv = apply_guardrails(draft("FLAG_FOR_INVESTIGATION"), "PN-1", {**base, "claim_history": hist})
    assert d.decision == "FLAG_FOR_INVESTIGATION" and d.flag_category == "FRAUD_RISK" and not iv
    assert "HISTORY_CONTRADICTION" in d.reason_codes and any("lists 3 previous" in c for c in d.conflicting_evidence)


def test_history_contradiction_only_for_under_reported_claims():
    from src.agent.evidence import history_contradiction
    def h(found, reported):
        return {"claim_history": {"ok": True, "status": "completed", "history_available": True,
                                  "previous_claim_count": found,
                                  "recorded_summary": {"fields": {"PastNumberOfClaims": reported}}}}
    assert history_contradiction(h(1, "none")) and history_contradiction(h(5, "2 to 4"))
    assert not history_contradiction(h(0, "2 to 4"))         # fewer found: the source may cover less time
    assert not history_contradiction(h(9, "more than 4")) and not history_contradiction(h(1, "1"))
    assert not history_contradiction({"claim_history": HIST_UNAVAILABLE})