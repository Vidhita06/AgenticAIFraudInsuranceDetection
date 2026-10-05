"""Graph paths with a scripted fake chat model (no API key, no network) and stub tools."""
import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.agent.nodes import NodeDeps
from src.agent.runner import TriageRunner


class ScriptedChatModel(GenericFakeChatModel):
    """Returns the scripted messages in order; tool binding is a no-op."""

    def bind_tools(self, tools, **kwargs):
        return self


class FailingChatModel(ScriptedChatModel):
    def _generate(self, *args, **kwargs):
        raise ConnectionError("LLM unavailable")


def scripted(*messages):
    return ScriptedChatModel(messages=iter(list(messages)))


def tool_call(name, args=None, i=0):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": f"call_{name}_{i}"}])


def answer(decision, *evidence):
    return AIMessage(content=json.dumps({"decision": decision, "rationale": "Summary of the evidence for the adjuster.",
                                         "evidence": [{"source_tool": s, "fact": f} for s, f in evidence]}))


def score_stub(band, p):
    return lambda claim: {"ok": True, "claim_id": "PN", "fraud_probability": p, "risk_band": band,
                          "top_factors": [], "model_version": "test"}


SIM = lambda claim, k=5: {"ok": True, "claim_id": "PN", "k": 3, "neighbour_frauds": 2, "neighbour_fraud_rate": 0.667,
                          "neighbours": [{"policy_number": i, "similarity": 0.9, "fraud_label": int(i < 3)} for i in (1, 2, 3)]}
POLICY = lambda claim: {"ok": True, "claim_id": "PN", "blocking": False, "failed_rules": [], "red_flags": [
    {"rule_id": "RF01_PH_FAULT_ALL_PERILS", "severity": "warning", "support": "15.7% fraud in 1944 claims"}]}


def runner(llm, band="low", p=0.0312, max_steps=8):
    deps = NodeDeps(llm=llm, scorer=score_stub(band, p), similar=SIM, policy=POLICY, max_steps=max_steps)
    return TriageRunner(deps=deps)


def test_approve_path(sample_claims):
    r = runner(scripted(tool_call("check_policy_rules"), AIMessage(content="Low risk, no red flags of note."),
                        answer("APPROVE", ("fraud_score", "Fraud probability 0.0312 in the low band"))))
    tid, d, trace = r.run_claim(sample_claims[0])
    assert d.decision == "APPROVE" and d.decided_by == "llm" and d.requires_human_review
    assert [e.fact for e in d.evidence] == ["Fraud probability 0.0312 in the low band"]
    nodes = [t["node"] for t in trace]
    assert nodes[:3] == ["ingest", "validate", "score"] and "tools" in nodes and nodes[-1] == "guardrails"


def test_flag_path_with_similar_claims(sample_claims):
    r = runner(scripted(tool_call("find_similar_claims", {"k": 3}), tool_call("check_policy_rules", i=1),
                        AIMessage(content="Evidence gathered."),
                        answer("FLAG_FOR_INVESTIGATION", ("fraud_score", "Fraud probability 0.4123 (high band)"),
                               ("similar_claims", "2 of 3 most similar past claims were fraudulent"),
                               ("policy_check", "Red flag RF01_PH_FAULT_ALL_PERILS fired"))),
               band="high", p=0.4123)
    tid, d, trace = r.run_claim(sample_claims[1])
    assert d.decision == "FLAG_FOR_INVESTIGATION" and len(d.evidence) == 3 and not d.guardrail_notes
    assert d.similar_claim_ids == ["PN-1", "PN-2", "PN-3"]
    calls = [c["name"] for t in trace if t["node"] == "tools" for c in t["calls"]]
    assert calls == ["find_similar_claims", "check_policy_rules"]


def test_request_more_info_overrides_llm_approve(sample_claims):
    r = runner(scripted(AIMessage(content="Validation is blocking; no tools needed."), answer("APPROVE")))
    tid, d, _ = r.run_claim(sample_claims[3])               # Age = 0
    assert d.decision == "REQUEST_MORE_INFO"
    assert any(n.startswith("blocking_validation") for n in d.guardrail_notes)


def test_incomplete_claim_is_not_scored_and_requests_info(sample_claims):
    r = runner(None)
    tid, d, trace = r.run_claim(sample_claims[-1])
    assert d.decision == "REQUEST_MORE_INFO" and d.fraud_probability is None
    assert any("missing field" in v for v in d.validation_issues)


def test_llm_failure_falls_back_to_rules(sample_claims):
    r = runner(FailingChatModel(messages=iter([])), band="high", p=0.4123)
    tid, d, trace = r.run_claim(sample_claims[1])
    assert d.decided_by == "rule_fallback" and d.decision == "FLAG_FOR_INVESTIGATION"
    assert any(t["node"] == "agent" and "error" in t for t in trace)


def test_invalid_output_twice_falls_back(sample_claims):
    r = runner(scripted(AIMessage(content="ok"), AIMessage(content="not json"),
                        AIMessage(content='{"decision": "APPROVE"}')))   # missing rationale
    tid, d, trace = r.run_claim(sample_claims[0])
    assert d.decided_by == "rule_fallback" and d.decision == "APPROVE"
    synth = [t for t in trace if t["node"] == "synthesize"][0]
    assert len(synth["failed"]) == 2


def test_invalid_output_once_then_valid(sample_claims):
    r = runner(scripted(AIMessage(content="ok"), AIMessage(content="I think approve"),
                        answer("APPROVE", ("fraud_score", "probability 0.0312"))))
    tid, d, _ = r.run_claim(sample_claims[0])
    assert d.decided_by == "llm" and d.decision == "APPROVE"


def test_max_steps_cutoff(sample_claims):
    loop = [tool_call("check_policy_rules", i=i) for i in range(10)]
    r = runner(scripted(*loop, answer("APPROVE", ("policy_check", "RF01_PH_FAULT_ALL_PERILS fired"))), max_steps=3)
    tid, d, trace = r.run_claim(sample_claims[0])
    assert sum(t["node"] == "agent" for t in trace) == 3
    assert sum(t["node"] == "tools" for t in trace) == 2
    assert any(n.startswith("max_steps") for n in d.guardrail_notes)
    assert d.decision == "APPROVE"


def test_hallucinated_evidence_removed_in_graph(sample_claims):
    r = runner(scripted(AIMessage(content="done"),
                        answer("APPROVE", ("fraud_score", "Fraud probability 0.0312"),
                               ("similar_claims", "0 of 5 similar claims were fraud"))))
    tid, d, _ = r.run_claim(sample_claims[0])
    assert [e.source_tool for e in d.evidence] == ["fraud_score"]
    assert any(n.startswith("evidence_grounding") for n in d.guardrail_notes)


def test_human_review_interrupt_and_resume(sample_claims, tmp_path):
    deps = NodeDeps(llm=None, scorer=score_stub("high", 0.4123), similar=SIM, policy=POLICY)
    r = TriageRunner(deps=deps, audit_log=tmp_path / "audit.jsonl")
    tid, d, _ = r.run_claim(sample_claims[1])
    state = r.graph.get_state({"configurable": {"thread_id": tid}})
    assert state.next == ("human_review",)                    # paused for the adjuster
    final = r.resume(tid, {"decision": "APPROVE", "adjuster_id": "adj-7", "override_reason": "known customer"})
    assert final["adjuster"]["decision"] == "APPROVE" and final["decision"]["decision"] == "FLAG_FOR_INVESTIGATION"
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert [json.loads(l)["event"] for l in lines] == ["recommendation", "adjuster_decision"]
    with pytest.raises(Exception):
        r.resume(tid, {"decision": "MAYBE"})                  # invalid adjuster decision
