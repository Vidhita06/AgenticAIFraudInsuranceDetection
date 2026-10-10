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


# ---------------------------------------------------------------------------
# Agent evaluation plumbing (evaluation/agent_eval.py) with the scripted model
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402

from evaluation import agent_eval as ae  # noqa: E402


def with_usage(msg, tin=1000, tout=200):
    msg.usage_metadata = {"input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout}
    return msg


def test_run_batch_records_usage_cost_and_model_only_decision(sample_claims, tmp_path):
    claims = pd.DataFrame([{**sample_claims[0], "FraudFound_P": 0}, {**sample_claims[1], "FraudFound_P": 1}])
    llm = scripted(with_usage(AIMessage(content="done")),
                   with_usage(answer("APPROVE", ("fraud_score", "Fraud probability 0.0312")), 2000, 500),
                   with_usage(tool_call("find_similar_claims", {"k": 3})), with_usage(AIMessage(content="done")),
                   with_usage(answer("FLAG_FOR_INVESTIGATION", ("similar_claims", "2 of 3 similar claims were fraud")), 2000, 500))
    r = runner(llm)
    out = tmp_path / "res.jsonl"
    recs = ae.run_batch(claims, r, "validation", out, model_name="claude-sonnet-5-5", progress=lambda m: None)
    assert [x["decision"] for x in recs] == ["APPROVE", "FLAG_FOR_INVESTIGATION"]
    assert recs[0]["input_tokens"] == 3000 and recs[0]["output_tokens"] == 700 and recs[0]["llm_calls"] == 2
    assert recs[0]["cost_usd"] == pytest.approx((3000 * 2 + 700 * 10) / 1e6)
    assert recs[1]["tools_called"] == ["find_similar_claims"]
    assert recs[0]["model_only_decision"] == "APPROVE"            # low band stub
    df = ae.load_results([out])
    assert len(df) == 2 and set(df.label) == {0, 1}
    s = ae.summarize(df)
    dq = s["decision_quality"].set_index(["policy", "metric"])
    assert dq.loc[("agent", "fraud -> FLAG (recall)"), "rate"] == 1.0
    assert dq.loc[("model-only threshold policy", "fraud -> FLAG (recall)"), "rate"] == 0.0
    assert s["process"]["tool_use_rate"]["find_similar_claims"] == 0.5
    path = ae.write_report(s, tmp_path / "report.md", "Test report", {"model": "fake"})
    assert "Decision quality" in path.read_text()


def test_consistency_and_manual_review(tmp_path):
    rows = []
    for pn, decisions in ((1, ["APPROVE"] * 3), (2, ["APPROVE", "FLAG_FOR_INVESTIGATION", "APPROVE"])):
        for k, d in enumerate(decisions):
            rows.append({"policy_number": pn, "run_index": k, "decision": d, "label": pn - 1, "claim_id": f"PN-{pn}",
                         "fraud_probability": 0.1, "risk_band": "low", "model_only_decision": "APPROVE",
                         "decided_by": "llm", "rationale": "r", "evidence": [], "guardrail_notes": [], "tools_called": []})
    df = pd.DataFrame(rows)
    c = ae.consistency(df)
    assert c == {"claims": 2, "runs_per_claim": 3, "identical_decision_share": 0.5, "unstable_claims": [2]}
    m = ae.manual_review_sample(df, n=2)
    assert len(m) == 2 and "reviewer_notes" in m.columns


def test_stratified_sample_and_estimate():
    from src.ml.data import load_real_validation
    v = load_real_validation()
    s = ae.stratified_sample(v, legit_per_fraud=1)
    assert s.FraudFound_P.sum() == v.FraudFound_P.sum() == len(s) / 2
    assert s.attrs["legit_weight"] == pytest.approx((len(v) - 139) / 139)
    est = ae.estimate_cost(s.head(10), "claude-sonnet-5-5", n_runs=10, n_judge=2)
    assert est.usd > 0 and est.per_claim_usd > 0 and "static" in est.basis
    meas = ae.estimate_cost(s.head(10), "claude-sonnet-5-5", 10,
                            measured={"input_tokens": 10_000, "output_tokens": 1_000})
    assert meas.usd == pytest.approx(10 * (10_000 * 2 + 1_000 * 10) / 1e6)
    with pytest.raises(KeyError):
        ae.pricing("unknown-model")


def test_confirm_requires_explicit_yes():
    assert ae.confirm("cost $1", assume_yes=False, input_fn=lambda _: "y") is True
    assert ae.confirm("cost $1", assume_yes=False, input_fn=lambda _: "") is False

    def no_tty(_):
        raise EOFError
    assert ae.confirm("cost $1", assume_yes=False, input_fn=no_tty) is False
    assert ae.confirm("cost $1", assume_yes=True, input_fn=no_tty) is True


def test_judge_parses_scores(tmp_path):
    df = pd.DataFrame([{"run_index": 0, "decided_by": "llm", "claim_id": "PN-1", "label": 1,
                        "decision": "FLAG_FOR_INVESTIGATION", "rationale": "High band.", "evidence": [],
                        "tool_results": {"fraud_score": {"ok": True, "fraud_probability": 0.4}}}])
    judge_llm = scripted(AIMessage(content='{"cites_evidence": 1, "consistent_with_score": 1, '
                                           '"no_invented_facts": 0, "clear_for_adjuster": 1, "comment": "x"}'))
    j = ae.judge_rationales(df, judge_llm, n=5)
    assert j.loc[0, "no_invented_facts"] == 0 and j.loc[0, "cites_evidence"] == 1
