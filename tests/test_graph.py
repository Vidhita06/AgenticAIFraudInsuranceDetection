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


# ---------------------------------------------------------------------------
# Ollama path, end to end against a local stand-in for the Ollama HTTP API.
# This checks the wiring (tools sent, num_ctx, JSON mode, tool-call parsing, token counts),
# not model quality: no model weights are involved.
# ---------------------------------------------------------------------------
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402


class FakeOllama:
    def __init__(self, replies, models=("qwen2.5:7b",)):
        self.replies, self.requests, self.models = list(replies), [], list(models)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"models": [{"name": m} for m in outer.models]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(req)
                msg = outer.replies.pop(0) if outer.replies else {"content": "{}"}
                lines = [{"model": req["model"], "created_at": "2026-01-01T00:00:00Z",
                          "message": {"role": "assistant", **msg}, "done": False},
                         {"model": req["model"], "created_at": "2026-01-01T00:00:00Z",
                          "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
                          "prompt_eval_count": 1500, "eval_count": 100}]
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for line in lines:
                    self.wfile.write((json.dumps(line) + "\n").encode())

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()


def ollama_runner(fake, band="high", p=0.4123):
    from src.agent.llm import get_llm
    import os
    os.environ["OLLAMA_BASE_URL"] = fake.url
    llm = get_llm(provider="ollama", model="qwen2.5:7b")
    synth = get_llm(provider="ollama", model="qwen2.5:7b", json_mode=True)
    deps = NodeDeps(llm=llm, synth_llm=synth, scorer=score_stub(band, p), similar=SIM, policy=POLICY)
    return TriageRunner(deps=deps)


def decision_json(decision, *evidence):
    return {"content": json.dumps({"decision": decision, "rationale": "High score and a policyholder-fault red flag.",
                                   "evidence": [{"source_tool": s, "fact": f} for s, f in evidence]})}


def test_ollama_native_tool_calls_end_to_end(sample_claims):
    fake = FakeOllama([
        {"content": "", "tool_calls": [{"function": {"name": "find_similar_claims", "arguments": {"k": 3}}}]},
        {"content": "Enough evidence."},
        decision_json("FLAG_FOR_INVESTIGATION", ("similar_claims", "2 of 3 similar claims were fraud"),
                      ("fraud_score", "Fraud probability 0.4123")),
    ])
    try:
        tid, d, trace = ollama_runner(fake).run_claim(sample_claims[1])
    finally:
        fake.close()
    assert d.decision == "FLAG_FOR_INVESTIGATION" and d.decided_by == "llm" and len(d.evidence) == 2
    agent_req, synth_req = fake.requests[0], fake.requests[-1]
    assert {t["function"]["name"] for t in agent_req["tools"]} == {"find_similar_claims", "check_policy_rules"}
    assert agent_req["options"]["num_ctx"] >= 16384 and agent_req["options"]["temperature"] == 0
    assert synth_req.get("format") == "json" and not synth_req.get("tools")      # no tools offered
    calls = [c["name"] for t in trace if t["node"] == "tools" for c in t["calls"]]
    assert calls == ["find_similar_claims"]
    auto = [t for t in trace if t["node"] == "max_steps_check"][0]["auto_tools"]
    assert auto == ["check_policy_rules"]          # model never called it: safety net ran


def test_ollama_tool_call_written_as_text_is_recovered(sample_claims):
    fake = FakeOllama([
        {"content": 'I will check the rules: {"name": "check_policy_rules", "arguments": {}}'},
        {"content": "No more tools needed."},
        decision_json("FLAG_FOR_INVESTIGATION", ("policy_check", "RF01_PH_FAULT_ALL_PERILS fired")),
    ])
    try:
        tid, d, trace = ollama_runner(fake).run_claim(sample_claims[1])
    finally:
        fake.close()
    agent_events = [t for t in trace if t["node"] == "agent"]
    assert agent_events[0].get("tool_calls_from_text") is True
    assert [c["name"] for t in trace if t["node"] == "tools" for c in t["calls"]] == ["check_policy_rules"]
    assert d.decided_by == "llm" and d.evidence[0].source_tool == "policy_check"


def test_ollama_malformed_json_retries_then_falls_back(sample_claims):
    fake = FakeOllama([{"content": "Looks risky."}, {"content": "I recommend flagging it."},
                       {"content": "```json\n{\"decision\": \"FLAG\"\n```"}])
    try:
        tid, d, trace = ollama_runner(fake).run_claim(sample_claims[1])
    finally:
        fake.close()
    assert d.decided_by == "rule_fallback" and d.decision == "FLAG_FOR_INVESTIGATION"
    assert len([t for t in trace if t["node"] == "synthesize"][0]["failed"]) == 2


def test_ollama_lenient_json_is_accepted(sample_claims):
    fake = FakeOllama([{"content": "ok"},
                       {"content": '```json\n{"Decision": "approve", "reasoning": "Low score, no red flags of note.",'
                                   ' "evidence": [{"source": "fraud_score_tool", "fact": "probability 0.0312"},],}\n```'}])
    try:
        tid, d, _ = ollama_runner(fake, band="low", p=0.0312).run_claim(sample_claims[0])
    finally:
        fake.close()
    assert d.decision == "APPROVE" and d.decided_by == "llm" and d.evidence[0].source_tool == "fraud_score"


def test_ollama_batch_is_free_and_counts_tokens(sample_claims, tmp_path):
    fake = FakeOllama([{"content": "ok"}, decision_json("APPROVE", ("fraud_score", "probability 0.0312"))])
    claims = pd.DataFrame([{**sample_claims[0], "FraudFound_P": 0}])
    try:
        recs = ae.run_batch(claims, ollama_runner(fake, band="low", p=0.0312), "validation", tmp_path / "r.jsonl",
                            model_name="qwen2.5:7b", provider="ollama", progress=lambda m: None)
    finally:
        fake.close()
    assert recs[0]["cost_usd"] == 0 and recs[0]["input_tokens"] == 3000 and recs[0]["provider"] == "ollama"
    est = ae.estimate_cost(claims, "qwen2.5:7b", 10, 2, provider="ollama")
    assert est.free and est.usd == 0 and "$0" in est.describe("qwen2.5:7b")
    assert ae.confirm("free", assume_yes=False, input_fn=lambda _: "", free=True) is True


def test_check_ollama_reports_missing_model():
    from src.agent.llm import check_ollama
    fake = FakeOllama([], models=("llama3.1:8b",))
    try:
        with pytest.raises(RuntimeError, match="ollama pull qwen2.5:7b"):
            check_ollama("qwen2.5:7b", fake.url)
        check_ollama("llama3.1:8b", fake.url)
    finally:
        fake.close()
    with pytest.raises(RuntimeError, match="not reachable"):
        check_ollama("qwen2.5:7b", "http://127.0.0.1:9")


def test_openai_base_url_and_free_tier_pricing(monkeypatch):
    from src.agent.llm import get_llm, llm_settings, needs_api_key
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    s = llm_settings()
    assert s["base_url"] == "https://api.groq.com/openai/v1" and needs_api_key(s)
    llm = get_llm(model="llama-3.3-70b-versatile")
    assert str(llm.openai_api_base).startswith("https://api.groq.com")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert not needs_api_key(llm_settings())                # local OpenAI-compatible server
    assert ae.pricing("llama-3.3-70b-versatile", "openai")["free"] is True
    assert ae.pricing("claude-sonnet-5-5", "anthropic")["input"] == 2.0
