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


def tool_calls(*names):
    return AIMessage(content="", tool_calls=[{"name": n, "args": {}, "id": f"call_{n}_{i}"} for i, n in enumerate(names)])


def answer(decision, *evidence):
    return AIMessage(content=json.dumps({"decision": decision, "rationale": "Summary of the evidence for the adjuster.",
                                         "evidence": [{"source_tool": s, "fact": f} for s, f in evidence]}))


def score_stub(band, p):
    flags = [{"rule_id": "RF01_PH_FAULT_ALL_PERILS", "severity": "warning", "support": "15.7% fraud in 1944 claims"}]
    return lambda claim: {"ok": True, "status": "completed", "claim_id": "PN", "fraud_probability": p,
                          "risk_band": band, "predicted_class": "fraud" if band == "high" else "not_fraud",
                          "risk_indicators": flags if band == "high" else [], "model_version": "test"}


EXPLAIN = lambda claim, top_k=5, fraud_score=None: {
    "ok": True, "status": "completed", "top_factors": [{"feature": "fault", "value": "Policy Holder", "shap_value": 0.3476,
                                                        "direction": "increases fraud score"}],
    "positive_factors": ["fault"], "negative_factors": []}
RETRIEVE = lambda claim, **kw: {"ok": True, "status": "partial", "policy_id": "PN-1", "document_found": False,
                          "document_lookup": "no policy document store", "missing_information": ["policy version"]}


def verify_stub(fail=False):
    """fail: False, "claim" (PV04 claim-record contradiction) or "policy" (SIM-ALL-04 deadline)."""
    failing = {"claim": "PV04_CLAIM_TIMELINE", "policy": "SIM-ALL-04", True: "PV04_CLAIM_TIMELINE"}.get(fail)

    def f(claim, policy=None, **kw):
        assert policy is not None and policy.get("ok"), "retrieval must run first"
        checks = [{"check_id": c, "level": "claim_record" if c.startswith("PV04") else "policy_condition",
                   "result": "FAIL" if c == failing else "PASS", "reason": "check"}
                  for c in ("PV04_CLAIM_TIMELINE", "SIM-ALL-04")]
        checks.append({"check_id": "PV08_COVERAGE_LIMITS", "level": "policy_condition", "result": "UNKNOWN",
                       "reason": "no amount"})
        fails = [c for c in checks if c["result"] == "FAIL"]
        return {"ok": True, "status": "completed", "overall_result": "FAIL" if fails else "UNKNOWN",
                "counts": {"PASS": 2 - len(fails), "FAIL": len(fails), "UNKNOWN": 1}, "checks": checks,
                "failed_checks": [c["check_id"] for c in fails],
                "claim_record_failures": [c["check_id"] for c in fails if c["level"] == "claim_record"],
                "policy_condition_failures": [c["check_id"] for c in fails if c["level"] == "policy_condition"],
                "unknown_checks": ["PV08_COVERAGE_LIMITS"]}
    return f


SIM = lambda claim, k=5, same_base_policy=False: {
    "ok": True, "status": "completed", "claim_id": "PN", "k": 3, "neighbour_frauds": 2, "neighbour_fraud_rate": 0.667,
    "historical_base_rate": 0.0599,
    "neighbours": [{"claim_id": f"PN-{i}", "policy_number": i, "similarity": 0.9, "fraud_label": int(i < 3)}
                   for i in (1, 2, 3)]}
HISTORY = lambda claim, years=5, **kw: {"ok": True, "status": "unavailable", "history_available": False,
                                  "previous_claim_count": None, "previous_claims": [],
                                  "limitation": "No authorised claim-history source is configured."}


def deps_for(llm, band="low", p=0.0312, max_steps=8, fail=False, **kw):
    return NodeDeps(llm=llm, scorer=score_stub(band, p), explainer=EXPLAIN, policy_retriever=RETRIEVE,
                    policy_verifier=verify_stub(fail), similar=SIM, history=HISTORY, max_steps=max_steps, **kw)


def runner(llm, band="low", p=0.0312, max_steps=8, fail=False):
    return TriageRunner(deps=deps_for(llm, band, p, max_steps, fail))


def agent_calls(trace):
    return [c["name"] for t in trace if t["node"] == "tools" for c in t["calls"] if c.get("origin") == "agent"]


def auto_tools(trace):
    return [t for t in trace if t["node"] == "complete_investigation"][0]["auto_tools"]


def test_approve_path(sample_claims):
    r = runner(scripted(tool_call("verify_policy"), AIMessage(content="Low risk, no policy failures."),
                        answer("APPROVE", ("fraud_score", "Fraud probability 0.0312 in the low band"))))
    tid, d, trace = r.run_claim(sample_claims[0])
    assert d.decision == "APPROVE" and d.decided_by == "llm" and d.requires_human_review
    assert [e.fact for e in d.evidence] == ["Fraud probability 0.0312 in the low band"]
    nodes = [t["node"] for t in trace]
    assert nodes[:3] == ["ingest", "validate", "score"] and "tools" in nodes
    assert nodes[-4:] == ["complete_investigation", "aggregate_evidence", "synthesize", "guardrails"]
    assert auto_tools(trace) == ["explain_fraud_score", "find_similar_claims"]   # required, not called by the agent
    assert d.investigation_id == f"INV-{tid}" and d.investigation_status == "complete"
    assert {"LOW_MODEL_RISK", "POLICY_TERMS_UNVERIFIED"} <= set(d.reason_codes) and d.flag_category is None
    st = {k: (v.status, v.origin) for k, v in d.tool_statuses.items()}
    assert st["policy_verification"] == ("completed", "agent") and st["policy_retrieval"] == ("partial", "dependency")
    assert st["fraud_score"] == ("completed", "orchestrator") and st["shap_explanation"] == ("completed", "auto")
    assert st["similar_claims"] == ("completed", "auto") and st["claim_history"] == ("not_run", None)


def test_flag_path_with_all_agent_tools(sample_claims):
    r = runner(scripted(tool_calls("find_similar_claims", "verify_policy", "explain_fraud_score", "get_claim_history"),
                        AIMessage(content="Evidence gathered."),
                        answer("FLAG_FOR_INVESTIGATION", ("fraud_score", "Fraud probability 0.4123 (high band)"),
                               ("similar_claims", "2 of 3 most similar past claims were fraudulent"),
                               ("fraud_score", "Red flag RF01_PH_FAULT_ALL_PERILS fired"),
                               ("shap_explanation", "fault = Policy Holder adds 0.3476"))),
               band="high", p=0.4123)
    tid, d, trace = r.run_claim(sample_claims[1])
    assert d.decision == "FLAG_FOR_INVESTIGATION" and len(d.evidence) == 4 and not d.guardrail_notes
    assert d.flag_category == "FRAUD_RISK"
    assert d.similar_claim_ids == ["PN-1", "PN-2", "PN-3"]
    assert agent_calls(trace) == ["find_similar_claims", "verify_policy", "explain_fraud_score", "get_claim_history"]
    assert auto_tools(trace) == []
    assert d.claim_history_summary["status"] == "unavailable"
    assert any(m.startswith("Claim history unavailable") for m in d.missing_information)
    assert {"HIGH_MODEL_RISK", "RED_FLAGS_PRESENT", "SIMILAR_CLAIMS_ELEVATED_FRAUD",
            "CLAIM_HISTORY_UNAVAILABLE"} <= set(d.reason_codes)


def test_dependency_retrieval_runs_before_verification(sample_claims):
    r = runner(scripted(tool_call("verify_policy"), AIMessage(content="done"), answer("APPROVE")))
    tid, d, trace = r.run_claim(sample_claims[0])
    calls = [t for t in trace if t["node"] == "tools"][0]["calls"]
    assert [(c["name"], c["origin"]) for c in calls] == [("retrieve_policy", "dependency"), ("verify_policy", "agent")]


def test_request_more_information_overrides_llm_approve(sample_claims):
    r = runner(scripted(AIMessage(content="Validation is blocking; no tools needed."), answer("APPROVE")))
    tid, d, _ = r.run_claim(sample_claims[3])               # Age = 0
    assert d.decision == "REQUEST_MORE_INFORMATION" and d.investigation_status == "blocked"
    assert any(n.startswith("blocking_validation") for n in d.guardrail_notes)
    assert "BLOCKING_VALIDATION" in d.reason_codes


def test_claim_record_contradiction_requests_information(sample_claims):
    r = runner(scripted(tool_call("verify_policy"), AIMessage(content="done"), answer("APPROVE")), fail="claim")
    tid, d, _ = r.run_claim(sample_claims[0])
    assert d.decision == "REQUEST_MORE_INFORMATION" and d.flag_category is None
    assert any(n.startswith("approve_needs_consistent_claim") for n in d.guardrail_notes)
    assert d.policy_findings["verification"]["failed"][0]["check_id"] == "PV04_CLAIM_TIMELINE"
    assert "CLAIM_RECORD_CONTRADICTION" in d.reason_codes


def test_policy_condition_fail_is_a_policy_flag_not_fraud(sample_claims):
    r = runner(scripted(tool_call("verify_policy"), AIMessage(content="done"), answer("APPROVE")), fail="policy")
    tid, d, _ = r.run_claim(sample_claims[0])
    assert d.decision == "FLAG_FOR_INVESTIGATION" and d.flag_category == "POLICY_CONDITION"
    assert any(n.startswith("approve_needs_policy_pass") for n in d.guardrail_notes)
    assert "POLICY_CONDITION_FAILED" in d.reason_codes
    assert d.conflicting_evidence            # low band but a policy-condition FAIL


def test_incomplete_claim_is_not_scored_and_requests_info(sample_claims):
    r = runner(None)
    tid, d, trace = r.run_claim(sample_claims[-1])
    assert d.decision == "REQUEST_MORE_INFORMATION" and d.fraud_probability is None
    assert any("missing field" in v for v in d.validation_issues)
    assert auto_tools(trace) == []                             # nothing runs on an invalid claim
    assert d.tool_statuses["fraud_score"].status == "failed"


def test_without_llm_the_orchestrator_runs_all_six_tools(sample_claims):
    r = runner(None, band="high", p=0.4123)
    tid, d, trace = r.run_claim(sample_claims[1])
    assert auto_tools(trace) == ["explain_fraud_score", "retrieve_policy", "verify_policy", "find_similar_claims",
                                 "get_claim_history"]
    assert all(v.status != "not_run" for v in d.tool_statuses.values())
    assert d.decided_by == "rule_fallback" and d.decision == "FLAG_FOR_INVESTIGATION"
    assert {e.source_tool for e in d.evidence} >= {"fraud_score", "shap_explanation", "policy_verification",
                                                   "similar_claims", "claim_history"}


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


def test_unresolved_questions_are_kept(sample_claims):
    msg = AIMessage(content=json.dumps({"decision": "REQUEST_MORE_INFO", "rationale": "Policy dates are unknown here.",
                                        "evidence": [], "unresolved_questions": ["What are the policy's validity dates?"]}))
    tid, d, _ = runner(scripted(AIMessage(content="ok"), msg)).run_claim(sample_claims[0])
    assert d.unresolved_questions == ["What are the policy's validity dates?"]
    # no claim-level reason for more information: the rule-based decision replaces it
    assert d.decision == "APPROVE" and any(n.startswith("request_info_not_claim_level") for n in d.guardrail_notes)


def test_max_steps_cutoff(sample_claims):
    loop = [tool_call("verify_policy", i=i) for i in range(10)]
    r = runner(scripted(*loop, answer("APPROVE", ("policy_verification", "PV04_CLAIM_TIMELINE: 0 FAIL"))), max_steps=3)
    tid, d, trace = r.run_claim(sample_claims[0])
    assert sum(t["node"] == "agent" for t in trace) == 3
    assert sum(t["node"] == "tools" for t in trace) == 2
    assert any(n.startswith("max_steps") for n in d.guardrail_notes)
    assert d.decision == "APPROVE"


def test_hallucinated_evidence_removed_in_graph(sample_claims):
    r = runner(scripted(AIMessage(content="done"),
                        answer("APPROVE", ("fraud_score", "Fraud probability 0.0312"),
                               ("similar_claims", "0 of 5 similar claims were fraud"),
                               ("claim_history", "No previous claims in 5 years"))))
    tid, d, _ = r.run_claim(sample_claims[0])
    assert [e.source_tool for e in d.evidence] == ["fraud_score"]
    assert sum(n.startswith("evidence_grounding") for n in d.guardrail_notes) == 2


def test_unknown_tool_is_reported_not_run(sample_claims):
    r = runner(scripted(tool_call("delete_claim"), AIMessage(content="done"), answer("APPROVE")))
    tid, d, trace = r.run_claim(sample_claims[0])
    calls = [t for t in trace if t["node"] == "tools"][0]["calls"]
    assert calls == [{"name": "delete_claim", "args": {}, "ok": False, "origin": "agent"}]


def test_human_review_interrupt_and_resume(sample_claims, tmp_path):
    r = TriageRunner(deps=deps_for(None, "high", 0.4123), audit_log=tmp_path / "audit.jsonl")
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
    assert dq.loc[("agent", "fraud -> FRAUD_RISK flag (recall)"), "rate"] == 1.0
    assert dq.loc[("model-only threshold policy", "fraud -> FRAUD_RISK flag (recall)"), "rate"] == 0.0
    assert recs[1]["flag_category"] == "FRAUD_RISK" and recs[0]["flag_category"] is None
    assert s["process"]["tool_use_rate"]["find_similar_claims"] == 0.5
    path = ae.write_report(s, tmp_path / "report.md", "Test report", {"model": "fake"})
    assert "Decision quality" in path.read_text()


def _rows(mode_on, decisions):
    rows = []
    for pn, (label, dec, cat, checks) in enumerate(decisions, 1):
        rows.append({"policy_number": pn, "run_index": 0, "label": label, "decision": dec, "flag_category": cat,
                     "model_only_decision": "APPROVE", "model_only_flag_category": None, "decided_by": "llm",
                     "tools_called": [], "auto_tools": [], "reason_codes": [], "conflicting_evidence": [],
                     "interventions": [], "grounding_removed": 0, "n_evidence": 1, "agent_steps": 1,
                     "max_steps_reached": False, "llm_failures": 0, "latency_s": 1.0, "input_tokens": 0,
                     "output_tokens": 0, "cost_usd": 0.0, "error": None, "simulated_data": mode_on,
                     "policy_checks": checks, "history_status": "completed" if mode_on else "unavailable",
                     "history_count": 2 if mode_on else None, "claim_id": f"PN-{pn}"})
    return pd.DataFrame(rows)


def test_policy_flags_are_not_fraud_caught():
    df = _rows(True, [(1, "FLAG_FOR_INVESTIGATION", "POLICY_CONDITION", {"SIM-ALL-04": "FAIL"}),
                      (1, "FLAG_FOR_INVESTIGATION", "FRAUD_RISK", {}),
                      (0, "FLAG_FOR_INVESTIGATION", "SYSTEM_FAILURE", {}), (0, "APPROVE", None, {})])
    dq = ae.decision_quality(df).set_index(["policy", "metric"])
    assert dq.loc[("agent", "fraud -> FRAUD_RISK flag (recall)"), "rate"] == 0.5
    assert dq.loc[("agent", "legit -> FRAUD_RISK flag (false-flag rate)"), "rate"] == 0.0
    assert dq.loc[("agent", "POLICY_CONDITION flags (all claims)"), "k"] == 1
    assert dq.loc[("agent", "SYSTEM_FAILURE flags (all claims)"), "k"] == 1
    assert dq.loc[("agent", "FRAUD_RISK flag precision (sample)"), "rate"] == 1.0


def test_mode_comparison_report(tmp_path):
    off = _rows(False, [(1, "FLAG_FOR_INVESTIGATION", "FRAUD_RISK", {}), (0, "APPROVE", None, {}),
                        (0, "APPROVE", None, {})])
    on = _rows(True, [(1, "FLAG_FOR_INVESTIGATION", "FRAUD_RISK", {"SIM-ALL-04": "PASS"}),
                      (0, "FLAG_FOR_INVESTIGATION", "POLICY_CONDITION", {"SIM-ALL-04": "FAIL"}),
                      (0, "APPROVE", None, {"SIM-ALL-04": "PASS"})])
    cmp = ae.compare_modes({"off": off, "on": on})
    t = cmp["decision_quality"].set_index("metric")
    assert t.loc["POLICY_CONDITION flags (all claims)", "Δ (on − off)"] == pytest.approx(1 / 3)
    assert t.loc["fraud -> FRAUD_RISK flag (recall)", "Δ (on − off)"] == 0
    assert list(cmp["changed"].policy_number) == [2] and "SIM-ALL-04" in cmp["changed"].cause.iloc[0]
    assert cmp["policy_checks"]["on"].set_index("check").loc["SIM-ALL-04", "FAIL"] == 1
    path = ae.write_report(None, tmp_path / "r.md", "T", {"model": "x"},
                           mode_summaries={m: ae.summarize(d) for m, d in (("off", off), ("on", on))},
                           comparison=cmp)
    text = path.read_text()
    assert "off vs on" in text and "Caveat" in text and "FRAUD_RISK flags only" in text


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
    return TriageRunner(deps=deps_for(llm, band, p, synth_llm=synth))


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
    assert {t["function"]["name"] for t in agent_req["tools"]} == {"explain_fraud_score", "retrieve_policy",
                                                                     "verify_policy", "find_similar_claims",
                                                                     "get_claim_history"}
    assert agent_req["options"]["num_ctx"] >= 16384 and agent_req["options"]["temperature"] == 0
    assert synth_req.get("format") == "json" and not synth_req.get("tools")      # no tools offered
    assert agent_calls(trace) == ["find_similar_claims"]
    # the model never called the required tools: complete_investigation ran them
    assert auto_tools(trace) == ["explain_fraud_score", "retrieve_policy", "verify_policy"]   # similar was called


def test_ollama_tool_call_written_as_text_is_recovered(sample_claims):
    fake = FakeOllama([
        {"content": 'I will check the policy: {"name": "verify_policy", "arguments": {}}'},
        {"content": 'And the old name: {"name": "check_policy_rules", "arguments": {}}'},
        {"content": "No more tools needed."},
        decision_json("FLAG_FOR_INVESTIGATION", ("policy_check", "PV08_COVERAGE_LIMITS is UNKNOWN (1 check)")),
    ])
    try:
        tid, d, trace = ollama_runner(fake).run_claim(sample_claims[1])
    finally:
        fake.close()
    agent_events = [t for t in trace if t["node"] == "agent"]
    assert agent_events[0].get("tool_calls_from_text") is True
    assert agent_calls(trace) == ["verify_policy", "verify_policy"]     # old name mapped to the new tool
    assert d.decided_by == "llm" and d.evidence[0].source_tool == "policy_verification"


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
