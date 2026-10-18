"""Adjuster dashboard (Streamlit AppTest, headless) and the runner's review rules."""
import sqlite3

import pytest

from src.agent.audit import AuditStore
from src.config import load_config, path_for

needs_models = pytest.mark.skipif(not (path_for("models_dir") / "fraud_model.joblib").exists(),
                                  reason="model artifacts not present")


@pytest.fixture
def audit_path(tmp_path):
    paths = load_config()["paths"]
    old = paths.get("audit_db")
    paths["audit_db"] = str(tmp_path / "audit.db")
    yield tmp_path / "audit.db"
    paths["audit_db"] = old


def test_runner_review_rules_and_audit(sample_claims, audit_path):
    from src.agent.nodes import NodeDeps
    from src.agent.runner import TriageRunner
    stub = lambda claim: {"ok": True, "status": "completed", "fraud_probability": 0.41, "risk_band": "high",
                          "predicted_class": "fraud", "risk_indicators": [], "model_version": "t"}
    ok = lambda *a, **k: {"ok": True, "status": "completed"}
    r = TriageRunner(deps=NodeDeps(llm=None, scorer=stub, explainer=ok, policy_retriever=ok, policy_verifier=ok,
                                   similar=ok, history=ok), audit_db=True)
    tid, d, _ = r.run_claim(sample_claims[1])
    assert d.decision == "FLAG_FOR_INVESTIGATION" and r.pending_review(tid)
    with pytest.raises(ValueError, match="override reason"):
        r.resume(tid, {"decision": "APPROVE", "adjuster_id": "a1"})       # override without a reason
    assert r.pending_review(tid)                                           # still paused
    r.resume(tid, {"decision": "APPROVE", "adjuster_id": "a1", "override_reason": "documents verified"})
    assert not r.pending_review(tid)
    with pytest.raises(RuntimeError, match="not waiting"):
        r.resume(tid, {"decision": "APPROVE", "adjuster_id": "a1", "override_reason": "again"})
    store = AuditStore(audit_path)
    rec, adj = store.rows("recommendations"), store.rows("adjuster_decisions")
    assert len(rec) == 1 and rec[0]["decision"] == "FLAG_FOR_INVESTIGATION" and rec[0]["agent_mode"] == "rule_based"
    assert len(adj) == 1 and adj[0]["adjuster_decision"] == "APPROVE" and adj[0]["overrides_recommendation"] == 1
    assert adj[0]["override_reason"] == "documents verified" and adj[0]["investigation_id"] == d.investigation_id


@needs_models
def test_dashboard_end_to_end(audit_path, vector_store):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(path_for("models_dir").parent / "dashboard" / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception and "Run an investigation" in at.info[0].value
    [b for b in at.button if "Run investigation" in b.label][0].click()
    at.run()
    assert not at.exception
    assert [h.value for h in at.header] == ["1 · Claim", "2 · Recommendation", "3 · Tool results", "4 · Your decision"]
    decision = [r for r in at.radio if r.label == "Decision"][0]
    recommended = decision.value
    other = next(x for x in ("APPROVE", "FLAG_FOR_INVESTIGATION") if x != recommended)
    decision.set_value(other)
    [b for b in at.button if b.label == "Submit decision"][0].click()
    at.run()
    assert any("reason" in e.value for e in at.error)                     # override without a reason refused
    [r for r in at.radio if r.label == "Decision"][0].set_value(other)
    [t for t in at.text_area if t.label.startswith("Override")][0].input("checked with the policyholder")
    [b for b in at.button if b.label == "Submit decision"][0].click()
    at.run()
    assert not at.exception and at.success and other in at.success[0].value
    rows = sqlite3.connect(audit_path).execute(
        "SELECT recommendation, adjuster_decision, overrides_recommendation, override_reason FROM adjuster_decisions"
    ).fetchall()
    assert rows == [(recommended, other, 1, "checked with the policyholder")]


def test_dashboard_llm_mode_without_configuration_explains_and_stops(audit_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = AppTest.from_file(str(path_for("models_dir").parent / "dashboard" / "app.py"), default_timeout=120)
    at.run()
    [r for r in at.radio if r.label == "Agent"][0].set_value("LLM from .env")
    at.run()
    assert not at.exception
    assert any("No API key" in e.value for e in at.sidebar.error)
    assert any("Rule-based" in e.value for e in at.error)
