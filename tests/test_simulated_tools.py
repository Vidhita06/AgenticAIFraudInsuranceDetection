"""Tools 3, 4 and 6 with the SIMULATED reference database on and off, and the run-time tables."""
import sqlite3

import pytest

from src.config import path_for
from src.tools import get_claim_history, retrieve_policy, verify_policy

SIM = "simulated reference data"
needs_models = pytest.mark.skipif(not (path_for("models_dir") / "fraud_model.joblib").exists(),
                                  reason="model artifacts not present")


def checks(res):
    return {c["check_id"]: c for c in res["checks"]}


def test_off_behaves_exactly_as_without_the_database(sample_claims, simulated_db):
    for flag in (False, None):                       # explicit off, and the config default (off)
        r = retrieve_policy(sample_claims[1], simulated=flag)
        assert r["status"] == "partial" and "source" not in r
        h = get_claim_history(sample_claims[1], simulated=flag)
        assert h["status"] == "unavailable" and h["previous_claim_count"] is None
        v = verify_policy(sample_claims[1], simulated=flag)
        assert "PV08_COVERAGE_LIMITS" in checks(v) and "SIM-ALL-04" not in checks(v)


def test_retrieval_from_the_simulated_database(sample_claims, simulated_db):
    r = retrieve_policy(sample_claims[1], simulated=True)
    assert r["ok"] and r["status"] == "found" and r["source"] == SIM
    assert r["wording_note"] == "illustrative wording written by the project team"
    assert r["simulated_policy"]["base_policy"] == sample_claims[1]["BasePolicy"]
    assert r["applicable_version"]["effective_from"] <= r["simulated_policy"]["accident_date_derived"]
    ids = {c["clause_id"] for cat in ("conditions", "exclusions", "reporting_requirements") for c in r["document"][cat]}
    assert {"SIM-ALL-01", "SIM-ALL-04", "SIM-AP-02"} <= ids          # All Perils claim
    assert any("claim amount" in m for m in r["missing_information"])


def test_verification_uses_the_clauses(sample_claims, simulated_db):
    v = verify_policy(sample_claims[1], simulated=True)
    c = checks(v)
    assert v["source"] == SIM and c["PV01_POLICY_IDENTIFIED"]["result"] == "PASS"
    assert c["PV02_POLICY_TYPE_CONSISTENT"]["level"] == "claim_record"
    assert c["SIM-ALL-04"]["result"] == "PASS" and c["SIM-ALL-04"]["source"] == SIM
    assert c["SIM-ALL-05"]["result"] == "UNKNOWN"                      # no claim amount
    assert c["SIM-AP-02"]["result"] == "UNKNOWN"                       # peril not recorded
    assert all(x["level"] == "policy_condition" for k, x in c.items() if k.startswith("SIM-"))
    late = {**sample_claims[1], "MonthClaimed": "Jun", "WeekOfMonthClaimed": 2}     # accident in March
    lv = verify_policy(late, simulated=True)
    assert checks(lv)["SIM-ALL-04"]["result"] == "FAIL"
    assert lv["policy_condition_failures"] == ["SIM-ALL-04"] and lv["claim_record_failures"] == []
    no_age = verify_policy({**sample_claims[1], "Age": 0}, simulated=True)
    assert checks(no_age)["SIM-COL-02"]["result"] == "FAIL"


def test_history_from_the_simulated_database(sample_claims, simulated_db):
    for claim in sample_claims[:4]:
        h = get_claim_history(claim, simulated=True)
        assert h["status"] == "completed" and h["source"] == SIM and h["history_available"]
        lo, hi = {"none": (0, 0), "1": (1, 1), "2 to 4": (2, 4), "more than 4": (5, 7)}[claim["PastNumberOfClaims"]]
        assert lo <= h["previous_claim_count"] <= hi
        for p in h["previous_claims"]:
            assert set(p) == {"prior_claim_id", "claim_date", "incident_type", "status"}   # no outcomes
        assert "No investigation outcomes" in h["note"]


def test_claims_missing_from_the_database_fall_back(sample_claims, simulated_db):
    unknown = {**sample_claims[1], "PolicyNumber": 99999}
    assert retrieve_policy(unknown, simulated=True)["status"] == "partial"
    assert get_claim_history(unknown, simulated=True)["status"] == "unavailable"


@needs_models
def test_graph_with_simulated_data_writes_runtime_tables(sample_claims, simulated_db, vector_store):
    from src.agent.nodes import NodeDeps
    from src.agent.runner import TriageRunner
    db_file = simulated_db[0]
    con = sqlite3.connect(db_file)
    before = con.execute("SELECT COUNT(*) FROM investigation_audit").fetchone()[0]
    r = TriageRunner(deps=NodeDeps(llm=None, simulated=True))
    tid, d, trace = r.run_claim(sample_claims[1])
    assert d.simulated_data is True and "SIMULATED_DATA_USED" in d.reason_codes
    assert d.tool_statuses["claim_history"].data_origin == SIM
    assert d.tool_statuses["fraud_score"].data_origin is None
    assert all(e.fact.endswith(f"({SIM})") for e in d.evidence if e.source_tool in ("claim_history",
                                                                                      "policy_verification"))
    assert [t for t in trace if t["node"] == "guardrails"][0]["runtime_tables"] == "written"
    con = sqlite3.connect(db_file)
    assert con.execute("SELECT COUNT(*) FROM investigation_audit").fetchone()[0] == before + 1
    rows = con.execute("SELECT check_id, result, source FROM verification_results WHERE investigation_id = ?",
                       (d.investigation_id,)).fetchall()
    assert {r[0] for r in rows} >= {"PV01_POLICY_IDENTIFIED", "SIM-ALL-04"}
    off = TriageRunner(deps=NodeDeps(llm=None, simulated=False))
    tid, d2, trace2 = off.run_claim(sample_claims[1])
    assert d2.simulated_data is False and "SIMULATED_DATA_USED" not in d2.reason_codes
    assert "runtime_tables" not in [t for t in trace2 if t["node"] == "guardrails"][0]
    assert con.execute("SELECT COUNT(*) FROM investigation_audit").fetchone()[0] == before + 1


@needs_models
def test_pn_5736_low_band_with_data_gaps_is_approved(vector_store):
    """Pilot regression (qwen2.5:72b, simulated data OFF): PN-5736 (legit, p ~ 0.009) was flagged
    FRAUD_RISK only because no policy document exists. The guardrail restores APPROVE."""
    import json as _json

    import pandas as pd
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from src.agent.nodes import NodeDeps
    from src.agent.runner import TriageRunner

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    v = pd.read_csv(path_for("real_validation"))
    claim = v[v.PolicyNumber == 5736].iloc[0].drop("FraudFound_P").to_dict()
    flag = AIMessage(content=_json.dumps({
        "decision": "FLAG_FOR_INVESTIGATION",
        "rationale": "Despite the low fraud score, the lack of a policy document is a critical limitation "
                     "that warrants further investigation.",
        "evidence": [{"source_tool": "policy_retrieval", "fact": "Policy document not available"}]}))
    llm = Scripted(messages=iter([AIMessage(content="Evidence gathered."), flag]))
    tid, d, trace = TriageRunner(deps=NodeDeps(llm=llm, simulated=False)).run_claim(claim)
    assert d.risk_band == "low" and d.simulated_data is False
    assert set(d.reason_codes) <= {"LOW_MODEL_RISK", "POLICY_TERMS_UNVERIFIED", "CLAIM_HISTORY_UNAVAILABLE",
                                   "SIMILAR_CLAIMS_NO_FRAUD", "NO_MATERIAL_CONCERNS"}
    assert d.decision == "APPROVE" and d.flag_category is None and d.decided_by == "llm"
    assert any(n.startswith("unsupported_fraud_flag") for n in d.guardrail_notes)


def test_simulated_history_never_contradicts_the_claim(sample_claims, simulated_db):
    """Simulated prior-claim counts follow PastNumberOfClaims by construction, so the history
    contradiction (a medium-band flag trigger) can only come from a real history source."""
    from src.agent.evidence import history_contradiction
    for claim in sample_claims[:4]:
        assert history_contradiction({"claim_history": get_claim_history(claim, simulated=True)}) is None
