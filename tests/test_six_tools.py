"""Contracts of the six investigation tools (docs/agent_restructure_plan.md): statuses,
PASS / FAIL / UNKNOWN checks, provenance, and "unavailable" rather than invented data."""
import json

import pytest

from src.config import path_for
from src.tools import (explain_claim, find_similar_claims, get_claim_history, retrieve_policy, score_claim,
                       verify_policy)

needs_models = pytest.mark.skipif(not (path_for("models_dir") / "fraud_model.joblib").exists(),
                                  reason="model artifacts not present")

POLICY_DOC = {
    "policy_id": "PN-11",
    "versions": [
        {"version": "1", "effective_from": "1993-01-01", "effective_to": "1993-12-31", "deductible": 400},
        {"version": "2", "effective_from": "1994-01-01", "effective_to": "1994-12-31", "deductible": 400,
         "coverage": {"base_policy": "All Perils"}, "limits": {"per_claim": 50000},
         "conditions": [{"id": "C1", "clause": "4.2", "text": "a police report is required",
                         "requires": {"PoliceReportFiled": ["Yes"]}}],
         "exclusions": [{"id": "E1", "clause": "7.1", "text": "sport vehicles excluded",
                         "when": {"VehicleCategory": ["Sport"]}}],
         "reporting_requirements": {"max_months_after_accident": 1, "clause": "5.1"},
         "clauses": [{"ref": "4.2", "text": "..."}]},
    ],
}


def checks(res):
    return {c["check_id"]: c for c in res["checks"]}


# Tool 2 -------------------------------------------------------------------------------------

@needs_models
def test_shap_explanation(sample_claims):
    res = explain_claim(sample_claims[1], top_k=4)
    assert res["ok"] and res["status"] == "completed" and len(res["top_factors"]) == 4
    for f in res["top_factors"]:
        assert {"feature", "value", "shap_value", "direction", "reason"} <= set(f)
        assert (f["shap_value"] > 0) == (f["direction"] == "increases fraud score")
    assert set(res["positive_factors"]) | set(res["negative_factors"]) == {f["feature"] for f in res["top_factors"]}
    assert res["explained_class"].startswith("fraud") and "log-odds" in res["scale"]
    assert any("not" in l and "fraud" in l for l in res["limitations"])
    json.dumps(res)


def test_shap_explanation_needs_a_score(sample_claims):
    res = explain_claim(sample_claims[1], fraud_score={"ok": False, "error": "x"})
    assert not res["ok"] and "fraud score unavailable" in res["error"]


# Tool 3 -------------------------------------------------------------------------------------

def test_policy_retrieval_without_documents_reports_missing(sample_claims, tmp_path):
    res = retrieve_policy(sample_claims[1], store=tmp_path / "none")
    assert res["ok"] and res["status"] == "partial" and not res["document_found"]
    assert res["recorded_policy_attributes"]["BasePolicy"] == "All Perils"
    assert len(res["missing_information"]) == 8                  # every plan category reported missing
    assert res["source_references"][0]["source"] == "claim record"


def test_policy_retrieval_picks_the_version_in_force(sample_claims, tmp_path):
    (tmp_path / "11.json").write_text(json.dumps(POLICY_DOC))
    res = retrieve_policy(sample_claims[1], store=tmp_path)       # accident Mar 1994
    assert res["status"] == "found" and res["applicable_version"]["version"] == "2"
    assert res["document"]["conditions"][0]["id"] == "C1"
    assert res["missing_information"] == []


# Tool 4 -------------------------------------------------------------------------------------

def test_policy_verification_without_document_is_unknown_not_pass(sample_claims, tmp_path):
    res = verify_policy(sample_claims[1], policy=retrieve_policy(sample_claims[1], store=tmp_path / "none"))
    c = checks(res)
    assert res["ok"] and res["overall_result"] == "UNKNOWN" and not res["failed_checks"]
    for cid in ("PV01_POLICY_IDENTIFIED", "PV05_DEDUCTIBLE_MATCHES_POLICY", "PV06_COVERAGE_CONDITIONS",
                "PV07_EXCLUSIONS", "PV08_COVERAGE_LIMITS", "PV09_REPORTING_REQUIREMENT"):
        assert c[cid]["result"] == "UNKNOWN" and c[cid]["reason"]
    for cid in ("PV02_POLICY_TYPE_CONSISTENT", "PV03_INCIDENT_IN_COVERAGE_PERIOD", "PV04_CLAIM_TIMELINE"):
        assert c[cid]["result"] == "PASS"
    assert all({"rule", "clause_reference", "reason", "evidence", "result"} <= set(x) for x in res["checks"])


def test_policy_verification_against_a_document(sample_claims, tmp_path):
    (tmp_path / "11.json").write_text(json.dumps(POLICY_DOC))
    claim = {**sample_claims[1], "PoliceReportFiled": "No"}
    res = verify_policy(claim, policy=retrieve_policy(claim, store=tmp_path))
    c = checks(res)
    assert c["PV01_POLICY_IDENTIFIED"]["result"] == "PASS"
    assert c["PV03_INCIDENT_IN_COVERAGE_PERIOD"]["result"] == "PASS"       # Mar 1994 inside v2
    assert c["PV05_DEDUCTIBLE_MATCHES_POLICY"]["result"] == "PASS"
    assert c["PV06_COVERAGE_CONDITIONS"]["result"] == "FAIL" and c["PV06_COVERAGE_CONDITIONS"]["clause_reference"] == "4.2"
    assert c["PV07_EXCLUSIONS"]["result"] == "PASS"
    assert c["PV08_COVERAGE_LIMITS"]["result"] == "UNKNOWN"                # no claim amount in the data
    assert c["PV09_REPORTING_REQUIREMENT"]["result"] == "PASS"
    assert res["overall_result"] == "FAIL" and res["failed_checks"] == ["PV06_COVERAGE_CONDITIONS"]


def test_policy_verification_fails_on_timeline_and_flags_unknown_period(sample_claims):
    before = {**sample_claims[0], "WeekOfMonthClaimed": 1, "WeekOfMonth": 4}       # same month, claim first
    assert checks(verify_policy(before))["PV04_CLAIM_TIMELINE"]["result"] == "FAIL"
    none = {**sample_claims[0], "Days_Policy_Accident": "none"}
    assert checks(verify_policy(none))["PV03_INCIDENT_IN_COVERAGE_PERIOD"]["result"] == "UNKNOWN"


# Tool 5 -------------------------------------------------------------------------------------

def test_similar_claims_provenance_and_constraints(sample_claims, vector_store):
    res = find_similar_claims(sample_claims[1], k=4, same_base_policy=True)
    assert res["ok"] and res["k"] == 4 and res["constraints"]["current_claim_excluded"]
    assert res["provenance"]["synthetic"] is False and res["provenance"]["record_type"] == "genuine historical claim"
    for n in res["neighbours"]:
        assert n["key_fields"]["BasePolicy"] == sample_claims[1]["BasePolicy"]
        assert n["source"] == "real_train" and n["outcome"] in ("fraud found", "no fraud found")
        assert set(n["shared"]) | set(n["differences"]) == set(n["key_fields"])


# Tool 6 -------------------------------------------------------------------------------------

def test_claim_history_unavailable_is_not_a_clean_history(sample_claims):
    res = get_claim_history(sample_claims[1])
    assert res["ok"] and res["status"] == "unavailable" and res["history_available"] is False
    assert res["previous_claim_count"] is None                 # not 0: unknown, not "no claims"
    assert "unique" in res["limitation"]
    assert "not individual previous claim records" in res["recorded_summary"]["note"]


def test_claim_history_from_an_authorised_source(sample_claims, tmp_path):
    src = tmp_path / "history.csv"
    src.write_text("policy_id,claim_id,claim_date,incident_type,status,outcome\n"
                   "PN-11,C-1,1992-06-01,collision,closed,paid\n"
                   "PN-11,C-2,1993-11-15,theft,closed,denied\n"
                   "PN-11,C-3,1980-01-01,collision,closed,paid\n"        # outside the 5-year window
                   "PN-12,C-4,1993-01-01,collision,closed,paid\n")     # another policy
    res = get_claim_history(sample_claims[1], years=5, source=src)
    assert res["status"] == "completed" and res["previous_claim_count"] == 2
    assert [r["claim_id"] for r in res["previous_claims"]] == ["C-1", "C-2"]
    empty = get_claim_history({**sample_claims[1], "PolicyNumber": 500}, source=src)
    assert empty["status"] == "completed" and empty["previous_claim_count"] == 0 and empty["limitation"]


def test_invalid_claims_fail_cleanly(sample_claims):
    for fn in (score_claim, retrieve_policy, verify_policy, get_claim_history):
        res = fn(sample_claims[-1])
        assert res["ok"] is False and "schema" in res["error"]
