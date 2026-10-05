"""Agent tools, artifact loading and retrieval (uses models/ placeholder or real artifacts)."""
import json

import numpy as np
import pytest

from src.config import path_for
from src.ingestion.normalizer import normalize_claim
from src.ingestion.parser import parse_claims
from src.ml.predict import load_artifacts, score_claims
from src.schemas.claim import ClaimSchema
from src.tools import TOOLS, agent_tools, check_policy_rules, find_similar_claims, score_claim

needs_models = pytest.mark.skipif(not (path_for("models_dir") / "fraud_model.joblib").exists(),
                                  reason="model artifacts not present")


@needs_models
def test_artifacts_load_and_score_fixtures(sample_claims):
    art = load_artifacts()
    assert {"medium", "high"} <= set(art.bands) and art.bands["medium"] <= art.bands["high"]
    complete = [c for c in sample_claims if normalize_claim(c).ok]
    out = score_claims(complete, art)
    assert len(out) == len(complete) == 4
    for o in out:
        assert 0.0 <= o["fraud_probability"] <= 1.0
        assert o["risk_band"] in ("low", "medium", "high")
    card = art.card
    for key in ("model_version", "thresholds", "risk_bands", "features", "data_sha256", "environment"):
        assert key in card


@needs_models
def test_score_claim_tool(sample_claims):
    res = score_claim(sample_claims[1])
    assert res["ok"] is True
    assert 0 <= res["fraud_probability"] <= 1
    assert res["risk_band"] in ("low", "medium", "high")
    assert 1 <= len(res["top_factors"]) <= 5
    for f in res["top_factors"]:
        assert {"feature", "value", "shap_contribution", "reason"} <= set(f)
    json.dumps(res)  # JSON-serialisable


def test_score_claim_rejects_invalid_claim(sample_claims):
    res = score_claim(sample_claims[-1])
    assert res["ok"] is False and "schema" in res["error"]


def test_policy_check_tool(sample_claims):
    missing_age = check_policy_rules(sample_claims[3])
    assert missing_age["ok"] and missing_age["blocking"]
    assert "V03_AGE_MISSING" in {r["rule_id"] for r in missing_age["failed_rules"]}
    tp = check_policy_rules(sample_claims[2])
    assert "RF04_THIRD_PARTY_DEDUCTIBLE_500" in {f["rule_id"] for f in tp["red_flags"]}
    assert all("support" in f for f in tp["red_flags"])
    incomplete = check_policy_rules(sample_claims[-1])
    assert incomplete["ok"] and incomplete["blocking"] and incomplete["schema_issues"]
    json.dumps(tp)


def test_similar_claims_tool(sample_claims, vector_store):
    res = find_similar_claims(sample_claims[0], k=5)
    assert res["ok"] and res["k"] == 5
    ids = [n["policy_number"] for n in res["neighbours"]]
    assert sample_claims[0]["PolicyNumber"] not in ids          # never the claim itself
    assert all(i <= 15420 for i in ids)                         # real claims only
    sims = [n["similarity"] for n in res["neighbours"]]
    assert sims == sorted(sims, reverse=True) and all(0 < s <= 1.0001 for s in sims)
    # Neighbours share the dominant risk factors of the query
    assert sum(n["key_fields"]["Fault"] == sample_claims[0]["Fault"] for n in res["neighbours"]) >= 3
    assert 0 <= res["neighbour_fraud_rate"] <= 1


def test_index_excludes_synthetic_validation_and_test(vector_store):
    import pandas as pd
    from src.ml.data import load_real_test, load_real_validation
    meta = pd.read_parquet(vector_store / "claims_meta.parquet")
    assert len(meta) == 10793 and meta.policy_number.max() <= 15420
    held_out = set(load_real_validation().PolicyNumber) | set(load_real_test().PolicyNumber)
    assert not held_out & set(meta.policy_number)


def test_langchain_wrappers(sample_claims, vector_store):
    names = {t.name for t in TOOLS}
    assert names == {"score_claim", "find_similar_claims", "check_policy_rules"}
    assert TOOLS[2].invoke({"claim": sample_claims[3]})["blocking"] is True
    bound = {t.name: t for t in agent_tools(sample_claims[0])}
    assert bound["find_similar_claims"].invoke({"k": 2})["k"] == 2
    assert bound["check_policy_rules"].invoke({})["ok"]


def test_parser_and_normalizer(sample_claims, tmp_path):
    p = tmp_path / "claims.json"
    p.write_text(json.dumps(sample_claims))
    parsed = parse_claims(p)
    assert len(parsed) == len(sample_claims)
    messy = {**sample_claims[0], "fault": "policy holder", "Age": "47", "Deductible": "400.0"}
    messy.pop("Fault")
    n = normalize_claim(messy)
    assert n.ok and n.claim.Fault == "Policy Holder" and n.claim.Age == 47 and n.claim.Deductible == 400
    bad = normalize_claim({**sample_claims[0], "Make": "Tesla"})
    assert not bad.ok and any("Make" in i for i in bad.invalid)
    assert isinstance(ClaimSchema(**sample_claims[0]).claim_id, str)
