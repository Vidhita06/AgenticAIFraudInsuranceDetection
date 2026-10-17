"""Agent tools, artifact loading and retrieval (uses models/ placeholder or real artifacts)."""
import json

import numpy as np
import pytest

from src.config import path_for
from src.ingestion.normalizer import normalize_claim
from src.ingestion.parser import parse_claims
from src.ml.predict import load_artifacts, score_claims
from src.schemas.claim import ClaimSchema
from src.tools import TOOLS, agent_tools, find_similar_claims, score_claim, verify_policy

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
    assert res["risk_band"] in ("low", "medium", "high") and res["status"] == "completed"
    assert res["predicted_class"] == ("fraud" if res["fraud_probability"] >= res["decision_threshold"] else "not_fraud")
    assert res["model"]["version"] == res["model_version"] and "top_factors" not in res     # SHAP is Tool 2
    assert {"RF01_PH_FAULT_ALL_PERILS"} <= {f["rule_id"] for f in res["risk_indicators"]}
    assert all("support" in f for f in res["risk_indicators"])
    json.dumps(res)  # JSON-serialisable


def test_score_claim_rejects_invalid_claim(sample_claims):
    res = score_claim(sample_claims[-1])
    assert res["ok"] is False and "schema" in res["error"]


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
    names = [t.name for t in TOOLS]
    assert names == ["score_claim", "explain_claim", "retrieve_policy", "verify_policy", "find_similar_claims",
                     "get_claim_history"]
    assert TOOLS[3].invoke({"claim": sample_claims[0]})["overall_result"] in ("PASS", "FAIL", "UNKNOWN")
    bound = {t.name: t for t in agent_tools(sample_claims[0])}
    assert set(bound) == {"explain_fraud_score", "retrieve_policy", "verify_policy", "find_similar_claims",
                          "get_claim_history"}
    assert bound["find_similar_claims"].invoke({"k": 2})["k"] == 2
    assert bound["verify_policy"].invoke({})["ok"]
    assert bound["get_claim_history"].invoke({})["status"] == "unavailable"


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


def test_index_matches_current_model_and_rules(vector_store):
    """Fails if models/, the policy rules or the feature config changed after the index was
    built: rerun scripts/build_vector_index.py."""
    import json as _json
    from src.retrieval.build_index import check_index_fresh
    check_index_fresh(_json.loads((vector_store / "index_meta.json").read_text()))


def test_stale_index_is_refused(vector_store, monkeypatch):
    import json as _json
    from src.retrieval import build_index as bi
    from src.retrieval.search import SimilarClaimSearcher
    info = _json.loads((vector_store / "index_meta.json").read_text())
    monkeypatch.setattr(bi, "dependency_fingerprint",
                        lambda: {**info["dependencies"], "model_preprocessor_sha256": "different"})
    with pytest.raises(bi.StaleIndexError):
        bi.check_index_fresh(info)
    from src.retrieval import search as srch
    monkeypatch.setattr(srch, "check_index_fresh", bi.check_index_fresh)
    with pytest.raises(bi.StaleIndexError):
        SimilarClaimSearcher.load(vector_store)
