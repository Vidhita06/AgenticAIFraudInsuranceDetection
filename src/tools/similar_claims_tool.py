"""Tool 5, Similar Claims Retrieval: the most similar historical claims from the FAISS index.

The index holds genuine real training claims only (no synthetic or augmented rows, no
validation or test claims), so every match is a real past claim with its recorded outcome.
Similarity is context, not proof of fraud."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from src.retrieval.search import get_searcher
from src.tools._common import jsonable, require_valid_claim, safe_tool

COMPARE = ["Fault", "BasePolicy", "VehicleCategory", "VehiclePrice", "Deductible",
           "AddressChange_Claim", "PastNumberOfClaims", "Days_Policy_Accident", "AgentType"]
PROVENANCE = {"dataset": "real_train", "record_type": "genuine historical claim", "synthetic": False,
              "outcome_field": "FraudFound_P (fraud found by the insurer's investigation)"}


@safe_tool
def find_similar_claims(claim: Mapping[str, Any], k: int = 5, same_base_policy: bool = False) -> dict[str, Any]:
    """Return the k most similar past claims (excluding this claim) with similarity, shared
    characteristics, differences, recorded outcomes and provenance."""
    k = int(max(1, min(int(k), 20)))
    n = require_valid_claim(claim)
    record = n.claim.to_record()
    searcher = get_searcher()
    if "real" not in str(searcher.info.get("source", "")):
        raise RuntimeError("similar-claims index is not marked as real claims only; refusing to use it")
    hits = searcher.search(record, k * 10 if same_base_policy else k)
    if same_base_policy:
        hits = [h for h in hits if h["key_fields"]["BasePolicy"] == record["BasePolicy"]][:k]
    fraud = sum(h["fraud_label"] for h in hits)
    neighbours = []
    for h in hits:
        kf = {f: h["key_fields"][f] for f in COMPARE}
        neighbours.append({
            "claim_id": f"PN-{h['policy_number']}", "policy_number": h["policy_number"],
            "similarity": h["similarity"], "fraud_label": h["fraud_label"],
            "outcome": "fraud found" if h["fraud_label"] else "no fraud found",
            "shared": [f for f in COMPARE if str(kf[f]) == str(record[f])],
            "differences": {f: {"this_claim": record[f], "match": kf[f]} for f in COMPARE
                            if str(kf[f]) != str(record[f])},
            "key_fields": kf, "source": "real_train",
        })
    diffs = []
    for f in COMPARE:
        values = Counter(str(h["key_fields"][f]) for h in hits)
        if not values:
            break
        common, cnt = values.most_common(1)[0]
        if str(record[f]) != common and cnt >= (len(hits) + 1) // 2:
            diffs.append(f"{f}: this claim has '{record[f]}', most similar claims have '{common}'")
    shared_all = [f for f in COMPARE if hits and all(str(h["key_fields"][f]) == str(record[f]) for h in hits)]
    return jsonable({
        "claim_id": n.claim_id, "status": "completed", "k": len(hits), "requested_k": k,
        "constraints": {"same_base_policy": bool(same_base_policy), "current_claim_excluded": True},
        "neighbours": neighbours,
        "neighbour_frauds": fraud,
        "neighbour_fraud_rate": round(fraud / len(hits), 3) if hits else None,
        "historical_base_rate": round(searcher.info["fraud_rate"], 4),
        "shared_by_all_matches": shared_all,
        "key_differences": diffs,
        "provenance": {**PROVENANCE, "index": searcher.info["source"], "index_size": searcher.info.get("n_claims")},
        "index": searcher.info["source"],
    })
