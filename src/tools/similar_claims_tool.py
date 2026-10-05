"""Similar historical claims (real training claims only) from the FAISS index."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from src.retrieval.build_index import KEY_FIELDS
from src.retrieval.search import get_searcher
from src.tools._common import jsonable, require_valid_claim, safe_tool

COMPARE = ["Fault", "BasePolicy", "VehicleCategory", "VehiclePrice", "Deductible",
           "AddressChange_Claim", "PastNumberOfClaims", "Days_Policy_Accident", "AgentType"]


@safe_tool
def find_similar_claims(claim: Mapping[str, Any], k: int = 5) -> dict[str, Any]:
    """Return the k most similar past claims, their fraud rate and the key differences."""
    k = int(max(1, min(int(k), 20)))
    n = require_valid_claim(claim)
    record = n.claim.to_record()
    searcher = get_searcher()
    hits = searcher.search(record, k)
    fraud = sum(h["fraud_label"] for h in hits)
    diffs = []
    for f in COMPARE:
        values = Counter(str(h["key_fields"][f]) for h in hits)
        common, cnt = values.most_common(1)[0]
        if str(record[f]) != common and cnt >= (len(hits) + 1) // 2:
            diffs.append(f"{f}: this claim has '{record[f]}', most similar claims have '{common}'")
    return jsonable({
        "claim_id": n.claim_id, "k": len(hits),
        "neighbours": [{"policy_number": h["policy_number"], "similarity": h["similarity"],
                        "fraud_label": h["fraud_label"],
                        "key_fields": {f: h["key_fields"][f] for f in COMPARE}} for h in hits],
        "neighbour_frauds": fraud,
        "neighbour_fraud_rate": round(fraud / len(hits), 3) if hits else None,
        "historical_base_rate": round(searcher.info["fraud_rate"], 4),
        "key_differences": diffs,
        "index": searcher.info["source"],
    })
