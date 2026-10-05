"""Top-k similar historical claims from the FAISS index."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.config import path_for
from src.retrieval.build_index import KEY_FIELDS


class SimilarClaimSearcher:
    def __init__(self, index, meta: pd.DataFrame, encoder, info: dict):
        self.index, self.meta, self.encoder, self.info = index, meta, encoder, info

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "SimilarClaimSearcher":
        import faiss
        import joblib

        d = Path(directory) if directory else path_for("vector_store_dir")
        if not (d / "claims.faiss").exists():
            raise FileNotFoundError(f"No index in {d}. Run: python scripts/build_vector_index.py")
        return cls(faiss.read_index(str(d / "claims.faiss")), pd.read_parquet(d / "claims_meta.parquet"),
                   joblib.load(d / "encoder.joblib"), json.loads((d / "index_meta.json").read_text()))

    def search(self, claim: Mapping[str, Any], k: int = 5) -> list[dict[str, Any]]:
        q = self.encoder.encode(claim)
        sims, idx = self.index.search(q, k + 1)
        own = claim.get("PolicyNumber")
        out = []
        for s, i in zip(sims[0], idx[0]):
            if i < 0:
                continue
            row = self.meta.iloc[int(i)]
            if own is not None and int(row["policy_number"]) == int(own):
                continue                     # never return the claim itself
            out.append({"policy_number": int(row["policy_number"]), "similarity": round(float(s), 4),
                        "fraud_label": int(row["fraud_label"]),
                        "key_fields": {f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in KEY_FIELDS}})
            if len(out) == k:
                break
        return out


@lru_cache(maxsize=2)
def get_searcher(directory: str | None = None) -> SimilarClaimSearcher:
    return SimilarClaimSearcher.load(directory)
