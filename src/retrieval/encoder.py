"""Claim -> L2-normalised vector for similar-claim retrieval.

Uses the "linear" preprocessor from src.ml.preprocess (one-hot nominal, scaled ordinal and
numeric, red-flag indicators), fitted on REAL training claims only and saved next to the
index. Cosine similarity on these vectors = inner product (FAISS IndexFlatIP)."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.ml.predict import claims_frame
from src.ml.preprocess import build_preprocessor


class ClaimEncoder:
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor

    def fit(self, real_train: pd.DataFrame) -> "ClaimEncoder":
        self.preprocessor = build_preprocessor("linear").fit(real_train)
        return self

    def encode(self, claims: pd.DataFrame | Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> np.ndarray:
        X = np.asarray(self.preprocessor.transform(claims_frame(claims)), dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-12)
