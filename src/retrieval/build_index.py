"""Build the FAISS similar-claims index over REAL training claims only
(no synthetic rows, no validation/test claims).

Writes to data/vector_store/: claims.faiss, claims_meta.parquet (policy number, label,
key fields), encoder.joblib, index_meta.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import load_config, path_for, sha256_file
from src.ml.data import ID_COL, TARGET, load_real_train
from src.retrieval.encoder import ClaimEncoder

KEY_FIELDS = ["Fault", "BasePolicy", "VehicleCategory", "VehiclePrice", "Make", "Deductible",
              "AddressChange_Claim", "PastNumberOfClaims", "Days_Policy_Accident", "Age",
              "AgentType", "PoliceReportFiled", "Month", "Year"]


def dependency_fingerprint() -> dict[str, str]:
    """Hashes of everything the index depends on. Stored in index_meta.json at build time
    and compared at load time (see check_index_fresh)."""
    import hashlib

    models = path_for("models_dir") / "preprocessor.joblib"
    feats = json.dumps(load_config()["features"], sort_keys=True).encode()
    return {
        "model_preprocessor_sha256": sha256_file(models) if models.exists() else "missing",
        "policy_rules_sha256": sha256_file(path_for("policy_rules")),
        "feature_config_sha256": hashlib.sha256(feats).hexdigest(),
    }


class StaleIndexError(RuntimeError):
    pass


def check_index_fresh(info: dict) -> None:
    """Raise StaleIndexError if models/preprocessor.joblib, the policy rules or the feature
    config changed since the index was built."""
    built = info.get("dependencies", {})
    now = dependency_fingerprint()
    changed = [k for k, v in now.items() if built.get(k) != v]
    if changed:
        raise StaleIndexError(
            f"similar-claims index is stale ({', '.join(changed)} changed since it was built); "
            "rebuild it with: python scripts/build_vector_index.py")


def build_index(out_dir: str | Path | None = None) -> dict:
    import faiss
    import joblib

    out = Path(out_dir) if out_dir else path_for("vector_store_dir")
    out.mkdir(parents=True, exist_ok=True)
    real = load_real_train()
    max_real = load_config()["data"]["real_policy_number_max"]
    assert real[ID_COL].max() <= max_real, "index must contain real claims only"
    enc = ClaimEncoder().fit(real)
    X = enc.encode(real)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(out / "claims.faiss"))
    meta = real[[ID_COL, TARGET] + KEY_FIELDS].rename(columns={ID_COL: "policy_number", TARGET: "fraud_label"})
    meta.to_parquet(out / "claims_meta.parquet", index=False)
    joblib.dump(enc, out / "encoder.joblib")
    info = {"n_claims": int(len(real)), "dim": int(X.shape[1]), "source": "real_train only",
            "fraud_rate": float(real[TARGET].mean()),
            "real_train_sha256": sha256_file(path_for("augmented_train")),
            "encoder_sha256": sha256_file(out / "encoder.joblib"),
            "dependencies": dependency_fingerprint()}
    (out / "index_meta.json").write_text(json.dumps(info, indent=2))
    return info
