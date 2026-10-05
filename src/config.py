"""Project configuration: paths, constants and reproducibility helpers."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
RANDOM_STATE = 42


@lru_cache(maxsize=1)
def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path_for(key: str) -> Path:
    """Absolute path for an entry under `paths:` in config/settings.yaml."""
    return PROJECT_ROOT / load_config()["paths"][key]


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes() -> dict[str, str]:
    """SHA-256 of every input data file, for logging in reports and model metadata."""
    keys = ["raw", "augmented_train", "real_validation", "real_test"]
    # The B200 bundle ships data/processed only, so absent files are recorded, not required.
    return {load_config()["paths"][k]: sha256_file(path_for(k)) if path_for(k).exists() else "missing"
            for k in keys}
