import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_claims() -> list[dict]:
    data = json.loads((ROOT / "tests" / "fixtures" / "sample_claims.json").read_text())
    return [{k: v for k, v in c.items() if not k.startswith("_")} for c in data["claims"]]


@pytest.fixture(scope="session")
def vector_store(tmp_path_factory):
    """Build the FAISS index into a temp dir if the repo copy (git-ignored) is missing."""
    from src.config import path_for
    from src.retrieval.search import get_searcher

    d = path_for("vector_store_dir")
    if not (d / "claims.faiss").exists():
        from src.retrieval.build_index import build_index
        build_index(d)
    get_searcher.cache_clear()
    return d
