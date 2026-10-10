"""Build the FAISS similar-claims index (real training claims only).

    python scripts/build_vector_index.py [--out data/vector_store]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.build_index import build_index  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    print(json.dumps(build_index(ap.parse_args().out), indent=2))
