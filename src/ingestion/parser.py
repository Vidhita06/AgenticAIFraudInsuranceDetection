"""Read claims from a CSV/JSON file, a JSON string, a dict (form) or a list of dicts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def parse_claims(source: Any) -> list[dict[str, Any]]:
    """Return a list of raw claim dicts. Values are passed through as given; the
    normalizer coerces and validates them."""
    if isinstance(source, pd.DataFrame):
        return [_clean(r) for r in source.to_dict("records")]
    if isinstance(source, Mapping):
        return [dict(source)]
    if isinstance(source, (str, Path)):
        text = str(source)
        path = Path(text)
        if len(text) < 4096 and path.suffix.lower() in (".csv", ".json", ".jsonl") and path.exists():
            if path.suffix.lower() == ".csv":
                return parse_claims(pd.read_csv(path, encoding="utf-8-sig", dtype=str))
            if path.suffix.lower() == ".jsonl":
                return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            return parse_claims(json.loads(path.read_text(encoding="utf-8")))
        return parse_claims(json.loads(text))
    if isinstance(source, Iterable):
        return [dict(c) for c in source]
    raise TypeError(f"cannot parse claims from {type(source).__name__}")


def _clean(record: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in record.items()}
