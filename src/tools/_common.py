"""Shared helpers: input validation against ClaimSchema and error capture."""
from __future__ import annotations

import functools
import math
from typing import Any, Callable, Mapping

from src.ingestion.normalizer import normalize_claim


def require_valid_claim(claim: Mapping[str, Any]):
    n = normalize_claim(claim)
    if not n.ok:
        raise ValueError("claim failed schema validation: " + "; ".join(n.issues[:12]))
    return n


def safe_tool(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Tools never raise: any error becomes {"ok": False, "error": ...}."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> dict:
        try:
            out = fn(*args, **kwargs)
            return {"ok": True, **out}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return wrapper


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x
