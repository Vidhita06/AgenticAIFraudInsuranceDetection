"""Raw claim dict -> normalised fields + validated ClaimSchema. Never raises on bad input:
missing or invalid fields are listed so the agent can ask for more information."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from pydantic import ValidationError

from src.ml.data import CATEGORIES, INTEGER_DOMAINS, RAW_COLUMNS
from src.schemas.claim import ClaimSchema

_LOOKUP = {col: {v.lower(): v for v in vals} for col, vals in CATEGORIES.items()}
_ALIASES = {col.lower().replace("_", ""): col for col in RAW_COLUMNS}


@dataclass
class NormalizedClaim:
    record: dict[str, Any]                 # best-effort normalised values (all 33 keys)
    claim: Optional[Any] = None            # ClaimSchema when every field is valid
    missing: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.claim is not None

    @property
    def claim_id(self) -> str:
        pn = self.record.get("PolicyNumber")
        return f"PN-{pn}" if pn not in (None, "") else "unidentified"

    @property
    def issues(self) -> list[str]:
        return [f"missing field: {m}" for m in self.missing] + [f"invalid value: {i}" for i in self.invalid]


def _coerce(col: str, value: Any) -> Any:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if col in CATEGORIES:
        s = str(value).strip()
        return _LOOKUP[col].get(s.lower(), s)
    if col in INTEGER_DOMAINS or col in ("Age", "PolicyNumber"):
        try:
            f = float(value)
            return int(f) if f.is_integer() else f
        except (TypeError, ValueError):
            return value
    return value


def normalize_claim(raw: Mapping[str, Any]) -> NormalizedClaim:
    keyed = {}
    for k, v in raw.items():
        col = _ALIASES.get(str(k).strip().lower().replace("_", "").replace(" ", ""), str(k))
        keyed[col] = v
    record = {col: _coerce(col, keyed.get(col)) for col in RAW_COLUMNS}
    required = [c for c in RAW_COLUMNS if c not in ("PolicyNumber", "FraudFound_P")]
    missing = [c for c in required if record[c] is None]
    try:
        claim = ClaimSchema(**{k: v for k, v in record.items() if v is not None})
        return NormalizedClaim(record, claim)
    except ValidationError as e:
        invalid = sorted({f"{err['loc'][0]}={record.get(err['loc'][0])!r}" for err in e.errors()
                          if err["loc"] and err["loc"][0] not in missing})
        return NormalizedClaim(record, None, missing, invalid)
