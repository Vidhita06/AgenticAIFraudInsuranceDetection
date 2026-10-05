"""Common claim schema (synopsis Phase 1): every raw field, its type and allowed values.
The input contract for ingestion and every agent tool."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.ml.data import CATEGORIES, INTEGER_DOMAINS, RAW_COLUMNS

AGE_MAX = 120


def _field_type(col: str):
    if col in CATEGORIES:
        return Literal[tuple(CATEGORIES[col])]
    domain = INTEGER_DOMAINS.get(col)
    if domain is not None:
        return Literal[tuple(sorted(domain))]
    return int


def _fields() -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for col in RAW_COLUMNS:
        if col == "PolicyNumber":
            fields[col] = (Optional[int], Field(None, description="Claim/policy identifier (optional)"))
        elif col == "FraudFound_P":
            fields[col] = (Optional[Literal[0, 1]], Field(None, description="Label; never a model input"))
        elif col == "Age":
            fields[col] = (int, Field(..., ge=0, le=AGE_MAX, description="0 = age not recorded"))
        else:
            fields[col] = (_field_type(col), Field(...))
    return fields


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    @property
    def claim_id(self) -> str:
        pn = getattr(self, "PolicyNumber", None)
        return f"PN-{pn}" if pn is not None else "unidentified"

    def to_record(self) -> dict[str, Any]:
        return self.model_dump()


ClaimSchema = create_model("ClaimSchema", __base__=_Base, **_fields())
ClaimSchema.__doc__ = "A vehicle insurance claim with all 33 raw fields of the Kaggle schema."
