"""Structured triage output. The system recommends; it never denies a claim, and every
decision goes to a human adjuster."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["APPROVE", "FLAG_FOR_INVESTIGATION", "REQUEST_MORE_INFO"]
ALLOWED_DECISIONS: tuple[str, ...] = ("APPROVE", "FLAG_FOR_INVESTIGATION", "REQUEST_MORE_INFO")
ToolName = Literal["validation", "fraud_score", "similar_claims", "policy_check"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source_tool: ToolName
    fact: str = Field(..., min_length=3, max_length=400)


class DecisionDraft(BaseModel):
    """What the synthesis LLM must return (validated before guardrails run)."""
    model_config = ConfigDict(extra="ignore")
    decision: str
    rationale: str = Field(..., min_length=10)
    evidence: list[Evidence] = Field(default_factory=list)


class TriageDecision(BaseModel):
    claim_id: str
    decision: Decision
    fraud_probability: Optional[float] = None
    risk_band: Optional[str] = None
    rationale: str
    evidence: list[Evidence] = Field(default_factory=list)
    validation_issues: list[str] = Field(default_factory=list)
    similar_claim_ids: list[str] = Field(default_factory=list)
    requires_human_review: Literal[True] = True
    decided_by: Literal["llm", "rule_fallback"] = "llm"
    guardrail_notes: list[str] = Field(default_factory=list)
    model_version: Optional[str] = None


class AdjusterDecision(BaseModel):
    decision: Literal["APPROVE", "FLAG_FOR_INVESTIGATION", "REQUEST_MORE_INFO", "DENY"]
    adjuster_id: str = "adjuster"
    override_reason: Optional[str] = None
    notes: Optional[str] = None
