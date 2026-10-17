"""Structured investigation output. The system recommends; it never denies a claim, and every
recommendation goes to a human adjuster."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Decision = Literal["APPROVE", "FLAG_FOR_INVESTIGATION", "REQUEST_MORE_INFORMATION"]
ALLOWED_DECISIONS: tuple[str, ...] = ("APPROVE", "FLAG_FOR_INVESTIGATION", "REQUEST_MORE_INFORMATION")
# Older or abbreviated labels accepted from model output and normalised to the above.
DECISION_ALIASES = {"REQUEST_MORE_INFO": "REQUEST_MORE_INFORMATION", "REQUEST_INFO": "REQUEST_MORE_INFORMATION",
                    "MORE_INFO": "REQUEST_MORE_INFORMATION", "FLAG": "FLAG_FOR_INVESTIGATION",
                    "INVESTIGATE": "FLAG_FOR_INVESTIGATION"}
ToolName = Literal["validation", "fraud_score", "shap_explanation", "policy_retrieval", "policy_verification",
                   "similar_claims", "claim_history"]
ToolStatus = Literal["completed", "partial", "unavailable", "failed", "not_run"]
# Why a claim was flagged. Only FRAUD_RISK flags count as "fraud caught" in the evaluation.
FlagCategory = Literal["FRAUD_RISK", "POLICY_CONDITION", "SYSTEM_FAILURE"]


def normalize_decision(label: Any) -> str:
    d = str(label).strip().upper().replace(" ", "_").replace("-", "_")
    return DECISION_ALIASES.get(d, d)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source_tool: ToolName
    fact: str = Field(..., min_length=3, max_length=400)


class DecisionDraft(BaseModel):
    """What the reasoning LLM must return (validated before guardrails run)."""
    model_config = ConfigDict(extra="ignore")
    decision: str
    rationale: str = Field(..., min_length=10)
    evidence: list[Evidence] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    flag_category: Optional[FlagCategory] = None    # set by the rule-based decision only


class ToolStatusEntry(BaseModel):
    status: ToolStatus
    origin: Optional[str] = None          # orchestrator | agent | dependency | auto
    detail: Optional[str] = None
    data_origin: Optional[str] = None     # "simulated reference data" when the output used it


class TriageDecision(BaseModel):
    """Final recommendation (plan §12). Fields other than decision / rationale / evidence /
    unresolved_questions are filled in code from the evidence record, never by the LLM."""
    claim_id: str
    investigation_id: Optional[str] = None
    investigation_status: Optional[str] = None
    decision: Decision
    flag_category: Optional[FlagCategory] = None    # set for every FLAG_FOR_INVESTIGATION
    simulated_data: bool = False                    # simulated reference data enabled for this run
    fraud_probability: Optional[float] = None
    risk_band: Optional[str] = None
    predicted_class: Optional[str] = None
    reason_codes: list[str] = Field(default_factory=list)
    rationale: str
    evidence: list[Evidence] = Field(default_factory=list)
    policy_findings: dict[str, Any] = Field(default_factory=dict)
    similar_claim_findings: dict[str, Any] = Field(default_factory=dict)
    claim_history_summary: dict[str, Any] = Field(default_factory=dict)
    missing_information: list[str] = Field(default_factory=list)
    conflicting_evidence: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    tool_statuses: dict[str, ToolStatusEntry] = Field(default_factory=dict)
    validation_issues: list[str] = Field(default_factory=list)
    similar_claim_ids: list[str] = Field(default_factory=list)
    requires_human_review: Literal[True] = True
    decided_by: Literal["llm", "rule_fallback"] = "llm"
    guardrail_notes: list[str] = Field(default_factory=list)
    model_version: Optional[str] = None


class AdjusterDecision(BaseModel):
    decision: Literal["APPROVE", "FLAG_FOR_INVESTIGATION", "REQUEST_MORE_INFORMATION", "DENY"]
    adjuster_id: str = "adjuster"
    override_reason: Optional[str] = None
    notes: Optional[str] = None

    _normalize = field_validator("decision", mode="before")(lambda v: normalize_decision(v))
