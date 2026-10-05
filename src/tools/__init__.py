"""Agent tools: pure, typed functions returning JSON-serialisable dicts ({"ok": False,
"error"} on failure), plus LangChain wrappers.

- TOOLS: generic LangChain tools taking the full claim as input.
- agent_tools(claim): claim-bound tools for the triage agent (the LLM only chooses whether to
  call them and with which k; it never re-types the claim, so it cannot alter it).
"""
from __future__ import annotations

from typing import Any, Mapping

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.tools.fraud_score_tool import score_claim
from src.tools.policy_check_tool import check_policy_rules
from src.tools.similar_claims_tool import find_similar_claims


class _ClaimInput(BaseModel):
    claim: dict[str, Any] = Field(..., description="Claim with the 33 raw fields")


class _SimilarInput(_ClaimInput):
    k: int = Field(5, ge=1, le=20)


TOOLS = [
    StructuredTool.from_function(score_claim, name="score_claim", args_schema=_ClaimInput,
                                 description="Fraud probability, risk band and top SHAP factors."),
    StructuredTool.from_function(find_similar_claims, name="find_similar_claims", args_schema=_SimilarInput,
                                 description="Most similar historical claims and their fraud rate."),
    StructuredTool.from_function(check_policy_rules, name="check_policy_rules", args_schema=_ClaimInput,
                                 description="Validation/consistency rules and red flags."),
]

AGENT_TOOL_TO_SOURCE = {"find_similar_claims": "similar_claims", "check_policy_rules": "policy_check"}


class _K(BaseModel):
    k: int = Field(5, ge=1, le=10, description="How many similar claims to retrieve")


class _NoArgs(BaseModel):
    pass


def agent_tools(claim: Mapping[str, Any]) -> list[StructuredTool]:
    claim = dict(claim)
    return [
        StructuredTool.from_function(
            lambda k=5: find_similar_claims(claim, k), name="find_similar_claims", args_schema=_K,
            description="Retrieve the k most similar historical (real) claims for this claim, with "
                        "their fraud labels, the neighbours' fraud rate and key differences."),
        StructuredTool.from_function(
            lambda: check_policy_rules(claim), name="check_policy_rules", args_schema=_NoArgs,
            description="Run the policy/consistency rules and red-flag checks on this claim."),
    ]


__all__ = ["TOOLS", "agent_tools", "score_claim", "find_similar_claims", "check_policy_rules",
           "AGENT_TOOL_TO_SOURCE"]
