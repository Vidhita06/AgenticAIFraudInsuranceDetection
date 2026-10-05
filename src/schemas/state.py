"""LangGraph state for one claim's triage run."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_dicts(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}


class ClaimState(TypedDict, total=False):
    raw_claim: dict[str, Any]                       # input as received
    claim: dict[str, Any]                           # normalised claim fields
    claim_id: str
    normalization_issues: list[str]
    messages: Annotated[list[AnyMessage], add_messages]   # agent conversation
    tool_results: Annotated[dict[str, Any], merge_dicts]  # tool name -> latest output
    steps: int                                      # agent (LLM) turns used
    max_steps_reached: bool
    llm_failures: int
    llm_error: Optional[str]
    draft: Optional[dict[str, Any]]                 # DecisionDraft from synthesis (or None)
    decision: Optional[dict[str, Any]]              # TriageDecision after guardrails
    adjuster: Optional[dict[str, Any]]              # human review outcome
    trace: Annotated[list[dict[str, Any]], operator.add]  # audit log
