"""The six specialised investigation tools: pure, typed functions returning JSON-serialisable
dicts (`{"ok": False, "error"}` on failure), plus LangChain wrappers.

    Tool 1  score_claim          fraud probability, predicted class, risk band   -> fraud_score
    Tool 2  explain_claim        SHAP contributions                              -> shap_explanation
    Tool 3  retrieve_policy      applicable policy information                   -> policy_retrieval
    Tool 4  verify_policy        PASS / FAIL / UNKNOWN policy checks             -> policy_verification
    Tool 5  find_similar_claims  similar genuine historical claims               -> similar_claims
    Tool 6  get_claim_history    previous claims of the same policy / customer   -> claim_history

- TOOLS: generic LangChain tools taking the full claim as input.
- agent_tools(claim): claim-bound tools for the orchestrator LLM (it chooses which to call and
  with which arguments; it never re-types the claim, so it cannot alter it). Tool 1 is run by
  the orchestrator before the agent starts, as the initial risk assessment.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.tools.claim_history_tool import get_claim_history
from src.tools.fraud_score_tool import score_claim
from src.tools.policy_retrieval_tool import retrieve_policy
from src.tools.policy_verification_tool import verify_policy
from src.tools.shap_explanation_tool import explain_claim
from src.tools.similar_claims_tool import find_similar_claims

# Evidence source names, in investigation order.
SOURCES = ("fraud_score", "shap_explanation", "policy_retrieval", "policy_verification",
           "similar_claims", "claim_history")
# Agent-callable tool name -> evidence source.
AGENT_TOOL_TO_SOURCE = {"explain_fraud_score": "shap_explanation", "retrieve_policy": "policy_retrieval",
                        "verify_policy": "policy_verification", "find_similar_claims": "similar_claims",
                        "get_claim_history": "claim_history"}
SOURCE_TO_AGENT_TOOL = {v: k for k, v in AGENT_TOOL_TO_SOURCE.items()}
# Prerequisites enforced in code: source -> sources that must have run first.
DEPENDENCIES = {"policy_verification": ("policy_retrieval",), "shap_explanation": ("fraud_score",)}


class _ClaimInput(BaseModel):
    claim: dict[str, Any] = Field(..., description="Claim with the 33 raw fields")


class _ExplainInput(_ClaimInput):
    top_k: int = Field(5, ge=1, le=15)


class _SimilarInput(_ClaimInput):
    k: int = Field(5, ge=1, le=20)
    same_base_policy: bool = False


class _HistoryInput(_ClaimInput):
    years: Optional[int] = Field(5, ge=1, le=20)


TOOLS = [
    StructuredTool.from_function(score_claim, name="score_claim", args_schema=_ClaimInput,
                                 description="Tool 1: fraud probability, predicted class, risk band and model metadata."),
    StructuredTool.from_function(explain_claim, name="explain_claim", args_schema=_ExplainInput,
                                 description="Tool 2: SHAP explanation of the fraud score."),
    StructuredTool.from_function(retrieve_policy, name="retrieve_policy", args_schema=_ClaimInput,
                                 description="Tool 3: the applicable policy information and what is missing."),
    StructuredTool.from_function(verify_policy, name="verify_policy", args_schema=_ClaimInput,
                                 description="Tool 4: PASS / FAIL / UNKNOWN checks of the claim against the policy."),
    StructuredTool.from_function(find_similar_claims, name="find_similar_claims", args_schema=_SimilarInput,
                                 description="Tool 5: most similar genuine historical claims and their outcomes."),
    StructuredTool.from_function(get_claim_history, name="get_claim_history", args_schema=_HistoryInput,
                                 description="Tool 6: previous claims of the same policy or customer, if a source exists."),
]


class _Explain(BaseModel):
    top_k: int = Field(5, ge=1, le=10, description="How many features to explain")


class _Similar(BaseModel):
    k: int = Field(5, ge=1, le=10, description="How many similar claims to retrieve")
    same_base_policy: bool = Field(False, description="Only claims with the same base policy")


class _History(BaseModel):
    years: int = Field(5, ge=1, le=20, description="Look-back window in years before the accident")


class _NoArgs(BaseModel):
    pass


AGENT_TOOL_DESCRIPTIONS = {
    "explain_fraud_score": "Tool 2. SHAP explanation of this claim's fraud score: which features raised or "
                           "lowered it, with their values and signed contributions.",
    "retrieve_policy": "Tool 3. Retrieve the policy that applies to this claim: recorded policy attributes, the "
                       "policy document if one exists, and which policy information is missing.",
    "verify_policy": "Tool 4. Check the claim against the policy conditions. Each check is PASS, FAIL or "
                     "UNKNOWN with its reason. Runs policy retrieval first if needed.",
    "find_similar_claims": "Tool 5. The k most similar genuine historical claims (never synthetic) with their "
                           "recorded fraud outcomes, shared characteristics and differences.",
    "get_claim_history": "Tool 6. Previous claims of the same policy or customer from an authorised history "
                         "source; reports 'unavailable' when no linked history exists.",
}


def agent_tools(claim: Mapping[str, Any]) -> list[StructuredTool]:
    claim = dict(claim)
    d = AGENT_TOOL_DESCRIPTIONS
    return [
        StructuredTool.from_function(lambda top_k=5: explain_claim(claim, top_k), name="explain_fraud_score",
                                     args_schema=_Explain, description=d["explain_fraud_score"]),
        StructuredTool.from_function(lambda: retrieve_policy(claim), name="retrieve_policy",
                                     args_schema=_NoArgs, description=d["retrieve_policy"]),
        StructuredTool.from_function(lambda: verify_policy(claim), name="verify_policy",
                                     args_schema=_NoArgs, description=d["verify_policy"]),
        StructuredTool.from_function(lambda k=5, same_base_policy=False: find_similar_claims(claim, k, same_base_policy),
                                     name="find_similar_claims", args_schema=_Similar, description=d["find_similar_claims"]),
        StructuredTool.from_function(lambda years=5: get_claim_history(claim, years), name="get_claim_history",
                                     args_schema=_History, description=d["get_claim_history"]),
    ]


__all__ = ["TOOLS", "agent_tools", "score_claim", "explain_claim", "retrieve_policy", "verify_policy",
           "find_similar_claims", "get_claim_history", "SOURCES", "AGENT_TOOL_TO_SOURCE",
           "SOURCE_TO_AGENT_TOOL", "DEPENDENCIES"]
