"""LangGraph investigation graph (docs/agent_restructure_plan.md).

    START -> ingest -> validate -> score (Tool 1) -> agent <-> tools (Tools 2-6, <= max_steps turns)
                                                      |
                                                      v
          complete_investigation -> aggregate_evidence -> synthesize -> guardrails -> human_review -> END

Layer 1 (ingest, validate) and Tool 1 always run. The orchestrator LLM chooses among Tools 2-6;
complete_investigation runs any required tool it skipped (logged as `auto`). Without an LLM the
graph still runs every required tool and uses the rule-based decision.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import NodeDeps, make_nodes
from src.schemas.state import ClaimState


def build_graph(deps: NodeDeps | None = None, checkpointer=None):
    deps = deps or NodeDeps()
    n = make_nodes(deps)
    g = StateGraph(ClaimState)
    for name in ("ingest", "validate", "score", "agent", "tools", "complete_investigation",
                 "aggregate_evidence", "synthesize", "guardrails", "human_review"):
        g.add_node(name, n[name])
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "validate")
    g.add_edge("validate", "score")
    g.add_edge("score", "agent")
    g.add_conditional_edges("agent", n["route_after_agent"], {"tools": "tools", "synthesize": "complete_investigation"})
    g.add_edge("tools", "agent")
    g.add_edge("complete_investigation", "aggregate_evidence")
    g.add_edge("aggregate_evidence", "synthesize")
    g.add_edge("synthesize", "guardrails")
    g.add_edge("guardrails", "human_review")
    g.add_edge("human_review", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
