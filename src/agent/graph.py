"""LangGraph triage graph.

    START -> ingest -> validate -> score -> agent <-> tools   (ReAct loop, <= MAX_STEPS turns)
                                              |
                                              v
                    mark_max_steps -> synthesize -> guardrails -> human_review (interrupt) -> END

Validate and score always run; the agent decides which further tools (similar claims,
policy checks) to call. Without an LLM the graph still runs and uses the rule-based decision.
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
    for name in ("ingest", "validate", "score", "agent", "tools", "mark_max_steps", "synthesize",
                 "guardrails", "human_review"):
        g.add_node(name, n[name])
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "validate")
    g.add_edge("validate", "score")
    g.add_edge("score", "agent")
    g.add_conditional_edges("agent", n["route_after_agent"], {"tools": "tools", "synthesize": "mark_max_steps"})
    g.add_edge("tools", "agent")
    g.add_edge("mark_max_steps", "synthesize")
    g.add_edge("synthesize", "guardrails")
    g.add_edge("guardrails", "human_review")
    g.add_edge("human_review", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
