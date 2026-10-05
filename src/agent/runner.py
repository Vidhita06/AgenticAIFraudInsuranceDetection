"""Run claims through the triage graph and resume them after human review.

    from src.agent.runner import TriageRunner
    runner = TriageRunner()                       # LLM from .env, or rule-based if none
    thread_id, decision, trace = runner.run_claim(claim_dict)
    final_state = runner.resume(thread_id, {"decision": "APPROVE", "adjuster_id": "a1"})
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

from langgraph.types import Command

from src.agent.graph import build_graph
from src.agent.nodes import NodeDeps
from src.schemas.decision import AdjusterDecision, TriageDecision


class TriageRunner:
    def __init__(self, llm: Any = "auto", deps: Optional[NodeDeps] = None, checkpointer=None,
                 audit_log: str | Path | None = None, purpose: str = "default"):
        if deps is None:
            if llm == "auto":
                llm = _try_llm(purpose)
            deps = NodeDeps(llm=llm)
        self.deps = deps
        self.graph = build_graph(deps, checkpointer)
        self.audit_log = Path(audit_log) if audit_log else None

    def run_claim(self, claim: Mapping[str, Any], thread_id: str | None = None
                  ) -> tuple[str, TriageDecision, list[dict]]:
        thread_id = thread_id or f"claim-{uuid.uuid4().hex[:12]}"
        cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 4 * self.deps.max_steps + 20}
        self.graph.invoke({"raw_claim": dict(claim)}, cfg)
        state = self.graph.get_state(cfg).values
        decision = TriageDecision.model_validate(state["decision"])
        self._log({"thread_id": thread_id, "event": "recommendation", "decision": state["decision"],
                   "trace": state.get("trace", [])})
        return thread_id, decision, state.get("trace", [])

    def resume(self, thread_id: str, adjuster_decision: Mapping[str, Any]) -> dict[str, Any]:
        adj = AdjusterDecision.model_validate(dict(adjuster_decision)).model_dump()
        cfg = {"configurable": {"thread_id": thread_id}}
        self.graph.invoke(Command(resume=adj), cfg)
        state = self.graph.get_state(cfg).values
        self._log({"thread_id": thread_id, "event": "adjuster_decision", "adjuster": adj,
                   "recommendation": state["decision"]["decision"]})
        return state

    def _log(self, record: dict) -> None:
        if self.audit_log:
            self.audit_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")


def _try_llm(purpose: str):
    try:
        from src.agent.llm import get_llm, llm_settings
        if not llm_settings(purpose)["api_key"] and llm_settings(purpose)["provider"] != "ollama":
            return None
        return get_llm(purpose=purpose)
    except Exception:  # noqa: BLE001
        return None


_default: Optional[TriageRunner] = None


def run_claim(claim: Mapping[str, Any], thread_id: str | None = None):
    global _default
    _default = _default or TriageRunner()
    return _default.run_claim(claim, thread_id)


def resume(thread_id: str, adjuster_decision: Mapping[str, Any]):
    if _default is None:
        raise RuntimeError("no run in progress")
    return _default.resume(thread_id, adjuster_decision)
