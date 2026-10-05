"""Graph nodes: ingest -> validate -> score -> agent <-> tools -> synthesize -> guardrails ->
human_review. Each node returns a partial ClaimState update and appends an audit-trace event."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt
from pydantic import ValidationError

from src.agent.guardrails import apply_guardrails
from src.agent.llm import message_text
from src.config import PROJECT_ROOT, load_config
from src.ingestion.normalizer import normalize_claim
from src.schemas.decision import DecisionDraft
from src.tools import AGENT_TOOL_TO_SOURCE, agent_tools, check_policy_rules, find_similar_claims, score_claim
from src.tools._common import jsonable
from src.validation.validators import has_blocking_failure, validate_claim

PROMPTS = PROJECT_ROOT / "config" / "prompts"
KEY_FIELDS = ["Fault", "BasePolicy", "VehicleCategory", "VehiclePrice", "Make", "Age", "Sex",
              "Deductible", "AddressChange_Claim", "PastNumberOfClaims", "Days_Policy_Accident",
              "Days_Policy_Claim", "PoliceReportFiled", "WitnessPresent", "AgentType",
              "Month", "MonthClaimed", "Year"]


def _event(node: str, **data) -> list[dict]:
    return [{"node": node, "ts": round(time.time(), 3), **jsonable(data)}]


@dataclass
class NodeDeps:
    """Injected dependencies (real tools by default; tests swap in stubs)."""
    llm: Any = None                                  # LangChain chat model or None
    scorer: Callable[[dict], dict] = score_claim
    similar: Callable[..., dict] = find_similar_claims
    policy: Callable[[dict], dict] = check_policy_rules
    max_steps: int = field(default_factory=lambda: int(load_config()["agent"]["max_steps"]))
    orchestrator_prompt: str = field(default_factory=lambda: (PROMPTS / "orchestrator_system.md").read_text())
    synthesis_prompt: str = field(default_factory=lambda: (PROMPTS / "synthesis.md").read_text())


def make_nodes(deps: NodeDeps) -> dict[str, Callable]:
    def ingest(state):
        n = normalize_claim(state["raw_claim"])
        return {"claim": n.record, "claim_id": n.claim_id, "normalization_issues": n.issues,
                "steps": 0, "llm_failures": 0, "tool_results": {},
                "trace": _event("ingest", claim_id=n.claim_id, schema_ok=n.ok, issues=n.issues)}

    def validate(state):
        results = validate_claim(state["claim"])
        failed = [r for r in results if not r["passed"]]
        out = {"ok": True, "claim_id": state["claim_id"], "blocking": has_blocking_failure(results),
               "failed_rules": failed, "schema_issues": state.get("normalization_issues", [])}
        return {"tool_results": {"validation": out},
                "trace": _event("validate", blocking=out["blocking"],
                                failed=[r["rule_id"] for r in failed])}

    def score(state):
        v = state["tool_results"]["validation"]
        schema_bad = any(r["rule_id"] in ("V01_REQUIRED_FIELDS", "V02_DOMAIN_VALUES") for r in v["failed_rules"])
        out = ({"ok": False, "error": "claim fields missing or invalid; not scored"} if schema_bad
               else deps.scorer(state["claim"]))
        return {"tool_results": {"fraud_score": out},
                "trace": _event("score", ok=out.get("ok"), p=out.get("fraud_probability"),
                                band=out.get("risk_band"))}

    def _context_message(state) -> str:
        claim = {k: state["claim"].get(k) for k in KEY_FIELDS}
        tr = state["tool_results"]
        return ("Claim " + state["claim_id"] + "\n\nKey fields:\n" + json.dumps(claim, default=str)
                + "\n\nValidation:\n" + json.dumps(tr.get("validation"), default=str)
                + "\n\nFraud score:\n" + json.dumps(tr.get("fraud_score"), default=str))

    def agent(state):
        if deps.llm is None:
            return {"trace": _event("agent", skipped="no LLM configured")}
        msgs = list(state.get("messages") or [])
        new: list = []
        if not msgs:
            new = [SystemMessage(deps.orchestrator_prompt), HumanMessage(_context_message(state))]
        try:
            model = deps.llm.bind_tools(agent_tools(state["claim"]))
            reply = model.invoke(msgs + new)
        except Exception as e:  # noqa: BLE001
            return {"messages": new, "llm_failures": state.get("llm_failures", 0) + 1,
                    "llm_error": f"{type(e).__name__}: {e}",
                    "trace": _event("agent", error=f"{type(e).__name__}: {str(e)[:300]}")}
        steps = state.get("steps", 0) + 1
        calls = [{"name": c["name"], "args": c.get("args", {})} for c in (getattr(reply, "tool_calls", None) or [])]
        return {"messages": new + [reply], "steps": steps,
                "trace": _event("agent", step=steps, tool_calls=calls, text=message_text(reply)[:500])}

    def route_after_agent(state) -> str:
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        if isinstance(last, AIMessage) and last.tool_calls and not state.get("llm_error"):
            if state.get("steps", 0) < deps.max_steps:
                return "tools"
        return "synthesize"

    def tools(state):
        last = state["messages"][-1]
        bound = {"find_similar_claims": lambda **a: deps.similar(state["claim"], int(a.get("k", 5))),
                 "check_policy_rules": lambda **a: deps.policy(state["claim"])}
        out_msgs, results, calls = [], {}, []
        for call in last.tool_calls:
            fn = bound.get(call["name"])
            res = fn(**(call.get("args") or {})) if fn else {"ok": False, "error": f"unknown tool {call['name']}"}
            if fn:
                results[AGENT_TOOL_TO_SOURCE[call["name"]]] = res
            out_msgs.append(ToolMessage(content=json.dumps(res, default=str)[:12000], tool_call_id=call["id"],
                                        name=call["name"]))
            calls.append({"name": call["name"], "args": call.get("args"), "ok": res.get("ok")})
        return {"messages": out_msgs, "tool_results": results, "trace": _event("tools", calls=calls)}

    def mark_max_steps(state):
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        hit = isinstance(last, AIMessage) and bool(last.tool_calls) and state.get("steps", 0) >= deps.max_steps
        return {"max_steps_reached": hit, "trace": _event("max_steps_check", reached=hit)}

    def synthesize(state):
        if deps.llm is None:
            return {"draft": None, "trace": _event("synthesize", skipped="no LLM configured")}
        failures = state.get("llm_failures", 0)
        tr = state["tool_results"]
        prompt = (deps.synthesis_prompt + "\n\n# Tool outputs\n"
                  + "\n".join(f"## {k}\n{json.dumps(v, default=str)}" for k, v in tr.items())
                  + "\n\nClaim id: " + state["claim_id"])
        errors = []
        while failures < 2:
            try:
                reply = deps.llm.invoke([HumanMessage(prompt + (f"\n\nYour previous answer was invalid: {errors[-1]}. "
                                                                 "Return only the JSON object." if errors else ""))])
                text = message_text(reply)
                start, end = text.find("{"), text.rfind("}")
                if start < 0 or end <= start:
                    raise ValueError("no JSON object in the response")
                draft = DecisionDraft.model_validate(json.loads(text[start:end + 1]))
                return {"draft": draft.model_dump(), "llm_failures": failures,
                        "trace": _event("synthesize", attempt=len(errors) + 1, decision=draft.decision)}
            except (ValidationError, ValueError, json.JSONDecodeError) as e:
                errors.append(f"{type(e).__name__}: {str(e)[:200]}")
            except Exception as e:  # noqa: BLE001  (API/network errors)
                errors.append(f"{type(e).__name__}: {str(e)[:200]}")
            failures += 1
        return {"draft": None, "llm_failures": failures, "llm_error": "; ".join(errors),
                "trace": _event("synthesize", failed=errors)}

    def guardrails(state):
        draft = DecisionDraft.model_validate(state["draft"]) if state.get("draft") else None
        reason = ("no LLM configured" if deps.llm is None else
                  f"LLM failed or returned invalid output twice ({state.get('llm_error', '')[:200]})")
        decision, interventions = apply_guardrails(draft, state["claim_id"], state["tool_results"],
                                                   fallback_reason=reason)
        if state.get("max_steps_reached"):
            decision.guardrail_notes.append(f"max_steps: agent stopped after {deps.max_steps} steps")
            interventions.append({"guardrail": "max_steps", "detail": f"{deps.max_steps} steps"})
        return {"decision": decision.model_dump(),
                "trace": _event("guardrails", decision=decision.decision, decided_by=decision.decided_by,
                                interventions=interventions)}

    def human_review(state):
        answer = interrupt({"claim_id": state["claim_id"], "recommendation": state["decision"],
                            "instructions": "Approve, modify or override. The adjuster has final authority."})
        return {"adjuster": answer, "trace": _event("human_review", adjuster=answer)}

    return {"ingest": ingest, "validate": validate, "score": score, "agent": agent, "tools": tools,
            "route_after_agent": route_after_agent, "mark_max_steps": mark_max_steps,
            "synthesize": synthesize, "guardrails": guardrails, "human_review": human_review}
