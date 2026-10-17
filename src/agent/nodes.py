"""Graph nodes (docs/agent_restructure_plan.md):

    ingest -> validate -> score (Tool 1) -> agent <-> tools (Tools 2-6) -> complete_investigation
    -> aggregate_evidence -> synthesize (reasoning) -> guardrails (recommendation) -> human_review

Each node returns a partial ClaimState update and appends an audit-trace event."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt
from pydantic import ValidationError

from src.agent.evidence import aggregate_evidence, compact_for_prompt
from src.agent.guardrails import apply_guardrails
from src.agent.llm import message_text
from src.config import PROJECT_ROOT, load_config
from src.ingestion.normalizer import normalize_claim
from src.schemas.decision import DecisionDraft
from src.tools import (AGENT_TOOL_TO_SOURCE, DEPENDENCIES, SOURCE_TO_AGENT_TOOL, SOURCES, agent_tools, explain_claim,
                       find_similar_claims, get_claim_history, retrieve_policy, score_claim, verify_policy)
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
    scorer: Callable[[dict], dict] = score_claim                 # Tool 1
    explainer: Callable[..., dict] = explain_claim               # Tool 2
    policy_retriever: Callable[[dict], dict] = retrieve_policy   # Tool 3
    policy_verifier: Callable[..., dict] = verify_policy         # Tool 4
    similar: Callable[..., dict] = find_similar_claims           # Tool 5
    history: Callable[..., dict] = get_claim_history             # Tool 6
    max_steps: int = field(default_factory=lambda: int(load_config()["agent"]["max_steps"]))
    orchestrator_prompt: str = field(default_factory=lambda: (PROMPTS / "orchestrator_system.md").read_text())
    synthesis_prompt: str = field(default_factory=lambda: (PROMPTS / "synthesis.md").read_text())
    synth_llm: Any = None                            # optional JSON-mode model for synthesis
    # Run by complete_investigation if the agent did not call them (cheap, local, deterministic).
    required_tools: tuple[str, ...] = field(default_factory=lambda: tuple(
        load_config()["agent"].get("required_tools", ["fraud_score", "shap_explanation", "policy_retrieval",
                                                      "policy_verification", "similar_claims"])))
    # SIMULATED reference data for Tools 3, 4 and 6: None = config (simulated_data.enabled).
    simulated: Optional[bool] = None

    @property
    def simulated_enabled(self) -> bool:
        from src.simulated.db import enabled
        return enabled(self.simulated)

    def run_source(self, source: str, claim: dict, args: dict, results: dict) -> dict:
        """Run one evidence source with its arguments and the results it depends on."""
        if source == "fraud_score":
            return self.scorer(claim)
        if source == "shap_explanation":
            return self.explainer(claim, int(args.get("top_k", 5)), fraud_score=results.get("fraud_score"))
        sim = {"simulated": self.simulated_enabled}
        if source == "policy_retrieval":
            return self.policy_retriever(claim, **sim)
        if source == "policy_verification":
            return self.policy_verifier(claim, policy=results.get("policy_retrieval"), **sim)
        if source == "similar_claims":
            return self.similar(claim, int(args.get("k", 5)), bool(args.get("same_base_policy", False)))
        if source == "claim_history":
            return self.history(claim, int(args.get("years", 5)), **sim)
        return {"ok": False, "error": f"unknown evidence source {source}"}


AGENT_TOOL_NAMES = tuple(AGENT_TOOL_TO_SOURCE)
# Names small models use for the tools (old names included) -> agent tool name.
_TOOL_NAME_ALIASES = {"check_policy_rules": "verify_policy", "policy_check": "verify_policy",
                      "explain_claim": "explain_fraud_score", "shap": "explain_fraud_score",
                      "claim_history": "get_claim_history", "similar_claims": "find_similar_claims",
                      "policy_retrieval": "retrieve_policy", "policy_verification": "verify_policy"}
_SOURCE_ALIASES = {
    **{k: "fraud_score" for k in ("fraud_score", "score", "score_claim", "fraud_score_tool", "model", "tool_1")},
    **{k: "shap_explanation" for k in ("shap_explanation", "shap", "explain_fraud_score", "explain_claim",
                                       "explanation", "tool_2")},
    **{k: "policy_retrieval" for k in ("policy_retrieval", "retrieve_policy", "policy", "tool_3")},
    **{k: "policy_verification" for k in ("policy_verification", "verify_policy", "policy_check",
                                          "check_policy_rules", "tool_4")},
    **{k: "similar_claims" for k in ("similar_claims", "find_similar_claims", "similar_claims_tool", "similar",
                                     "tool_5")},
    **{k: "claim_history" for k in ("claim_history", "get_claim_history", "history", "tool_6")},
    **{k: "validation" for k in ("validation", "validate", "validator", "input_validation")},
}


def _json_objects(text: str) -> list[Any]:
    """Every top-level JSON object in `text` (code fences and trailing commas tolerated)."""
    import re
    text = re.sub(r"```(?:json)?", "", text)
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            esc = (not esc) and ch == "\\"
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                chunk = text[start:i + 1]
                for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
                    try:
                        out.append(json.loads(candidate))
                        break
                    except json.JSONDecodeError:
                        continue
    return out


def parse_decision_json(text: str) -> DecisionDraft:
    """Lenient parser for the synthesis answer: accepts fenced JSON, key aliases
    (reasoning/explanation -> rationale, source -> source_tool) and drops individual invalid
    evidence items instead of rejecting the whole answer."""
    objs = [o for o in _json_objects(text) if isinstance(o, dict) and "decision" in {k.lower() for k in o}]
    if not objs:
        raise ValueError("no JSON object with a 'decision' field in the response")
    o = {k.lower(): v for k, v in objs[0].items()}
    rationale = o.get("rationale") or o.get("reasoning") or o.get("explanation") or ""
    evidence = []
    for ev in o.get("evidence") or []:
        if isinstance(ev, dict):
            e = {k.lower(): v for k, v in ev.items()}
            src = _SOURCE_ALIASES.get(str(e.get("source_tool") or e.get("source") or e.get("tool") or "").lower())
            fact = e.get("fact") or e.get("text") or e.get("detail")
            if src and isinstance(fact, str) and 3 <= len(fact) <= 400:
                evidence.append({"source_tool": src, "fact": fact})
    questions = o.get("unresolved_questions") or o.get("open_questions") or []
    questions = [str(q) for q in questions] if isinstance(questions, list) else [str(questions)]
    return DecisionDraft.model_validate({"decision": str(o["decision"]), "rationale": str(rationale),
                                         "evidence": evidence, "unresolved_questions": questions})


def text_tool_calls(text: str) -> list[dict[str, Any]]:
    """Tool calls that a model wrote as JSON text instead of native tool calls
    (common with small local models), e.g. {"name": "verify_policy", "arguments": {}}."""
    calls = []
    for o in _json_objects(text):
        if not isinstance(o, dict):
            continue
        fn = o.get("function") if isinstance(o.get("function"), dict) else o
        name = fn.get("name") or o.get("tool")
        name = _TOOL_NAME_ALIASES.get(name, name)
        if name in AGENT_TOOL_NAMES:
            args = fn.get("arguments", fn.get("parameters", fn.get("args", {}))) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": name, "args": args if isinstance(args, dict) else {},
                          "id": f"text_call_{len(calls)}", "type": "tool_call"})
    return calls


def _write_runtime_tables(state, decision) -> str:
    """Record the real verification checks and the recommendation in the simulated database's
    run-time tables (verification_results, investigation_audit). Never fails the run."""
    from src.simulated.db import get_db, settings
    if not settings().get("write_runtime_tables", True):
        return "disabled"
    try:
        db = get_db()
        pv = (state.get("tool_results") or {}).get("policy_verification") or {}
        inv = state.get("investigation_id") or ""
        if pv.get("ok"):
            db.write_verification(inv, state["claim_id"], pv.get("checks", []))
        db.write_audit({"investigation_id": inv, "claim_id": state["claim_id"], "decision": decision.decision,
                        "flag_category": decision.flag_category, "decided_by": decision.decided_by,
                        "reason_codes": decision.reason_codes,
                        "tool_statuses": {k: v.status for k, v in decision.tool_statuses.items()},
                        "simulated_data": True, "model_version": decision.model_version})
        return "written"
    except Exception as e:  # noqa: BLE001
        return f"failed: {type(e).__name__}: {str(e)[:200]}"


def make_nodes(deps: NodeDeps) -> dict[str, Callable]:
    def ingest(state):
        n = normalize_claim(state["raw_claim"])
        inv = state.get("investigation_id") or f"INV-{uuid.uuid4().hex[:12]}"
        return {"claim": n.record, "claim_id": n.claim_id, "investigation_id": inv,
                "normalization_issues": n.issues, "steps": 0, "llm_failures": 0, "tool_results": {},
                "tool_origin": {},
                "trace": _event("ingest", claim_id=n.claim_id, investigation_id=inv, schema_ok=n.ok, issues=n.issues)}

    def validate(state):
        results = validate_claim(state["claim"])
        failed = [r for r in results if not r["passed"]]
        out = {"ok": True, "claim_id": state["claim_id"], "blocking": has_blocking_failure(results),
               "failed_rules": failed, "schema_issues": state.get("normalization_issues", [])}
        return {"tool_results": {"validation": out}, "tool_origin": {"validation": "orchestrator"},
                "trace": _event("validate", blocking=out["blocking"],
                                failed=[r["rule_id"] for r in failed])}

    def _schema_bad(state) -> bool:
        v = state["tool_results"]["validation"]
        return any(r["rule_id"] in ("V01_REQUIRED_FIELDS", "V02_DOMAIN_VALUES") for r in v["failed_rules"])

    def score(state):
        out = ({"ok": False, "error": "claim fields missing or invalid; not scored"} if _schema_bad(state)
               else deps.scorer(state["claim"]))
        return {"tool_results": {"fraud_score": out}, "tool_origin": {"fraud_score": "orchestrator"},
                "trace": _event("score", ok=out.get("ok"), p=out.get("fraud_probability"),
                                band=out.get("risk_band"))}

    def _context_message(state) -> str:
        claim = {k: state["claim"].get(k) for k in KEY_FIELDS}
        tr = state["tool_results"]
        return ("Claim " + state["claim_id"] + "\n\nKey fields:\n" + json.dumps(claim, default=str)
                + "\n\nValidation:\n" + json.dumps(compact_for_prompt("validation", tr.get("validation")), default=str)
                + "\n\nFraud score (Tool 1):\n"
                + json.dumps(compact_for_prompt("fraud_score", tr.get("fraud_score")), default=str))

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
        recovered = False
        if not getattr(reply, "tool_calls", None):
            parsed = text_tool_calls(message_text(reply))
            if parsed:
                reply = AIMessage(content=message_text(reply), tool_calls=parsed)
                recovered = True
        calls = [{"name": c["name"], "args": c.get("args", {})} for c in (getattr(reply, "tool_calls", None) or [])]
        return {"messages": new + [reply], "steps": steps,
                "trace": _event("agent", step=steps, tool_calls=calls, text=message_text(reply)[:500],
                                **({"tool_calls_from_text": True} if recovered else {}))}

    def route_after_agent(state) -> str:
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        if isinstance(last, AIMessage) and last.tool_calls and not state.get("llm_error"):
            if state.get("steps", 0) < deps.max_steps:
                return "tools"
        return "synthesize"

    def _run_with_dependencies(source, args, state, results, origins, origin, calls):
        """Run `source`, first running any prerequisite that has not run yet (origin
        'dependency'). Results are written into `results`."""
        have = {**(state.get("tool_results") or {}), **results}
        for dep in DEPENDENCIES.get(source, ()):
            if dep not in have and dep != "fraud_score":       # Tool 1 always runs before the agent
                results[dep] = deps.run_source(dep, state["claim"], {}, have)
                origins[dep] = "dependency"
                calls.append({"name": SOURCE_TO_AGENT_TOOL.get(dep, dep), "args": {}, "ok": results[dep].get("ok"),
                              "origin": "dependency"})
                have[dep] = results[dep]
        results[source] = deps.run_source(source, state["claim"], args, have)
        origins[source] = origin
        return results[source]

    def tools(state):
        last = state["messages"][-1]
        out_msgs, results, calls, origins = [], {}, [], {}
        for call in last.tool_calls:
            source = AGENT_TOOL_TO_SOURCE.get(call["name"])
            args = call.get("args") or {}
            if source is None:
                res = {"ok": False, "error": f"unknown tool {call['name']}"}
            else:
                res = _run_with_dependencies(source, args, state, results, origins, "agent", calls)
            out_msgs.append(ToolMessage(content=json.dumps(compact_for_prompt(source, res) if source else res,
                                                           default=str)[:12000],
                                        tool_call_id=call["id"], name=call["name"]))
            calls.append({"name": call["name"], "args": args, "ok": res.get("ok"), "origin": "agent"})
        return {"messages": out_msgs, "tool_results": results, "tool_origin": dict(origins),
                "trace": _event("tools", calls=calls)}

    def complete_investigation(state):
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        hit = isinstance(last, AIMessage) and bool(last.tool_calls) and state.get("steps", 0) >= deps.max_steps
        update: dict[str, Any] = {"max_steps_reached": hit}
        # Required tools the orchestrator LLM skipped (or every required tool when no LLM is
        # configured) run here, logged as automatic rather than as agent choices. Nothing runs
        # on a claim whose fields are missing or invalid.
        results, calls, origins = {}, [], {}
        # Without an LLM nobody chooses tools, so the orchestrator runs all six.
        plan = deps.required_tools if deps.llm is not None else SOURCES
        if not _schema_bad(state):
            for source in plan:
                if source not in (state.get("tool_results") or {}) and source not in results:
                    _run_with_dependencies(source, {}, state, results, origins, "auto", calls)
        if results:
            update["tool_results"] = results
            update["tool_origin"] = dict(origins)
        update["trace"] = _event("complete_investigation", max_steps_reached=hit,
                                 auto_tools=[SOURCE_TO_AGENT_TOOL.get(s, s) for s, o in origins.items() if o == "auto"],
                                 dependency_tools=[c["name"] for c in calls])
        return update

    def aggregate(state):
        record = aggregate_evidence(state.get("investigation_id", ""), state["claim_id"], state["tool_results"],
                                    state.get("tool_origin") or {}, deps.required_tools,
                                    bool(state.get("max_steps_reached")))
        return {"evidence_record": record,
                "trace": _event("aggregate_evidence", investigation_status=record["investigation_status"],
                                statuses={k: v["status"] for k, v in record["tool_statuses"].items()},
                                conflicts=len(record["conflicts"]), missing=len(record["missing_information"]))}

    def synthesize(state):
        if deps.llm is None:
            return {"draft": None, "trace": _event("synthesize", skipped="no LLM configured")}
        failures = state.get("llm_failures", 0)
        tr = state["tool_results"]
        rec = state.get("evidence_record") or {}
        summary = {k: rec.get(k) for k in ("investigation_status", "missing_information", "conflicts", "warnings")}
        summary["tool_statuses"] = {k: v["status"] for k, v in (rec.get("tool_statuses") or {}).items()}
        prompt = (deps.synthesis_prompt + "\n\n# Evidence record (aggregated)\n" + json.dumps(summary, default=str)
                  + "\n\n# Tool outputs\n"
                  + "\n".join(f"## {k}\n{json.dumps(compact_for_prompt(k, v), default=str)}" for k, v in tr.items())
                  + "\n\nClaim id: " + state["claim_id"])
        errors = []
        while failures < 2:
            try:
                model = deps.synth_llm or deps.llm
                reply = model.invoke([HumanMessage(prompt + (f"\n\nYour previous answer was invalid: {errors[-1]}. "
                                                              "Return only the JSON object." if errors else ""))])
                draft = parse_decision_json(message_text(reply))
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
                                                   fallback_reason=reason, record=state.get("evidence_record"),
                                                   required=deps.required_tools,
                                                   simulated_data=deps.simulated_enabled)
        if state.get("max_steps_reached"):
            decision.guardrail_notes.append(f"max_steps: agent stopped after {deps.max_steps} steps")
            interventions.append({"guardrail": "max_steps", "detail": f"{deps.max_steps} steps"})
        written = _write_runtime_tables(state, decision) if deps.simulated_enabled else None
        return {"decision": decision.model_dump(),
                "trace": _event("guardrails", decision=decision.decision, flag_category=decision.flag_category,
                                decided_by=decision.decided_by, interventions=interventions,
                                **({"runtime_tables": written} if written else {}))}

    def human_review(state):
        answer = interrupt({"claim_id": state["claim_id"], "recommendation": state["decision"],
                            "instructions": "Approve, modify or override. The adjuster has final authority."})
        return {"adjuster": answer, "trace": _event("human_review", adjuster=answer)}

    return {"ingest": ingest, "validate": validate, "score": score, "agent": agent, "tools": tools,
            "route_after_agent": route_after_agent, "complete_investigation": complete_investigation,
            "aggregate_evidence": aggregate,
            "synthesize": synthesize, "guardrails": guardrails, "human_review": human_review}
