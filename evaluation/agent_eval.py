"""Agent evaluation (Step D): sampling, cost estimation, batch runs, metrics, LLM-as-judge
and the report.

Data rules: prompts are developed on validation claims; the final evaluation runs once on
test claims (a marker file in evaluation/results/ prevents silent re-runs). The sample is
all frauds in the split plus a matched number of legitimate claims, so precision is also
reported re-weighted to the true prevalence.

Every LLM spend goes through `estimate_cost` + `confirm` first (see scripts/run_batch_triage.py).
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from langchain_core.callbacks import BaseCallbackHandler

from src.config import PROJECT_ROOT, load_config, path_for
from src.ml.data import TARGET, load_real_test, load_real_validation
from src.ml.evaluate import wilson_ci

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
TEST_MARKER = RESULTS_DIR / ".test_evaluation_done"
CHARS_PER_TOKEN = 3.5
PAUSE_BETWEEN_CLAIMS_S = 0.0     # set by --pause (free-tier API rate limits)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def load_split(split: str) -> pd.DataFrame:
    if split == "validation":
        return load_real_validation()
    if split == "test":
        return load_real_test()
    raise ValueError(split)


def stratified_sample(df: pd.DataFrame, legit_per_fraud: float = 1.0, n: int | None = None,
                      seed: int = 42, n_frauds: int | None = None) -> pd.DataFrame:
    """All frauds (or `n_frauds` of them) + legit_per_fraud x as many legit claims; with `n`,
    a balanced sample of n claims (half fraud, half legit) instead."""
    fraud, legit = df[df[TARGET] == 1], df[df[TARGET] == 0]
    n_fraud_total = len(fraud)
    if n_frauds is not None:
        fraud = fraud.sample(min(n_frauds, len(fraud)), random_state=seed)
        n = None
    if n is not None:
        nf = min(len(fraud), max(1, n // 2))
        f = fraud.sample(nf, random_state=seed)
        l = legit.sample(min(len(legit), n - nf), random_state=seed)
    else:
        f = fraud
        l = legit.sample(min(len(legit), int(round(legit_per_fraud * len(fraud)))), random_state=seed)
    out = pd.concat([f, l]).sample(frac=1, random_state=seed).reset_index(drop=True)
    # Re-weighting to the true prevalence: legit rows count (legit sampling fraction)^-1
    # relative to fraud rows' (fraud sampling fraction)^-1.
    out.attrs["fraud_weight"] = n_fraud_total / max(len(f), 1)
    out.attrs["legit_weight"] = (len(legit) / max(len(l), 1)) / out.attrs["fraud_weight"]
    return out


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def pricing(model: str, provider: str = "anthropic") -> dict[str, Any]:
    """USD per million tokens. Local Ollama models are free. For OpenAI-compatible gateways
    (Groq, OpenRouter, Gemini) a model missing from llm_pricing_usd_per_mtok is assumed to be
    on a free tier; add it to settings.yaml if it is billed."""
    if provider == "ollama":
        return {"input": 0.0, "output": 0.0, "free": True, "note": "local Ollama model"}
    table = load_config()["llm_pricing_usd_per_mtok"]
    for k, v in table.items():
        if model == k or model.startswith(k):
            return {**v, "free": v["input"] == 0 and v["output"] == 0, "note": "list price"}
    if provider == "openai":
        return {"input": 0.0, "output": 0.0, "free": True,
                "note": "price unknown, assumed free tier (add it to llm_pricing_usd_per_mtok if billed)"}
    raise KeyError(f"no price for model {model!r}; add it to llm_pricing_usd_per_mtok in settings.yaml")


MODES = {"off": False, "on": True}      # simulated reference data


def test_marker(model: str, mode: str = "off") -> Path:
    """The test set is evaluated once per agent model and simulated-data mode."""
    slug = "".join(c if c.isalnum() or c in "-." else "_" for c in model)
    return RESULTS_DIR / (f".test_evaluation_done_{slug}" + ("" if mode == "off" else f"_sim{mode}"))


@dataclass
class CostEstimate:
    n_claims: int
    n_runs: int
    n_judge: int
    input_tokens: float
    output_tokens: float
    usd: float
    per_claim_usd: float
    basis: str
    free: bool = False
    price_note: str = ""
    seconds_per_run: Optional[float] = None

    def describe(self, model: str) -> str:
        cost = (f"$0 ({self.price_note})" if self.free else
                f"~${self.usd:.2f} (~${self.per_claim_usd:.4f} per run)")
        t = ""
        if self.seconds_per_run:
            hours = (self.n_runs + self.n_judge * 0.5) * self.seconds_per_run / 3600
            t = f"; ~{hours:.1f} h at the measured {self.seconds_per_run:.0f} s per run"
        return (f"{self.n_runs} agent runs ({self.n_claims} claims) + {self.n_judge} judge calls on {model}: "
                f"~{self.input_tokens / 1e6:.2f}M input + ~{self.output_tokens / 1e6:.2f}M output tokens "
                f"= {cost}{t} (basis: {self.basis})")


def _static_tokens_per_claim(claim: Mapping[str, Any]) -> tuple[float, float]:
    """Token estimate for one agent run from the actual prompt sizes: the tools are local,
    so their outputs are computed (free) and measured."""
    from src.agent.evidence import compact_for_prompt
    from src.agent.nodes import KEY_FIELDS, PROMPTS
    from src.tools import (explain_claim, find_similar_claims, get_claim_history, retrieve_policy, score_claim,
                           verify_policy)
    cfg = load_config()["evaluation"]
    orch = len((PROMPTS / "orchestrator_system.md").read_text())
    synth = len((PROMPTS / "synthesis.md").read_text())
    ctx = len(json.dumps({k: claim.get(k) for k in KEY_FIELDS}, default=str))
    size = lambda src, out: len(json.dumps(compact_for_prompt(src, out), default=str))
    score = size("fraud_score", score_claim(claim))
    # agent-callable tool outputs, in a typical call order
    later = [size("policy_verification", verify_policy(claim)), size("similar_claims", find_similar_claims(claim, 5)),
             size("shap_explanation", explain_claim(claim)), size("policy_retrieval", retrieve_policy(claim)),
             size("claim_history", get_claim_history(claim))]
    tool_schemas = 2400
    calls = cfg["est_agent_calls_per_claim"]
    out_agent = cfg["est_output_tokens_agent_call"]
    # agent turn i sees the system prompt, context, tool schemas, all earlier outputs and tool results
    base = orch + ctx + score + 800 + tool_schemas
    agent_in = sum(base + sum(later[:i]) + i * out_agent * CHARS_PER_TOKEN for i in range(calls)) / CHARS_PER_TOKEN
    synth_in = (synth + score + sum(later) + 1500 + 800) / CHARS_PER_TOKEN
    return agent_in + synth_in, calls * out_agent + cfg["est_output_tokens_synthesis"]


def estimate_cost(sample: pd.DataFrame, model: str, n_runs: int, n_judge: int = 0,
                  measured: Optional[Mapping[str, float]] = None,
                  probe: Optional[pd.DataFrame] = None, provider: str = "anthropic") -> CostEstimate:
    """Measured per-run tokens (from a pilot) when available, otherwise a static estimate
    from the actual prompt and tool-output sizes, with a safety margin."""
    price = pricing(model, provider)
    if measured:
        tin, tout, basis = measured["input_tokens"], measured["output_tokens"], "measured in pilot run"
    else:
        # `probe`: claims whose prompt sizes are measured (validation claims for the test
        # estimate, so estimating never scores test claims)
        probe = (probe if probe is not None else sample).head(5)
        per = [_static_tokens_per_claim(r) for r in probe.to_dict("records")]
        margin = load_config()["evaluation"]["est_safety_margin"]
        tin, tout = margin * np.mean([p[0] for p in per]), margin * np.mean([p[1] for p in per])
        basis = f"static estimate from prompt sizes x{margin} margin"
    judge_in, judge_out = 2500, 400
    total_in = n_runs * tin + n_judge * judge_in
    total_out = n_runs * tout + n_judge * judge_out
    usd = (total_in * price["input"] + total_out * price["output"]) / 1e6
    per = (tin * price["input"] + tout * price["output"]) / 1e6
    return CostEstimate(len(sample), n_runs, n_judge, total_in, total_out, usd, per, basis,
                        free=price["free"], price_note=price["note"],
                        seconds_per_run=(measured or {}).get("latency_s"))


def confirm(message: str, assume_yes: bool = False, input_fn: Callable[[str], str] = input,
            free: bool = False) -> bool:
    print(message)
    if free:
        print("No spend (free model): running without confirmation.")
        return True
    if assume_yes:
        print("--yes given: proceeding.")
        return True
    try:
        return input_fn("Proceed and spend this? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        print("No interactive input available: not spending. Re-run with --yes to approve.")
        return False


# ---------------------------------------------------------------------------
# Batch runs
# ---------------------------------------------------------------------------

class UsageTracker(BaseCallbackHandler):
    """Collects token usage from every chat-model call inside a graph run."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def on_llm_end(self, response, **kwargs):
        self.calls += 1
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.input_tokens += int(usage.get("input_tokens", 0) or 0)
                self.output_tokens += int(usage.get("output_tokens", 0) or 0)


def model_only(tool_results: Mapping[str, Any]) -> tuple[str, Optional[str]]:
    """The threshold policy without the agent: blocking -> REQUEST_MORE_INFORMATION,
    high band -> FLAG (FRAUD_RISK), no score -> FLAG (SYSTEM_FAILURE), otherwise APPROVE
    (the rule-based fallback restricted to validation and the fraud score)."""
    from src.agent.guardrails import rule_based_decision
    d = rule_based_decision("", {"validation": tool_results.get("validation"),
                                 "fraud_score": tool_results.get("fraud_score")}, "model-only")
    return d.decision, d.flag_category


def model_only_decision(tool_results: Mapping[str, Any]) -> str:
    return model_only(tool_results)[0]


def run_batch(claims: pd.DataFrame, runner, split: str, out_path: Path, run_index: int = 0,
              model_name: str = "none", progress: Callable[[str], None] = print,
              provider: str = "anthropic") -> list[dict]:
    """Triage every claim (stops at human review) and append one JSON record per claim."""
    tracker = UsageTracker()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    price = None
    try:
        price = pricing(model_name, provider)
    except KeyError:
        pass
    records = []
    for i, row in enumerate(claims.to_dict("records")):
        label = int(row.pop(TARGET))
        tracker.reset()
        thread = f"{split}-{row.get('PolicyNumber')}-r{run_index}-{uuid.uuid4().hex[:6]}"
        cfg = {"configurable": {"thread_id": thread}, "callbacks": [tracker],
               "recursion_limit": 4 * runner.deps.max_steps + 20}
        t0 = time.time()
        error = None
        try:
            runner.graph.invoke({"raw_claim": row}, cfg)
            state = runner.graph.get_state(cfg).values
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            state = {}
        latency = time.time() - t0
        d = state.get("decision") or {}
        tr = state.get("tool_results") or {}
        trace = state.get("trace") or []
        calls = [c["name"] for t in trace if t["node"] == "tools" for c in t.get("calls", [])
                 if c.get("origin", "agent") == "agent"]
        auto = [n for t in trace if t["node"] == "complete_investigation" for n in t.get("auto_tools", [])]
        record = state.get("evidence_record") or {}
        mo_decision, mo_category = model_only(tr) if tr else (None, None)
        pv = tr.get("policy_verification") or {}
        ch = tr.get("claim_history") or {}
        interventions = [iv["guardrail"] for t in trace if t["node"] == "guardrails" for iv in t.get("interventions", [])]
        cost = ((tracker.input_tokens * price["input"] + tracker.output_tokens * price["output"]) / 1e6
                if price else 0.0)
        rec = {
            "split": split, "run_index": run_index, "claim_id": state.get("claim_id", f"PN-{row.get('PolicyNumber')}"),
            "policy_number": row.get("PolicyNumber"), "label": label, "thread_id": thread,
            "decision": d.get("decision"), "decided_by": d.get("decided_by"),
            "flag_category": d.get("flag_category"),
            "model_only_decision": mo_decision, "model_only_flag_category": mo_category,
            "simulated_data": bool(getattr(runner.deps, "simulated_enabled", False)),
            "policy_checks": {c["check_id"]: c["result"] for c in pv.get("checks", [])} if pv.get("ok") else {},
            "history_status": ch.get("status") if ch.get("ok") else ("failed" if ch else "not_run"),
            "history_count": ch.get("previous_claim_count") if ch.get("ok") else None,
            "fraud_probability": d.get("fraud_probability"), "risk_band": d.get("risk_band"),
            "tools_called": calls, "auto_tools": auto, "agent_steps": int(state.get("steps", 0) or 0),
            "investigation_id": d.get("investigation_id"), "investigation_status": d.get("investigation_status"),
            "reason_codes": d.get("reason_codes", []),
            "tool_statuses": {k: v["status"] for k, v in (record.get("tool_statuses") or {}).items()},
            "n_missing_information": len(d.get("missing_information", [])),
            "conflicting_evidence": d.get("conflicting_evidence", []),
            "unresolved_questions": d.get("unresolved_questions", []),
            "max_steps_reached": bool(state.get("max_steps_reached", False)),
            "llm_failures": int(state.get("llm_failures", 0) or 0),
            "interventions": interventions,
            "grounding_removed": interventions.count("evidence_grounding"),
            "n_evidence": len(d.get("evidence", [])), "evidence": d.get("evidence", []),
            "rationale": d.get("rationale"), "guardrail_notes": d.get("guardrail_notes", []),
            "validation_issues": d.get("validation_issues", []),
            "latency_s": round(latency, 3), "llm_calls": tracker.calls,
            "input_tokens": tracker.input_tokens, "output_tokens": tracker.output_tokens,
            "cost_usd": round(cost, 6), "model": model_name, "provider": provider, "error": error,
            "tool_results": tr,
        }
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        records.append(rec)
        if PAUSE_BETWEEN_CLAIMS_S:
            time.sleep(PAUSE_BETWEEN_CLAIMS_S)
        if (i + 1) % 10 == 0 or i + 1 == len(claims):
            spent = sum(r["cost_usd"] for r in records)
            progress(f"  {i + 1}/{len(claims)} claims, ${spent:.3f} so far")
    return records


def measured_per_run(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    recs = [r for r in records if r.get("llm_calls")]
    if not recs:
        return {}
    return {"input_tokens": float(np.mean([r["input_tokens"] for r in recs])),
            "output_tokens": float(np.mean([r["output_tokens"] for r in recs])),
            "cost_usd": float(np.mean([r["cost_usd"] for r in recs])),
            "latency_s": float(np.mean([r["latency_s"] for r in recs]))}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def load_results(paths: Iterable[str | Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    from src.schemas.decision import normalize_decision     # results written before the label rename
    for col in ("decision", "model_only_decision"):
        if col in df:
            df[col] = df[col].map(lambda x: normalize_decision(x) if isinstance(x, str) else x)
    for col in ("auto_tools", "reason_codes", "conflicting_evidence"):
        if col in df:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    # Results written before flag categories existed: every flag then came from the risk band.
    for dec, cat in (("decision", "flag_category"), ("model_only_decision", "model_only_flag_category")):
        if dec in df:
            if cat not in df:
                df[cat] = None
            df[cat] = [c if isinstance(c, str) else ("FRAUD_RISK" if d == "FLAG_FOR_INVESTIGATION" else None)
                       for d, c in zip(df[dec], df[cat])]
    if "simulated_data" not in df:
        df["simulated_data"] = False
    if "policy_checks" not in df:
        df["policy_checks"] = [{} for _ in range(len(df))]
    df["policy_checks"] = df["policy_checks"].apply(lambda x: x if isinstance(x, dict) else {})
    return df


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_ci(int(k), int(n))
    return {"k": int(k), "n": int(n), "rate": k / n if n else np.nan, "lo": lo, "hi": hi}


def decision_quality(df: pd.DataFrame, legit_weight: float = 1.0) -> pd.DataFrame:
    """Agent vs model-only threshold policy on the primary runs (run_index 0). Fraud recall and
    precision count FRAUD_RISK flags only; policy-condition and system-failure flags are
    reported as separate counts, so a coverage problem never counts as fraud caught."""
    d = df[df.run_index == 0]
    rows = []
    for col, cat_col, name in (("decision", "flag_category", "agent"),
                               ("model_only_decision", "model_only_flag_category", "model-only threshold policy")):
        fr, lg = d[d.label == 1], d[d.label == 0]
        fraud_flag = lambda x: (x[col] == "FLAG_FOR_INVESTIGATION") & (x[cat_col] == "FRAUD_RISK")
        flag_f = int(fraud_flag(fr).sum())
        flag_l = int(fraud_flag(lg).sum())
        prec_sample = flag_f / (flag_f + flag_l) if flag_f + flag_l else np.nan
        prec_pop = flag_f / (flag_f + legit_weight * flag_l) if flag_f + flag_l else np.nan
        other = lambda x, c: int(((x[col] == "FLAG_FOR_INVESTIGATION") & (x[cat_col] == c)).sum())
        for metric, (k, n) in {
            "fraud -> FRAUD_RISK flag (recall)": (flag_f, len(fr)),
            "legit -> APPROVE": (int((lg[col] == "APPROVE").sum()), len(lg)),
            "legit -> FRAUD_RISK flag (false-flag rate)": (flag_l, len(lg)),
            "REQUEST_MORE_INFORMATION (all claims)": (int((d[col] == "REQUEST_MORE_INFORMATION").sum()), len(d)),
            "POLICY_CONDITION flags (all claims)": (other(d, "POLICY_CONDITION"), len(d)),
            "SYSTEM_FAILURE flags (all claims)": (other(d, "SYSTEM_FAILURE"), len(d)),
        }.items():
            rows.append({"policy": name, "metric": metric, **_rate(k, n)})
        for c in ("POLICY_CONDITION", "SYSTEM_FAILURE"):
            rows.append({"policy": name, "metric": f"{c} flags: frauds / legit", "k": other(fr, c),
                         "n": other(lg, c), "rate": np.nan, "lo": np.nan, "hi": np.nan})
        rows.append({"policy": name, "metric": "FRAUD_RISK flag precision (sample)", "k": flag_f, "n": flag_f + flag_l,
                     "rate": prec_sample, "lo": np.nan, "hi": np.nan})
        rows.append({"policy": name, "metric": "FRAUD_RISK flag precision (re-weighted to true prevalence)",
                     "k": flag_f, "n": flag_f + flag_l, "rate": prec_pop, "lo": np.nan, "hi": np.nan})
    return pd.DataFrame(rows)


def agreement_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.run_index == 0]
    return pd.crosstab(d["model_only_decision"], d["decision"], margins=True)


def process_metrics(df: pd.DataFrame) -> dict[str, Any]:
    d = df[df.run_index == 0]
    n = len(d)
    from src.tools import AGENT_TOOL_TO_SOURCE
    tool_rate = {t: float(d.tools_called.apply(lambda c: t in c).mean()) for t in AGENT_TOOL_TO_SOURCE}
    auto = d["auto_tools"] if "auto_tools" in d else pd.Series([[]] * n, index=d.index)
    auto_rate = {t: float(auto.apply(lambda c: t in c).mean()) for t in AGENT_TOOL_TO_SOURCE}
    codes = pd.Series([c for lst in (d["reason_codes"] if "reason_codes" in d else []) for c in lst]).value_counts()
    iv = pd.Series([x for lst in d.interventions for x in lst]).value_counts().to_dict()
    lat = d.latency_s
    return {
        "claims": n, "errors": int(d.error.notna().sum()),
        "tool_use_rate": tool_rate,
        "auto_run_rate": auto_rate,
        "reason_code_counts": {k: int(v) for k, v in codes.items()},
        "claims_with_conflicts": int(d["conflicting_evidence"].apply(len).gt(0).sum()) if "conflicting_evidence" in d else 0,
        "flag_categories": {k: int(v) for k, v in d.loc[d.decision == "FLAG_FOR_INVESTIGATION", "flag_category"]
                            .value_counts().items()} if "flag_category" in d else {},
        "mean_tool_calls": float(d.tools_called.apply(len).mean()),
        "mean_agent_steps": float(d.agent_steps.mean()),
        "max_steps_hits": int(d.max_steps_reached.sum()),
        "fallback_rate": float((d.decided_by == "rule_fallback").mean()),
        "llm_failure_claims": int((d.llm_failures > 0).sum()),
        "guardrail_interventions": {k: int(v) for k, v in iv.items()},
        "claims_with_grounding_removals": int((d.grounding_removed > 0).sum()),
        "evidence_items_removed": int(d.grounding_removed.sum()),
        "mean_evidence_kept": float(d.n_evidence.mean()),
        "latency_p50_s": float(lat.median()), "latency_p95_s": float(lat.quantile(0.95)),
        "mean_input_tokens": float(d.input_tokens.mean()), "mean_output_tokens": float(d.output_tokens.mean()),
        "mean_cost_usd": float(d.cost_usd.mean()), "total_cost_usd": float(df.cost_usd.sum()),
    }


def consistency(df: pd.DataFrame) -> dict[str, Any]:
    """Run-to-run consistency on claims that were run more than once."""
    multi = df.groupby("policy_number").filter(lambda g: g.run_index.nunique() > 1)
    if multi.empty:
        return {"claims": 0}
    g = multi.groupby("policy_number")
    same_decision = g.decision.nunique().eq(1)
    return {"claims": int(len(same_decision)), "runs_per_claim": int(g.size().median()),
            "identical_decision_share": float(same_decision.mean()),
            "unstable_claims": [int(x) for x in same_decision[~same_decision].index]}


# ---------------------------------------------------------------------------
# LLM-as-judge and manual review
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are auditing an insurance-claim triage recommendation. Grade it against the
tool outputs it was based on. Answer each question with 1 (yes) or 0 (no):

- cites_evidence: does the rationale rely on specific facts from the tool outputs (scores,
  rule ids, red flags, SHAP factors, policy checks, similar-claim outcomes)?
- consistent_with_score: is the decision consistent with the fraud score and risk band? A
  high band should be flagged. If not, does the rationale explain why?
- no_invented_facts: is every fact in the rationale and evidence supported by the tool outputs?
  Treating missing information (e.g. an unavailable policy document or claim history) as a
  negative finding counts as an invented fact.
- clear_for_adjuster: could an adjuster act on it without re-reading the raw data?

Respond with ONLY a JSON object:
{"cites_evidence": 0|1, "consistent_with_score": 0|1, "no_invented_facts": 0|1,
 "clear_for_adjuster": 0|1, "comment": "<one sentence>"}

# Tool outputs
{tools}

# Recommendation
Decision: {decision}
Rationale: {rationale}
Evidence: {evidence}
"""


def judge_rationales(df: pd.DataFrame, llm, n: int = 40, seed: int = 42,
                     tracker: Optional[UsageTracker] = None) -> pd.DataFrame:
    from langchain_core.messages import HumanMessage
    from src.agent.llm import message_text
    d = df[(df.run_index == 0) & (df.decided_by == "llm")]
    d = d.sample(min(n, len(d)), random_state=seed) if len(d) else d
    rows = []
    for r in d.to_dict("records"):
        from src.agent.evidence import compact_for_prompt
        tools = {k: compact_for_prompt(k, v) for k, v in (r.get("tool_results") or {}).items()}
        prompt = JUDGE_PROMPT.replace("{tools}", json.dumps(tools, default=str)[:16000]) \
            .replace("{decision}", str(r["decision"])).replace("{rationale}", str(r["rationale"])) \
            .replace("{evidence}", json.dumps(r["evidence"], default=str))
        try:
            reply = llm.invoke([HumanMessage(prompt)], config={"callbacks": [tracker]} if tracker else None)
            text = message_text(reply)
            scores = json.loads(text[text.find("{"): text.rfind("}") + 1])
        except Exception as e:  # noqa: BLE001
            scores = {"error": f"{type(e).__name__}: {e}"}
        rows.append({"claim_id": r["claim_id"], "label": r["label"], "decision": r["decision"], **scores})
    return pd.DataFrame(rows)


def manual_review_sample(df: pd.DataFrame, n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Stratified across label x decision so disagreements are represented."""
    d = df[df.run_index == 0].copy()
    d["stratum"] = d.label.astype(str) + "_" + d.decision.astype(str)
    picks = d.groupby("stratum", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, n // max(d.stratum.nunique(), 1))), random_state=seed))
    if len(picks) < n:
        rest = d.drop(picks.index)
        picks = pd.concat([picks, rest.sample(min(n - len(picks), len(rest)), random_state=seed)])
    cols = ["claim_id", "label", "fraud_probability", "risk_band", "decision", "model_only_decision",
            "decided_by", "rationale", "evidence", "guardrail_notes", "tools_called"]
    out = picks.head(n)[cols].copy()
    out["evidence"] = out.evidence.apply(lambda e: " | ".join(f"[{x['source_tool']}] {x['fact']}" for x in e))
    for c in ("reviewer_agrees_with_decision", "rationale_accurate", "reviewer_notes"):
        out[c] = ""
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    def f(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v).replace("|", "\\|")
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)])


def summarize(df: pd.DataFrame, legit_weight: float = 1.0, judge: pd.DataFrame | None = None) -> dict[str, Any]:
    return {"decision_quality": decision_quality(df, legit_weight), "agreement": agreement_table(df),
            "process": process_metrics(df), "consistency": consistency(df),
            "judge": None if judge is None or judge.empty else
            judge[[c for c in ("cites_evidence", "consistent_with_score", "no_invented_facts", "clear_for_adjuster")
                   if c in judge]].mean().to_dict()}


def _policy_check_shares(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    checks = sorted({c for pc in d.policy_checks for c in pc})
    for c in checks:
        res = d.policy_checks.apply(lambda pc: pc.get(c))
        ran = res.notna().sum()
        rows.append({"check": c, "claims": int(ran), **{r: int((res == r).sum()) for r in ("PASS", "FAIL", "UNKNOWN")}})
    return pd.DataFrame(rows)


def compare_modes(dfs: Mapping[str, pd.DataFrame], legit_weight: float = 1.0) -> dict[str, Any]:
    """Side by side: the same claims with simulated reference data off and on."""
    off, on = (dfs[m][dfs[m].run_index == 0] for m in ("off", "on"))
    dq = {m: decision_quality(dfs[m], legit_weight) for m in ("off", "on")}
    agent = lambda m: dq[m][dq[m].policy == "agent"].set_index("metric")
    a_off, a_on = agent("off"), agent("on")
    table = pd.DataFrame({"off": a_off["rate"], "on": a_on["rate"],
                          "off k/n": [f"{k}/{n}" for k, n in zip(a_off.k, a_off.n)],
                          "on k/n": [f"{k}/{n}" for k, n in zip(a_on.k, a_on.n)]})
    table["Δ (on − off)"] = table["on"] - table["off"]
    table = table.reset_index()
    proc = {m: process_metrics(dfs[m]) for m in ("off", "on")}
    keys = ["mean_tool_calls", "fallback_rate", "claims_with_conflicts", "mean_evidence_kept", "latency_p50_s",
            "mean_input_tokens"]
    process = pd.DataFrame([{"metric": k, "off": proc["off"].get(k), "on": proc["on"].get(k)} for k in keys])
    hist = pd.DataFrame([{"mode": m, **d.history_status.value_counts().to_dict(),
                          "mean prior claims": d.history_count.dropna().mean() if d.history_count.notna().any() else None}
                         for m, d in (("off", off), ("on", on)) if "history_status" in d])
    j = off[["policy_number", "label", "decision", "flag_category"]].merge(
        on[["policy_number", "decision", "flag_category", "reason_codes", "policy_checks", "history_count"]],
        on="policy_number", suffixes=(" (off)", " (on)"))
    cat_off, cat_on = j["flag_category (off)"].fillna(""), j["flag_category (on)"].fillna("")
    changed = j[(j["decision (off)"] != j["decision (on)"]) | (cat_off != cat_on)]
    cause = lambda r: ", ".join([c for c, v in r["policy_checks"].items() if v == "FAIL"]
                                + ([f"{int(r['history_count'])} simulated prior claims"]
                                   if pd.notna(r["history_count"]) and r["history_count"] > 0 else [])) or "LLM judgement"
    changed = changed.assign(cause=[cause(r) for _, r in changed.iterrows()])
    changed = changed[["policy_number", "label", "decision (off)", "flag_category (off)", "decision (on)",
                       "flag_category (on)", "cause"]]
    return {"decision_quality": table, "process": process, "history": hist,
            "policy_checks": {m: _policy_check_shares(d) for m, d in (("off", off), ("on", on))},
            "changed": changed, "n_claims": int(len(j))}


def _summary_lines(summary: Mapping[str, Any], level: str = "##") -> list[str]:
    p = summary["process"]
    dq = summary["decision_quality"].copy()
    dq["95% CI"] = [f"[{lo:.3f}, {hi:.3f}]" if not np.isnan(lo) else "" for lo, hi in zip(dq.lo, dq.hi)]
    dq = dq[["policy", "metric", "k", "n", "rate", "95% CI"]]
    lines = [f"{level} Decision quality vs the model-only threshold policy", "",
             "Fraud recall and precision count **FRAUD_RISK flags only**. POLICY_CONDITION and SYSTEM_FAILURE "
             "flags are reported as separate counts; they are not fraud caught.", "", _md(dq), "",
             "Agreement between the model-only policy (rows) and the agent (columns):", "",
             _md(summary["agreement"].reset_index()), "",
             f"{level} Process", "", _md(pd.DataFrame([{"metric": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
                                                      for k, v in p.items()]), "{:.4f}"), "",
             f"{level} Run-to-run consistency", "", json.dumps(summary["consistency"]), ""]
    if summary.get("judge"):
        lines += [f"{level} Rationale quality (LLM-as-judge, share of 1s)", "",
                  _md(pd.DataFrame([{"criterion": k, "score": v} for k, v in summary["judge"].items()])), ""]
    return lines


def comparison_lines(cmp: Mapping[str, Any]) -> list[str]:
    lines = ["## Simulated reference data: off vs on (same claims)", "",
             "> **Caveat.** With simulated data **on**, policy terms, policy dates, coverage limits and prior "
             "claims come from the SIMULATED reference database (`data/simulated/insurance.db`, "
             "illustrative wording written by the project team, never derived from the fraud label). "
             "Differences show how the agent *uses* such evidence, not real fraud-detection gains.", "",
             f"Claims compared: {cmp['n_claims']}.", "",
             "### Agent decision quality", "", _md(cmp["decision_quality"]), "",
             "### Process", "", _md(cmp["process"], "{:.3f}"), ""]
    if len(cmp["history"]):
        lines += ["### Claim history availability", "", _md(cmp["history"], "{:.2f}"), ""]
    for m in ("off", "on"):
        t = cmp["policy_checks"][m]
        if len(t):
            lines += [f"### Policy verification results, simulated data {m}", "", _md(t), ""]
    ch = cmp["changed"]
    lines += [f"### Claims whose decision or flag category changed ({len(ch)})", "",
              _md(ch.head(50)) if len(ch) else "None.", ""]
    return lines


def write_report(summary: Mapping[str, Any] | None, path: Path, title: str, meta: Mapping[str, Any],
                 mode_summaries: Optional[Mapping[str, Mapping[str, Any]]] = None,
                 comparison: Optional[Mapping[str, Any]] = None) -> Path:
    """One summary (single mode), or one per simulated-data mode plus their comparison."""
    lines = [f"# {title}", "", "_Generated by `evaluation/agent_eval.py`._", "",
             "## Run", "", _md(pd.DataFrame([{"key": k, "value": v} for k, v in meta.items()])), ""]
    if comparison is not None:
        lines += comparison_lines(comparison)
    if mode_summaries:
        for m, sm in mode_summaries.items():
            lines += [f"## Simulated reference data {m}: full results", ""] + _summary_lines(sm, "###")
    elif summary is not None:
        lines += _summary_lines(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
