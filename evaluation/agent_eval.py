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
                      seed: int = 42) -> pd.DataFrame:
    """All frauds + legit_per_fraud x as many legit claims; with `n`, a balanced sample of
    n claims (half fraud, half legit) instead."""
    fraud, legit = df[df[TARGET] == 1], df[df[TARGET] == 0]
    if n is not None:
        nf = min(len(fraud), max(1, n // 2))
        f = fraud.sample(nf, random_state=seed)
        l = legit.sample(min(len(legit), n - nf), random_state=seed)
    else:
        f = fraud
        l = legit.sample(min(len(legit), int(round(legit_per_fraud * len(fraud)))), random_state=seed)
    out = pd.concat([f, l]).sample(frac=1, random_state=seed).reset_index(drop=True)
    out.attrs["legit_weight"] = len(legit) / max(len(l), 1)     # for prevalence re-weighting
    out.attrs["fraud_weight"] = len(fraud) / max(len(f), 1)
    return out


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def pricing(model: str) -> dict[str, float]:
    table = load_config()["llm_pricing_usd_per_mtok"]
    for k, v in table.items():
        if model == k or model.startswith(k):
            return v
    raise KeyError(f"no price for model {model!r}; add it to llm_pricing_usd_per_mtok in settings.yaml")


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

    def describe(self, model: str) -> str:
        return (f"{self.n_runs} agent runs ({self.n_claims} claims) + {self.n_judge} judge calls on {model}: "
                f"~{self.input_tokens / 1e6:.2f}M input + ~{self.output_tokens / 1e6:.2f}M output tokens "
                f"= ~${self.usd:.2f} (~${self.per_claim_usd:.4f} per run; basis: {self.basis})")


def _static_tokens_per_claim(claim: Mapping[str, Any]) -> tuple[float, float]:
    """Token estimate for one agent run from the actual prompt sizes: the tools are local,
    so their outputs are computed (free) and measured."""
    from src.agent.nodes import KEY_FIELDS, PROMPTS
    from src.tools import check_policy_rules, find_similar_claims, score_claim
    cfg = load_config()["evaluation"]
    orch = len((PROMPTS / "orchestrator_system.md").read_text())
    synth = len((PROMPTS / "synthesis.md").read_text())
    ctx = len(json.dumps({k: claim.get(k) for k in KEY_FIELDS}, default=str))
    score = len(json.dumps(score_claim(claim), default=str))
    policy = len(json.dumps(check_policy_rules(claim), default=str))
    similar = len(json.dumps(find_similar_claims(claim, 5), default=str))
    tool_schemas = 1200
    calls = cfg["est_agent_calls_per_claim"]
    out_agent = cfg["est_output_tokens_agent_call"]
    # agent turn i sees the system prompt, context, tool schemas, all earlier outputs and tool results
    base = orch + ctx + score + 800 + tool_schemas
    agent_in = sum(base + (policy if i >= 1 else 0) + (similar if i >= 2 else 0) + i * out_agent * CHARS_PER_TOKEN
                   for i in range(calls)) / CHARS_PER_TOKEN
    synth_in = (synth + score + policy + similar + 800) / CHARS_PER_TOKEN
    return agent_in + synth_in, calls * out_agent + cfg["est_output_tokens_synthesis"]


def estimate_cost(sample: pd.DataFrame, model: str, n_runs: int, n_judge: int = 0,
                  measured: Optional[Mapping[str, float]] = None,
                  probe: Optional[pd.DataFrame] = None) -> CostEstimate:
    """Measured per-run tokens (from a pilot) when available, otherwise a static estimate
    from the actual prompt and tool-output sizes, with a safety margin."""
    price = pricing(model)
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
    return CostEstimate(len(sample), n_runs, n_judge, total_in, total_out, usd, per, basis)


def confirm(message: str, assume_yes: bool = False, input_fn: Callable[[str], str] = input) -> bool:
    print(message)
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


def model_only_decision(tool_results: Mapping[str, Any]) -> str:
    """The threshold policy without the agent: blocking -> REQUEST_MORE_INFO,
    high band -> FLAG, otherwise APPROVE (same rules as the rule-based fallback)."""
    from src.agent.guardrails import rule_based_decision
    return rule_based_decision("", {"validation": tool_results.get("validation"),
                                    "fraud_score": tool_results.get("fraud_score")}, "model-only").decision


def run_batch(claims: pd.DataFrame, runner, split: str, out_path: Path, run_index: int = 0,
              model_name: str = "none", progress: Callable[[str], None] = print) -> list[dict]:
    """Triage every claim (stops at human review) and append one JSON record per claim."""
    tracker = UsageTracker()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    price = None
    try:
        price = pricing(model_name)
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
        calls = [c["name"] for t in trace if t["node"] == "tools" for c in t.get("calls", [])]
        interventions = [iv["guardrail"] for t in trace if t["node"] == "guardrails" for iv in t.get("interventions", [])]
        cost = ((tracker.input_tokens * price["input"] + tracker.output_tokens * price["output"]) / 1e6
                if price else 0.0)
        rec = {
            "split": split, "run_index": run_index, "claim_id": state.get("claim_id", f"PN-{row.get('PolicyNumber')}"),
            "policy_number": row.get("PolicyNumber"), "label": label, "thread_id": thread,
            "decision": d.get("decision"), "decided_by": d.get("decided_by"),
            "model_only_decision": model_only_decision(tr) if tr else None,
            "fraud_probability": d.get("fraud_probability"), "risk_band": d.get("risk_band"),
            "tools_called": calls, "agent_steps": int(state.get("steps", 0) or 0),
            "max_steps_reached": bool(state.get("max_steps_reached", False)),
            "llm_failures": int(state.get("llm_failures", 0) or 0),
            "interventions": interventions,
            "grounding_removed": interventions.count("evidence_grounding"),
            "n_evidence": len(d.get("evidence", [])), "evidence": d.get("evidence", []),
            "rationale": d.get("rationale"), "guardrail_notes": d.get("guardrail_notes", []),
            "validation_issues": d.get("validation_issues", []),
            "latency_s": round(latency, 3), "llm_calls": tracker.calls,
            "input_tokens": tracker.input_tokens, "output_tokens": tracker.output_tokens,
            "cost_usd": round(cost, 6), "model": model_name, "error": error,
            "tool_results": tr,
        }
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        records.append(rec)
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
    return pd.DataFrame(rows)


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_ci(int(k), int(n))
    return {"k": int(k), "n": int(n), "rate": k / n if n else np.nan, "lo": lo, "hi": hi}


def decision_quality(df: pd.DataFrame, legit_weight: float = 1.0) -> pd.DataFrame:
    """Agent vs model-only threshold policy on the primary runs (run_index 0)."""
    d = df[df.run_index == 0]
    rows = []
    for col, name in (("decision", "agent"), ("model_only_decision", "model-only threshold policy")):
        fr, lg = d[d.label == 1], d[d.label == 0]
        flag_f = int((fr[col] == "FLAG_FOR_INVESTIGATION").sum())
        flag_l = int((lg[col] == "FLAG_FOR_INVESTIGATION").sum())
        prec_sample = flag_f / (flag_f + flag_l) if flag_f + flag_l else np.nan
        prec_pop = flag_f / (flag_f + legit_weight * flag_l) if flag_f + flag_l else np.nan
        for metric, (k, n) in {
            "fraud -> FLAG (recall)": (flag_f, len(fr)),
            "legit -> APPROVE": (int((lg[col] == "APPROVE").sum()), len(lg)),
            "legit -> FLAG (false-flag rate)": (flag_l, len(lg)),
            "REQUEST_MORE_INFO (all claims)": (int((d[col] == "REQUEST_MORE_INFO").sum()), len(d)),
        }.items():
            rows.append({"policy": name, "metric": metric, **_rate(k, n)})
        rows.append({"policy": name, "metric": "FLAG precision (sample)", "k": flag_f, "n": flag_f + flag_l,
                     "rate": prec_sample, "lo": np.nan, "hi": np.nan})
        rows.append({"policy": name, "metric": "FLAG precision (re-weighted to true prevalence)",
                     "k": flag_f, "n": flag_f + flag_l, "rate": prec_pop, "lo": np.nan, "hi": np.nan})
    return pd.DataFrame(rows)


def agreement_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.run_index == 0]
    return pd.crosstab(d["model_only_decision"], d["decision"], margins=True)


def process_metrics(df: pd.DataFrame) -> dict[str, Any]:
    d = df[df.run_index == 0]
    n = len(d)
    tool_rate = {t: float(d.tools_called.apply(lambda c: t in c).mean())
                 for t in ("check_policy_rules", "find_similar_claims")}
    iv = pd.Series([x for lst in d.interventions for x in lst]).value_counts().to_dict()
    lat = d.latency_s
    return {
        "claims": n, "errors": int(d.error.notna().sum()),
        "tool_use_rate": tool_rate,
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
  rule ids, red flags, similar-claim outcomes)?
- consistent_with_score: is the decision consistent with the fraud score and risk band? A
  high band should be flagged. If not, does the rationale explain why?
- no_invented_facts: is every fact in the rationale and evidence supported by the tool outputs?
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
        tools = {k: v for k, v in (r.get("tool_results") or {}).items()}
        prompt = JUDGE_PROMPT.replace("{tools}", json.dumps(tools, default=str)[:12000]) \
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


def write_report(summary: Mapping[str, Any], path: Path, title: str, meta: Mapping[str, Any]) -> Path:
    p = summary["process"]
    dq = summary["decision_quality"].copy()
    dq["95% CI"] = [f"[{lo:.3f}, {hi:.3f}]" if not np.isnan(lo) else "" for lo, hi in zip(dq.lo, dq.hi)]
    dq = dq[["policy", "metric", "k", "n", "rate", "95% CI"]]
    lines = [f"# {title}", "", "_Generated by `evaluation/agent_eval.py`._", "",
             "## Run", "", _md(pd.DataFrame([{"key": k, "value": v} for k, v in meta.items()])), "",
             "## Decision quality vs the model-only threshold policy", "", _md(dq), "",
             "Agreement between the model-only policy (rows) and the agent (columns):", "",
             _md(summary["agreement"].reset_index()), "",
             "## Process", "", _md(pd.DataFrame([{"metric": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
                                                for k, v in p.items()]), "{:.4f}"), "",
             "## Run-to-run consistency", "", json.dumps(summary["consistency"]), ""]
    if summary.get("judge"):
        lines += ["## Rationale quality (LLM-as-judge, share of 1s)", "",
                  _md(pd.DataFrame([{"criterion": k, "score": v} for k, v in summary["judge"].items()])), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
