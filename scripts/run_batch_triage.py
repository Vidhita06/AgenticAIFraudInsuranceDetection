"""Batch triage runs for the agent evaluation (Step D).

Paid models (Anthropic, billed OpenAI-compatible APIs) print a cost estimate and ask before
spending (or need --yes). Free models (Ollama, free-tier gateways) show $0 and run without
a prompt; for slow local models use --sample-size and read the time estimate.

    python scripts/run_batch_triage.py offline              # rule-based path, no LLM, free
    python scripts/run_batch_triage.py estimate [--sample-size 40]
    python scripts/run_batch_triage.py all [--sample-size 40]   # pilot -> final test run -> report
    python scripts/run_batch_triage.py pilot [--n 5]
    python scripts/run_batch_triage.py final [--sample-size 40] [--max-cost 20]

--simulated-data off|on|both (default both) runs the same claims without and with the SIMULATED
reference database (scripts/build_simulated_db.py) and reports both side by side. Each mode
is its own once-only test evaluation per model.

The provider and model come from .env (LLM_PROVIDER, MODEL_NAME; see .env.example) or
--provider/--model. --batch uses BATCH_MODEL_NAME. --sample-size N = N test frauds + N legit
claims (default: all 138 frauds + 138 legit). The test set is evaluated once per model.
Results: evaluation/results/<stage>_<split>_<model>_<timestamp>.jsonl, the report in
docs/report/03_agent_evaluation.md (+ a per-model copy).
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.agent_eval import (  # noqa: E402
    MODES, RESULTS_DIR, UsageTracker, band_stratified_sample, compare_modes, confirm, estimate_cost, judge_rationales, load_results,
    load_split, manual_review_sample, measured_per_run, pricing, run_batch, stratified_sample, summarize,
    test_marker, write_report,
)
from src.config import load_config  # noqa: E402


@dataclass
class Plan:
    provider: str
    model: str
    sample_size: int | None      # frauds in the final sample (None = all)
    consistency_claims: int
    consistency_runs: int
    judge_claims: int
    modes: tuple[str, ...] = ("off",)

    @property
    def slug(self) -> str:
        return "".join(c if c.isalnum() or c in "-." else "_" for c in self.model)


def make_plan(args) -> Plan:
    from src.agent.llm import llm_settings
    s = llm_settings("batch" if args.batch else "default")
    provider = (args.provider or s["provider"]).lower()
    model = args.model or s["model"]
    ev = load_config()["evaluation"]
    n = args.sample_size
    total = 2 * n if n else None
    cons = args.consistency_claims if args.consistency_claims is not None else ev["consistency_claims"]
    judge = args.judge_claims if args.judge_claims is not None else ev["judge_claims"]
    if total:
        cons, judge = min(cons, total), min(judge, total)
    modes = ("off", "on") if args.simulated_data == "both" else (args.simulated_data,)
    return Plan(provider, model, n, cons, ev["consistency_runs"], judge, modes)


def check_simulated(plan: Plan) -> None:
    if "on" in plan.modes:
        from src.simulated.db import db_path
        if not db_path().exists():
            sys.exit(f"--simulated-data {'both' if len(plan.modes) == 2 else 'on'} needs the simulated reference "
                     "database: run python scripts/build_simulated_db.py (or use --simulated-data off)")


def make_runner(plan: Plan, need_llm: bool, mode: str = "off"):
    from src.agent.runner import TriageRunner
    if not need_llm:
        r = TriageRunner(llm=None)
        r.deps.simulated = MODES[mode]
        return r
    from src.agent.llm import check_ollama, get_llm, llm_settings, needs_api_key
    s = {**llm_settings(), "provider": plan.provider}
    if plan.provider == "ollama":
        try:
            check_ollama(plan.model, s["base_url"] or "http://localhost:11434")
        except RuntimeError as e:
            sys.exit(str(e))
    elif needs_api_key(s) and not s["api_key"]:
        sys.exit("No API key: set LLM_API_KEY in .env (see .env.example), or use the free "
                 "LLM_PROVIDER=ollama.")
    llm = get_llm(provider=plan.provider, model=plan.model)
    synth = get_llm(provider=plan.provider, model=plan.model, json_mode=True) if plan.provider == "ollama" else None
    runner = TriageRunner(llm=llm)
    runner.deps.synth_llm = synth
    runner.deps.simulated = MODES[mode]
    return runner


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def latest_pilot(plan: Plan) -> dict:
    """Measured per-run usage and latency from the most recent pilot with the same model."""
    for p in sorted(RESULTS_DIR.glob("pilot_validation_*.jsonl"), reverse=True):
        df = load_results([p])
        if not df.empty and (df.model == plan.model).all():
            m = measured_per_run(df.to_dict("records"))
            if m:
                return {**m, "file": p.name}
    return {}


def final_sample(plan: Plan, seed: int):
    """Test claim ids and labels only; nothing is scored until the final run."""
    ev = load_config()["evaluation"]
    test = load_split("test")
    if plan.sample_size:
        return stratified_sample(test, legit_per_fraud=ev["legit_per_fraud"], n_frauds=plan.sample_size, seed=seed)
    return stratified_sample(test, legit_per_fraud=ev["legit_per_fraud"], seed=seed)


def final_runs(plan: Plan, sample) -> int:
    return len(plan.modes) * (len(sample) + plan.consistency_claims * (plan.consistency_runs - 1))


def probe_claims(seed: int):
    return stratified_sample(load_split("validation"), n=6, seed=seed + 1)


def estimate_final(plan: Plan, seed: int, measured: dict | None):
    sample = final_sample(plan, seed)
    return sample, estimate_cost(sample, plan.model, final_runs(plan, sample), plan.judge_claims * len(plan.modes),
                                 measured or None, probe=probe_claims(seed), provider=plan.provider)


def report_modes(dfs: dict, legit_weight: float, path_title_meta, judges: dict | None = None) -> dict:
    """Write docs/report/03_agent_evaluation.md (+ per-model copy): one mode, or both side by side."""
    summaries = {m: summarize(d, legit_weight=legit_weight, judge=(judges or {}).get(m)) for m, d in dfs.items()}
    cmp = compare_modes(dfs, legit_weight) if set(dfs) == {"off", "on"} else None
    for path, title, meta in path_title_meta:
        if cmp is None:
            write_report(next(iter(summaries.values())), path, title, meta)
        else:
            write_report(None, path, title, meta, mode_summaries=summaries, comparison=cmp)
    return {"summaries": summaries, "comparison": cmp}


def stage_offline(args, plan: Plan) -> None:
    check_simulated(plan)
    df = load_split(args.split)
    sample = stratified_sample(df, n=args.n, seed=args.seed) if args.n else stratified_sample(df, seed=args.seed)
    dfs = {}
    for mode in plan.modes:
        runner = make_runner(plan, need_llm=False, mode=mode)
        out = RESULTS_DIR / f"offline_{args.split}_sim{mode}_{stamp()}.jsonl"
        print(f"offline (rule-based, no LLM, simulated data {mode}) on {len(sample)} {args.split} claims "
              f"-> {out.relative_to(ROOT)}")
        run_batch(sample, runner, args.split, out)
        dfs[mode] = load_results([out])
        s = summarize(dfs[mode], legit_weight=sample.attrs.get("legit_weight", 1.0))
        print(s["decision_quality"].round(3).to_string(index=False))
    if len(dfs) == 2:
        cmp = compare_modes(dfs, sample.attrs.get("legit_weight", 1.0))
        print("\nSimulated data off vs on (agent):")
        print(cmp["decision_quality"].round(3).to_string(index=False))
        print(f"{len(cmp['changed'])} claims changed decision or flag category")


def stage_estimate(args, plan: Plan) -> None:
    ev = load_config()["evaluation"]
    pilot = band_stratified_sample(load_split("validation"), n=args.n or ev["pilot_claims"], seed=args.seed)
    print(f"provider={plan.provider} model={plan.model}")
    print("PILOT  ", estimate_cost(pilot, plan.model, len(pilot), provider=plan.provider).describe(plan.model))
    measured = latest_pilot(plan)
    _, est = estimate_final(plan, args.seed, measured)
    print("FINAL  ", est.describe(plan.model))
    if measured:
        print(f"         (calibrated with {measured['file']})")


def stage_pilot(args, plan: Plan) -> dict:
    check_simulated(plan)
    n = args.n or load_config()["evaluation"]["pilot_claims"]
    sample = band_stratified_sample(load_split("validation"), n=n, seed=args.seed)
    band_of = sample.attrs["band_of"]
    print(f"Pilot sample: {n} validation claims by risk band {sample.attrs['bands']}, "
          f"{int(sample.FraudFound_P.sum())} frauds")
    est = estimate_cost(sample, plan.model, len(sample) * len(plan.modes), provider=plan.provider)
    if args.max_cost and est.usd > args.max_cost:
        sys.exit(f"Estimate ${est.usd:.2f} exceeds --max-cost {args.max_cost}")
    runners = {m: make_runner(plan, need_llm=True, mode=m) for m in plan.modes}   # fails fast, no spend
    if not confirm(f"PILOT (simulated data: {'/'.join(plan.modes)}): " + est.describe(plan.model), args.yes,
                   free=est.free):
        sys.exit("Not run.")
    all_recs, dfs = [], {}
    for mode in plan.modes:
        out = RESULTS_DIR / f"pilot_validation_{plan.slug}_sim{mode}_{stamp()}.jsonl"
        print(f"\nPilot, simulated data {mode}:")
        recs = run_batch(sample, runners[mode], "validation", out, model_name=plan.model, provider=plan.provider)
        all_recs += recs
        dfs[mode] = load_results([out])
        agree = 0
        for r in recs:
            tools = ",".join(r["tools_called"]) or "none"
            auto = ",".join(r.get("auto_tools") or []) or "none"
            cat = f"/{r['flag_category']}" if r.get("flag_category") else ""
            same = r["decision"] == r["model_only_decision"]
            agree += same
            guard = [i for i in r.get("interventions", []) if i not in ("evidence_grounding",)]
            print(f"  {r['claim_id']:>9} label={r['label']} band={band_of.get(r['policy_number'], '?'):<6} "
                  f"-> {str(r['decision']) + cat:<42} model-only={r['model_only_decision']:<24} "
                  f"{'=' if same else '≠'}  ({r['decided_by']}; agent tools={tools}; auto={auto}; "
                  f"guardrails={','.join(guard) or 'none'}; removed evidence={r['grounding_removed']}; "
                  f"llm failures={r['llm_failures']})")
        print(f"  Agreement with model_only_decision (simulated data {mode}): {agree}/{len(recs)}")
    meas = measured_per_run(all_recs) or {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0,
                                           "latency_s": sum(r["latency_s"] for r in all_recs) / max(len(all_recs), 1)}
    print(f"\nPilot done: {len(all_recs)} runs, ${sum(r['cost_usd'] for r in all_recs):.4f} spent; per run: "
          f"{meas['input_tokens']:.0f} input / {meas['output_tokens']:.0f} output tokens, "
          f"${meas['cost_usd']:.4f}, {meas['latency_s']:.1f}s")
    if len(dfs) == 2:
        ch = compare_modes(dfs)["changed"]
        print(f"Simulated data changed the decision or flag category of {len(ch)}/{len(sample)} pilot claims"
              + (":\n" + ch.to_string(index=False) if len(ch) else "."))
    fallback = sum(r["decided_by"] == "rule_fallback" for r in all_recs)
    if fallback:
        print(f"  NOTE: {fallback}/{len(all_recs)} runs fell back to the rule-based decision "
              "(the model failed or returned invalid JSON twice). Consider a larger model.")
    _, est_f = estimate_final(plan, args.seed, meas)
    print("\nProjected FINAL run: " + est_f.describe(plan.model))
    return meas


def stage_final(args, plan: Plan, measured: dict | None = None) -> None:
    check_simulated(plan)
    markers = {m: test_marker(plan.model, m) for m in plan.modes}
    done = [m for m, mk in markers.items() if mk.exists()]
    if done and not args.force_test_rerun:
        sys.exit(f"The final test evaluation for {plan.model} (simulated data {', '.join(done)}) already ran "
                 f"({markers[done[0]].read_text().strip()}). The test set is evaluated once per model and mode; "
                 "pass --force-test-rerun only if that run was invalid.")
    measured = measured or latest_pilot(plan)
    sample, est = estimate_final(plan, args.seed, measured)
    if args.max_cost and est.usd > args.max_cost:
        sys.exit(f"Estimate ${est.usd:.2f} exceeds --max-cost {args.max_cost}")
    runners = {m: make_runner(plan, need_llm=True, mode=m) for m in plan.modes}   # fails fast, no spend
    if not confirm(f"FINAL (test set, once per model and mode; simulated data {'/'.join(plan.modes)}): "
                   f"{est.describe(plan.model)}", args.yes, free=est.free):
        sys.exit("Not run.")
    ts = stamp()
    n_fraud = int(sample.FraudFound_P.sum())
    rep = sample.sample(min(plan.consistency_claims, len(sample)), random_state=args.seed)
    dfs, judges, outs, judge_cost = {}, {}, {}, 0.0
    pr = pricing(plan.model, plan.provider)
    for mode in plan.modes:
        runner, marker = runners[mode], markers[mode]
        out = RESULTS_DIR / f"final_test_{plan.slug}_sim{mode}_{ts}.jsonl"
        outs[mode] = out.name
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{out.name} started {ts}\n")
        print(f"Primary run (simulated data {mode}): {len(sample)} test claims ({n_fraud} frauds, "
              f"{len(sample) - n_fraud} legit) on {plan.model}")
        run_batch(sample, runner, "test", out, run_index=0, model_name=plan.model, provider=plan.provider)
        for k in range(1, plan.consistency_runs if plan.consistency_claims else 1):
            print(f"Consistency run {k + 1}/{plan.consistency_runs} on {len(rep)} claims (simulated data {mode})")
            run_batch(rep, runner, "test", out, run_index=k, model_name=plan.model, provider=plan.provider)
        dfs[mode] = load_results([out])
        if plan.judge_claims:
            tracker = UsageTracker()
            print(f"LLM-as-judge on up to {plan.judge_claims} rationales (simulated data {mode})")
            judges[mode] = judge_rationales(dfs[mode], runner.deps.llm, n=plan.judge_claims, tracker=tracker)
            judges[mode].to_csv(RESULTS_DIR / f"judge_{plan.slug}_sim{mode}_{ts}.csv", index=False)
            judge_cost += (tracker.input_tokens * pr["input"] + tracker.output_tokens * pr["output"]) / 1e6
        manual_review_sample(dfs[mode], load_config()["evaluation"]["manual_review_claims"], args.seed).to_csv(
            RESULTS_DIR / f"manual_review_{plan.slug}_sim{mode}_{ts}.csv", index=False)
    all_df = pd.concat(dfs.values(), ignore_index=True)
    meta = {"model": plan.model, "provider": plan.provider,
            "simulated reference data": " and ".join(plan.modes) + (" (side by side)" if len(plan.modes) == 2 else ""),
            "sample": (f"{n_fraud} frauds + {len(sample) - n_fraud} legit test claims"
                       + (" (--sample-size)" if plan.sample_size else " (all test frauds)")),
            "consistency": f"{len(rep) if plan.consistency_claims else 0} claims x {plan.consistency_runs} runs per mode",
            "judge rationales": sum(len(j) for j in judges.values()),
            "agent runs": len(all_df), "results": ", ".join(outs.values()),
            "agent cost USD": round(float(all_df.cost_usd.sum()), 4), "judge cost USD": round(judge_cost, 4),
            "price basis": pr["note"],
            "mean latency s": round(float(all_df.latency_s.mean()), 1),
            "model artifacts": json.loads((ROOT / "models" / "model_card.json").read_text())["model_version"]}
    title = f"Agent Evaluation (test set): {plan.model}, {len(sample)} claims"
    report_dir = ROOT / "docs" / "report"
    res = report_modes(dfs, sample.attrs["legit_weight"],
                       [(report_dir / "03_agent_evaluation.md", title, meta),
                        (report_dir / f"03_agent_evaluation_{plan.slug}.md", title, meta)], judges)
    (RESULTS_DIR / f"summary_{plan.slug}_{ts}.json").write_text(json.dumps(
        {"meta": meta, **{m: {"process": sm["process"], "consistency": sm["consistency"], "judge": sm["judge"],
                              "decision_quality": sm["decision_quality"].to_dict("records")}
                          for m, sm in res["summaries"].items()}}, indent=2, default=str))
    for mode in plan.modes:
        markers[mode].write_text(f"{outs[mode]} completed {stamp()}\n")
        print(f"\nSimulated data {mode}:")
        print(res["summaries"][mode]["decision_quality"].round(3).to_string(index=False))
    print(f"\nReport: docs/report/03_agent_evaluation.md (+ 03_agent_evaluation_{plan.slug}.md) | "
          f"manual review sheets: evaluation/results/manual_review_{plan.slug}_sim*_{ts}.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["offline", "estimate", "pilot", "final", "all"])
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--n", type=int, default=None, help="claims in the offline/pilot run")
    ap.add_argument("--sample-size", type=int, default=None,
                    help="final run: N test frauds + N legit claims (default: all 138 frauds + 138 legit)")
    ap.add_argument("--consistency-claims", type=int, default=None, help="claims re-run for consistency (0 = skip)")
    ap.add_argument("--judge-claims", type=int, default=None, help="rationales graded by LLM-as-judge (0 = skip)")
    ap.add_argument("--provider", default=None, choices=["anthropic", "openai", "ollama"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch", action="store_true", help="use BATCH_MODEL_NAME")
    ap.add_argument("--yes", action="store_true", help="approve spending without prompting (paid models)")
    ap.add_argument("--max-cost", type=float, default=None, help="abort if the estimate exceeds this (USD)")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to wait between claims (free-tier rate limits)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-test-rerun", action="store_true")
    ap.add_argument("--simulated-data", default="both", choices=["off", "on", "both"],
                    help="SIMULATED reference database for Tools 3, 4, 6: off, on, or both side by side")
    args = ap.parse_args()
    if args.pause:
        import evaluation.agent_eval as _ae
        _ae.PAUSE_BETWEEN_CLAIMS_S = args.pause
    plan = make_plan(args)
    if args.stage == "offline":
        if args.split == "test":
            sys.exit("The offline run uses validation claims; the test set is reserved for the final run.")
        stage_offline(args, plan)
    elif args.stage == "estimate":
        stage_estimate(args, plan)
    elif args.stage == "pilot":
        stage_pilot(args, plan)
    elif args.stage == "final":
        stage_final(args, plan)
    else:
        stage_final(args, plan, stage_pilot(args, plan))


if __name__ == "__main__":
    main()
