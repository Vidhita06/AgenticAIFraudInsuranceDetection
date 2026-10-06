"""Batch triage runs for the agent evaluation (Step D). Every stage that calls an LLM prints
a cost estimate first and asks before spending (or needs --yes).

    # free: rule-based path, no LLM, validation claims
    python scripts/run_batch_triage.py offline --split validation

    # estimates only, spends nothing
    python scripts/run_batch_triage.py estimate

    # THE single command for the live evaluation on a laptop with a .env file:
    # pilot on 5 validation claims (measured cost) -> projected cost of the final run ->
    # final run on test (all frauds + matched legit, consistency re-runs, LLM-as-judge) -> report
    python scripts/run_batch_triage.py all

    # or the stages separately
    python scripts/run_batch_triage.py pilot [--n 5]
    python scripts/run_batch_triage.py final [--max-cost 20]

Options: --provider/--model override .env; --batch uses BATCH_MODEL_NAME (cheaper Haiku);
--yes skips the confirmation prompts; --max-cost aborts if the estimate is higher.
Results: evaluation/results/<stage>_<split>_<timestamp>.jsonl (+ summary .json),
docs/report/03_agent_evaluation.md for the final run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.agent_eval import (  # noqa: E402
    RESULTS_DIR, TEST_MARKER, UsageTracker, confirm, estimate_cost, judge_rationales, load_results,
    load_split, manual_review_sample, measured_per_run, run_batch, stratified_sample, summarize,
    write_report,
)
from src.config import load_config  # noqa: E402


def make_runner(args, need_llm: bool):
    from src.agent.runner import TriageRunner
    if not need_llm:
        return TriageRunner(llm=None), "none"
    from src.agent.llm import get_llm, llm_settings
    s = llm_settings("batch" if args.batch else "default")
    provider, model = args.provider or s["provider"], args.model or s["model"]
    if provider != "ollama" and not s["api_key"]:
        sys.exit("No API key: set LLM_API_KEY (or ANTHROPIC_API_KEY) in .env. See .env.example.")
    llm = get_llm(provider=provider, model=model)
    return TriageRunner(llm=llm), model


def model_name(args) -> str:
    from src.agent.llm import llm_settings
    return args.model or llm_settings("batch" if args.batch else "default")["model"]


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def latest_pilot(model: str) -> dict:
    """Measured per-run usage from the most recent pilot with the same model."""
    for p in sorted(RESULTS_DIR.glob("pilot_validation_*.jsonl"), reverse=True):
        df = load_results([p])
        if not df.empty and (df.model == model).all():
            m = measured_per_run(df.to_dict("records"))
            if m:
                return {**m, "file": p.name}
    return {}


def final_plan(seed: int):
    """The test sample (claim ids and labels only; nothing is scored until the final run)."""
    ev = load_config()["evaluation"]
    test = load_split("test")
    sample = stratified_sample(test, legit_per_fraud=ev["legit_per_fraud"], seed=seed)
    extra = ev["consistency_claims"] * (ev["consistency_runs"] - 1)
    return sample, len(sample) + extra, ev


def probe_claims(seed: int):
    return stratified_sample(load_split("validation"), n=6, seed=seed + 1)


def stage_offline(args) -> None:
    df = load_split(args.split)
    sample = stratified_sample(df, n=args.n, seed=args.seed) if args.n else df
    runner, _ = make_runner(args, need_llm=False)
    out = RESULTS_DIR / f"offline_{args.split}_{stamp()}.jsonl"
    print(f"offline (rule-based, no LLM) run on {len(sample)} {args.split} claims -> {out.relative_to(ROOT)}")
    run_batch(sample, runner, args.split, out)
    s = summarize(load_results([out]), legit_weight=sample.attrs.get("legit_weight", 1.0))
    print(s["decision_quality"].round(3).to_string(index=False))


def stage_estimate(args) -> None:
    m = model_name(args)
    ev = load_config()["evaluation"]
    pilot = stratified_sample(load_split("validation"), n=args.n or ev["pilot_claims"], seed=args.seed)
    print("PILOT  ", estimate_cost(pilot, m, len(pilot)).describe(m))
    sample, runs, ev = final_plan(args.seed)
    measured = latest_pilot(m)
    print("FINAL  ", estimate_cost(sample, m, runs, ev["judge_claims"], measured or None,
                                  probe=probe_claims(args.seed)).describe(m))
    if measured:
        print(f"         (calibrated with {measured['file']})")


def stage_pilot(args) -> dict:
    m = model_name(args)
    n = args.n or load_config()["evaluation"]["pilot_claims"]
    sample = stratified_sample(load_split("validation"), n=n, seed=args.seed)
    est = estimate_cost(sample, m, len(sample))
    if args.max_cost and est.usd > args.max_cost:
        sys.exit(f"Estimate ${est.usd:.2f} exceeds --max-cost {args.max_cost}")
    runner, m = make_runner(args, need_llm=True)      # fails fast if no API key (no spend)
    if not confirm("PILOT: " + est.describe(m), args.yes):
        sys.exit("Not run.")
    out = RESULTS_DIR / f"pilot_validation_{stamp()}.jsonl"
    recs = run_batch(sample, runner, "validation", out, model_name=m)
    meas = measured_per_run(recs)
    spent = sum(r["cost_usd"] for r in recs)
    print(f"\nPilot done: {len(recs)} claims, ${spent:.4f} spent, measured per claim: "
          f"{meas.get('input_tokens', 0):.0f} input / {meas.get('output_tokens', 0):.0f} output tokens, "
          f"${meas.get('cost_usd', 0):.4f}, {meas.get('latency_s', 0):.1f}s")
    for r in recs:
        print(f"  {r['claim_id']:>9} label={r['label']} -> {r['decision']:<24} ({r['decided_by']}, "
              f"tools={r['tools_called']}, removed evidence={r['grounding_removed']})")
    sample_f, runs, ev = final_plan(args.seed)
    print("\nProjected FINAL run: " + estimate_cost(sample_f, m, runs, ev["judge_claims"], meas,
                                                  probe=probe_claims(args.seed)).describe(m))
    return meas


def stage_final(args, measured: dict | None = None) -> None:
    if TEST_MARKER.exists() and not args.force_test_rerun:
        sys.exit(f"The final test evaluation already ran ({TEST_MARKER.read_text().strip()}). "
                 "The test set is evaluated once; pass --force-test-rerun only if that run was invalid.")
    m = model_name(args)
    sample, runs, ev = final_plan(args.seed)
    measured = measured or latest_pilot(m)
    est = estimate_cost(sample, m, runs, ev["judge_claims"], measured or None, probe=probe_claims(args.seed))
    if args.max_cost and est.usd > args.max_cost:
        sys.exit(f"Estimate ${est.usd:.2f} exceeds --max-cost {args.max_cost}")
    runner, m = make_runner(args, need_llm=True)      # fails fast if no API key (no spend)
    if not confirm(f"FINAL (test set, runs once): {est.describe(m)}", args.yes):
        sys.exit("Not run.")
    ts = stamp()
    out = RESULTS_DIR / f"final_test_{ts}.jsonl"
    TEST_MARKER.parent.mkdir(parents=True, exist_ok=True)
    TEST_MARKER.write_text(f"{out.name} started {ts}\n")
    print(f"Primary run: {len(sample)} test claims ({int(sample.FraudFound_P.sum())} frauds)")
    run_batch(sample, runner, "test", out, run_index=0, model_name=m)
    rep = sample.sample(min(ev["consistency_claims"], len(sample)), random_state=args.seed)
    for k in range(1, ev["consistency_runs"]):
        print(f"Consistency run {k + 1}/{ev['consistency_runs']} on {len(rep)} claims")
        run_batch(rep, runner, "test", out, run_index=k, model_name=m)
    df = load_results([out])
    tracker = UsageTracker()
    print(f"LLM-as-judge on {ev['judge_claims']} rationales")
    judge = judge_rationales(df, runner.deps.llm, n=ev["judge_claims"], tracker=tracker)
    from evaluation.agent_eval import pricing
    pr = pricing(m)
    judge_cost = (tracker.input_tokens * pr["input"] + tracker.output_tokens * pr["output"]) / 1e6
    judge.to_csv(RESULTS_DIR / f"judge_{ts}.csv", index=False)
    manual_review_sample(df, ev["manual_review_claims"], args.seed).to_csv(
        RESULTS_DIR / f"manual_review_{ts}.csv", index=False)
    s = summarize(df, legit_weight=sample.attrs["legit_weight"], judge=judge)
    meta = {"results": out.name, "model": m, "test claims": len(sample),
            "frauds": int(sample.FraudFound_P.sum()), "agent runs": len(df),
            "agent cost USD": round(float(df.cost_usd.sum()), 4), "judge cost USD": round(judge_cost, 4),
            "model artifacts": json.loads((ROOT / "models" / "model_card.json").read_text())["model_version"]}
    path = write_report(s, ROOT / "docs" / "report" / "03_agent_evaluation.md", "Agent Evaluation (test set)", meta)
    (RESULTS_DIR / f"summary_{ts}.json").write_text(json.dumps(
        {"meta": meta, "process": s["process"], "consistency": s["consistency"], "judge": s["judge"],
         "decision_quality": s["decision_quality"].to_dict("records")}, indent=2, default=str))
    TEST_MARKER.write_text(f"{out.name} completed {stamp()}\n")
    print(s["decision_quality"].round(3).to_string(index=False))
    print(f"\nReport: {path.relative_to(ROOT)} | manual review sheet: evaluation/results/manual_review_{ts}.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["offline", "estimate", "pilot", "final", "all"])
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--n", type=int, default=None, help="claims (offline/pilot)")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch", action="store_true", help="use BATCH_MODEL_NAME (Haiku) for all calls")
    ap.add_argument("--yes", action="store_true", help="approve spending without prompting")
    ap.add_argument("--max-cost", type=float, default=None, help="abort if the estimate exceeds this (USD)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-test-rerun", action="store_true")
    args = ap.parse_args()
    if args.stage == "offline":
        if args.split == "test":
            sys.exit("The offline run uses validation claims; the test set is reserved for the final run.")
        stage_offline(args)
    elif args.stage == "estimate":
        stage_estimate(args)
    elif args.stage == "pilot":
        stage_pilot(args)
    elif args.stage == "final":
        stage_final(args)
    else:
        stage_final(args, stage_pilot(args))


if __name__ == "__main__":
    main()
