# Simulated reference database: implementation plan

**Status: proposal, awaiting team approval. Nothing below is built yet.**

The team decided to give Tools 3 (policy retrieval), 4 (policy verification) and 6 (claim
history) a **clearly labelled simulated** data source. This plan says how to build it, how the
tools use it, how the evaluation compares runs with and without it, and the two earlier fixes.

## 1. The database: `data/simulated/insurance.db` (SQLite)

It is built by `python scripts/build_simulated_db.py [--seed 42]`, which is deterministic for a
given seed.

- **Randomness:** each policy gets its own RNG stream from `(seed, PolicyNumber)`, so adding or
  removing claims never reshuffles the other policies.
- **Version control:** the file is git-ignored (a few MB) and rebuilt by the script.
  `data/simulated/manifest.json` is committed, with the seed, row counts and a SHA-256 of each
  generated table, so a rebuild can be checked against it.

| Table | Rows | Generated from | Simulated parts |
|---|---|---|---|
| `meta` | key/value | the script | `data_origin = "simulated reference data"`, seed, build time, generator version, the label-free column list |
| `policies` | one per `PolicyNumber` in train, validation and test (15,420) | real fields: `BasePolicy`, `VehicleCategory`, `Deductible`, `AgentType`, `VehiclePrice` (limit tier only) | inception date, current term start/end, coverage limits, `customer_id` |
| `policy_rules` | one per clause, for Liability, Collision and All Perils | `config/simulated_policy_clauses.yaml` (hand-written, reviewable in PRs) | all of it; every row has `wording_note = "illustrative wording written by the project team"` |
| `claim_history` | 0–7 per customer | count from the real `PastNumberOfClaims`: `none` → 0, `1` → 1, `2 to 4` → 2–4, `more than 4` → 5–7 | dates, incident types, statuses. **No fraud or outcome-of-investigation column.** |
| `verification_results` | written at run time | Tool 4's real outputs, one row per check per run | none (real outputs) |
| `investigation_audit` | written at run time | the final recommendation per investigation (decision, decided_by, reason codes, tool statuses, data mode) | none (real outputs) |

Every row of every simulated table carries `data_origin = 'simulated reference data'`.

### Dates

- **Accident date:** derived deterministically, with no randomness, from `Year`, `Month`,
  `WeekOfMonth` and `DayOfWeek`. Week 5 is clamped to the month's last such weekday.
- **Claim date:** derived the same way from `MonthClaimed`, `WeekOfMonthClaimed` and
  `DayOfWeekClaimed`. The claim year is `Year`, or `Year + 1` if the claim month is earlier than
  the accident month (the same assumption as rule V06/V07).
- **Policy inception:** `accident_date − d`, with `d` drawn uniformly from the
  `Days_Policy_Accident` bin:
  - `none` → 0;
  - `1 to 7`, `8 to 15` and `15 to 30` → within the bin;
  - `more than 30` → 31–1,095 days.
  - `d` is constrained so that `claim_date − inception` falls in the `Days_Policy_Claim` bin.
- **Contradictory records:** when the real fields contradict each other (V06/V09/V10, 3 of the
  4,626 validation and test claims), no consistent `d` exists. The row gets
  `dates_consistent = 0` and the best-effort `d`; the generator never "repairs" a claim.
- **Term:** annual and renewing. The current term is the last anniversary on or before the
  accident date, and ends one year later. Accidents on the inception day are inside the term.

### Coverage limits

These are drawn per policy from fixed tiers per coverage type:

- **Liability:** bodily injury per person / per accident, and property damage.
- **Collision and All Perils:** vehicle damage, capped near the top of the real `VehiclePrice` bin.
- **All Perils also:** a comprehensive (theft, fire, weather) limit.

**The data has no claim amount, so limit checks stay `UNKNOWN`.** The limits are shown in
retrieval but cannot PASS or FAIL.

### Clauses (`config/simulated_policy_clauses.yaml`, a draft for the team to edit)

Each clause has an id, a coverage type, a category (`condition`, `exclusion`, `reporting` or
`limit`), the wording, and, where the fields allow, a machine-checkable test. Draft set:

- **All coverages:**
  - incident inside the current term (dates);
  - claim reported within the deadline, measured from the derived dates (Collision and All
    Perils 30 days, Liability 60 days; illustrative values);
  - claim deductible equals the policy deductible. This always PASSes by construction, which
    the output says;
  - insured vehicle category equals the claim's `VehicleCategory`;
  - `PolicyType` consistent with the vehicle and base policy.
- **Liability:** covers damage the policyholder causes to third parties. Exclusion: damage to
  the insured's own vehicle. The data does not say whose damage is claimed, so this one is
  `UNKNOWN`, not guessed.
- **Collision:** a condition that the policyholder age is recorded (Age ≠ 0).
- **All Perils:** comprehensive perils listed as wording only; no field identifies the peril,
  so the check is `UNKNOWN`.

Clauses that need data we don't have stay `UNKNOWN`, with the reason.

## 2. The label is never read: hard rule and tests

- **Reading the inputs:** the generator reads the CSVs with an explicit `usecols` allow-list
  (`PolicyNumber`, the policy fields, the date fields and `PastNumberOfClaims`). Neither
  `FraudFound_P` nor `is_synthetic` is ever loaded.
- **Test 1, instrumented read:** patch `pandas.read_csv` and assert that every call passed
  `usecols` without the label.
- **Test 2, labels don't matter:** build twice from copies of the CSVs, once with
  `FraudFound_P` randomly permuted and once with it deleted. The table hashes must be
  byte-identical.
- **Test 3, schema:** no column in any simulated table is named like an outcome, label or fraud
  field.
- **Test 4, static check:** the generator's source does not reference `FraudFound_P` or
  `TARGET`.

## 3. Tool behaviour

The switch is `simulated_data.enabled` in `config/settings.yaml`. It defaults to `false`, and
the evaluation can override it per run.

| | `enabled: false` (default) | `enabled: true` |
|---|---|---|
| Tool 3, retrieval | exactly as now: claim-recorded attributes, `partial` | policy row + clauses for its base policy, `status: found`, `source: "simulated reference data"`, plus the wording note |
| Tool 4, verification | exactly as now | PASS/FAIL from the clause tests (term dates, reporting deadline, deductible, vehicle category, age condition); `UNKNOWN` where fields don't allow a check (limits, peril, whose damage). Each check names its source (`claim record` or `simulated reference data`) and its clause id. Writes `verification_results`. |
| Tool 6, history | exactly as now: `unavailable` | prior claims for the simulated customer within the window: dates, types, statuses, count, `source: "simulated reference data"`. No outcomes. |
| Final output | — | simulated evidence is suffixed "(simulated reference data)" by code; `investigation_audit` row written; `data_mode` recorded |

- The "behave exactly as now" requirement is enforced by keeping the current test suite
  running with `enabled: false`.
- New tests run the tools against a small DB built in a temp directory.

## 4. Earlier fixes

1. **REQUEST_MORE_INFORMATION only for claim-level missing or contradictory data.**
   - It is allowed when validation is blocking (V01–V03), or when a check on the claim record
     itself FAILs: PolicyType vs vehicle/base policy (PV02), or the claim timeline (PV04).
   - Any other REQUEST_MORE_INFORMATION from the LLM is replaced by the guardrail's rule-based
     decision, logged as `request_info_not_claim_level`.
   - A failed tool or missing fraud score is a system failure, not missing claim data. The
     rule-based decision then becomes FLAG_FOR_INVESTIGATION (manual assessment), with reason
     code `TOOL_FAILURE`, never REQUEST_MORE_INFORMATION.
   - A clause FAIL from the simulated policy (outside the term, late reporting, exclusion) is
     a coverage concern. It blocks APPROVE, and the fallback is FLAG_FOR_INVESTIGATION.
   - **Team: please confirm FLAG for these two cases**; the only alternative within the three
     labels is APPROVE, which the guardrails forbid.
2. **`similar_claims` becomes required:** `agent.required_tools` gains `similar_claims`, so
   `complete_investigation` runs it if the agent skips it.

## 5. Evaluation with and without simulated data

- **Command:** `scripts/run_batch_triage.py ... --simulated-data off|on|both` (default `both`).
  - `both` runs the *same* claim sample twice, once per mode, with the same model.
  - The results files are tagged `_simoff` / `_simon`.
  - The test set is evaluated once per (model, data mode).
- **Report:** `docs/report/03_agent_evaluation.md` gets side-by-side columns (off | on | Δ) for:
  - decision quality (recall, false-flag rate, FLAG precision, APPROVE and
    REQUEST_MORE_INFORMATION rates);
  - policy-verification outcomes (PASS / FAIL / UNKNOWN shares by check);
  - history availability, reason-code counts and conflicts;
  - claims whose decision changed between modes, with the cause (a clause FAIL or history).
- **Caveat in the report:** with simulated data **on**, the policy and history evidence is
  invented for the prototype. Differences show how the agent *uses* such evidence, not real
  fraud-detection gains.
- Notebook 03 runs the free offline baseline in both modes.
- Cost and time double with `both`; Ollama stays $0.

## 6. Files

- **New:**
  - `scripts/build_simulated_db.py` and `src/simulated/{db.py,generate.py,dates.py}`;
  - `config/simulated_policy_clauses.yaml` and `data/simulated/manifest.json`;
  - `tests/test_simulated_db.py` (label independence, determinism, bin consistency, history
    counts, schema) and `tests/test_simulated_tools.py` (tools on/off, runtime writes).
- **Changed:**
  - the three tools, `nodes.py` (audit write and data mode), `guardrails.py` and `evidence.py`;
  - `config/settings.yaml`, `.gitignore`, `evaluation/agent_eval.py`,
    `scripts/run_batch_triage.py`;
  - notebook 03, the README and `docs/agent_restructure_plan.md`.

## 7. For the B200 pilot

After this lands, run `python scripts/build_simulated_db.py` once (seconds), then
`run_batch_triage.py pilot --simulated-data both`.
