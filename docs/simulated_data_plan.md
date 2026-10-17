# Simulated reference database: implementation plan

**Status: implemented** (approved with changes: flag categories, a clause-failure check before
finishing). Results are in §8.

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
  - claim reported within the deadline, measured from the derived dates (60 days for every
    coverage after the measurement in §8; the first draft had 30 days for Collision and All
    Perils);
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
   - Confirmed by the team, with a reason category on every flag (§7): SYSTEM_FAILURE for
     tool failures, POLICY_CONDITION for clause FAILs.
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

## 7. Flag categories (added at approval)

- Every FLAG_FOR_INVESTIGATION carries `flag_category`:
  - `FRAUD_RISK`: high band, or the LLM's flag backed by fraud evidence (medium band, red
    flags, or elevated similar-claim fraud);
  - `POLICY_CONDITION`: a policy-condition FAIL (e.g. the reporting deadline) without fraud
    evidence;
  - `SYSTEM_FAILURE`: no fraud score, or a required tool failed.
- Who sets it: guardrail conversions and the rule-based fallback set it explicitly; for an
  LLM's own flag, code derives it from the evidence (`src/agent/evidence.py: flag_category`).
- In the evaluation, fraud recall and precision count FRAUD_RISK flags only. POLICY_CONDITION
  and SYSTEM_FAILURE flags are separate counts, also split by label.

## 8. Clause failure rates (measured before finishing)

Measured on the 2,313 real validation claims (`scripts/simulated_clause_report.py`, written to
`docs/report/tables/03_simulated_clause_results.csv`).

**The first draft's 30-day reporting deadline exceeded the ~5% bar.** These are the shares of
validation claims reported more than N days after the accident:

| Coverage | > 30 days | > 60 days | > 90 days |
|---|---|---|---|
| Liability (745) | 2.6% | 1.3% | 0.8% |
| Collision (891) | **5.3%** | 2.2% | 1.5% |
| All Perils (677) | **7.8%** | 3.5% | 2.4% |
| All (2,313) | **5.1%** | 2.3% | 1.5% |

The lag is measured conservatively. The data records only the week and weekday of the accident
and of the claim, so a day-level reading puts the claim before the accident for 305 validation
claims (13%), all of them in the same week. The tool therefore uses the smallest lag both weeks
allow, floored at 0. Rule V06, which works at week level, flags only 2 of those 305.

**Fix adopted:** a 60-day deadline for every coverage (one line in
`config/simulated_policy_clauses.yaml`).

**Result, simulated data on, validation:**

| Check | Level | FAIL | FAIL % | UNKNOWN |
|---|---|---|---|---|
| PV04_CLAIM_TIMELINE (claim record) | claim_record | 3 | 0.13% | 0 |
| SIM-ALL-04 reporting deadline (60 d) | policy_condition | 54 | 2.33% | 0 |
| SIM-COL-02 driver age recorded (Collision, All Perils) | policy_condition | 22 | 1.40% | 0 |
| SIM-ALL-01 term, SIM-ALL-02 vehicle, SIM-ALL-03 deductible | policy_condition | 0 | 0% | 0 |
| SIM-ALL-05 limits, SIM-LIA-02 own damage, SIM-AP-02 wear | policy_condition | 0 | 0% | all |

No clause fails on more than 5% of claims. Notes:

- SIM-ALL-01/02/03 cannot fail on these claims. The simulated policies are built from the
  claims' own fields, so these checks only catch claims that were edited or don't match their
  policy record.
- The 22 SIM-COL-02 failures are claims with Age = 0, which validation already blocks (V03 →
  REQUEST_MORE_INFORMATION).

## 9. For the B200 pilot

After this lands, run `python scripts/build_simulated_db.py` once (seconds), then
`run_batch_triage.py pilot --simulated-data both`.
