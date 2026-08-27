<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# PromptBudget v3.3 Request-Level Tail-Aware Cost Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose whether base-predicted-cost buckets contain AX31/Think underprediction-tail signal, then evaluate one fixed Train-only request-level p90 tail guard for 45/45 actual safety and at least 20% non-Light retention.

**Architecture:** A pure v3.3 helper reuses v3.2's immutable materialization, Raw quality fitting, base log-cost fitting, and official routing/scoring. Inside every fitting partition it creates grouped OOF base-cost residuals, derives per-model four-bucket p90 nonnegative log guards, applies them only to upgrade costs after the normal cost order clamp, and routes through the unchanged allocator and Premium fill. A diagnostic CLI may kill the experiment before any guard evaluator runs. A guarded evaluator emits isolated reports; only a 45/45, no-fallback, >=20%-retention candidate may produce a versioned runtime artifact and one frozen Dev sanity report.

**Tech Stack:** Python, NumPy, unittest, `baselines/hash_regex.py`, `tools/hash_regex_cost_stabilization_nested.py`, `ossp_router.scoring`, and `promptbudget.safety`.

---

## Fixed experiment contract

```python
QUALITY_KIND = "raw"
HASH_BINS = 256
RIDGE_ALPHA = 100.0
OUTER_SEEDS = (137, 271, 811)
OUTER_FOLDS = 5
INNER_FOLDS = 4
TAIL_BUCKET_COUNT = 4
TAIL_RESIDUAL_QUANTILE = 0.90
TAIL_HETEROGENEITY_MIN_SPREAD = math.log(1.10)
TAIL_HETEROGENEITY_REQUIRED_SEEDS = 2
TIER_SAFETY_RATIOS = {"fast": 1.0, "balanced": 1.0, "premium": 1.0}
RETENTION_PROMOTION_MINIMUM = Decimal("0.20")
FALLBACK_ARTIFACT = "baselines/hash-regex-public.v1.json"
```

Read before editing: `tools/hash_regex_cost_stabilization_nested.py`, `tools/evaluate_hash_regex_cost_stabilization_nested.py`, `baselines/hash_regex.py`, `baselines/train_hash_regex.py`, `src/promptbudget/safety.py`, and `src/ossp_router/scoring.py`.

Do not alter the Raw score target/features/alpha, tier safety, allocator/tie-break, Premium fill rule, or public v1 artifact behavior. Do not use Dev in diagnosis, fitting, selection, calibration, or full evaluation.

## File structure

| File | Responsibility |
| --- | --- |
| Create `tools/hash_regex_tail_guard_nested.py` | Pure grouped-OOF guard fitting, guarded routing, inner/outer evaluation, retention/status aggregation; no file I/O. |
| Create `tools/diagnose_hash_regex_tail_guard.py` | Train-only diagnostic guard, dry-run/execute mode, isolated atomic report. |
| Create `tools/evaluate_hash_regex_tail_guard_nested.py` | Guarded Train-only smoke/full evaluator and report contract. |
| Create `tests/promptbudget/test_hash_regex_tail_guard_nested.py` | Pure guard, leakage, routing, 12/12, outer/promotion tests. |
| Create `tests/test_diagnose_hash_regex_tail_guard.py` | Diagnostic CLI boundary/report tests. |
| Create `tests/test_evaluate_hash_regex_tail_guard_nested.py` | Evaluator CLI boundary/report tests. |
| Conditional modify `baselines/hash_regex.py` | Backward-compatible v1 parsing plus versioned tail-guard artifact parsing/runtime application. |
| Conditional create `tools/train_hash_regex_tail_guard_final.py` | Locked full-Train finalizer for safe candidates only. |
| Conditional create `tools/score_hash_regex_tail_guard_artifact.py` | One-pass frozen Dev scorer. |
| Conditional create corresponding finalizer/scorer tests | Parser, lock, route, and Dev boundary tests. |

Use this interpreter in every command because the project venv lacks NumPy:

```powershell
$env:PYTHONPATH='tools;src;baselines'
$py='C:\Users\alvin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
```

### Task 1: Build immutable OOF tail-guard primitives

**Files:**

- Create: `tools/hash_regex_tail_guard_nested.py`
- Create: `tests/promptbudget/test_hash_regex_tail_guard_nested.py`

- [ ] **Step 1: Write failing primitive tests.**

```python
def test_guard_uses_model_specific_predicted_log_cost_buckets() -> None:
    guard = nested.TailGuard(
        ax31_edges=(-2.0, 0.0, 2.0), think_edges=(-1.0, 1.0, 3.0),
        ax31_log_guards=(0.0, 0.1, 0.2, 0.3),
        think_log_guards=(0.0, 0.4, 0.5, 0.6),
    )
    result = nested.apply_tail_guard(
        base_costs=np.asarray([[1.0, 2.0, 3.0]]),
        base_log_costs=np.asarray([[0.0, 0.2, 1.2]]),
        guard=guard,
    )
    self.assertEqual(1.0, result[0, 0])
    self.assertAlmostEqual(2.0 * math.exp(0.2), result[0, 1])
    self.assertAlmostEqual(3.0 * math.exp(0.5), result[0, 2])

def test_guard_is_nonnegative_and_reapplies_cost_order() -> None:
    guard = nested.TailGuard(
        ax31_edges=(0.0, 1.0, 2.0), think_edges=(0.0, 1.0, 2.0),
        ax31_log_guards=(0.0, 0.0, 0.0, 0.0),
        think_log_guards=(0.0, 0.0, 0.0, 0.0),
    )
    with self.assertRaises(ValueError):
        nested.TailGuard((0.0, 1.0, 2.0), (0.0, 1.0, 2.0), (-0.1, 0, 0, 0), (0, 0, 0, 0))
    guarded = nested.apply_tail_guard(np.asarray([[2.0, 1.0, 1.5]]), np.zeros((1, 3)), guard)
    self.assertGreater(guarded[0, 1], guarded[0, 0])
    self.assertGreater(guarded[0, 2], guarded[0, 1])
```

- [ ] **Step 2: Verify failure.**

Run:

```powershell
& $py -m unittest tests.promptbudget.test_hash_regex_tail_guard_nested -v
```

Expected: import failure because the v3.3 module does not yet exist.

- [ ] **Step 3: Implement fixed guard representation and application.**

Implement an immutable `TailGuard` dataclass with exactly three finite nondecreasing edges and exactly four finite nonnegative guards for each upgrade model. Implement `bucket_index(value, edges)` with `bisect_right`, `apply_tail_guard(base_costs, base_log_costs, guard)`, and a shared `apply_ordering_clamp(costs)`.

`apply_tail_guard` must validate finite positive `(rows, 3)` base costs and finite `(rows, 3)` base log costs; copy input; leave Light column unchanged; multiply AX31/Think by `exp(guard)` selected from their own guard-before-clamp log prediction; reject nonfinite exponentiation, multiplication, or floor results; then enforce `AX31 >= Light*(1+1e-12)` and `Think >= AX31*(1+1e-12)`.

- [ ] **Step 4: Verify and commit.**

Run the Task 1 module command again. Expected: primitive tests pass.

Commit:

```powershell
git add tools/hash_regex_tail_guard_nested.py tests/promptbudget/test_hash_regex_tail_guard_nested.py
git commit -m "feat: add v3.3 tail guard primitives"
```

### Task 2: Fit grouped-OOF residual bucket guards and diagnostic signal

**Files:**

- Modify: `tools/hash_regex_tail_guard_nested.py`
- Modify: `tests/promptbudget/test_hash_regex_tail_guard_nested.py`

- [ ] **Step 1: Write failing OOF/diagnostic tests.**

```python
def test_fit_tail_guard_uses_only_partition_oof_residuals() -> None:
    first = nested.fit_tail_guard(data, train_indices, seed=137)
    changed = data.with_log_costs_changed_at(held_out_indices, delta=9.0)
    second = nested.fit_tail_guard(changed, train_indices, seed=137)
    self.assertEqual(first, second)

def test_equal_count_edges_use_bisect_right_and_report_tied_counts() -> None:
    summary = nested.bucket_residual_summary(
        np.asarray([0.0, 0.0, 0.0, 1.0, 2.0, 3.0]),
        np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    )
    self.assertEqual(4, len(summary["counts"]))
    self.assertEqual(sum(summary["counts"]), 6)
    self.assertEqual(2, nested.bucket_index(1.0, (0.0, 1.0, 2.0)))

def test_signal_requires_spread_for_two_grouped_oof_seeds() -> None:
    reports = {
        137: {"ax31": {"p90": (0.0, 0.0, math.log(1.11), math.log(1.11)}},
        271: {"ax31": {"p90": (0.0, 0.0, math.log(1.11), math.log(1.11)}},
        811: {"ax31": {"p90": (0.0, 0.0, 0.0, 0.0)}},
    }
    self.assertEqual("tail-signal-present", nested.diagnostic_status(reports))
```

- [ ] **Step 2: Verify failure.**

Run the focused module. Expected: missing `fit_tail_guard`, bucket summary, and diagnostic status APIs.

- [ ] **Step 3: Implement leakage-safe calibration.**

Implement `fit_tail_guard(data, partition_indices, seed)` as follows:

```python
def fit_tail_guard(data, partition_indices, seed):
    folds = grouped_folds(tuple(data.groups[i] for i in partition_indices), folds=4, seed=seed)
    oof = np.empty((len(partition_indices), 3), dtype=float)
    for fold in folds:
        fit_rows = tuple(partition_indices[i] for i in fold.train_indices)
        valid_rows = tuple(partition_indices[i] for i in fold.validation_indices)
        head = v32.fit_log_cost_heads(data.matrix[list(fit_rows)], data.log_costs[list(fit_rows)])
        oof[list(fold.validation_indices)] = v32.predict_heads(head, data.matrix[list(valid_rows)])
    residuals = data.log_costs[list(partition_indices)] - oof
    return TailGuard.from_oof_predictions(oof, residuals)
```

Map local grouped-fold indices back to `partition_indices`; ensure no row/group crosses a local fold. `TailGuard.from_oof_predictions` derives three higher empirical quartile edges and four p90 values per AX31/Think. It rejects empty buckets, nonfinite residuals, fewer than four content groups, or inconsistent OOF coverage. Its report includes edges, counts, p50/p90/p95/max residuals and exp(p90) factors, but never prompt, episode ID, or outcome text.

Implement `diagnose_tail_heterogeneity(data)` over seeds 137/271/811. It creates a complete grouped OOF report per seed, then returns `tail-signal-present` only if AX31 or Think reaches a p90 range `>= log(1.10)` in at least two seeds; else `tail-no-signal`.

- [ ] **Step 4: Verify and commit.**

Run the focused module. Expected: held-out mutation cannot change guard, ties are deterministic, and the two-seed threshold is enforced.

Commit:

```powershell
git add tools/hash_regex_tail_guard_nested.py tests/promptbudget/test_hash_regex_tail_guard_nested.py
git commit -m "feat: fit v3.3 grouped tail guards"
```

### Task 3: Route guarded batches and retain the v3.2 safety discipline

**Files:**

- Modify: `tools/hash_regex_tail_guard_nested.py`
- Modify: `tests/promptbudget/test_hash_regex_tail_guard_nested.py`

- [ ] **Step 1: Write failing policy and admission tests.**

```python
def test_guarded_policy_keeps_light_and_passes_guarded_cost_to_premium_fill() -> None:
    report = nested.score_guarded_batch_policy(data, indices, scores, base_logs, guard)
    self.assertTrue(report["tiers"]["premium"]["includes_premium_fill"])
    self.assertEqual(1.0, report["guard_metadata"]["light_multiplier"])

def test_inner_guard_admission_requires_twelve_independent_actual_checks() -> None:
    report = nested.evaluate_inner_guard(data, outer_train, seed=137)
    self.assertEqual(4, len(report["inner_folds"]))
    self.assertEqual(12, report["admission"]["required_checks"])
    self.assertFalse(report["pooled_for_routing"])

def test_outer_test_log_cost_mutation_cannot_change_guard_or_inner_admission() -> None:
    original = nested.select_inner_guard(data, outer_train, 137)
    changed = data.with_log_costs_changed_at(outer_test, delta=4.0)
    self.assertEqual(original, nested.select_inner_guard(changed, outer_train, 137))
```

- [ ] **Step 2: Verify failure.**

Run the focused module. Expected: missing guarded policy/evaluation APIs.

- [ ] **Step 3: Implement guarded routing.**

Implement `score_guarded_batch_policy` by calling v3.2's public `score_batch_policy` with `(1.0, 1.0)` and `guarded_log_costs`; do not duplicate official scorer cap semantics. Build `guarded_log_costs` only by applying `TailGuard` to base costs/logs and taking finite logs of the guarded costs. Include guard metadata, unrounded official scorer values, actual/predicted ratios, final model counts, and Premium-fill status.

Implement `evaluate_inner_guard` and `select_inner_guard`. There is exactly one fixed guard candidate: each validation fold independently fits its guard on its train complement, refits Raw quality/base cost heads on that complement, routes the validation batch, and feeds the four official reports to v3.2's `admit_inner_candidate`. Return an exact fallback dict `{status: "no-admitted-tail-guard", route: "all-light", guard: None}` if 12/12 fails. Never pool validation predictions or routes.

- [ ] **Step 4: Verify and commit.**

Run the focused module. Expected: Premium receives guarded costs, all-Light remains unchanged, 12/12 is fold-wise, and outer-test mutation cannot affect selection.

Commit:

```powershell
git add tools/hash_regex_tail_guard_nested.py tests/promptbudget/test_hash_regex_tail_guard_nested.py
git commit -m "feat: route v3.3 tail guarded batches"
```

### Task 4: Evaluate outer folds and enforce 20% promotion retention

**Files:**

- Modify: `tools/hash_regex_tail_guard_nested.py`
- Modify: `tests/promptbudget/test_hash_regex_tail_guard_nested.py`

- [ ] **Step 1: Write failing outer/promotion tests.**

```python
def test_outer_guard_locks_train_guard_and_scores_outer_test_once() -> None:
    report = nested.evaluate_outer_guard_fold(data, outer_train, outer_test, seed=137, fold=0)
    self.assertEqual(1, report["outer_test_evaluations"])
    self.assertIn("raw_comparator", report)

def test_promotion_requires_45_checks_no_fallback_and_twenty_percent_retention() -> None:
    result = nested.aggregate_outer_guard_folds(forty_five_passing_folds_with_retention("0.19"))
    self.assertEqual("safe-but-collapse", result["status"])
    result = nested.aggregate_outer_guard_folds(forty_five_passing_nonfallback_folds("0.20"))
    self.assertEqual("safe-candidate", result["status"])

def test_retention_counts_final_premium_fill_decisions_against_same_raw_routes() -> None:
    result = nested.aggregate_outer_guard_folds(sample_folds)
    self.assertEqual(Decimal("0.20"), result["retention"]["non_light_retention"]["value"])
```

- [ ] **Step 2: Verify failure.**

Run the focused module. Expected: missing outer guard/aggregation APIs.

- [ ] **Step 3: Implement outer report and terminal status.**

`evaluate_outer_guard_fold` must derive guard/heads from outer-train only; score the selected outer-test route once per tier; score unguarded Raw comparator separately for diagnostics; and store final routes after Premium fill. `aggregate_outer_guard_folds` must preserve Decimal arithmetic and produce all 45 actual checks, maximum actual ratio, paired official/cap-neutral Raw differences, frozen v3.2 report provenance, tier/total retention, fallback count, and guard/bucket summary.

Use this exact status logic:

```python
if not aggregate["promotion"]["outer_45_of_45_pass"]:
    status = "cost-calibration-no-go"
elif aggregate["fallback_all_light_folds"]:
    status = "safe-but-collapse"
elif aggregate["retention"]["non_light_retention"]["not_applicable"]:
    status = "safe-but-collapse"
elif aggregate["retention"]["non_light_retention"]["value"] < Decimal("0.20"):
    status = "safe-but-collapse"
else:
    status = "safe-candidate"
```

Do not add a cap-neutral quality or official-score threshold. Keep them diagnostic only.

- [ ] **Step 4: Verify and commit.**

Run the focused module. Expected: 44/45 fails, 45/45 with fallback fails promotion, 45/45 at 19% fails promotion, and 45/45 at 20% is safe candidate.

Commit:

```powershell
git add tools/hash_regex_tail_guard_nested.py tests/promptbudget/test_hash_regex_tail_guard_nested.py
git commit -m "feat: report v3.3 guard promotion diagnostics"
```

### Task 5: Add Train-only diagnostic CLI and execute the kill screen

**Files:**

- Create: `tools/diagnose_hash_regex_tail_guard.py`
- Create: `tests/test_diagnose_hash_regex_tail_guard.py`

- [ ] **Step 1: Write failing CLI tests.**

```python
def test_dev_path_is_rejected_before_load() -> None:
    with self.assertRaises(ValueError):
        cli.diagnose(args_for("data/materialized/dev/inputs.json"))

def test_dry_run_writes_nothing() -> None:
    self.assertEqual("dry-run", cli.diagnose(valid_args(execute=False))["mode"])
    self.assertFalse(report_path.exists())

def test_tail_no_signal_is_terminal_without_evaluator_artifact() -> None:
    report = cli.diagnose(valid_args(execute=True))
    self.assertIn(report["terminal_status"], {"tail-signal-present", "tail-no-signal"})
```

- [ ] **Step 2: Verify failure.**

Run:

```powershell
& $py -m unittest tests.test_diagnose_hash_regex_tail_guard -v
```

Expected: module import failure.

- [ ] **Step 3: Implement guarded diagnostic CLI.**

Validate aligned exactly 1,760 Train rows, Train paths, grouped schedule, report path below `build/hash-regex-tail-guard`, and no `dev` component before loading. Default is no-write dry run. `--execute` writes atomic JSON only. Include input/outcome SHA-256, fixed contract, OOF counts/edges/p50/p90/p95/max/factors for both upgrades and all seeds, the deterministic signal calculation, and `tail-signal-present`/`tail-no-signal`. Do not serialize prompts, episode IDs, or outcome text. Do not create an artifact or invoke the evaluator.

- [ ] **Step 4: Verify focused tests and run diagnostic once.**

Run:

```powershell
& $py -m unittest tests.test_diagnose_hash_regex_tail_guard -v
& $py tools/diagnose_hash_regex_tail_guard.py --input data/materialized/train/inputs.json --outcomes data/train/outcomes.json --report build/hash-regex-tail-guard/tail-diagnostic.json --execute
```

Expected: one Train-only diagnostic report with a terminal signal status. Inspect the report for three seeds, four buckets per upgrade model, no prompts/IDs, and no artifact.

- [ ] **Step 5: Commit source/tests only and stop on no signal.**

Commit:

```powershell
git add tools/diagnose_hash_regex_tail_guard.py tests/test_diagnose_hash_regex_tail_guard.py
git commit -m "feat: diagnose v3.3 cost tails"
```

If terminal status is `tail-no-signal` or `invalid-diagnostic`, preserve the report, record `baselines/hash-regex-public.v1.json` fallback, and do not implement/run Tasks 6–8.

### Task 6: Add guarded evaluator, run tests, and execute one-fold smoke only after signal

**Files:**

- Create: `tools/evaluate_hash_regex_tail_guard_nested.py`
- Create: `tests/test_evaluate_hash_regex_tail_guard_nested.py`

- [ ] **Step 1: Write failing evaluator boundary tests.**

```python
def test_evaluator_rejects_diagnostic_without_signal() -> None:
    with self.assertRaises(ValueError):
        cli.evaluate(args_with_diagnostic_status("tail-no-signal"))

def test_output_root_is_isolated() -> None:
    with self.assertRaises(ValueError):
        cli.require_report_path(Path("build/hash-regex-cost-stabilization/report.json"))

def test_dry_run_does_not_write_full_report() -> None:
    self.assertEqual("dry-run", cli.evaluate(valid_args(execute=False))["mode"])
    self.assertFalse(report_path.exists())
```

- [ ] **Step 2: Verify failure.**

Run the evaluator test module. Expected: import failure.

- [ ] **Step 3: Implement evaluator/report contract.**

Require a valid `tail-signal-present` diagnostic whose input/outcome SHA-256 matches evaluator paths. Reject Dev, wrong output root, malformed report, missing four bucket guards, nonfinite/negative guard values, and selectors not paired with `--execute`. Use atomic JSON and Decimal-to-string normalization. `--execute` requires exactly one of `--one-outer-fold` or `--full`; default dry run writes nothing.

Full report fields: diagnostic hash/provenance, inputs/outcomes hashes, schedule, fixed constants, every inner guard report and 12 actual checks, final outer routes without prompts/IDs, Raw and frozen-v3.2 comparator diagnostics, 45 checks, retention numerator/denominator, terminal status, and forbidden-artifact assertion.

- [ ] **Step 4: Verify and execute one-fold smoke.**

Run:

```powershell
& $py -m unittest tests.test_evaluate_hash_regex_tail_guard_nested -v
& $py tools/evaluate_hash_regex_tail_guard_nested.py --input data/materialized/train/inputs.json --outcomes data/train/outcomes.json --diagnostic build/hash-regex-tail-guard/tail-diagnostic.json --report build/hash-regex-tail-guard/nested-evaluation-smoke.json --execute --one-outer-fold
```

Expected: one outer fold; four independently routed inner batches; 12 actual checks; guarded Premium fill; Raw comparator; no artifact; and no Dev path.

- [ ] **Step 5: Commit source/tests only.**

```powershell
git add tools/evaluate_hash_regex_tail_guard_nested.py tests/test_evaluate_hash_regex_tail_guard_nested.py
git commit -m "feat: add v3.3 guarded evaluator"
```

### Task 7: Perform exactly one full Train-only evaluation and decide finalization

**Files:**

- Generated: `build/hash-regex-tail-guard/nested-evaluation.json`

- [ ] **Step 1: Run all focused regression tests.**

Run:

```powershell
& $py -m unittest tests.test_hash_regex_baseline tests.promptbudget.test_hash_regex_cost_stabilization_nested tests.promptbudget.test_hash_regex_tail_guard_nested tests.test_diagnose_hash_regex_tail_guard tests.test_evaluate_hash_regex_tail_guard_nested -v
```

Expected: pass. Any failed test, bad group split, missing report field, schema/serialization error, or report contamination is a hard stop. Do not change bucket count, quantile, diagnostic threshold, retention threshold, or folds.

- [ ] **Step 2: Run one full evaluation only.**

Run exactly once:

```powershell
& $py tools/evaluate_hash_regex_tail_guard_nested.py --input data/materialized/train/inputs.json --outcomes data/train/outcomes.json --diagnostic build/hash-regex-tail-guard/tail-diagnostic.json --report build/hash-regex-tail-guard/nested-evaluation.json --execute --full
```

Expected: 15 outer folds, 45 actual checks, all guard/bucket provenance, Raw/v3.2 paired diagnostics, retention, and exactly one terminal status.

- [ ] **Step 3: Stop on no-go/collapse.**

For `cost-calibration-no-go`, `safe-but-collapse`, or `invalid-evaluation`, preserve the full report; record the frozen v1 fallback; do not create artifact/finalizer or read Dev. A 45/45 result with any fallback or retention below 20% is not promotable.

### Task 8: Conditionally add runtime artifact, full-Train finalization, and one Dev sanity

**Files:**

- Modify: `baselines/hash_regex.py`
- Modify: `tests/test_hash_regex_baseline.py`
- Create: `tools/train_hash_regex_tail_guard_final.py`
- Create: `tools/score_hash_regex_tail_guard_artifact.py`
- Create: `tests/test_train_hash_regex_tail_guard_final.py`
- Create: `tests/test_score_hash_regex_tail_guard_artifact.py`
- Generated conditionally: `build/hash-regex-tail-guard/final-artifact.json`, `finalization.json`, `dev-sanity.json`

- [ ] **Step 1: Write failing parser/runtime/finalizer tests.**

```python
def test_v1_artifact_remains_parser_and_route_compatible() -> None:
    artifact = hash_regex.load_artifact(PUBLIC_V1)
    scores, costs = hash_regex.predict_episode(episode, artifact)
    self.assertEqual(set(MODEL_IDS), set(costs))

def test_tail_guard_artifact_uses_base_log_bucket_before_cost_order_clamp() -> None:
    artifact = hash_regex.parse_artifact(tail_guard_artifact_dict())
    _scores, costs = hash_regex.predict_episode(episode, artifact)
    self.assertGreater(costs["ax31"], 1.0)

def test_finalizer_rejects_non_safe_candidate_report() -> None:
    with self.assertRaises(ValueError):
        finalizer.locked_configuration(report_with_status("safe-but-collapse"))
```

- [ ] **Step 2: Verify failure.**

Run the three new modules. Expected: missing tail-guard parser/finalizer/scorer APIs.

- [ ] **Step 3: Implement versioned artifact and locked finalizer.**

Keep `ossp-hash-regex-linear-v1` exact-key parsing and routing unchanged. Add a new `ossp-hash-regex-tail-guard-v1` parser branch whose exact keys include `tail_cost_guard`. The guard object has `bucket_count: 4`, `quantile: 0.90`, model keys exactly `ax31`/`axk1-think`, three finite nondecreasing base-log edges and four finite nonnegative guards each. Extend the artifact representation so `predict_episode` obtains base log costs, exponentiates/clamps, applies the optional guard to AX31/Think, then reclamps. Test ID/order permutation invariance.

The finalizer accepts only `safe-candidate`, verifies nested/diagnostic/input/outcome hashes, reruns the locked full-Train grouped OOF guard derivation, refuses no-signal/empty bucket/fallback, fits full-Train Raw quality/base cost heads, writes the versioned parser-loadable artifact, and records raw vs guarded metadata. It never reads Dev.

- [ ] **Step 4: Verify and run finalization plus exactly one Dev sanity.**

Run:

```powershell
& $py -m unittest tests.test_hash_regex_baseline tests.test_train_hash_regex_tail_guard_final tests.test_score_hash_regex_tail_guard_artifact -v
& $py tools/train_hash_regex_tail_guard_final.py --input data/materialized/train/inputs.json --outcomes data/train/outcomes.json --nested-report build/hash-regex-tail-guard/nested-evaluation.json --artifact build/hash-regex-tail-guard/final-artifact.json --report build/hash-regex-tail-guard/finalization.json
& $py tools/score_hash_regex_tail_guard_artifact.py --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json --artifact build/hash-regex-tail-guard/final-artifact.json --report build/hash-regex-tail-guard/dev-sanity.json
```

Dev scorer only parses, routes, and scores the frozen artifact. Dev cap/protocol/parser/runtime failure writes `fallback-required` and cannot change edges, guards, route, or model choice.

- [ ] **Step 5: Final verification and handoff.**

Run:

```powershell
& $py -m unittest tests.test_hash_regex_baseline tests.promptbudget.test_hash_regex_tail_guard_nested tests.test_diagnose_hash_regex_tail_guard tests.test_evaluate_hash_regex_tail_guard_nested tests.test_train_hash_regex_tail_guard_final tests.test_score_hash_regex_tail_guard_artifact -v
git diff --check
```

Report worktree/branch, diagnostic status and p90 spread, 45/45 result, fallback count, retention, Raw/v3.2 paired deltas, finalization/Dev action, and frozen fallback. Do not claim a runtime/submission replacement without this conditional finalization path passing.

## Plan self-review

- One diagnostic form, one guard form, four buckets, p90, and 20% promotion threshold are fixed before execution.
- OOF residuals fit guards within every relevant Train-only partition; no in-sample residual calibration or outer/Dev leakage is permitted.
- All routing/cap checks use unchanged official scorer and independent batches.
- Diagnostic signal selects whether to run the guard experiment, not bucket count, quantile, thresholds, or a winning route.
- Quality is reported but cannot tune or promote a candidate.
- Full Train evaluation runs once only after focused tests and smoke; finalization/Dev are strictly conditional.

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-27-promptbudget-v3.3-tail-aware-cost-guard.md`.

1. **Subagent-Driven (recommended)** — dispatch a fresh agent per task with spec and code-quality review.
2. **Inline Execution** — execute task-by-task with `executing-plans` checkpoints.
