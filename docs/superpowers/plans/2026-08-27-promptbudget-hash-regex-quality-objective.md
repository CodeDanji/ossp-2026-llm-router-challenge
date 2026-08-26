<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# PromptBudget hash-regex quality-objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Train-only grouped nested evaluator that compares only the approved quality objectives for the existing hash-regex router, chooses a winner through pre-registered routing/cap gates, and leaves Dev untouched until the candidate is frozen.

**Architecture:** Do not change `baselines/hash_regex.py`, the public artifact, or the legacy trainer while comparing candidates. A new pure evaluator helper materializes the current 270-dimensional feature matrix once, refits the unchanged cost-head family per training partition, fits quality heads with fixed `alpha=100.0`, and converts every uplift fit back to ordinary raw score heads before calling the existing runtime allocator. A thin CLI validates Train-only boundaries and writes reports under an isolated build directory.

**Tech Stack:** Python standard library, NumPy, existing `ossp_router` protocol/scorer, `promptbudget.safety` grouped folds/content groups, existing hash-regex runtime primitives, `unittest`.

---

## Fixed experiment contract

```python
QUALITY_CANDIDATES = ("raw", "weighted-uplift", "regret-weighted-raw")
GAMMA_GRID = (0.0, 1.0, 2.0, 4.0)
ETA_GRID = (0.0, 1.0, 2.0, 4.0)
HASH_BINS = 256
RIDGE_ALPHA = 100.0
OUTER_SEEDS = (137, 271, 811)
OUTER_FOLDS = 5
INNER_FOLDS = 4
SAFETY_GRID_SIZE = 121
MIN_WEIGHTED_IMPROVEMENT = 0.002
```

`direct-uplift` is an invariance control, not a winner candidate. The evaluator uses only materialized Train inputs and Train outcomes, rejects every path containing a `dev` component, and never creates runtime artifacts. The complete fixed policy always means `hash_regex.select_models` plus `hash_regex.fill_ax31_upgrades` for Premium.

The report schedule is exactly **3 seeds x 5 grouped outer folds** (15 outer tests);
each outer-train partition uses four grouped inner folds. This schedule is a fixed
evaluation constant, not a CLI option.

## File structure

| File | Responsibility |
| --- | --- |
| `tools/hash_regex_quality_nested.py` | Pure data materialization, partition-only ridge/WLS fits, direct control, lambda/regret construction, inner selection, provisional safety, outer scoring, and winner gates. No file I/O. |
| `tools/evaluate_hash_regex_quality_nested.py` | Train-path and output-root guards, dry-run/one-fold/full modes, atomic JSON report writing. |
| `tests/promptbudget/test_hash_regex_quality_nested.py` | Mathematical contracts, group isolation, fitting boundaries, full-policy safety, candidate selection, and winner gates. |
| `tests/test_evaluate_hash_regex_quality_nested.py` | CLI guardrails, no-write dry-run, and report schema smoke tests. |
| `tests/test_hash_regex_baseline.py` | Existing runtime/parser/permutation regression; append only artifact compatibility coverage after a candidate is frozen. |
| `tools/train_hash_regex_quality_final.py` | **Post-winner only.** Train-only finalization from a locked report configuration into a build artifact using the existing `HashRegexArtifact` schema. |
| `tools/score_hash_regex_artifact.py` | Frozen-artifact Dev sanity scorer. Builds every tier submission and calls the official scorer; it never fits, calibrates, or rewrites an artifact. |

The legacy `baselines/train_hash_regex.py` is deliberately not the nested evaluator: its OOF split is row modulo, its alpha choice combines score and cost losses, and its safety calibration omits Premium fill. Its default artifact/training behavior remains untouched.

## Autonomous execution contract

This document by itself authorizes no expensive run. A subsequent explicit user request
such as **"implement through Task 6"**, **"run the complete nested evaluation"**, or
**"run this overnight"** authorizes the following bounded sequence without further
questions:

1. Complete Tasks 1--5, including the focused tests and the one-outer-fold smoke run.
2. Run `--full` only if those checks pass. A failed check is a hard stop: preserve its
   report/log and do not run a partial or altered full evaluation.
3. If the full report retains Raw or any winner gate fails, stop successfully with
   `raw-retained`; do not train a new artifact or inspect Dev.
4. Only an admitted non-Raw winner may enter finalization. If finalizer tests or
   Train-only finalization fail, stop before Dev.
5. Score exactly one frozen final artifact on Dev. A cap, protocol, parser, or runtime
   failure produces `fallback-required`; do not retune. The deployable fallback is
   exactly `baselines/hash-regex-public.v1.json`.

The autonomous request never authorizes changing candidate grids, folds, feature family,
cost family, allocator, or acceptance gates in response to an intermediate result.

### Task 1: Add pure fixed-capacity quality fitting primitives

**Files:**

- Create: `tools/hash_regex_quality_nested.py`
- Create: `tests/promptbudget/test_hash_regex_quality_nested.py`

- [ ] **Step 1: Write failing tests for the target transformations and WLS normalization.**

```python
def test_direct_uplift_is_raw_prediction_invariant() -> None:
    matrix = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    scores = np.asarray([[.2, .3, .5], [.4, .6, .7], [.1, .2, .1], [.9, .8, .9]])
    raw = nested.fit_quality_heads(matrix, scores, kind="raw")
    direct = nested.fit_quality_heads(matrix, scores, kind="direct-uplift")
    np.testing.assert_allclose(nested.predict_heads(raw, matrix), nested.predict_heads(direct, matrix), atol=1e-10)

def test_uplift_weights_are_positive_and_mean_one_per_training_head() -> None:
    weights = nested.positive_uplift_weights(np.asarray([-.5, 0.0, .5, 1.0]), gamma=4.0)
    self.assertTrue(np.all(weights > 0.0))
    self.assertAlmostEqual(1.0, float(weights.mean()))

def test_zero_regret_strength_is_raw_invariant() -> None:
    regret = np.zeros((4, 3))
    raw = nested.fit_quality_heads(matrix, scores, kind="raw")
    weighted = nested.fit_quality_heads(matrix, scores, kind="regret-weighted-raw", strength=0.0, regret=regret)
    np.testing.assert_allclose(nested.predict_heads(raw, matrix), nested.predict_heads(weighted, matrix), atol=1e-10)
```

- [ ] **Step 2: Run the test to verify the missing helper fails.**

Run:

```powershell
$env:PYTHONPATH='src;baselines;tools'; python -m unittest tests.promptbudget.test_hash_regex_quality_nested -v
```

Expected: import failure for `hash_regex_quality_nested`.

- [ ] **Step 3: Implement data-free model types and fits.**

The module exposes an immutable `LinearScoreHeads(mean, scale, intercept,
coefficients)` value, plus `fit_quality_heads`, `fit_log_cost_heads`, and
`predict_heads`. `intercept` has three entries and `coefficients` has one column per
model. `fit_quality_heads` accepts exactly `raw`, `direct-uplift`,
`weighted-uplift`, and `regret-weighted-raw`; the latter accepts a `(rows, 3)`
regret array.

Use `HASH_BINS=256` and `RIDGE_ALPHA=100.0`; do not expose alpha or hash size as candidate grids. Standardize each fit partition from that partition only. Raw and direct uplift must share the ordinary unweighted ridge path so their equivalent reconstructed raw heads match. For weighted fits, use an augmented standardized design with an unpenalized intercept and a ridge penalty only on feature coefficients.

Weighted uplift uses `delta = scores[:, 1:] - scores[:, [0]]`, an unweighted Light head, and per-upgrade `w=1+gamma*maximum(delta, 0)`, normalized to mean one **using only fit rows**. Reconstruct AX31/Think intercepts and coefficients by adding the Light and uplift fits. Regret-weighted Raw fits all three raw score heads separately with per-action mean-one WLS weights. Cost heads are always ordinary unweighted log-cost ridge.

- [ ] **Step 4: Add input validation tests and implement matching errors.**

```python
def test_weighted_fit_rejects_nonfinite_strength_or_wrong_regret_shape() -> None:
    with self.assertRaises(ValueError):
        nested.fit_quality_heads(matrix, scores, kind="weighted-uplift", strength=-1.0)
    with self.assertRaises(ValueError):
        nested.fit_quality_heads(matrix, scores, kind="regret-weighted-raw", strength=1.0, regret=np.zeros((4, 2)))
```

Reject nonfinite arrays, wrong `(rows, 3)` target shapes, empty partitions, negative/nonfinite strength, and regret missing for nonzero regret-weighted fits.

- [ ] **Step 5: Run focused tests and commit.**

Run:

```powershell
$env:PYTHONPATH='src;baselines;tools'; python -m unittest tests.promptbudget.test_hash_regex_quality_nested -v
```

Expected: all primitive tests pass.

```powershell
git add tools/hash_regex_quality_nested.py tests/promptbudget/test_hash_regex_quality_nested.py
git commit -m "feat: add hash-regex quality objective fits"
```

### Task 2: Materialize Train-only grouped data and tier-derived regret

**Files:**

- Modify: `tools/hash_regex_quality_nested.py`
- Modify: `tests/promptbudget/test_hash_regex_quality_nested.py`

- [ ] **Step 1: Write failing boundary, group, and lambda-unit tests.**

```python
def test_group_disjointness_rejects_a_shared_canonical_content_group() -> None:
    with self.assertRaises(ValueError):
        nested.require_group_disjoint((0, 1), (2,), ("a", "b", "a"))

def test_lambda_uses_the_batch_light_total_not_per_row_light_cost() -> None:
    selected, penalty = nested.lagrange_selection_and_penalty(scores, costs, multiplier=1.25)
    expected, _ratio = hash_regex.select_models(score_rows, cost_rows, budget_multiplier=1.25, safety_ratio=1.0)
    self.assertEqual(tuple(MODEL_IDS[index] for index in selected), expected)
    self.assertGreaterEqual(penalty, 0.0)
```

- [ ] **Step 2: Implement immutable evaluation data and split helpers.**

```python
@dataclass(frozen=True)
class EvaluationData:
    matrix: np.ndarray
    groups: Sequence[str]
    scores: np.ndarray       # (rows, 3)
    log_costs: np.ndarray    # (rows, 3)
    costs: np.ndarray        # (rows, 3)
    policy: RoutingPolicy

```

Implement `make_evaluation_data`, `require_group_disjoint`, and
`grouped_inner_folds(groups, seed)`. Build the matrix once with
`hash_regex.raw_feature_vector(episode, HASH_BINS)`. Build canonical groups from
`canonical_content_group(to_prompt_record(episode).text)`. Use
`repeated_outer_folds` for the fixed 3x5 schedule and
`grouped_folds(local_groups, folds=4, seed=seed)` for inner folds. Do not use
`train_hash_regex._oof_predictions`.

- [ ] **Step 3: Implement production-unit Lagrange and regret.**

Implement `lagrange_selection_and_penalty(scores, costs, multiplier)` and
`tier_regrets(scores, costs, policy)`. The former returns zero-based selected model
indices and the final bisection penalty; the latter returns a `(rows, 3)` regret
matrix and a mapping from every tier name to its penalty.

`lagrange_selection_and_penalty` mirrors `select_models`: compare `score - penalty * cost / light_total`, use `MODEL_IDS` order for ties, apply the official multiplier at safety 1.0, and perform the same 80 bisection iterations. `tier_regrets` derives a lambda for every tier from the fit partition's actual Train arrays, calculates nonnegative action regret in the same batch-light-total units, and combines tier regrets by official policy weights. It returns `(rows, 3)` regret and three reportable lambdas. Premium fill is intentionally excluded from lambda derivation but required in actual route/safety scoring.

- [ ] **Step 4: Prove no outer-test data enters inner construction.**

```python
def test_mutating_outer_test_scores_cannot_change_inner_lambdas() -> None:
    before = nested.inner_oof_bundle(data, outer_train=(0, 1, 2, 3), seed=137)
    after = nested.inner_oof_bundle(data.with_scores_changed_at((4, 5)), outer_train=(0, 1, 2, 3), seed=137)
    self.assertEqual(before["lambdas"], after["lambdas"])
```

- [ ] **Step 5: Run focused tests and commit.**

Run:

```powershell
$env:PYTHONPATH='src;baselines;tools'; python -m unittest tests.promptbudget.test_hash_regex_quality_nested -v
```

Expected: all grouped/lambda tests pass.

```powershell
git add tools/hash_regex_quality_nested.py tests/promptbudget/test_hash_regex_quality_nested.py
git commit -m "feat: add grouped hash-regex quality evaluation data"
```

### Task 3: Generate shared inner OOF predictions and candidate provisional safety

**Files:**

- Modify: `tools/hash_regex_quality_nested.py`
- Modify: `tests/promptbudget/test_hash_regex_quality_nested.py`

- [ ] **Step 1: Write failing cache, grid, and Premium-fill safety tests.**

```python
def test_cost_heads_are_cross_fit_once_per_inner_split_not_per_quality_candidate() -> None:
    with mock.patch.object(nested, "fit_log_cost_heads", wraps=nested.fit_log_cost_heads) as fit:
        nested.inner_oof_bundle(data, outer_train=tuple(range(12)), seed=137)
    self.assertEqual(4, fit.call_count)

def test_provisional_premium_safety_uses_runtime_ax31_fill() -> None:
    result = nested.select_provisional_safety(score_oof, cost_oof, data, tier="premium")
    self.assertTrue(result["includes_premium_fill"])
```

- [ ] **Step 2: Implement OOF bundle and full-policy scorer.**

Implement `inner_oof_bundle(data, outer_train, seed)`,
`route_complete_policy(score_rows, cost_rows, policy, tier, safety_ratio)`, and
`select_provisional_safety(score_rows, cost_rows, data, tier)`. The safety function
returns selected ratio, all-tier official metrics, and a boolean recording whether
Premium fill was applied.

For each inner fold, fit one unweighted cost model on inner-train and reuse its validation cost rows for every quality candidate/strength. Fit each candidate only on its inner-train rows; derive regret weights from those same rows. Collect full outer-train OOF score/cost predictions. `route_complete_policy` must call `select_models`, then `fill_ax31_upgrades` only for Premium. `select_provisional_safety` evaluates all 121 values through that complete route and the official scorer; it admits Fast `<=`, Balanced `<`, Premium `<` exactly as the scorer does.

- [ ] **Step 3: Implement deterministic candidate/strength selection.**

Implement `select_inner_configuration(data, outer_train, seed)`. It returns one
record for each real candidate containing its selected strength, provisional safety,
lambdas, route metrics, and either `admitted=True` or an explicit all-Light fallback.

For Raw use strength 0. For weighted uplift and regret-weighted Raw enumerate only their approved grids. First require all three OOF tier caps pass; among admitted values maximize official weighted final score; ties use lower maximum actual cost ratio then lower strength. If a candidate has no admitted strength, record its all-Light provisional policy. Return only inner-OOF-derived strength, safety, lambdas, route metrics, and no raw prompts/IDs.

- [ ] **Step 4: Run focused tests and commit.**

Run:

```powershell
$env:PYTHONPATH='src;baselines;tools'; python -m unittest tests.promptbudget.test_hash_regex_quality_nested -v
```

Expected: cost fits are shared; all selection/safety tests pass without touching outer-test targets.

```powershell
git add tools/hash_regex_quality_nested.py tests/promptbudget/test_hash_regex_quality_nested.py
git commit -m "feat: select hash-regex quality objectives by grouped oof"
```

### Task 4: Score outer folds and enforce the pre-registered winner gate

**Files:**

- Modify: `tools/hash_regex_quality_nested.py`
- Modify: `tests/promptbudget/test_hash_regex_quality_nested.py`

- [ ] **Step 1: Write failing full-allocator and gate tests.**

```python
def test_outer_fold_scores_each_real_candidate_with_all_three_tiers() -> None:
    report = nested.evaluate_outer_fold(data, outer_train=(0, 1, 2, 3), outer_test=(4, 5), seed=137, fold=0)
    self.assertEqual(set(nested.QUALITY_CANDIDATES), set(report["candidates"]))
    self.assertEqual(set(TIERS), set(report["candidates"]["raw"]["tiers"]))

def test_winner_rejects_a_candidate_with_one_cap_violation() -> None:
    self.assertEqual("raw", nested.choose_winner(report_with_one_failed_tier)["winner"])
```

- [ ] **Step 2: Implement outer refit, official submissions, and records.**

Implement `evaluate_outer_fold(data, outer_train, outer_test, seed, fold)` and
`choose_winner(folds)`. The first returns the three real candidate reports and one
Direct-uplift control report; the second returns the named winner and every passed or
failed gate.

For each real candidate, refit its selected quality objective and the fixed-family log-cost heads on all outer-train rows. Regret weights are recalculated only from outer-train actual arrays. Route outer-test once per candidate with its frozen provisional safety and the complete runtime policy. Construct all three `Submission` objects and use `score_submissions` so official Decimal cap semantics, including score-zero behavior, are not reimplemented.

Each candidate/fold report must contain unrounded final/tier score, actual cost ratio, budget pass, model counts, selected strength, safety, lambdas, paired score difference from Raw, and a content-group digest only. Direct uplift is run as a fit/prediction/route equality assertion, reported under `controls`, and cannot appear in `candidates`.

`choose_winner` selects a non-Raw candidate only if every one of 15 outer-fold/tier budget checks passes, at least two of three seed means are positive versus Raw, at least eight of 15 paired folds are positive, and mean paired weighted difference is at least `0.002`. Tie break: larger mean difference, larger minimum seed difference, lower maximum cost ratio, then `raw`, `weighted-uplift`, `regret-weighted-raw`. Otherwise return Raw with machine-readable failed gates.

- [ ] **Step 3: Run focused tests and commit.**

Run:

```powershell
$env:PYTHONPATH='src;baselines;tools'; python -m unittest tests.promptbudget.test_hash_regex_quality_nested -v
```

Expected: all outer scoring/gate tests pass.

```powershell
git add tools/hash_regex_quality_nested.py tests/promptbudget/test_hash_regex_quality_nested.py
git commit -m "feat: evaluate nested hash-regex quality candidates"
```

### Task 5: Add safe CLI, dry-run, and one-fold smoke report

**Files:**

- Create: `tools/evaluate_hash_regex_quality_nested.py`
- Create: `tests/test_evaluate_hash_regex_quality_nested.py`

- [ ] **Step 1: Write failing CLI boundary tests.**

```python
def test_evaluator_rejects_dev_path_before_loading() -> None:
    with self.assertRaises(ValueError):
        cli.evaluate(args_for("data/dev/inputs.json"))

def test_default_dry_run_writes_no_report() -> None:
    report = cli.evaluate(valid_train_args(execute=False))
    self.assertEqual("dry-run", report["mode"])
    self.assertFalse(report_path.exists())

def test_report_root_is_isolated() -> None:
    with self.assertRaises(ValueError):
        cli.require_output_path(Path("build/promptbudget-v3/report.json"))
```

- [ ] **Step 2: Implement the command contract.**

```text
--input data/materialized/train/inputs.json
--outcomes data/train/outcomes.json
--report build/hash-regex-quality/nested-evaluation.json
--execute --one-outer-fold | --full
```

Default mode prints a dry-run schedule/digests only and writes nothing. Execute mode requires exactly one run selector, validates Train split and row count 1,760, and writes only below `build/hash-regex-quality/` atomically. `--full` executes exactly the fixed 15 folds; it must not create an artifact, read Dev, or auto-finalize a winner.

- [ ] **Step 3: Run CLI tests, then a one-fold Train smoke report.**

Run:

```powershell
$env:PYTHONPATH='src;baselines;tools'; python -m unittest tests.test_evaluate_hash_regex_quality_nested -v
$env:PYTHONPATH='src;baselines;tools'; python tools/evaluate_hash_regex_quality_nested.py --input data/materialized/train/inputs.json --outcomes data/train/outcomes.json --report build/hash-regex-quality/nested-evaluation-smoke.json --execute --one-outer-fold
```

Expected: tests pass; smoke report has Raw/Weighted uplift/Regret-weighted candidates, controls, per-tier cap metrics, no raw prompt/episode ID, and no final winner claim.

- [ ] **Step 4: Commit CLI/test changes, not the generated smoke report.**

```powershell
git add tools/evaluate_hash_regex_quality_nested.py tests/test_evaluate_hash_regex_quality_nested.py
git commit -m "feat: add safe hash-regex quality evaluator cli"
```

### Task 6: Full evaluation, locked finalization, and one-pass Dev sanity

**Files:**

- Create only after full-evaluation approval: `tools/train_hash_regex_quality_final.py`
- Create only after full-evaluation approval: `tests/test_train_hash_regex_quality_final.py`
- Create only after full-evaluation approval: `tools/score_hash_regex_artifact.py`
- Create only after full-evaluation approval: `tests/test_score_hash_regex_artifact.py`
- Generated: `build/hash-regex-quality/nested-evaluation.json`
- Generated: `build/hash-regex-quality/finalization.json`
- Generated: `build/hash-regex-quality/final-artifact.json`
- Generated: `build/hash-regex-quality/dev-sanity.json`

- [ ] **Step 1: Obtain explicit approval before a full 3x5 execution.**

Do not run `--full` merely because this document exists. An explicit user request for a
complete/overnight run is the approval specified by the autonomous execution contract.
Otherwise, the approval must occur after focused tests and the one-fold smoke report
have been reviewed.

- [ ] **Step 2: Execute and inspect the full Train-only report.**

Run:

```powershell
$env:PYTHONPATH='src;baselines;tools'; python tools/evaluate_hash_regex_quality_nested.py --input data/materialized/train/inputs.json --outcomes data/train/outcomes.json --report build/hash-regex-quality/nested-evaluation.json --execute --full
```

Expected: 15 folds and a machine-readable winner/gate section. If Raw wins, or any
gate fails, the nested report itself has terminal status `raw-retained` and
`fallback_artifact: baselines/hash-regex-public.v1.json`. Do not create a candidate
artifact, `finalization.json`, or a Dev sanity report.

- [ ] **Step 3: Only if a non-Raw winner passes, write failing finalizer tests.**

```python
def test_locked_winner_accepts_only_admitted_nonraw_report(tmp_path: Path) -> None:
    report = tmp_path / "nested.json"
    report.write_text(json.dumps({"winner": {"name": "weighted-uplift", "all_gates_pass": True}}))
    self.assertEqual("weighted-uplift", finalizer.locked_winner_from_report(report))

def test_locked_winner_rejects_raw_failed_or_unknown_reports(tmp_path: Path) -> None:
    for winner in (
        {"name": "raw", "all_gates_pass": True},
        {"name": "weighted-uplift", "all_gates_pass": False},
        {"name": "unknown", "all_gates_pass": True},
    ):
        report = tmp_path / f"{winner['name']}-{winner['all_gates_pass']}.json"
        report.write_text(json.dumps({"winner": winner}))
        with self.assertRaises(ValueError):
            finalizer.locked_winner_from_report(report)

def test_final_artifact_roundtrips_existing_parser(finalized_artifact: Path) -> None:
    artifact = hash_regex.load_artifact(finalized_artifact)
    self.assertIsInstance(artifact, hash_regex.HashRegexArtifact)
    self.assertEqual(100.0, artifact.alpha)
    self.assertEqual(256, artifact.hash_bins)

def test_finalizer_rejects_dev_paths(tmp_path: Path) -> None:
    with self.assertRaises(ValueError):
        finalizer.validate_train_paths(Path("data/materialized/dev/inputs.json"), tmp_path / "outcomes.json")
```

- [ ] **Step 4: Implement a Train-only finalizer that consumes the locked winner config.**

Implement these public finalizer operations:

1. `locked_winner_from_report(report_path) -> str` accepts only an admitted
   `weighted-uplift` or `regret-weighted-raw` full nested report with every gate passed.
2. `locked_full_train_configuration(data, winner, seed=20260827)` calls the same
   inner-OOF selector on all Train rows and returns only that winner's pre-registered
   strength, provisional joint safety, and tier lambdas. It may not compare candidates
   again or use Dev.
3. `write_final_artifact(data, config, artifact_path)` fits quality and fixed-family
   cost heads on all Train, converts any uplift representation to raw score-head
   coefficients, and writes the exact `HashRegexArtifact` dictionary schema. Its
   `alpha` is `100.0`, `hash_bins` is `256`, and the existing parser must load it.
4. `write_finalization_report(status, train_input_path, train_outcome_path, nested_report_path, configuration, artifact_path, report_path)` records `status`, Train input/outcome SHA-256,
   nested-report SHA-256, winner, selected strength, safety, lambdas, artifact SHA-256,
   and the immutable public fallback path. Never serialize prompts, IDs, or row-level
   labels.

The finalizer CLI is exactly:

```powershell
$env:PYTHONPATH='src;baselines;tools'; python tools/train_hash_regex_quality_final.py --input data/materialized/train/inputs.json --outcomes data/train/outcomes.json --nested-report build/hash-regex-quality/nested-evaluation.json --artifact build/hash-regex-quality/final-artifact.json --report build/hash-regex-quality/finalization.json
```

It rejects Dev paths and reports outside `build/hash-regex-quality/`. It does not alter
the public artifact, runtime resources, or nested report. It exits nonzero on malformed
or non-admitted reports. It is never called for a valid Raw/no-gate winner because the
nested evaluator has already recorded that terminal `raw-retained` result.

- [ ] **Step 5: Freeze before Dev and apply the one-pass fallback rule.**

Write tests for the Dev scorer: it accepts a parser-loadable artifact, generates all
three official tiers, calls `score_submissions`, writes a report, and has no imports or
calls into any fitting, strength-selection, safety-selection, or artifact-writing
function. A malformed artifact or Dev path/outcome mismatch raises before scoring.

`tools/score_hash_regex_artifact.py` takes only `--input`, `--outcomes`, `--artifact`,
and `--report`. It uses `hash_regex.make_hash_regex_submission` for Fast, Balanced, and
Premium, invokes `score_submissions`, and writes unrounded score/cost/pass metrics plus
`artifact_sha256`. Its `sanity_status` is `pass` only when all three caps and all
technical/parser checks pass; otherwise it is `fallback-required`. It must never modify
the artifact or generate a replacement.

Run it exactly once after the finalizer and focused tests pass:

```powershell
$env:PYTHONPATH='src;baselines;tools'; python tools/score_hash_regex_artifact.py --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json --artifact build/hash-regex-quality/final-artifact.json --report build/hash-regex-quality/dev-sanity.json
```

On `fallback-required`, retain the report and use exactly
`baselines/hash-regex-public.v1.json`; do not change parameters or rerun Dev. A lower
Dev score or a different route mix with `sanity_status=pass` is logged only and cannot
change the frozen candidate.

- [ ] **Step 6: Verify source/test work before integration.**

Run:

```powershell
$env:PYTHONPATH='src;baselines;tools'; python -m unittest tests.test_hash_regex_baseline tests.promptbudget.test_hash_regex_quality_nested tests.test_evaluate_hash_regex_quality_nested tests.test_train_hash_regex_quality_final tests.test_score_hash_regex_artifact -v
```

Also verify that the final artifact runs through the unchanged baseline entry point for
each tier and that the Dev sanity report has no fitting/calibration fields. This task
does not modify a packaged runtime, so the relevant runtime verification is existing
parser/permutation coverage plus three-tier submission construction, not a container
deployment check. Generated reports/artifacts are not committed unless the user
explicitly asks for them to be versioned.

## Plan self-review

- Direct uplift is a control, not a competing candidate.
- Cost family is fixed but refit in every inner/outer training partition.
- Every candidate uses the exact runtime allocator and Premium fill.
- Outer-test labels never enter fit, lambda, strength, safety, or tie-break selection.
- Final joint calibration is Train-only; Dev has a cap/technical-failure fallback only.
- No task adds exact knapsack, new feature family, new router family, or runtime artifact schema.

## Execution handoff

Plan complete. After reviewing the decision document and this plan, choose one execution mode:

1. **Subagent-Driven (recommended):** invoke `subagent-driven-development` and review each task before the next.
2. **Inline execution:** invoke `executing-plans` and perform the checked tasks in order with checkpoints.
