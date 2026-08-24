<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# PromptBudget v2 Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement leakage-safe v2 candidate selection, Fast 0.10 budget-margin admission, Dev-only operational calibration, and a single-use locked-holdout evaluator with an append-only SQLite ledger.

**Architecture:** Add pure grouped-CV and candidate-selection helpers under `promptbudget` so training can fit transforms only on each inner-train partition and produce machine-readable provenance. Keep Dev calibration structurally frozen and make it retain the v1 settings when Fast cannot prove its margin. The only holdout-scoring CLI will reserve a canonical holdout digest in SQLite before reading outcomes and will record append-only reservation/completion/failure events.

**Tech Stack:** Python 3.9+, standard library plus NumPy for training, SQLite (`sqlite3`), unittest.

---

## File map

| Path | Responsibility |
| --- | --- |
| `src/promptbudget/safety.py` | Content-grouping, deterministic grouped folds, safe candidate admission/one-SE choice, bucketed conformal and aggregate upper ratio helpers. |
| `src/promptbudget/locked_eval.py` | Canonical holdout digest, append-only ledger schema, provenance capture and locked evaluation core. |
| `tools/train_oof.py` | Nested grouped-CV training report and final artifact fitting. |
| `tools/calibrate_policy.py` | Frozen-structure Dev calibration certificate with Fast fallback. |
| `tools/evaluate_locked.py` | Sole CLI entry point for outcome-bearing locked evaluations. |
| `tests/promptbudget/test_safety.py` | Regression coverage for group separation, 0.10 admission, one-SE order and bucket fallback. |
| `tests/promptbudget/test_locked_eval.py` | Ledger reservation, concurrency, failed-run, provenance and mutation guards. |
| `docs/PROMPTBUDGET_V2_OPERATIONS.md` | Artifact/report paths, local-ledger threat boundary and operator procedure. |

### Task 1: Add deterministic safety primitives and tests

**Files:**
- Create: `src/promptbudget/safety.py`
- Create: `tests/promptbudget/test_safety.py`

- [ ] Write failing tests asserting that the same content hash never occurs in both sides of any inner/outer split; an upper ratio of `tier_cap - Decimal("0.095792")` is rejected for Fast; candidates must pass every validation fold; and a small `korean`, `code`, or `long` bucket returns the global multiplier.
- [ ] Run: `PYTHONPATH=src python3 -m unittest tests.promptbudget.test_safety -v`; expect the module import to fail.
- [ ] Implement canonical text/content SHA-256 grouping, three fixed-seed repeated outer five-fold splits, inner four-fold splits derived only from the outer-train indices, `admit_fast_candidate`, grouped standard-error aggregation and a deterministic key `(active_features, family_rank, residual_multiplier, grid_index)` for the one-SE set.
- [ ] Implement an explicit aggregate upper-ratio calculation that takes the maximum of per-request conformal upper-cost ratio and the group-level one-sided ratio upper bound; do not use raw `budget_ratio` as an admission substitute.
- [ ] Re-run the focused test; expect `OK`.

### Task 2: Make training selection nested and provenance-rich

**Files:**
- Modify: `tools/train_oof.py`
- Modify: `src/promptbudget/artifact.py`
- Modify: `tests/promptbudget/test_safety.py`

- [ ] Add failing tiny-fixture tests proving that sparse selection, ridge fitting and residual quantiles receive only inner-train rows; an outer outcome sentinel cannot reach candidate-selection functions; and the report includes seed, fold, group and row counts.
- [ ] Run the focused test and observe the intended assertion failure.
- [ ] Replace positional folds with content-grouped nested folds. For each outer fold, select the candidate solely from its inner folds, re-fit on outer-train, score only outer-test, and retain outer predictions exclusively for the comparison report.
- [ ] Fit the final artifact using the same grouped selection procedure over all Train data, then store structure, quantile index, target miscoverage, bucket threshold/global fallback and split provenance in `training_provenance`. Preserve strict canonical artifact parsing by storing only JSON-safe provenance.
- [ ] Add fixed v1 comparison metrics: group-clustered paired bootstrap quality CI, fixed bootstrap seed/count and primary endpoint metadata. Do not let any comparison output alter final selection.
- [ ] Re-run focused tests; expect `OK`.

### Task 3: Freeze Dev to calibration and enforce Fast fallback

**Files:**
- Modify: `tools/calibrate_policy.py`
- Modify: `tests/promptbudget/test_safety.py`

- [ ] Write failing tests that use a `0.095792` Fast margin and a Dev-fast failure fixture. The former must not select a candidate; the latter must preserve the draft Fast settings and label the v1 fallback.
- [ ] Run the test and confirm failure before behavior changes.
- [ ] Restrict Dev candidates to safety multipliers for the already selected artifact settings. Compute both the conformal and aggregate upper ratios and require Fast `<= cap - 0.10`; do not use quality, family, feature count, alpha, residual family or grid selection from Dev.
- [ ] Emit `promptbudget-dev-calibration-certificate-v2` with input/outcome/artifact/policy hashes, frozen-structure declaration, operational-only limitation, selected multiplier or fallback, and per-tier safety evidence. Write artifact only after the certificate inputs are captured.
- [ ] Re-run focused tests; expect `OK`.

### Task 4: Add the append-only locked evaluator

**Files:**
- Create: `src/promptbudget/locked_eval.py`
- Create: `tools/evaluate_locked.py`
- Create: `tests/promptbudget/test_locked_eval.py`

- [ ] Write failing tests for canonical digest stability, one successful concurrent reservation, rejection after a failed reservation, rejection when only an artifact changes, immutable ledger rows, and a direct core call without a reservation token.
- [ ] Run: `PYTHONPATH=src python3 -m unittest tests.promptbudget.test_locked_eval -v`; expect the module import to fail.
- [ ] Implement `holdout_digest` as a domain-separated SHA-256 over input/outcome bytes, schema/version and canonical episode/content-group manifest. Initialize a SQLite schema containing reservation and event tables plus triggers that abort update/delete.
- [ ] In `evaluate_locked`, record a `reserved` event inside `BEGIN IMMEDIATE` before outcomes are loaded/scored. On success append `completed`; on any post-reservation exception append `failed` and leave the reservation in place. Disallow evaluator code paths that call scoring without a verified reservation.
- [ ] Capture UTC time, digest components, artifact/reference/policy/report hashes, Git commit and clean state, evaluator source/tree hash, argument list, interpreter and dependency versions. Verify input artifact/report byte hashes before and after scoring and atomically replace the output report.
- [ ] Re-run focused tests; expect `OK`.

### Task 5: Document, inspect, and verify the complete path

**Files:**
- Create: `docs/PROMPTBUDGET_V2_OPERATIONS.md`
- Modify: `README.md`
- Test: `tests/promptbudget/test_safety.py`, `tests/promptbudget/test_locked_eval.py`

- [ ] Document the Train/Dev/locked boundary, report and ledger commands, that a local SQLite ledger does not resist a user who can modify its files, and backup/access-control responsibilities.
- [ ] Add the only supported locked-evaluation command to the README; do not advertise a direct outcome-scoring CLI.
- [ ] Run targeted v2 tests, then the platform-applicable full suite using the project Python environment. Record any Windows-only test limitations separately from code failures.
- [ ] Run `git diff --check` and inspect artifact/report paths against the operations document before integration.

## Self-review

- **Spec coverage:** Tasks 1–2 implement grouped nested CV, outer-test isolation, conformal/aggregate gates and comparison reporting. Task 3 separates operational Dev calibration from structural selection. Task 4 provides the reserved-only locked path and append-only audit ledger. Task 5 documents the local WORM limitation and validates user-visible/runtime paths.
- **Non-goals:** No external WORM service, Dev quality selection, runtime tier promotion or break-glass tool is added.
- **Verification:** Every behavioral addition begins with a failing unittest; final verification includes focused tests, a full applicable suite and `git diff --check`.
