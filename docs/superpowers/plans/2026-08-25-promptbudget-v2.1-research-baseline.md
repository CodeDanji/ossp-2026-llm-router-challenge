# PromptBudget v2.1 Research Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the leakage-controlled v2.1 absolute-linear research artifact, Train-only policy selection, diagnostics, and one-time exploratory Dev confirmation.

**Architecture:** Keep v1 artifact parsing and the default runtime resource unchanged. Add a versioned v2 artifact cost-calibration payload, reuse the existing standard-library policy/runtime and SQLite ledger patterns, and replace the v2 training script's raw-MSE selection with Train-only nested policy scoring.

**Tech Stack:** Python 3.9+, NumPy training-only, standard library runtime, SQLite, unittest.

---

## File map

| Path | Responsibility |
| --- | --- |
| `src/promptbudget/artifact.py` | Read/write v1 and v2 artifacts, including model×bucket cost calibration. |
| `src/promptbudget/policy.py` | Select the length bucket and apply the persisted cost multiplier. |
| `src/promptbudget/safety.py` | Fixed bucket names, conservative policy tie-break, and reusable calibration helpers. |
| `tools/train_oof.py` | Train-only nested selection, direct monetary calibration, aggregate reports. |
| `tools/evaluate_locked.py` | Reserve and write the append-only exploratory Dev confirmation. |
| `tests/promptbudget/test_artifact_policy.py` | Artifact compatibility and bucketed runtime-cost assertions. |
| `tests/promptbudget/test_safety.py` | Calibration and deterministic conservative-choice assertions. |
| `tests/promptbudget/test_train_oof_v2.py` | Train-only selection and outer-test isolation fixtures. |
| `tests/promptbudget/test_locked_eval.py` | One-use Dev reservation-key assertions. |
| `docs/PROMPTBUDGET_V2_OPERATIONS.md` | v2.1 artifact, Train and Dev operator commands. |

### Task 1: Add v2 artifact calibration contract

**Files:**
- Modify: `src/promptbudget/artifact.py`
- Modify: `src/promptbudget/policy.py`
- Test: `tests/promptbudget/test_artifact_policy.py`

- [ ] Add a failing test that writes a v2 artifact with `{"ax31": {"global": 1.5, "short": 1.1, "medium": 1.2, "long": 1.3}}`, reloads it, and asserts short/medium/long prompts apply their respective multiplier while a v1 artifact still reloads unchanged.

```python
self.assertEqual(1.1, prediction_for("x" * 512).c_upper_multiplier)
self.assertEqual(1.2, prediction_for("x" * 513).c_upper_multiplier)
self.assertEqual(1.3, prediction_for("x" * 2049).c_upper_multiplier)
```

- [ ] Run `PYTHONPATH=src python -m unittest tests.promptbudget.test_artifact_policy -v`; expect failure because v2 calibration is unknown.
- [ ] Add the minimal `cost_calibration` mapping to `PromptBudgetArtifact`, accept artifact versions 1 and 2, and serialize a v2 manifest using its matching format version. Retain v1 field validation without alteration. Add a private policy helper that maps character count to `short`, `medium`, or `long` and falls back to `global`.
- [ ] Re-run the focused test; expect `OK`.
- [ ] Commit `feat: support PromptBudget v2 cost calibration`.

### Task 2: Make direct monetary OOF calibration testable

**Files:**
- Modify: `src/promptbudget/safety.py`
- Test: `tests/promptbudget/test_safety.py`

- [ ] Add failing tests for exact length boundaries, the higher empirical 99th percentile, per-bucket 100-row fallback to global, and rejection of non-finite/non-positive costs.

```python
values, fallback = monetary_cost_multipliers(
    predicted=[1.0] * 101, actual=[1.2] * 101,
    character_counts=[512] * 101, quantile=0.99, minimum_samples=100,
)
self.assertEqual(1.2, values["short"])
```

- [ ] Run `PYTHONPATH=src python -m unittest tests.promptbudget.test_safety -v`; expect the new helper import to fail.
- [ ] Implement the helper using only `math` and sorted values. It must expose model-independent bucket logic and return a global fallback marker; callers create one mapping per model.
- [ ] Re-run the focused test; expect `OK`.
- [ ] Commit `feat: add monetary OOF calibration helpers`.

### Task 3: Replace raw-MSE selection with inner policy scoring

**Files:**
- Modify: `tools/train_oof.py`
- Modify: `tests/promptbudget/test_train_oof_v2.py`

- [ ] Add failing fixtures where lower raw MSE loses to higher official policy score, an outer-test outcome sentinel cannot affect final head/policy/calibration selection, and a candidate that exceeds one validation-fold tier bound falls back to v1-all-light.

```python
chosen = select_tier_policy(candidates, tier="fast")
self.assertEqual("v1-all-light", chosen.fallback)
```

- [ ] Run `PYTHONPATH=src python -m unittest tests.promptbudget.test_train_oof_v2 -v`; expect missing selection helper failures.
- [ ] Enumerate the specified common-head and per-tier grids exactly. For each inner fit, make grouped cross-fit predictions only from inner-fit rows, derive model×bucket 99% multipliers, score the validation decisions with the frozen scorer, and admit only the specified Fast/Balanced/Premium upper-cost ratios. Fit selected heads on each outer-train, write outer outcomes only to the report, and perform final seed-137 selection plus separate seed-137 OOF calibration on all Train rows.
- [ ] Implement one-SE ordering with grouped mean upgrade fraction, max relative cost, higher minimum gain, higher lambda, then grid index. Use `TierSettings(..., safety_multiplier=1.0, ...)` and assign one common min gain to both non-Light models.
- [ ] Re-run the focused test; expect `OK`.
- [ ] Commit `feat: select PromptBudget v2.1 policies on Train only`.

### Task 4: Emit bounded research diagnostics

**Files:**
- Modify: `tools/train_oof.py`
- Test: `tests/promptbudget/test_train_oof_v2.py`

- [ ] Add failing tests asserting the report contains candidate/gate, calibration, routing, comparison and claim-label sections but no source prompt, episode identifier, or row-level outcome.

```python
self.assertFalse(any("prompt" in key for key in report["diagnostics"]))
self.assertFalse(report["claims"]["external_generalization_claim"])
```

- [ ] Run the focused test; expect an assertion failure for the absent sections.
- [ ] Write only aggregate metrics and digests: fixed 15 outer folds; model quality errors; p50/p90/p99 cost ratios; calibration fallback counts; tier model distributions, actual cost, upgrade gain/loss and oracle regret; 10,000-repetition group paired-bootstrap CI with seed `20260825`; and the four required labels.
- [ ] Write artifact, manifest, and report only below `build/promptbudget-v2.1/`; reject output paths outside that root before writing.
- [ ] Re-run the focused test; expect `OK`.
- [ ] Commit `feat: report PromptBudget v2.1 research diagnostics`.

### Task 5: Add one-time exploratory Dev confirmation

**Files:**
- Modify: `tools/evaluate_locked.py`
- Modify: `tests/promptbudget/test_locked_eval.py`
- Modify: `docs/PROMPTBUDGET_V2_OPERATIONS.md`

- [ ] Add a failing test that a reservation key derived from artifact, manifest, Dev input, and Dev outcome hashes cannot be reserved twice, and that modifying either artifact or manifest causes confirmation rejection before scoring.

```python
digest = dev_confirmation_digest(artifact, manifest, dev_input, dev_outcomes)
self.assertRaises(LockedEvaluationError, ledger.reserve, digest, metadata)
```

- [ ] Run `PYTHONPATH=src python -m unittest tests.promptbudget.test_locked_eval -v`; expect the v2.1 Dev digest function to be absent.
- [ ] Add an explicit `--exploratory-dev-confirmation` entry point that only accepts a v2.1 artifact, reserves the key before reading/scoring outcomes, checks artifact/manifest hashes before and after, appends the observed-split label and reservation id, and never writes an artifact. Reuse the existing SQLite append-only ledger.
- [ ] Document the exact Train and single Dev commands, build-only output root, and that Dev cannot alter any claim.
- [ ] Re-run the focused test; expect `OK`.
- [ ] Commit `feat: add one-time PromptBudget v2.1 Dev confirmation`.

### Task 6: Verify the full implementation

**Files:**
- Test: `tests/promptbudget/test_artifact_policy.py`
- Test: `tests/promptbudget/test_safety.py`
- Test: `tests/promptbudget/test_train_oof_v2.py`
- Test: `tests/promptbudget/test_locked_eval.py`

- [ ] Run the four targeted suites: `PYTHONPATH=src python -m unittest tests.promptbudget.test_artifact_policy tests.promptbudget.test_safety tests.promptbudget.test_train_oof_v2 tests.promptbudget.test_locked_eval -v`.
- [ ] Run `PYTHONPATH=src python -m unittest discover -s tests -v` and record only known Windows compatibility failures separately.
- [ ] Run `git diff --check`, verify the default runtime resource hash is unchanged, and inspect `build/promptbudget-v2.1/` as the only new artifact destination.
- [ ] Commit `test: verify PromptBudget v2.1 research baseline` if verification documentation changes.

## Self-review

- **Spec coverage:** Tasks 1–2 implement v2-only bucketed 99% monetary calibration and v1 compatibility. Tasks 3–4 enforce train-only nested policy selection, precise gates, diagnostics, and report-only outer tests. Task 5 supplies the one-use exploratory Dev confirmation. Task 6 proves the requested boundaries.
- **Scope:** No delta-quality model, runtime promotion, default artifact replacement, or deployment work is included.
- **Type consistency:** `cost_calibration` is a model→bucket→positive-float mapping throughout; v1 artifacts continue to use their existing model multiplier mapping.
