# PromptBudget v3 Delta-Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` with focused TDD checks.

**Goal:** Add a research-only `delta-linear` PromptBudget artifact family that routes from direct Light-relative quality, win probability, and incremental-cost predictions.

**Architecture:** Keep the v1/v2 `absolute-linear` artifact and selection path byte-compatible. Add a versioned delta artifact schema and dispatch runtime selection by family; Train/Dev tools write only below `build/promptbudget-v3/` and never copy into `src/promptbudget/resources/`.

**Tech Stack:** Python stdlib, existing NumPy training dependency, existing prompt-only hashed features and grouped-fold helpers.

---

### Task 1: Versioned delta artifact and deterministic runtime selection

**Files:**

- Modify: `src/promptbudget/artifact.py`
- Modify: `src/promptbudget/policy.py`
- Modify: `tests/promptbudget/test_artifact_policy.py`

- [ ] **Step 1: Add failing contract tests.** Construct a `delta-linear` artifact with heads only for `ax31` and `axk1-think`, then assert a round trip succeeds, malformed probabilities/non-finite values/missing heads fail, and each individual gate (`p_win`, delta gain, relative-cost upper, utility) selects Light.

- [ ] **Step 2: Run the focused test.**

  Run: `$env:PYTHONPATH='src'; & '<bundled-python>' -m unittest tests.promptbudget.test_artifact_policy -v`

  Expected: failures because no delta schema or runtime path exists.

- [ ] **Step 3: Implement the smallest schema and selector.** Add `DeltaTierSettings` (`lambda_cost`, `min_win_probability`, `min_delta_gain`, `max_relative_cost`) and a version-3 `DeltaLinearArtifact` containing the two upgrade heads for delta quality, probability, incremental relative cost, and per-model bucket residual offsets. Parse version 1/2 exactly as today and version 3 only when its exact field set validates. In `policy.py`, calculate `r_upper = max(0, r_hat + residual)`; retain Light unconditionally; choose the eligible upgrade by `(-utility, r_upper, MODEL_IDS index)`.

- [ ] **Step 4: Re-run the focused test.** Expected: pass, including prompt+ tier determinism.

### Task 2: Train-only delta data, head fitting, and research report

**Files:**

- Create: `tools/train_delta.py`
- Modify: `src/promptbudget/safety.py`
- Create: `tests/promptbudget/test_train_delta.py`

- [ ] **Step 1: Add failing focused tests.** Verify paired targets use outcome score/cost differences, direct delta ridge matches subtracting same-spec absolute ridge, undersized length buckets use global residual offsets, and all artifact/report outputs outside `build/promptbudget-v3/` are rejected.

- [ ] **Step 2: Run the new test module.** Expected: failure because the tool and delta calibration helpers do not exist.

- [ ] **Step 3: Implement the Train tool.** Reuse feature extraction, grouped folds, and NumPy ridge routines. Select per-head `(sparse feature count, alpha)` from fixed grids by inner grouped folds; fit a deterministic logistic head for win probability; form cross-fit residual offsets at the selected 0.90/0.95/0.99 quantile with the documented 100-row bucket fallback. Report signal metrics, cost coverage/slack, tier policy comparison, grouped-bootstrap provenance, and an explicit Go/No-Go result. Do not run Dev calibration or touch runtime resources when Go is false.

- [ ] **Step 4: Re-run the focused test.** Expected: pass without public data materialization.

### Task 3: Public-Dev policy calibration and operational boundary

**Files:**

- Create: `tools/calibrate_delta_policy.py`
- Modify: `docs/PROMPTBUDGET_V2_OPERATIONS.md`
- Modify: `tests/promptbudget/test_runtime_integration.py`

- [ ] **Step 1: Add failing contract tests.** Assert calibration rejects non-v3 output paths, searches tier knobs only (not head grids), evaluates actual cost for admission, and a permuted input selects the same model per prompt/tier from a delta artifact.

- [ ] **Step 2: Run the two focused modules.** Expected: calibration import/contract failures.

- [ ] **Step 3: Implement Dev calibration.** Load a Train-frozen delta artifact, sweep only tier settings, require Fast `<=1.15` and Balanced/Premium strict caps using actual public-Dev cost, then write the calibrated artifact/manifest/report below `build/promptbudget-v3/`. Add the matching Train and Dev commands to the operations document and state that the bundled default artifact is unchanged.

- [ ] **Step 4: Re-run focused modules and the PromptBudget suite.** Expected: all pass.

### Task 4: Focused verification and review

**Files:**

- Verify: `tests/promptbudget/`
- Verify: `src/promptbudget/resources/artifact.json`

- [ ] **Step 1:** Run the focused delta modules plus the full `tests/promptbudget` suite using the bundled Python runtime.
- [ ] **Step 2:** Confirm `git diff -- src/promptbudget/resources/artifact.json src/promptbudget/resources/manifest.json` is empty and `git status --short` contains no generated `build/` output.
- [ ] **Step 3:** Review the final diff against the v3 design’s artifact, deterministic-runtime, and output-location acceptance criteria.
