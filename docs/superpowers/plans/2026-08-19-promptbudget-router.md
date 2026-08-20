# PromptBudget Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, prompt-only model router that emits one valid challenge submission for a supplied tier, with reproducible training, calibration, artifacts, and release checks.

**Architecture:** Keep challenge-wire parsing in `ossp_router.protocol` and add `promptbudget` as a closed runtime package.  The adapter transforms an `Episode` into content-only text, feature and policy functions accept only that text and a validated artifact, and the runtime reattaches the opaque episode ID only when assembling the submission.  Development tools may use NumPy to fit models but the shipped runtime uses only the Python standard library.

**Tech Stack:** Python 3.9+ standard library at runtime; NumPy (pinned development-only dependency) for ridge fitting; unittest; Docker linux/arm64.

---

## File map

| Path | Responsibility |
| --- | --- |
| `src/promptbudget/schema.py` | Artifact, tier, model and prediction value contracts. |
| `src/promptbudget/input_adapter.py` | Protocol `Episode` conversion, output assembly, JSON validation boundary. |
| `src/promptbudget/text_features.py` | NFC whitespace normalization, structural/dense features, stable signed sparse hashing. |
| `src/promptbudget/linear.py` | Standard-library sparse/dense ridge-head scoring. |
| `src/promptbudget/artifact.py` | Strict artifact loading, canonical SHA-256 manifest and atomic serialization. |
| `src/promptbudget/policy.py` | Quality/cost estimates and tier-scoped, deterministic model choice. |
| `src/promptbudget/runtime.py` | One-tier execution and atomic challenge output. |
| `tools/*.py` | Train OOF, calibration, evaluation, artifact build, audit and release verification commands. |
| `tests/promptbudget/` | Unit, integration and compliance coverage for the new policy. |
| `artifacts/promptbudget-v1/` | Versioned artifact, manifest and notices copied into the image. |

### Task 1: Freeze G0 protocol facts and the content-only boundary

**Files:**
- Modify: `docs/skt/07_promptbudget_router_design_spec.md` in the planning workspace
- Create: `src/promptbudget/schema.py`
- Create: `src/promptbudget/input_adapter.py`
- Test: `tests/promptbudget/test_input_adapter.py`

- [ ] **Step 1: Write failing protocol-boundary tests.**

```python
def test_messages_are_flattened_with_role_boundaries():
    record = to_prompt_record(Episode("opaque", messages=(Message("system", "A"),)))
    self.assertEqual("<role>system</role>\nA", record.text)

def test_output_key_never_enters_feature_or_policy_arguments():
    self.assertEqual("ax31-light", select_model("hello", "fast", artifact))
```

- [ ] **Step 2: Run the test and confirm import failure.**

Run: `$env:PYTHONPATH='src'; python -m unittest tests.promptbudget.test_input_adapter -v`

Expected: `ModuleNotFoundError: promptbudget`.

- [ ] **Step 3: Implement immutable `PromptRecord`, model/tier constants, exact role flattening and submission assembly.**

```python
def to_prompt_record(episode: Episode) -> PromptRecord:
    text = episode.prompt if episode.prompt is not None else "\n".join(
        f"<role>{message.role}</role>\n{message.content}" for message in episode.messages or ()
    )
    return PromptRecord(text=text, output_key=episode.episode_id)
```

- [ ] **Step 4: Correct the design document to say the frozen v1 baseline rejects records containing both `prompt` and `messages`; add this fixture as the G0 evidence.**

- [ ] **Step 5: Re-run the focused tests.**

Run: `$env:PYTHONPATH='src'; python -m unittest tests.promptbudget.test_input_adapter -v`

Expected: `OK`.

### Task 2: Implement deterministic text features

**Files:**
- Create: `src/promptbudget/text_features.py`
- Test: `tests/promptbudget/test_text_features.py`

- [ ] **Step 1: Write failing fixtures for NFC/whitespace normalization, signed BLAKE2b hashing, and order-independent vectors.**

```python
def test_hash_is_process_independent():
    self.assertEqual(vectorize("e\u0301", 1 << 16), vectorize("é", 1 << 16))

def test_signed_hash_uses_only_text():
    self.assertEqual(vectorize("same", 1 << 16), vectorize("same", 1 << 16))
```

- [ ] **Step 2: Run the new test and confirm it fails because the feature module is absent.**

- [ ] **Step 3: Add dense structural features and sparse word/character n-grams.**

```python
digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, person=b"PBRouter").digest()
value = int.from_bytes(digest, "big")
index = value & (dimension - 1)
sign = -1.0 if value >> 63 else 1.0
```

- [ ] **Step 4: Re-run the focused test.**

Expected: `OK`; no Python `hash()` use in the module.

### Task 3: Add artifact, linear scoring and policy selection

**Files:**
- Create: `src/promptbudget/artifact.py`
- Create: `src/promptbudget/linear.py`
- Create: `src/promptbudget/policy.py`
- Test: `tests/promptbudget/test_artifact_policy.py`

- [ ] **Step 1: Write failing tests for manifest mismatch, coefficient length validation, clamp, cost upper bound, minimum gain and stable tie breaking.**

```python
def test_tie_break_prefers_lower_cost_then_model_order():
    self.assertEqual("ax31-light", select_model("x", "fast", tied_artifact))

def test_manifest_mismatch_refuses_runtime_load():
    with self.assertRaises(ArtifactError):
        load_artifact(artifact_path, manifest_path)
```

- [ ] **Step 2: Run the focused test and observe failures.**

- [ ] **Step 3: Implement canonical JSON SHA-256, sparse linear dot products and the pure `select_model(text, tier, artifact)` function.**

```python
utility = quality - tier.lambda_cost * cost_upper
return min(candidates, key=lambda item: (-item.utility, item.cost_upper, MODEL_ORDER.index(item.model_id))).model_id
```

- [ ] **Step 4: Re-run the focused test.**

Expected: `OK`.

### Task 4: Wire the submission runtime and a safe bundled artifact

**Files:**
- Create: `src/promptbudget/runtime.py`
- Create: `artifacts/promptbudget-v1/artifact.json`
- Create: `artifacts/promptbudget-v1/manifest.json`
- Modify: `src/ossp_router/heuristic.py`
- Modify: `container/entrypoint.py`
- Modify: `container/Dockerfile`
- Test: `tests/promptbudget/test_runtime_integration.py`

- [ ] **Step 1: Write an end-to-end toy test that calls `router-run` twice, permutes records, and validates output with `load_submission`.**

- [ ] **Step 2: Confirm the test fails before the runtime exists.**

- [ ] **Step 3: Implement atomic temporary-file replacement and bridge the existing `router-run` console script to `promptbudget.runtime.main`.**

```python
with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".submission-", delete=False) as stream:
    stream.write(payload)
os.replace(stream.name, output)
```

- [ ] **Step 4: Generate a conservative all-light fallback artifact with a verified manifest.  It is a runnable safety baseline, not the selected Train/Dev artifact.**

- [ ] **Step 5: Re-run the integration test.**

Expected: two byte-identical valid submissions; no temporary files remain.

### Task 5: Build reproducible training, calibration and evaluation tools

**Files:**
- Create: `tools/validate_data.py`
- Create: `tools/train_oof.py`
- Create: `tools/calibrate_policy.py`
- Create: `tools/evaluate_policy.py`
- Create: `tools/build_artifact.py`
- Create: `tools/requirements-train.txt`
- Test: `tests/promptbudget/test_training_pipeline.py`

- [ ] **Step 1: Write a tiny public fixture test for `validate_data → train_oof → calibrate_policy → build_artifact → router-run`.**

- [ ] **Step 2: Run it and confirm the tools are missing.**

- [ ] **Step 3: Implement fold-stable fitting from Train outcomes only, OOF score/cost reports, fixed candidate grids, strict tier-cost rejection and artifact construction.**

```python
if any(result.actual_cost_ratio >= result.budget_multiplier for result in candidate.tiers):
    candidate.status = "fail"
```

- [ ] **Step 4: Re-run the fixture pipeline.**

Expected: `OK`, no Train/Dev text, IDs or outcomes in the emitted artifact.

### Task 6: Add compliance and release validation

**Files:**
- Create: `tools/audit_runtime.py`
- Create: `tools/verify_release.py`
- Create: `reports/README.md`
- Modify: `README.md`
- Test: `tests/promptbudget/test_compliance.py`

- [ ] **Step 1: Write failing tests that scan runtime policy inputs, exercise ID/order permutation, reject invalid tier/artifact and check expected report fields.**

- [ ] **Step 2: Implement static-policy audit, report creation and release evidence validation.**

- [ ] **Step 3: Run focused compliance tests.**

Expected: `OK`.

### Task 7: Materialize public data and select the final artifact

**Files:**
- Modify: `artifacts/promptbudget-v1/artifact.json`
- Modify: `artifacts/promptbudget-v1/manifest.json`
- Create: `reports/dev_policy_comparison.json`
- Create: `reports/dev_policy_comparison.md`

- [ ] **Step 1: Create the documented data environment and materialize the official Train and Dev inputs.**

Run: `python -m pip install -r data/sources/requirements-materialize-public-data.txt` then `python tools/materialize_public_data.py`.

- [ ] **Step 2: Verify every materialized file SHA-256 against `data/public-data.v1.json`.**

- [ ] **Step 3: Run OOF training and fixed-grid Dev calibration, then compare all-light, hash-regex, absolute-linear and delta-linear.**

- [ ] **Step 4: Promote only the highest weighted candidate whose Fast, Balanced and Premium cost is strictly below its tier limit; otherwise retain the conservative artifact and record the failure.**

- [ ] **Step 5: Build the selected artifact and re-run the Train/Dev report.**

### Task 8: Verify the release candidate

**Files:**
- Modify: `NOTICE` and `SBOM.spdx.json` only if new development or runtime dependencies require disclosure
- Create: `submission-ossp-skt.json` only after the user supplies the public fork URL and pushed image digest

- [ ] **Step 1: Run all platform-applicable unit, integration and compliance tests.**

- [ ] **Step 2: Build and execute the `linux/arm64` image with `tools/check_runtime.py` on an ARM64 Docker host.**

- [ ] **Step 3: Run `tools/verify_release.py` and `tools/validate_technical_submission.py` once public repository URL, commit SHA and immutable image digest exist.**

- [ ] **Step 4: Commit each completed cohesive task; do not create the final submission JSON or claim G6 until external release evidence exists.**

## Self-review

- **Spec coverage:** Tasks 1–4 cover the runtime data boundary, features, artifact, policy, output contract and container. Tasks 5 and 7 cover Train-only fitting, OOF, Dev-only grid calibration and final candidate choice. Tasks 6 and 8 cover determinism, banned signals, container and release evidence.
- **Scope:** No answer generation, online learning, batch budget allocation, embeddings or kNN implementation is included. kNN remains intentionally deferred until the linear candidates have a measured safe improvement.
- **Dependencies:** Public-data materialization, NumPy and ARM64 Docker are isolated to Task 7/8. The safe runtime baseline and all toy tests do not need them.
- **Verification:** Every task starts with a failing test and ends with a focused success command; final G5/G6 commands require the stated external platform/release data.
