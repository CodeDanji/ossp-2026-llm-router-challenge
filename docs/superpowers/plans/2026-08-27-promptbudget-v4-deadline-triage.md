# PromptBudget v4 Deadline Triage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, with a fixed Train-only grouped screen, whether extending the existing AX31 fill to Fast and/or Balanced materially improves frozen v3.3 without weakening actual-cost safety.

**Architecture:** A new pure module reconstructs the complete guarded v3.3 route and exposes selected models, so it can calculate row-level observed diagnostics and score three fill variants through the official scorer. A thin CLI fixes the four grouped Train folds and writes one isolated JSON report; it never reads Dev, modifies an artifact, or runs a nested evaluation. Runtime/artifact work is conditional on one screen winner only.

**Tech Stack:** Python 3.12, NumPy, stdlib `unittest`, existing hash-regex v3.3 helpers, official local scorer.

---

## Fixed file map

| File | Responsibility |
| --- | --- |
| `tools/hash_regex_v4_fill_screen.py` | Pure guarded route reconstruction, blocked-upgrade diagnostics, candidate fill selection, official-score report aggregation. |
| `tools/screen_hash_regex_v4_fill.py` | Train-only CLI, input/report boundary enforcement, fixed grouped four-fold execution and atomic JSON output. |
| `tests/promptbudget/test_hash_regex_v4_fill_screen.py` | Focused invariants: fill only modifies Light to AX31, leaves Premium legacy behavior intact, computes the 12 actual checks. |
| `tests/test_screen_hash_regex_v4_fill.py` | Focused CLI boundaries: reject Dev/output leakage; dry-run does not execute or write an output report. |
| `build/hash-regex-v4-deadline-triage/screening.json` | Generated evidence only; never committed and never an artifact. |

## Task 1: Build the pure Train-only route and diagnostic kernel

**Files:**
- Create: `tools/hash_regex_v4_fill_screen.py`
- Create: `tests/promptbudget/test_hash_regex_v4_fill_screen.py`

- [ ] **Step 1: Write focused failing selection tests**

```python
def test_fast_fill_only_promotes_light_to_ax31():
    routes = route_guarded_candidate(data, indices, scores, log_costs, guard, "fast-ax31-fill")
    assert all(model != "axk1-think" for model in routes["fast"].added_models)
    assert routes["balanced"].choices == routes["baseline"]["balanced"].choices
    assert routes["premium"].choices == routes["baseline"]["premium"].choices

def test_screen_requires_twelve_actual_cap_checks():
    result = screen_candidate(data, folds, "balanced-ax31-fill")
    assert result["actual_checks_required"] == 12
    assert len(result["actual_checks"]) == 12
```

- [ ] **Step 2: Run the one new module before implementation**

Run:

```powershell
$env:PYTHONPATH='src;baselines;tools'
& "$env:LocalAppData\Programs\Python\Python312\python.exe" -m unittest tests.promptbudget.test_hash_regex_v4_fill_screen -v
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the minimal pure API**

```python
CANDIDATES = ("fast-ax31-fill", "balanced-ax31-fill", "fast-balanced-ax31-fill")

def route_guarded_candidate(data, indices, score_prediction, cost_prediction, guard, candidate):
    """Return baseline/candidate selections and official reports for all three tiers."""

def blocked_upgrade_diagnostic(data, indices, raw_routes, guarded_routes):
    """Use held-out Train outcomes only to summarize Raw non-Light -> guarded Light rows."""

def screen_candidate(data, folds, candidate):
    """Fit only fold-train state, score each validation batch once per tier, and return 12 checks."""
```

Implementation requirements:

- call `nested.apply_tail_guard` with the v3.3 guard and preserve the existing Light < AX31 < Think clamp;
- reconstruct baseline choices with `hash_regex.select_models`, preserving existing Premium `fill_ax31_upgrades(..., safety_ratio=0.65)`;
- for Fast/Balanced candidates, call the same `fill_ax31_upgrades` once after baseline selection with `safety_ratio=1.0`; it must lock all prior non-Light choices and may only promote Light to AX31;
- make submissions from these selected choices and call `score_submissions`, returning actual and predicted ratios plus selection vectors;
- calculate per-tier blocked row count, observed score gain vs Light, actual incremental cost, guard bucket, and an oracle-greedy actual-slack recovery upper bound solely for held-out Train reporting;
- return `actual_checks_required=12`, `actual_checks_passed`, fallback count, per-tier cap-neutral quality deltas, fold deltas, and a `screen_passed` boolean. `screen_passed` requires 12/12 cap pass, fallback 0, changed-tier pooled delta > 0, changed-tier median fold delta > 0, and no new Think decisions.

- [ ] **Step 4: Run the focused kernel test**

Run the command in Step 2.

Expected: PASS. Do not run the repository-wide suite.

- [ ] **Step 5: Commit the kernel**

```powershell
git add tools/hash_regex_v4_fill_screen.py tests/promptbudget/test_hash_regex_v4_fill_screen.py
git commit -m "feat: add PromptBudget v4 fill screen kernel"
```

## Task 2: Add the isolated screen CLI and report contract

**Files:**
- Create: `tools/screen_hash_regex_v4_fill.py`
- Create: `tests/test_screen_hash_regex_v4_fill.py`

- [ ] **Step 1: Write failing CLI boundary tests**

```python
def test_dry_run_is_train_only_and_does_not_write_report():
    result = cli.screen(cli._parser().parse_args(["--input", str(train_input), "--outcomes", str(train_outcomes), "--report", str(report)]))
    assert result["terminal_status"] == "not-executed"
    assert not report.exists()

def test_cli_rejects_dev_and_nonisolated_output():
    with self.assertRaises(ValueError):
        cli.screen(cli._parser().parse_args(["--input", "data/materialized/dev/inputs.json", "--outcomes", str(train_outcomes), "--report", "build/out.json"]))
```

- [ ] **Step 2: Run the failing CLI test**

Run:

```powershell
$env:PYTHONPATH='src;baselines;tools'
& "$env:LocalAppData\Programs\Python\Python312\python.exe" -m unittest tests.test_screen_hash_regex_v4_fill -v
```

Expected: import failure because the CLI does not exist.

- [ ] **Step 3: Implement the minimal CLI**

```python
def screen(args: argparse.Namespace) -> Mapping[str, object]:
    reject_dev_path(args.input)
    reject_dev_path(args.outcomes)
    report_path = require_v4_report_path(args.report)
    # validate exactly 1,760 Train rows
    # --execute: make EvaluationData, use grouped_folds(..., folds=4, seed=137),
    # run diagnostic first, then only the allowed candidates, and atomically write JSON
```

The report must include input/outcome hashes, `v3_3_comparator="tail-guarded"`, fixed seed/fold count, diagnostic result, candidate reports, terminal status, and `forbidden_artifact_assertion=true`. It must stop with `no-recovery-signal` before any candidate call when the diagnostic reports no target-tier recovery signal. It may run the combination candidate only after a single-tier candidate passes. `--execute` is required to create a report and the output must be below `build/hash-regex-v4-deadline-triage/`.

- [ ] **Step 4: Run exactly the two focused test modules**

Run:

```powershell
$env:PYTHONPATH='src;baselines;tools'
& "$env:LocalAppData\Programs\Python\Python312\python.exe" -m unittest tests.promptbudget.test_hash_regex_v4_fill_screen tests.test_screen_hash_regex_v4_fill -v
```

Expected: PASS. This is the only new automated test command before executing the screen.

- [ ] **Step 5: Commit the CLI**

```powershell
git add tools/screen_hash_regex_v4_fill.py tests/test_screen_hash_regex_v4_fill.py
git commit -m "feat: add Train-only PromptBudget v4 fill screen"
```

## Task 3: Execute the bounded Train-only screen

**Files:**
- Create at runtime: `build/hash-regex-v4-deadline-triage/screening.json`

- [ ] **Step 1: Run the diagnostic and allowed candidates once**

```powershell
$env:PYTHONPATH='src;baselines;tools'
& "$env:LocalAppData\Programs\Python\Python312\python.exe" tools/screen_hash_regex_v4_fill.py `
  --input data/materialized/train/inputs.json `
  --outcomes data/train/outcomes.json `
  --report build/hash-regex-v4-deadline-triage/screening.json `
  --execute
```

Expected: one JSON report with `no-recovery-signal`, `no-safe-recovery-candidate`, or one named screen winner. Do not rerun it with altered thresholds, folds, candidates, or seed.

- [ ] **Step 2: Inspect only terminal gates**

```powershell
$screen = Get-Content -Raw build/hash-regex-v4-deadline-triage/screening.json | ConvertFrom-Json
$screen.terminal_status
$screen.candidates | Select-Object candidate,screen_passed,actual_checks_passed,actual_checks_required
```

Expected: only `screen_passed=true` can unlock Task 4; any other result preserves v3.3.

- [ ] **Step 3: Commit only source/tests, never generated evidence**

```powershell
git status --short
git add tools/hash_regex_v4_fill_screen.py tools/screen_hash_regex_v4_fill.py tests/promptbudget/test_hash_regex_v4_fill_screen.py tests/test_screen_hash_regex_v4_fill.py
git commit -m "test: record v4 fill screening implementation"
```

## Task 4: Conditional winner expansion before T-1h only

**Files:**
- Conditional modify: `baselines/hash_regex.py`
- Conditional modify: artifact parsing/training/finalization support only if the winning screen candidate requires a persisted Fast/Balance fill tier.
- Conditional create: `build/hash-regex-v4-deadline-triage/one-fold-smoke.json`, `nested-evaluation.json`, and a final artifact.

- [ ] **Step 1: Enforce the hard stop**

At T-1h, do not start this task. If Task 3 has no fully passing winner before T-1h, record `v3.3-retained-time-stop`, run the v3.3 submission-path check, and stop all v4 work.

- [ ] **Step 2: If a winner exists before T-1h, add only its tier membership to the runtime**

```python
fill_tiers = artifact.ax31_fill_tiers
if tier in fill_tiers:
    selected, ratio = fill_ax31_upgrades(
        selected, scores, costs,
        budget_multiplier=float(policy.tiers[tier].budget_multiplier),
        safety_ratio=1.0 if tier != "premium" else PREMIUM_AX31_FILL_SAFETY_RATIO,
    )
```

The legacy artifact default must remain Premium-only. Do not add a tunable safety value, a Think fill, or any new prompt feature.

- [ ] **Step 3: Run one focused runtime contract test, then one one-fold smoke**

```powershell
$env:PYTHONPATH='src;baselines;tools'
& "$env:LocalAppData\Programs\Python\Python312\python.exe" -m unittest tests.test_hash_regex_baseline -v
```

Then use the corresponding v4 evaluator once with a single outer fold. Continue only if its three actual caps pass.

- [ ] **Step 4: Run a full evaluation only when measured remaining time allows submission validation**

Start a full grouped nested run only when the one-fold smoke passes and there is enough measured time left for the run plus 20 minutes of artifact/submission validation. Promote only on 45/45 actual cap pass, fallback 0, and mean cap-neutral quality delta `>= +0.005`; otherwise retain v3.3.

- [ ] **Step 5: Commit and hand off the outcome**

Commit source/tests and report the selected submission artifact. Do not push unless explicitly requested.

## Plan self-review

- Spec coverage: Tasks 1–3 implement the frozen guard, Train-only row diagnosis, three restricted candidates, actual-cap checks, isolated report, and no-rerun rule. Task 4 covers the conditional runtime/finalization and `+0.005` materiality gate.
- Scope: no guard/head/feature/threshold search, Dev access, Think fill, or repository-wide test suite is present.
- Type/contract consistency: all new paths are isolated under `hash-regex-v4-deadline-triage`; candidate names and the 4-fold/12-check contract are shared across tasks.
