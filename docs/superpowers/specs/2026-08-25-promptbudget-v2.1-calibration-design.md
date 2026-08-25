<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# PromptBudget v2.1 Calibration Design

## Decision

PromptBudget v2.1 remains a research-only absolute-linear baseline.  It creates
artifacts only below `build/promptbudget-v2.1/`; it neither overwrites nor
changes the default runtime artifact at
`src/promptbudget/resources/artifact.json`.

The existing v1 artifact format remains readable and unchanged.  A v2.1
artifact carries a separate, backward-compatible cost-calibration section that
stores a one-sided multiplier for every model and for each fixed prompt-length
bucket: `<=512`, `513-2048`, `>2048`, plus `global`.  Runtime prediction uses
the multiplier for the prompt bucket; buckets with fewer than 100 Train OOF
samples use the model's global multiplier.

## Cost Calibration and Admission

For a model and prompt, predicted monetary cost is calculated from the existing
input/output absolute heads and the frozen routing-policy rate card.  The
nonconformity score is actual monetary cost divided by predicted monetary cost.
The higher empirical 99th percentile of Train-only grouped cross-fit OOF scores
is the persisted multiplier. `TierSettings.safety_multiplier` is a separate
policy factor selected on public Dev with the tier threshold, cost lambda, and
maximum relative cost.

During Train diagnostics, a candidate's relative predicted cost is compared to
the matching predicted all-Light route, never actual Light cost. This keeps the
all-Light fallback at ratio 1 even with a conservative multiplier. If
calibration cannot produce a positive, finite multiplier, the candidate must
choose Light.

## Selection

The outer evaluation is the fixed three-seed, five-fold schedule, yielding 15
report-only outer tests.  The final artifact runs a fixed seed-137 grouped
four-fold inner selection on all Train rows, then runs a separate seed-137
grouped cross-fit pass to calibrate costs and fits the selected head on all
Train rows.

Train selects and calibrates common absolute heads without any Dev outcome. It
may emit a conservative starter tier policy, but that is not the submission
policy. On public Dev, each tier searches the complete fixed grid of
`lambda_cost`, common minimum gain (assigned to both non-Light models), maximum
relative cost, and safety multiplier. The official scorer supplies the quality
and actual-cost result; candidates must meet the actual tier budget with the
Fast 0.10 margin. Ties prefer the larger safety multiplier and then lower actual
cost. Dev calibration can be repeated as policies improve; it is official
public validation model selection, not an independent generalization claim.

## Integrity, Dev, and Diagnostics

The Train report contains only aggregate diagnostics and digests.  It records
the candidate ledger, fold admission decisions, cost calibration distributions,
routing analysis, paired-bootstrap confidence intervals, code/policy/grid
digests, and the four fixed claim labels from the research handoff.

Dev is the official public validation/tuning split. Its score, token, and cost
outcomes may be used repeatedly to select tier policy parameters and compare
candidate artifacts. They never become runtime inputs, model-training targets,
or prompt features. Hidden evaluation inputs remain the independent final
evaluation.

## Rejected Alternatives

- A sidecar calibration file is rejected because it can become detached from
  the artifact it calibrates.
- A global-only multiplier is rejected because it deliberately discards the
  fixed length-bucket safety rule.
- Treating Dev as a one-use independent test is rejected because the official
  workflow explicitly assigns it to threshold and cost-safety calibration.
