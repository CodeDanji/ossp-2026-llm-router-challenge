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
is the multiplier.  `TierSettings.safety_multiplier` is fixed at `1.0` for
every v2.1 candidate.

During an inner validation fold, a candidate is admissible only when the ratio
of summed predicted upper costs to summed actual Light costs meets the tier
bound in that fold: Fast `<=1.15`, Balanced `<2`, Premium `<4`.  The prior
group-level aggregate upper-ratio calculation is retained only as a diagnostic
and never changes admission.  If calibration cannot produce a positive,
finite multiplier, the candidate must choose Light.

## Selection

The outer evaluation is the fixed three-seed, five-fold schedule, yielding 15
report-only outer tests.  The final artifact runs a fixed seed-137 grouped
four-fold inner selection on all Train rows, then runs a separate seed-137
grouped cross-fit pass to calibrate costs and fits the selected head on all
Train rows.

Each common absolute head candidate is evaluated with independent tier-policy
candidates.  The common minimum-gain setting is assigned to both `ax31` and
`axk1-think`.  The official tier score is calculated from the existing frozen
policy scorer.  One-standard-error selection ranks eligible policies by:

1. lower grouped mean upgrade fraction;
2. lower maximum relative cost;
3. higher common minimum gain;
4. higher cost lambda;
5. pre-registered grid order.

The common head is selected by the sum of the three selected tier validation
scores.  Comparisons use 10,000 content-group paired-bootstrap repetitions and
seed `20260825`.

## Integrity, Dev, and Diagnostics

The Train report contains only aggregate diagnostics and digests.  It records
the candidate ledger, fold admission decisions, cost calibration distributions,
routing analysis, paired-bootstrap confidence intervals, code/policy/grid
digests, and the four fixed claim labels from the research handoff.

The observed Dev split is a one-time `exploratory_confirmation`.  Its SQLite
reservation key is the artifact hash, manifest hash, Dev input digest, and Dev
outcome digest.  It records the code/policy digest and its before/after artifact
and manifest hashes, and rejects any second use of the same key.  It cannot
alter model, policy, multiplier, or claim selection.

## Rejected Alternatives

- A sidecar calibration file is rejected because it can become detached from
  the artifact it calibrates.
- A global-only multiplier is rejected because it deliberately discards the
  fixed length-bucket safety rule.
- Using Dev for parameter selection is rejected because Dev is already
  observed and must remain exploratory.
