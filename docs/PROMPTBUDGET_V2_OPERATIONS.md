<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# PromptBudget v2 안전 운영 절차

v2는 Train에서만 grouped nested-CV로 구조를 선택한다. Dev는 구조 선택에
사용하지 않고, 고정된 artifact의 운영 safety multiplier를 보정하는
certificate만 만든다. Fast는 모든 보정 후보에서 `tier cap - 0.10` 이하의
upper cost ratio를 만족하지 못하면 v1 fallback을 유지한다.

```console
PYTHONPATH=src python3 tools/train_oof.py \
  --input data/materialized/train-inputs.json \
  --outcomes data/materialized/train-outcomes.json \
  --artifact build/promptbudget-v2/artifact.json \
  --manifest build/promptbudget-v2/manifest.json \
  --report build/promptbudget-v2/train-nested-cv.json

PYTHONPATH=src python3 tools/calibrate_policy.py \
  --input data/materialized/dev-inputs.json \
  --outcomes data/materialized/dev-outcomes.json \
  --draft-artifact build/promptbudget-v2/artifact.json \
  --draft-manifest build/promptbudget-v2/manifest.json \
  --artifact build/promptbudget-v2/calibrated-artifact.json \
  --manifest build/promptbudget-v2/calibrated-manifest.json \
  --report build/promptbudget-v2/dev-calibration-certificate.json
```

Locked holdout scoring must use the sole evaluator entry point. It reserves the
canonical holdout digest before outcomes are parsed for scoring; a crash or
failure leaves the digest permanently reserved.

```console
PYTHONPATH=src python3 tools/evaluate_locked.py \
  --input /secure/holdout-inputs.json \
  --outcomes /secure/holdout-outcomes.json \
  --artifact build/promptbudget-v2/calibrated-artifact.json \
  --manifest build/promptbudget-v2/calibrated-manifest.json \
  --ledger /secure/promptbudget-locked-ledger.sqlite \
  --report /secure/promptbudget-v2-locked-report.json \
  --operator operator@example.org
```

The local SQLite ledger rejects evaluator-mediated `UPDATE` and `DELETE`, but
it cannot protect against a person who already has direct filesystem write
access. Operators must therefore place the ledger in an access-controlled,
backed-up location and treat its database and journal files as audit records.
No break-glass command is included in this repository.
