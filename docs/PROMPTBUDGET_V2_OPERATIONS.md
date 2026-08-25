<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# PromptBudget v2.1 연구 기준선 운영 절차

v2.1은 Train에서만 grouped nested-CV로 common head·tier 정책·모델×문자수
bucket 화폐 비용 multiplier를 선택한다. 산출물은 반드시
`build/promptbudget-v2.1/` 아래에만 쓴다. 기본 runtime artifact와 배포 경로는
변경하지 않는다. Fast는 모든 inner validation fold에서 predicted upper cost
ratio가 `<=1.15`여야 하며, 통과 후보가 없으면 all-Light fallback을 기록한다.

```console
PYTHONPATH=src python3 tools/train_oof.py \
  --input data/materialized/train/inputs.json \
  --outcomes data/train/outcomes.json \
  --artifact build/promptbudget-v2.1/artifact.json \
  --manifest build/promptbudget-v2.1/manifest.json \
  --report build/promptbudget-v2.1/train-nested-cv.json
```

Dev는 이미 관측된 split이므로 독립 holdout이 아니다. 아래 confirmation은 artifact,
manifest, Dev input, Dev outcome digest 조합을 먼저 예약한다. 같은 조합은 성공·실패와
무관하게 다시 실행할 수 없고, 결과는 `exploratory_confirmation`으로만 해석한다.

```console
PYTHONPATH=src python3 tools/evaluate_locked.py \
  --exploratory-dev-confirmation \
  --input data/materialized/dev/inputs.json \
  --outcomes data/dev/outcomes.json \
  --artifact build/promptbudget-v2.1/artifact.json \
  --manifest build/promptbudget-v2.1/manifest.json \
  --ledger build/promptbudget-v2.1/dev-confirmation-ledger.sqlite \
  --report build/promptbudget-v2.1/dev-confirmation.json \
  --operator operator@example.org
```

The local SQLite ledger rejects evaluator-mediated `UPDATE` and `DELETE`, but
it cannot protect against a person who already has direct filesystem write
access. Operators must therefore place the ledger in an access-controlled,
backed-up location and treat its database and journal files as audit records.
No Dev execution may change the artifact, manifest, multiplier, policy, or
research claim labels.
