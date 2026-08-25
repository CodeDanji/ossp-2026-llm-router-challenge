<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# PromptBudget v2.1 연구 기준선 운영 절차

v2.1은 Train에서 grouped CV로 common score/비용 head와 모델×문자수 bucket 화폐
비용 multiplier를 학습한다. 공개 Dev는 tier별 `lambda_cost`, 최소 품질 이득,
최대 상대 비용, 비용 안전계수를 반복 보정하는 validation set이다. 산출물은 반드시
`build/promptbudget-v2.1/` 아래에만 쓴다. 기본 runtime artifact와 배포 경로는
변경하지 않는다.

```console
PYTHONPATH=src python3 tools/train_oof.py \
  --input data/materialized/train/inputs.json \
  --outcomes data/train/outcomes.json \
  --artifact build/promptbudget-v2.1/artifact.json \
  --manifest build/promptbudget-v2.1/manifest.json \
  --report build/promptbudget-v2.1/train-nested-cv.json
```

Dev는 숨겨진 test가 아닌 공식 공개 validation이다. 따라서 아래 보정은 반복 실행할
수 있고 결과 artifact를 갱신한다. Train에서 학습한 head와 OOF 비용 multiplier는
고정하며, Dev outcome은 실행 시점 router feature나 runtime artifact 보정에 사용하지
않는다. Fast는 실제 Dev 비용비율이 `<=1.15`(1.25 한도에서 0.10 여유), 다른 tier는
각 공식 한도보다 엄격히 작아야 후보가 통과한다.

```console
PYTHONPATH=src python3 tools/calibrate_policy.py \
  --input data/materialized/dev/inputs.json \
  --outcomes data/dev/outcomes.json \
  --draft-artifact build/promptbudget-v2.1/artifact.json \
  --draft-manifest build/promptbudget-v2.1/manifest.json \
  --artifact build/promptbudget-v2.1/dev-calibrated-artifact.json \
  --manifest build/promptbudget-v2.1/dev-calibrated-manifest.json \
  --report build/promptbudget-v2.1/dev-policy-calibration.json
```

이 결과는 제출 후보 선택 근거이지, 외부 일반화 성능의 독립 증거는 아니다. 진짜 독립
평가는 제출 뒤 숨겨진 입력이 담당한다. `episode_id`, split, 입력 순서, 데이터 출처,
실제 Dev score·토큰·비용·후보 답변은 runtime 선택 feature로 사용할 수 없다.
