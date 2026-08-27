<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# PromptBudget v4 제출 시한 triage 설계

**상태:** 승인된 설계 · 구현 전

**결정일:** 2026-08-27

## 1. 목적과 시간 제약

제출까지 3시간 이내다. v3.3 tail-aware cost guard는 `45/45` Train outer actual-cap pass, fallback `0`을 달성한 제출 가능한 baseline으로 **동결**한다. v4의 목표는 새 모델 가족을 찾는 것이 아니라, v3.3 guard가 남긴 안전 예산에서 기존 신호만으로 품질을 회복할 수 있는지 빠르게 반증하는 것이다.

이 설계는 다음 순서를 강제한다.

1. Train-only row-level 진단으로 guard가 차단한 upgrade의 관측 품질 이득과 actual 비용을 확인한다.
2. 회복 가능성이 있으면, 기존 quality/cost head와 고정 guard를 그대로 쓴 AX31 fill 후보 최대 3개를 독립 Train screen에서 비교한다.
3. screen이 사전등록 gate를 통과한 후보 하나에만 구현 및 확대 검증을 허용한다. 그렇지 않으면 v3.3을 그대로 제출한다.

## 2. 고정 comparator와 금지 범위

Comparator는 다음의 frozen v3.3 complete policy다.

- Tail guard: 4 bucket, model별 p90 nonnegative log-cost guard.
- Raw hash-regex quality head 및 log-cost head: hash bins 256, ridge alpha 100.
- 기존 batch Lagrange allocator, tie-break, Premium AX31 fill (safety `0.65`).
- complete runtime route와 official scorer; Premium fill을 제거한 부분 route 비교는 금지한다.

아래는 이번 triage의 범위 밖이다.

- guard bucket/quantile/edge/값, safety ratio, feature, ridge alpha, cost head, quality head를 바꾸거나 탐색하는 일;
- Dev 실행 또는 Dev 점수 기반 조정;
- full 15-fold × 3-seed nested evaluation을 후보 선택에 재사용하는 일;
- 기본 artifact, public v1 fallback, v3.3 artifact의 덮어쓰기;
- exact DP, 새 ML ranker, 새 runtime feature 또는 Think fill.

## 3. 최소 row-level 진단

새 diagnostic은 고정된 grouped Train screen split에서 각 held-out row와 tier에 대해 Raw complete route 및 v3.3 guarded complete route를 각각 한 번 재구성한다. 각 row는 다음 중 하나로 분류한다.

| 분류 | 조건 | 기록할 값 |
| --- | --- | --- |
| `blocked-upgrade` | Raw는 AX31/Think, guarded는 Light | Raw 선택 모델, observed score gain vs Light, observed incremental actual cost vs Light, guard bucket |
| `downgraded-upgrade` | Raw와 guarded 모두 non-Light이나 모델이 다름 | 모델 전환, observed quality/cost delta, guard bucket |
| `retained-upgrade` | Raw와 guarded 모델이 같고 non-Light | 모델, observed quality/cost delta 0, guard bucket |
| `unchanged-light` | 둘 다 Light | 후보 fill pool 여부와 predicted incremental utility/cost |

집계는 tier, 원래 모델(AX31/Think), guard bucket별 count, total/mean observed gain, total/mean actual incremental cost, 그리고 guarded route의 actual budget slack을 출력한다. 진단 결과는 관측 outcome을 **오직 Train held-out analysis에만** 사용하며 runtime feature나 artifact에 포함하지 않는다.

진단의 go/no-go는 다음이다.

- Fast 또는 Balanced에서 `blocked-upgrade`의 aggregate observed quality gain이 엄격히 양수여야 한다.
- 같은 tier에서 guarded route의 actual slack으로 설명 가능한 AX31 후보 pool이 존재해야 한다. 즉 guarded Light 중 실제 AX31 gain이 양수인 행이 하나 이상 있고, 그 actual incremental cost가 해당 batch의 actual slack보다 작거나 같은 경우가 하나 이상 있어야 한다.
- 둘 중 하나라도 성립하지 않으면 `no-recovery-signal`로 끝내고 어떤 후보 screen도 실행하지 않는다.

이 gate는 “oracle 회복”의 필요조건일 뿐 generalization 증명은 아니다. 통과해도 아래 예측 기반 screen을 반드시 통과해야 한다.

## 4. 후보: 기존 AX31 fill의 제한적 확장

후보는 새 ranker가 아니라 기존 `fill_ax31_upgrades`의 구조를 재사용한다. guarded Lagrange selection 뒤 Light 행만 AX31로 올리고, 기존 predicted incremental score 및 guarded predicted incremental cost로 penalty search를 한다. 기존 선택된 non-Light 모델은 바꾸지 않는다. Think upgrade는 추가하지 않는다.

각 후보는 guarded predicted cap을 넘으면 all-Light가 아니라 원래 guarded selection으로 되돌아간다. 실제 cap은 그 뒤 official scorer로만 판정한다.

| ID | 변경 | 이유 |
| --- | --- | --- |
| `fast-ax31-fill` | Fast에만 AX31 fill 추가 | v3.3 retention 손실이 가장 큰 tier를 최소 변경으로 확인한다. |
| `balanced-ax31-fill` | Balanced에만 AX31 fill 추가 | Fast 결과에 의존하지 않는 중간 예산 tier의 독립 확인이다. |
| `fast-balanced-ax31-fill` | Fast와 Balanced에 추가 | 두 독립 단일-tier 후보가 유효할 때만 확인하는 조합 후보다. Premium은 기존 fill을 유지한다. |

조합 후보는 단일-tier 후보 중 적어도 하나가 screen gate를 통과할 때만 실행한다. 따라서 최대 세 후보지만, 불필요한 실행은 하지 않는다.

## 5. Train-only screen과 승격 gate

Screen은 새로 고정한 4-fold grouped Train split(seed `137`)에서 수행한다. 각 fold에서 train complement로 guard와 heads를 fit하고, validation batch는 candidate마다 tier별 정확히 한 번 complete-route/official-score한다. 후보 사이와 frozen comparator는 같은 folds를 공유한다. 이 split은 후보 선택 용도이며 outer-test나 Dev가 아니다.

후보가 확대 검증 대상으로 승격되려면 다음을 전부 만족해야 한다.

1. 4 folds × 3 tiers의 모든 actual cap check가 pass한다.
2. fallback은 0이다.
3. 변경한 tier의 pooled cap-neutral quality가 frozen v3.3 comparator보다 **엄격히** 높다.
4. 변경한 tier의 각 fold cap-neutral quality delta의 중앙값이 양수다.
5. candidate가 Fast/Balance에서 새 Think route를 만들지 않고, guard/head/artifact metadata가 comparator와 byte-for-byte 동일하다는 contract assertion을 통과한다.

후보가 하나도 통과하지 않으면 terminal status는 `no-safe-recovery-candidate`이며 v3.3 제출을 유지한다. 여러 후보가 통과하면 weighted cap-neutral quality delta가 가장 큰 후보 하나를 선택한다. 동점은 더 낮은 maximum actual ratio, 그 다음 후보 ID 사전순으로 결정한다.

## 6. 시간 예산과 중단 규칙

| 단계 | 최대 시간 | 결과 |
| --- | ---: | --- |
| 진단 구현·실행 | 25분 | `recovery-signal` 또는 `no-recovery-signal` |
| 단일-tier 후보 screen | 후보당 20분 | pass/fail report |
| 조합 후보 screen | 20분 | 단일-tier 통과 시에만 실행 |
| 선택 후보 구현 및 Train 검증 확대 | 70분 | 새 isolated build 또는 fail evidence |
| 제출 artifact/path 확인 | 20분 | v3.3 또는 승격 후보 확정 |

진단 또는 screen이 시간 제한, contract 검증, official-score 실행, cap gate에서 실패하면 즉시 중단한다. 진단/후보 소스와 report는 격리된 v4 build 경로에만 남기며 v3.3을 수정하지 않는다.

## 7. 조건부 확대 검증과 제출

screen winner가 있을 때만 focused test, one-fold smoke, 그리고 한 번의 full grouped nested evaluation을 실행한다. full evaluation의 승격 조건은 v3.3과 동일하게 outer `45/45` actual cap pass 및 fallback `0`이며, v4 추가 gate로 cap-neutral quality가 frozen v3.3보다 양수여야 한다. 하나라도 실패하면 winner를 finalization하지 않는다.

full gate를 모두 통과한 경우에만 새 isolated artifact를 동결하고 Dev sanity를 정확히 한 번 수행한다. Dev는 tuning 근거가 아니며, Dev cap/parser/runtime failure 시 v3.3 artifact로 되돌아간다.

제출 시점의 기본값은 항상 frozen v3.3이다. 어떤 v4 candidate도 위 gate를 통과해 explicit replacement로 기록되기 전에는 제출 대상이 아니다.

## 8. 증거와 경로

- v3.3 handoff: `docs/skt/2026-08-26-promptbudget-v3-screening-handoff.html`
- primary nested evidence: `.worktrees/promptbudget-v3/build/hash-regex-tail-guard/nested-evaluation.json`
- v3.3 evaluator: `.worktrees/promptbudget-v3/tools/hash_regex_tail_guard_nested.py`
- v3.3 runtime: `baselines/hash_regex.py`

모든 v4 report는 `build/hash-regex-v4-deadline-triage/` 아래에 생성한다. public fallback 또는 frozen v3.3 artifact는 이 경로의 산출물로 교체하지 않는다.
