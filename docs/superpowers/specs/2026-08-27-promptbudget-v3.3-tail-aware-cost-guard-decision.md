<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# PromptBudget v3.3 Request-Level Tail-Aware Cost Guard 결정 기록

**상태:** 설계 확정 · 구현 전.

**결정일:** 2026-08-27

**독자:** 이 실험을 구현·검토·제출하거나, v3.2에서 v3.3으로 무엇을 바꾸었는지 학습할 사람.

## 1. 한 문장 질문과 결론

v3.3이 답할 질문은 다음 하나다.

> **Raw AX31/Think cost predictor의 grouped-OOF underprediction log residual p90이 base predicted log-cost의 사전 고정된 4개 equal-count 구간에 따라 이질적인가? 그리고 각 Train-only fit partition에서 산정한 구간별 nonnegative guard를 적용했을 때, v3.2 global multiplier보다 non-Light upgrade를 실질적으로 보존하면서 grouped outer 45/45 actual budget safety를 달성할 수 있는가?**

여기서 residual은 `log(actual_cost) - base_predicted_log_cost`다. 양수는 base predictor가 비용을 과소예측했다는 뜻이다. v3.3은 새 quality model, 새 allocator 또는 여러 cost-grid를 찾는 실험이 아니다. **base predicted-cost 위치만으로 conditional tail correction이 가능한지**를 시험하는 단일 guard 실험이다.

## 2. v3.2가 남긴 근거와 이 결정의 의도

v3.2는 AX31과 Think의 모든 request predicted cost에 하나의 model-level multiplier pair를 적용했다. Train-only grouped nested evaluation 결과는 다음과 같았다.

| v3.2 관측 | 값 | 의미 |
| --- | ---: | --- |
| terminal status | `cost-calibration-no-go` | global correction은 promotion되지 않음 |
| outer actual checks | 43 / 45 | 한 outer fold의 Balanced/Premium이 실패 |
| all-Light fallback folds | 14 / 15 | 대다수 outer-train에 12/12 admitted global pair가 없음 |
| non-Light retention vs Raw | 6.16% | 안전 시도는 거의 all-Light collapse |
| 최대 actual ratio | 5.746 | Premium tail이 매우 큼 |
| Raw 대비 cap-neutral quality delta | -0.06754 | official zeroing을 제외하면 quality도 하락 |

따라서 v3.2가 반증한 것은 “모델마다 전체적으로 비용을 더 높게 잡으면 충분한가”라는 **global correction** 가설이다. 이것은 prompt-side feature가 tail 정보를 전혀 갖지 않는다는 뜻은 아니다. v3.3은 이미 Raw cost predictor가 출력하는 per-request predicted log-cost의 위치를 condition으로 삼아, 실제 tail이 특정 위치에 집중하는지를 먼저 확인한다.

시간이 부족한 상황에서 이 설계의 목적은 두 가지다.

1. signal이 없으면 복잡한 runtime/guard 구현을 즉시 중단한다.
2. signal이 있으면 bucket count와 quantile을 다시 탐색하지 않고 하나의 고정 guard만 nested evaluation한다.

## 3. 고정된 실험 계약

```text
QUALITY_KIND = "raw"
HASH_BINS = 256
RIDGE_ALPHA = 100.0
OUTER_SEEDS = (137, 271, 811)
OUTER_FOLDS = 5
INNER_FOLDS = 4
TAIL_BUCKET_COUNT = 4
TAIL_RESIDUAL_QUANTILE = 0.90
TAIL_HETEROGENEITY_MIN_SPREAD = log(1.10)
TAIL_HETEROGENEITY_REQUIRED_SEEDS = 2
TIER_SAFETY_RATIOS = {"fast": 1.0, "balanced": 1.0, "premium": 1.0}
RETENTION_PROMOTION_MINIMUM = 0.20
FALLBACK_ARTIFACT = "baselines/hash-regex-public.v1.json"
```

유지하는 것:

- Raw quality ridge head, 14 dense + 256 signed-hash feature, alpha 100;
- 모델별 Raw log-cost ridge family와 actual-cost target;
- `select_models`의 batch Lagrange allocator와 결정적 tie-break;
- Premium의 기존 `fill_ax31_upgrades` 및 내부 safety 0.65;
- canonical prompt-content digest grouped split, official scorer, Train-only/Dev one-pass 원칙.

바꾸는 것은 AX31/Think의 **predicted upgrade cost**뿐이다. Light는 어떠한 guard도 받지 않는다. score label, quality target, routing feature, actual score/cost, tier safety와 allocator의 목적식은 바꾸지 않는다.

## 4. guard의 정확한 의미와 순서

각 fit partition `P`에서 AX31 및 Think 각각 다음을 수행한다.

1. `P` 내부의 4-fold grouped OOF로 base log-cost prediction을 만든다.
2. OOF predicted log-cost의 25/50/75% higher empirical quantile을 세 edge로 하여 4개 quantile bucket을 만든다. 같은 numeric prediction이 edge와 같으면 **`bisect_right`** 규칙으로 edge의 오른쪽 bucket에 둔다. repeated value로 exact equal count가 불가능할 수 있으므로 bucket count를 report하며, 빈 bucket은 invalid evaluation이다.
3. 각 bucket에서 `ceil(0.90*n)-1` 위치의 sorted OOF residual을 p90으로 취한다.
4. guard는 `max(0, p90_residual)`이다. 비용을 낮추는 보정은 허용하지 않는다.

새 request의 비용은 다음 순서로 계산한다.

```text
base predicted log-cost
  → exp
  → 기존 Light < AX31 < Think monotonic clamp
  → base predicted log-cost bucket에 따라 AX31/Think cost × exp(nonnegative guard)
  → 동일 ordering clamp 재적용
  → 기존 allocator
  → 기존 Premium fill (같은 guarded cost 사용)
```

bucket membership은 clamp 뒤 cost가 아니라 **guard 전 base predicted log-cost**로 결정한다. 이는 guard가 무엇을 condition으로 하는지 명확히 하고, monotonic clamp의 부수 효과가 bucket을 바꾸지 않게 한다.

## 5. diagnostic-first kill rule

구현 전 diagnostic은 Train 전체에서 seeds `(137, 271, 811)` 각각의 4-fold grouped OOF residual을 생성한다. model/seed마다 네 bucket이 모두 비어 있지 않아야 한다.

AX31 또는 Think 중 하나가 적어도 두 seed에서 다음을 만족하면 signal이 있다.

```text
max(bucket p90 residual) - min(bucket p90 residual) >= log(1.10)
```

이 rule은 bucket 수, quantile 또는 guard 값을 고르는 데 쓰이지 않는다. 결과는 다음 중 하나다.

| diagnostic 결과 | 의미와 행동 |
| --- | --- |
| `tail-signal-present` | 고정 4-bucket/p90 guard를 one-fold smoke 후 한 번의 full nested evaluation으로 진행 |
| `tail-no-signal` | base predicted-cost scalar의 p90 conditional discrimination이 부족함. guard 구현·full evaluation·artifact·Dev를 중단 |
| `invalid-diagnostic` | group, OOF, bucket, finite-value, serialization contract 오류. evidence를 보존하고 중단 |

signal이 없어도 “모든 prompt-side feature에는 tail 정보가 없다”고 결론 내리지 않는다. 오직 **base-predicted-cost bucket guard**의 근거가 없다는 결론이다. signal이 있어도 safety를 보장하지 않는다. 이후 45/45 실패는 이 고정 guard가 batch-level safety에 충분하지 않았다는 결론이다.

## 6. nested selection과 information boundary

각 outer fold의 outer-test는 selection 전에 봉인한다. outer-train만 네 grouped inner fold로 분할한다. 각 inner validation batch에 대해:

1. inner-train complement 안에서 다시 grouped OOF residual을 생성하여 bucket edges/p90 guards를 fit한다.
2. 같은 inner-train complement로 Raw quality와 base log-cost head를 fit한다.
3. validation request의 base predicted log-cost로 bucket을 배정하고 guarded costs를 만든다.
4. Fast/Balanced/Premium을 각각 독립 batch로 route하고 official scorer로 actual cap을 확인한다.

단일 guard candidate의 admission은 4 inner folds × 3 tiers = **12/12 actual cap pass**다. pooled rows, predicted ratio, rounded ratio, average ratio 또는 average pass는 admission 대체물이 아니다. 12/12 실패는 explicit all-Light fallback이며 finalizable configuration이 아니다.

12/12 admitted guard는 inner independent routes의 official weighted points 합/row 합으로 점수화한다. 이 점수는 retention, quality diagnostic, outer-test 또는 Dev로 바꾸지 않는다.

## 7. outer evaluation, comparator, promotion

각 outer fold는 outer-train에서 guard와 Raw quality/base cost head를 다시 fit한 뒤 outer-test를 tier별 정확히 한 번 route/official-score한다. 동일 split의 unguarded Raw `(1, 1)` route는 diagnostic comparator로 한 번 score한다. frozen v3.2 global report는 같은 15 outer fold schedule의 역사적 comparator로 report에 연결하되, v3.3 selection에는 쓰지 않는다.

promotion은 아래를 **모두** 만족할 때만 가능하다.

1. 15 outer folds × 3 tiers의 45 actual cap check가 모두 pass.
2. outer all-Light fallback이 0개이고 모든 outer selection이 admitted non-Light guard.
3. aggregate non-Light retention이 0.20 이상.

retention은 fold 평균이 아니다. Premium fill 뒤 최종 route에서 모든 outer test/tier의 AX31 또는 Think decision을 합한 것을, 같은 outer split의 Raw comparator AX31 또는 Think decision 합으로 나눈다. Raw denominator가 0이면 `not_applicable`로 기록하고 promotion하지 않는다.

Raw 대비 cap-neutral weighted quality, official weighted score, model count, tier/total retention, maximum actual ratio는 반드시 report한다. quality는 tuning target이나 추가 hard gate가 아니다. official score의 개선은 cap failure zeroing을 피한 효과일 수 있으므로 cap-neutral quality와 분리해 해석한다.

| terminal status | 조건 | 후속 행동 |
| --- | --- | --- |
| `tail-no-signal` | diagnostic signal 없음 | guard/full/artifact/Dev 중단 |
| `invalid-evaluation` | schema/group/OOF/bucket/fit/report 오류 | evidence 보존 후 중단 |
| `cost-calibration-no-go` | full outer 45/45 실패 | artifact/Dev 없이 frozen fallback 기록 |
| `safe-but-collapse` | 45/45 pass이나 fallback 존재 또는 retention <20% | 자동 finalization 금지 |
| `safe-candidate` | 45/45, fallback 0, retention ≥20% | locked finalization과 Dev one-pass 허용 |

## 8. runtime artifact, finalization, Dev

request-level guard는 model-level intercept에 흡수할 수 없다. 안전 후보의 artifact는 따라서 기존 v1 artifact를 깨지 않으면서 versioned optional runtime data를 표현해야 한다.

새 tail-guard artifact에는 AX31/Think별 세 bucket edge와 네 nonnegative log guard, bucket count 4, quantile 0.90, feature/hash/alpha, policy digest, Train hashes, nested report hash, raw/base cost head metadata를 기록한다. existing `ossp-hash-regex-linear-v1` parser/runtime은 계속 읽혀야 한다. new guarded artifact는 명시적 새 type/version으로 parse하며, `predict_episode`가 guard 전 log-cost로 bucket을 배정하고 guarded cost를 반환한다.

`safe-candidate`일 때만 full Train 내부 grouped OOF로 guard를 lock하고, full Train Raw quality/base log-cost heads를 fit한다. full-Train guard가 invalid 또는 12/12 fallback이면 finalizer는 거부한다. 실패한 evaluation은 artifact, published resource 또는 runtime resource를 덮어쓰지 않는다.

Dev는 frozen final artifact에 대해 정확히 한 번 parser/runtime/three-tier cap sanity를 한다. Dev는 bucket edge, p90, quality, retention, safety 또는 route mix를 변경할 수 없다. Dev cap/parser/runtime failure는 `fallback-required`이며 `baselines/hash-regex-public.v1.json`을 쓴다. Dev score 차이는 retune 이유가 아니다.

## 9. 비목표와 해석 한계

- p90을 p95로 바꾸거나 bucket 수를 3/5로 재탐색하지 않는다.
- prompt feature 기반 새 quantile/ML tail head, tier별 guard, global multiplier grid, allocator 교체, exact DP, Dev calibration을 섞지 않는다.
- 45 correlated checks를 독립 p-value나 hidden-budget guarantee로 과장하지 않는다.
- v3.3은 Train-only stress screen이다. safe-candidate는 finalization/one-pass Dev sanity의 자격이지 hidden generalization 보장은 아니다.

## 10. 추적 자료

- v3.2 결정: `docs/superpowers/specs/2026-08-27-promptbudget-v3.2-cost-stabilization-decision.md`
- v3.2 full report: `build/hash-regex-cost-stabilization/nested-evaluation.json`
- v3.2 evaluator: `tools/hash_regex_cost_stabilization_nested.py`, `tools/evaluate_hash_regex_cost_stabilization_nested.py`
- baseline/runtime: `baselines/hash_regex.py`, `baselines/train_hash_regex.py`
- grouping/scorer: `src/promptbudget/safety.py`, `src/ossp_router/scoring.py`
