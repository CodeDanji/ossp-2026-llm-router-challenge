<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# PromptBudget v3 정식 Train Nested Evaluation 설계서

**상태:** 방향 승인 · 구현 계획 작성 전 검토

**결정일:** 2026-08-26

**범위:** 현재 연구용 `delta-linear` v3의 평가 파이프라인만 정정한다. 새 모델
family, feature 확장, Dev 정책 보정, default artifact 변경은 범위 밖이다.

## 1. 결정 요약

공개 Dev 880 screening에서 v3는 bundled v2.1의 0.670085227273보다 낮은
0.619318181818을 냈다. v3는 세 tier 모두 Light 878, Think 2, ax31 0으로
수렴했다. 이 수치는 v3를 바로 버려야 한다는 독립적인 일반화 증거는 아니지만,
현재 artifact를 기준으로 계산비용이 큰 full nested-CV를 실행할 이유도 아니다.

다음 구현은 기존 `tools/train_delta.py`를 정식 평가기로 바꾸지 않는다. 새
`tools/evaluate_delta_nested.py`가 Train-only formal evaluation을 담당하고,
`train_delta.py`는 현재 artifact를 재현하는 provisional trainer로 남긴다. 이렇게
하면 현재의 불완전한 `train-nested-cv.json`과 새 formal report를 혼동하지 않는다.

정식 report가 Go가 되기 전에는 final v3 artifact를 refit하지 않고, 공개 Dev를
다시 사용하지 않으며, `src/promptbudget/resources/`도 변경하지 않는다.

## 2. 문제와 비판적 전제

현재 v3 implementation에는 다음 결함이 있다.

1. outer fold는 선택된 head spec만 기록하고 outer-test 정책 score/cost를 측정하지
   않는다.
2. final cost offset은 in-sample residual을 쓰므로 coverage/slack 주장이 낙관적일 수
   있다.
3. inner selection이 dense-only surrogate를 쓰므로 선언된 sparse grid와 다른 모델을
   고른다.
4. current bundled v2.1 artifact는 전체 Train으로 만들어진 artifact다. 이를 Train
   outer-test에 그대로 비교하면 comparator가 해당 test prompt를 이미 학습한 Train
   leakage가 된다.
5. 정식 signal, cost-risk, policy comparison, group-bootstrap CI report가 없다.

따라서 “현재 v3가 낮다”는 screening 결과와 “v3가 Train에서 v2.1보다 나쁜가”는
아직 다른 질문이다. 새 evaluator는 후자만 답한다. evaluator가 Go라고 해도 hidden
성능 또는 예산 통과를 보장하지 않는다.

## 3. 목표와 명시적 비목표

### 목표

1. content group이 train/test 양쪽에 나타나지 않는 fixed 3 seed × 5 outer-fold
   Train evaluation을 만든다.
2. 각 outer-train 안에서만 delta head, cost residual quantile, tier policy profile을
   선택하고 outer-test에는 선택 결과를 한 번만 적용한다.
3. v3와 v2.1을 동일 outer-train 정보로 각각 refit하여 outer-test에서 공정하게
   비교한다.
4. 실제 outer-test cost로 tier admission, quality score, coverage, slack, upgrade
   precision과 content-group bootstrap CI를 기록한다.
5. evidence 기반 Go/No-Go만 출력한다. evaluator는 artifact 또는 runtime resource를
   쓰지 않는다.

### 비목표

- v3 score를 올리기 위한 feature, model, threshold 탐색 확대
- Dev score를 보고 model family/head grid를 다시 바꾸는 작업
- automatic Dev calibration, automatic final artifact refit, automatic merge/push
- formal conformal guarantee 또는 hidden-budget guarantee 주장
- baseline v2.1 shipped artifact를 outer test comparator로 그대로 사용하는 작업

## 4. 경계와 산출물

| 구분 | 경로/책임 | 허용 | 금지 |
| --- | --- | --- | --- |
| provisional trainer | `tools/train_delta.py` | 기존 research artifact 재현 | formal report 이름·Go/No-Go 생성 |
| formal evaluator | `tools/evaluate_delta_nested.py` | Train outer/inner split, report 생성 | artifact/manifest 생성, Dev input/outcome 읽기 |
| shared evaluator helpers | `tools/delta_nested_eval.py` | fit/predict/metrics/bootstrap 순수 함수 | filesystem output, global mutable state |
| formal output | `build/promptbudget-v3/nested-evaluation.json` | report 하나 | `src/promptbudget/resources/` 또는 다른 build root 쓰기 |
| default runtime | `src/promptbudget/resources/*` | 읽기만 가능 | 변경 |

새 output path는 `build/promptbudget-v3/` 아래인지 검증하고, 기존
`train-nested-cv.json`을 덮어쓰지 않는다. Dev file path 또는 `dev` split을 인자로
받으면 evaluator는 실패한다.

## 5. 평가 후보를 사전 고정하는 방법

이 작업의 목적은 모델 개선이 아니라 평가 신뢰성이다. 계산량과 선택 자유도를
통제하기 위해 다음 후보를 고정한다.

### 5.1 v3 head 후보

각 upgrade model(`ax31`, `axk1-think`)과 각 target(`delta_quality`,
`win_probability`, `incremental_relative_cost`)에 독립적으로 다음 정확한 후보를
평가한다.

| 후보 축 | 값 |
| --- | --- |
| sparse feature count | 8, 16 |
| ridge/logistic L2 alpha | 1.0, 10.0 |
| length bucket minimum | 100 |
| residual quantile | 0.90, 0.95, 0.99 |

head selection은 outer-train 4-fold inner OOF prediction에서 target별 loss로 한다.
delta/cost는 mean squared error, win-probability는 Brier score를 최소화한다. loss가
최소값의 standard error 안에 있는 후보 중 더 작은 feature count, 더 큰 alpha,
그 다음 사전 순으로 고른다. 후보의 sparse feature ranking은 반드시 해당 inner
training partition만 사용하며, 같은 feature count의 alpha들은 준비된 matrix를
공유해 계산을 줄인다. dense-only surrogate는 사용하지 않는다.

### 5.2 v3 tier policy profile

head를 선택한 뒤, inner OOF prediction과 실제 inner-validation cost로 다음 세
profile × 세 residual quantile을 평가한다. 하나의 profile은 세 tier의 설정을 함께
묶으므로 9개 후보만 비교한다.

| profile | Fast `(lambda, p-win, gain, max relative)` | Balanced | Premium |
| --- | --- | --- | --- |
| reference | `(1.00, .60, .03, 1.15)` | `(.50, .55, .01, 2.00)` | `(.10, .50, .00, 4.00)` |
| conservative | `(1.00, .65, .04, 1.10)` | `(.75, .60, .02, 1.75)` | `(.25, .55, .01, 3.00)` |
| quality-limited | `(.75, .55, .02, 1.15)` | `(.25, .50, .00, 2.00)` | `(.05, .45, .00, 4.00)` |

각 profile/quantile은 Fast actual cost ratio `<= 1.15`, Balanced `< 2.0`, Premium
`< 4.0`를 inner OOF에서 모두 통과해야 한다. 통과 후보 중 weighted official score가
가장 높은 것을 선택한다. 동점은 더 낮은 p90 upper-cost slack, 더 낮은 quantile,
reference → conservative → quality-limited 순으로 고른다. 이 선택은 outer-test를
전혀 보지 않는다.

## 6. 한 outer fold의 정확한 데이터 흐름

각 fixed `(seed, fold)` outer split에서 canonical prompt group은 outer-train 또는
outer-test 중 한쪽에만 있다.

1. **v3 inner OOF head selection.** outer-train을 4개의 grouped inner fold로 나눈다.
   각 candidate head는 inner-train에서 sparse ranking·fit하고 inner-validation에
   prediction을 남긴다. 모든 model/target의 OOF prediction으로 5.1의 규칙에 따라
   여섯 head spec을 고른다.
2. **v3 residual calibration candidate.** 고정된 selected cost head spec으로
   outer-train 4-fold cross-fit prediction을 다시 만든다. model과 length bucket별
   residual `observed_relative_cost - predicted_relative_cost`의 0.90/0.95/0.99
   empirical quantile을 계산한다. bucket row가 100 미만이면 같은 model global
   residual quantile을 쓰고 fallback을 record한다.
3. **v3 inner policy selection.** 위 OOF prediction, quantile offset, 실제
   inner-validation outcome/cost로 9개의 profile/quantile 후보를 채점한다. profile은
   5.2의 admission과 tie-break로 하나만 선택한다. 후보가 하나도 admission을
   통과하지 않으면 `all-light`를 outer-fold v3 fallback으로 선택하고 이 사실을
   report한다.
4. **v3 outer-test evaluation.** selected head spec을 사용해 outer-train 전체에서
   여섯 head를 refit한다. step 2의 cross-fit offset, step 3의 policy를 고정하여
   outer-test prompt를 routing한다. outer-test의 실제 outcome/cost는 이 시점에서
   처음 score, cost ratio, coverage, slack을 계산하는 데만 사용한다.
5. **v2.1 fair comparator.** v2.1 absolute-linear heads와 monetary calibration도
   outer-train 내부에서 refit한다. feature/alpha selection과 residual calibration은
   `tools/train_oof.py`의 existing grouped-OOF semantics를 재사용하되 outer-test
   data를 읽지 않는다. shipped default artifact는 사용하지 않는다. tier settings는
   bundled v2.1 artifact의 frozen settings를 사용하며 outer-test actual cost와 score만
   계산한다.
6. **baselines.** all-Light와 all-ax31은 cost/score reference로 함께 계산한다.
   all-ax31이 tier budget을 넘으면 score zero rule을 그대로 report한다. 이들은
   candidate 선택에 쓰지 않는다.

각 outer-test row의 report에는 seed/fold, canonical group digest, route, actual
score/cost, v3 `p_win`/delta/r-upper/utility, v2.1 route를 기록한다. raw prompt와
episode ID는 report에 쓰지 않는다.

## 7. Metrics, confidence interval, and Go/No-Go

### 7.1 필수 metrics

| 영역 | outer-test report 값 |
| --- | --- |
| signal | model별 delta MAE, delta rank correlation, win Brier, win AUROC, chosen-upgrade precision/recall |
| cost risk | model/length별 relative-cost MAE, p90 absolute error, empirical coverage, Wilson 95% CI, p50/p90/p99 slack |
| policy | candidate/tier별 actual cost ratio, official score, upgrade model count, win/tie/loss, budget pass, all-Light 대비 quality delta |
| comparison | v3 − outer-refit v2.1 weighted score, tier별 score difference, p90 slack difference |

`slack = r_upper - observed_relative_cost`이며 coverage는
`r_upper >= observed_relative_cost` 비율이다. “upper”라는 이름은 empirical
calibration일 뿐 finite-sample guarantee가 아니다.

### 7.2 Bootstrap

content group 단위로 2,000회 bootstrap한다. 같은 group의 서로 다른 outer repeat
record는 항상 함께 resample한다. RNG seed는 20260826으로 고정한다. report는 v3-v2.1
weighted score difference와 p90 slack difference의 95% percentile CI를 기록한다.

### 7.3 formal verdict

아래 네 조건이 모두 참일 때만 `go`다.

1. v3−v2.1 weighted score point estimate가 `> 0`이고 95% bootstrap CI lower bound도
   `> 0`이다.
2. v3의 모든 outer-fold/tier가 actual-cost admission을 통과한다: Fast `<= 1.15`,
   Balanced `< 2.0`, Premium `< 4.0`.
3. v3 p90 slack point estimate가 v2.1보다 작고, p90 slack difference CI upper bound가
   `< 0`이다.
4. v3 chosen-upgrade precision이 해당 model의 Train all-positive baseline win rate보다
   높다. 둘 다 upgrade를 선택하지 않으면 `no-go`다.

이 중 하나라도 거짓이면 `no-go`다. report는 어떤 조건이 실패했는지 machine-readable
field와 한국어 reason으로 모두 쓴다. `inconclusive` verdict는 만들지 않는다. 넓은
CI는 조건 1 또는 3을 통과하지 못하므로 `no-go`로 기록한다.

## 8. 실행 안전장치와 계산비용 통제

1. evaluator 기본 모드는 `--dry-run`이다. 입력 digest, split row/group count, fixed
   fold schedule, 9 policy candidates, 예상 fit 수만 출력하고 artifact/report를 쓰지
   않는다.
2. `--execute`가 있어야 Train evaluation을 실제로 실행한다. 실행 전에는
   `--input` path가 materialized Train이고 `--outcomes` path가 Train outcomes인지,
   digest와 row count 1,760을 검사한다.
3. 3×5 full evaluation은 구현·focused tests·one outer fold smoke run을 통과한 뒤,
   사용자가 명시적으로 full run을 승인했을 때만 실행한다. 이 승인 전에는
   `--execute --full`을 실행하지 않는다.
4. sparse feature ranking은 fold/candidate별 결과를 evaluation output root 아래의
   ephemeral cache에만 저장하고, digest·grid·code version이 맞을 때만 재사용한다.
   report 완료 또는 실패 뒤 cache 삭제는 자동으로 하지 않는다.
5. evaluator는 process count가 1인 deterministic NumPy 실행만 지원한다. 병렬
   subagent 또는 process pool은 다른 fold가 shared artifact/cache를 바꾸지 않도록
   구현 범위에서 제외한다.

## 9. 수용 조건

1. unit test가 canonical group disjointness, outer-test 미사용, exact sparse ranking,
   cross-fit-only residual, 9 profile admission, no-admission all-Light fallback을
   고정한다.
2. synthetic grouped fixture에서 outer-test outcome을 바꾸어도 candidate selection이
   변하지 않음을 확인한다.
3. outer-refit v2.1 comparator가 bundled artifact를 load하지 않고 outer-train만으로
   fit됨을 확인한다.
4. output path outside `build/promptbudget-v3/`, Dev input, default resource write,
   no-`--execute` run은 모두 실패하거나 dry-run으로 끝난다.
5. one-outer-fold smoke run은 report schema, all required metric table, bootstrap fields,
   verdict field를 생성한다.
6. full run은 별도 승인 전에는 실행하지 않는다. full run 뒤 Go여도 final artifact,
   Dev calibration, merge/push는 별도 사용자 결정이다.

## 10. 후속 순서

1. 이 설계서 검토 후 별도 상세 implementation plan을 작성한다.
2. plan은 evaluator helper, v3 exact inner OOF, v2.1 outer-refit comparator, metrics/
   bootstrap, CLI/output guard, focused tests 순서로 나눈다.
3. plan 구현 후 one-outer-fold smoke 결과를 검토한다.
4. 사용자가 full execution을 별도로 승인할 때만 fixed 3×5 evaluation을 실행한다.
5. formal verdict가 Go인 경우에만 Dev-calibration 또는 final-refit 설계를 새로
   논의한다.
