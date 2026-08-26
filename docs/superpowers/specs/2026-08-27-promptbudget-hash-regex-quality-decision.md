<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# PromptBudget hash-regex quality-objective 결정 기록

**상태:** 구현 승인됨 · 아직 구현/후보 선택/최종 artifact refit은 시작하지 않음

**결정일:** 2026-08-27

**독자:** 이후 이 결정을 다시 검토하거나 구현·제출을 맡는 사람

## 1. 한 문장 결정

기존 **hash-regex feature family, cost predictor family, batch-level Lagrange
allocator와 Premium AX31 fill**은 유지하고, quality 학습 목적만 `Raw`, `Weighted
uplift`, `regret-weighted raw quality loss`로 제한해 Train-only grouped nested
evaluation에서 비교한다. Direct uplift는 독립 후보가 아니라 선형 동치성 control이다.

## 2. 문제의 맥락과 이 결정을 내린 이유

대회의 제출은 prompt와 tier만으로 한 batch의 각 문항에 모델 하나를 골라야 한다.
Fast/Balanced/Premium의 비용 상한은 각각 Light 전체 비용의 1.25/2/4배이고, 어느
tier든 상한을 넘으면 그 tier 점수는 0이 된다. 최종 점수의 tier weight는
Fast/Balanced/Premium 순서로 0.4/0.3/0.3이다. 따라서 평균 prediction loss가 낮은
것만으로 좋은 router라고 할 수 없고, **batch route의 실제 official score와 cap
pass**가 최우선이다.

현재 공개 hash-regex baseline은 14개의 구조·regex feature와 256-bin signed word
unigram/bigram hash를 사용한다. 세 raw quality ridge head와 세 log-cost ridge
head를 만들고, batch Lagrange로 route한 뒤 Premium에서 AX31 fill을 수행한다. 공개
Dev report의 final score는 `0.695369318182`이며 Fast/Balanced/Premium actual ratio는
각각 `1.235989/1.961506/3.985205`다. 이는 유용한 재현 기준이지만, Dev로 safety를
보정해 만들어진 값이므로 hidden 성능 추정이나 후보 선택 근거가 아니다.

별도로 수행한 Dev counterfactual 진단에서는 full hash-regex `0.695369`에 대해
quality oracle은 safety를 다시 보정했을 때 `0.790653` (+`0.095284`), cost oracle은
`0.705625` (+`0.010256`), 둘 다 oracle인 경우 Lagrange `0.803438`, exact allocation
`0.803778`이었다. 이 수치는 Dev outcome을 사용한 상한 진단일 뿐이지만,
quality prediction이 가장 큰 병목이고 exact knapsack의 순수 추가 이득은 약
`0.000341`임을 보여 준다.

이 결론은 delta-linear v3의 formal nested-CV no-go와 혼동하면 안 된다. v3는 다른
router family였으며, 그 no-go는 v3가 hash-regex보다 나쁘다는 증거가 아니다. 여기서
v3 결과를 쓰는 유일한 이유는 **Train-only nested selection, cap admission, Dev
leakage 금지**가 필요하다는 운영상 교훈이다.

## 3. 대회 규칙과 고정 경계

### 유지하는 것

| 영역 | 고정 결정 | 이유 |
| --- | --- | --- |
| feature | 현 `hash_regex.raw_feature_vector`: 14 dense feature + 256 signed hash bin | Dev leave-one-family 진단에서 hash가 더 큰 기여를 보였고, 대규모 feature 탐색은 이번 질문을 흐린다. |
| fit capacity | 256 hash bin, ridge `alpha=100.0` | 공개 artifact의 선택값을 고정한다. alpha를 다시 고르면 quality objective 효과와 regularization 탐색이 섞인다. |
| cost | log-cost ridge **family와 target** | cost oracle의 잠재 이득은 있지만 새 cost family보다 quality objective가 우선이다. 각 fold에서는 반드시 outer-train으로 refit한다. |
| allocation | `select_models` batch Lagrange, deterministic tie break, Premium `fill_ax31_upgrades` | batch-dependent routing은 규칙상 허용되고 baseline도 채택한다. exact DP 추가 이득이 매우 작다. |
| runtime output | 기존 `HashRegexArtifact`의 raw score head와 log-cost head | uplift 결과는 raw-score coefficient로 합성해 저장한다. 새 runtime dependency나 artifact schema 분기는 만들지 않는다. |

### 명시적 비목표

- 새 router family 또는 대규모 feature 재설계
- exact knapsack을 제출 runtime에 넣기
- allocator/Premium fill 규칙을 후보마다 바꾸기
- Dev score로 quality family, weight, lambda, safety를 고르기
- published full-Train artifact를 Train outer-test의 seed/comparator로 사용하기
- hidden budget guarantee 또는 Dev score의 일반화 보장 주장

Batch composition에는 의존할 수 있지만 prompt ID, input 순서, split/episode metadata에는
의존할 수 없다. 모든 새 artifact와 runtime route는 동일 prompt multiset에 대해
결정적이며 ID/order permutation invariant여야 한다.

## 4. 비교할 quality objective

모든 후보는 같은 input feature, fixed `alpha=100.0`, outer-train-refit log-cost head,
same batch allocator, same Premium fill을 쓴다. 최종 allocator 입력은 항상 모델별
Q-like raw score와 predicted cost다. regret 값을 allocator에 직접 넣지 않는다.

### 4.1 Raw quality baseline

각 모델의 공개 Train score `Q_{i,m}`를 현재처럼 ordinary ridge로 예측한다. 이 후보는
항상 포함되며, 새 후보가 모든 gate를 통과하지 못하면 최종 선택이다.

### 4.2 Direct uplift: 동치성 control

Light score와 upgrade gain을 따로 fit한다.

\[
\Delta_{i,m}=Q_{i,m}-Q_{i,L}, \qquad
\widehat Q_{i,m}=\widehat Q_{i,L}+\widehat\Delta_{i,m}
\]

여기서 `m`은 AX31 또는 Think다. 동일 feature, alpha, unweighted ridge라면 ridge의
선형성 때문에 이 방식은 Raw score head와 동치여야 한다. 따라서 이는 winner 후보가
아니라 coefficient/prediction/route가 tolerance 내에서 Raw와 일치하는지 확인하는
control이다. 불일치는 구현, centering, clipping 또는 regularization 처리의 오류다.

### 4.3 Weighted uplift

Light head는 raw loss를 유지하고, AX31/Think uplift head만 weighted least squares로
fit한다.

\[
w_{i,m}=1+\gamma\max(0,\Delta_{i,m}),\qquad
\widetilde w_{i,m}=w_{i,m}/\operatorname{mean}(w_{\cdot,m})
\]

\[
\mathcal L=(Q_{i,L}-\widehat Q_{i,L})^2+
\sum_{m\in\{AX,Think\}}\widetilde w_{i,m}
(\Delta_{i,m}-\widehat\Delta_{i,m})^2
\]

`gamma`는 `{0, 1, 2, 4}`만 평가한다. head별 mean-one 정규화는 gamma가 데이터
loss와 ridge penalty의 상대적 크기, 즉 사실상의 alpha까지 바꾸는 혼입을 막는다.
`gamma=0`은 Direct uplift control과 같다.

### 4.4 Regret-weighted raw quality loss

이 후보는 raw score를 계속 출력하되, 오류가 비용 제약 route에서 더 큰 손실을
만들 action에 WLS 가중치를 준다. 각 inner OOF batch와 tier `t`에서 실제 Train
score/cost로 그 tier cap을 만족시키는 batch Lagrange penalty `lambda_t`를 구한다.
penalty의 cost normalization은 production `select_models`와 동일하게 해당 batch의
Light total predicted-cost 단위로 정의한다.

\[
r_{i,m,t}=\max_a\{Q_{i,a}-\lambda_t C_{i,a}/C_{L,\mathrm{total}}\}
-\{Q_{i,m}-\lambda_t C_{i,m}/C_{L,\mathrm{total}}\}
\]

\[
R_{i,m}=0.4r_{i,m,Fast}+0.3r_{i,m,Balanced}+0.3r_{i,m,Premium}
\]

`R`을 zero-safe mean scaling한 뒤, `eta in {0, 1, 2, 4}`로 head별 mean-one WLS
weight를 만든다. 구체적으로 all-zero `R`이면 모든 weight는 1이고, 그렇지 않으면
`u=1+eta*R/mean(R)`, `w=u/mean(u)`다. 이 loss는 세 raw quality head 모두에
action별로 적용한다. lambda 자체는 hyperparameter search가 아니라 inner OOF의
official tier budget에서 파생하므로 대규모 lambda search를 하지 않는다.

## 5. Train-only nested protocol

### 5.1 Split과 정보 경계

고정 `3 seeds x 5 grouped outer folds`를 사용한다. group은 canonicalized prompt
content digest다. 기존 `train_hash_regex.py`의 `np.arange(rows) % folds` OOF는
group-aware가 아니므로 이 연구에 사용할 수 없다.

각 outer fold에서 다음만 outer-test 전에 허용된다.

1. outer-train을 4 grouped inner folds로 나눈다.
2. 공통 feature matrix, group schedule, cost target을 한 번 materialize한다.
3. cost head family를 inner-training/outer-train에서 refit하고, 각 quality candidate의
   inner OOF prediction을 만든다.
4. candidate별 gamma 또는 eta를 inner OOF official routing score와 safety admission만
   사용해 고른다.
5. candidate별 provisional safety ratio를 **그 candidate의 inner OOF quality/cost
   prediction과 final Premium fill을 포함한 route**로 고른다.
6. 선택된 candidate와 cost head를 outer-train 전체로 refit하고, 고정한 safety와
   allocator로 outer-test를 정확히 한 번 routing/score한다.

outer-test outcome, cost, score, budget pass는 6번 이후 metrics에만 쓰며 fit, lambda,
gamma/eta, safety, tie break에 쓰지 않는다. 같은 규칙으로 Raw도 outer-train에서
refit한다. published hash-regex artifact는 outer-test에 절대 load하지 않는다.

### 5.2 Provisional safety와 최종 joint calibration

Quality target이 route mix를 바꾸므로 safety를 winner 뒤까지 완전히 미루면 후보가
불공정해진다. 그래서 비교 동안에는 candidate별 provisional safety가 필요하다. 이는
inner OOF에서 tier별 실제 cap을 모두 통과하는 safety grid 값 중 weighted official
score가 최대인 값을 택하는 규칙이며, final Premium fill까지 실제로 채점한다.

winner가 정해진 뒤에만 full-Train grouped OOF로 final gamma/eta와 **joint
cost-safety calibration**을 한 번 수행한다. 이 결과를 고정하고 full Train으로
quality/cost heads를 refit해 제출 artifact를 만든다.

## 6. 선택, Dev, fallback 규칙

새 quality family가 Raw를 교체하려면 모두 충족해야 한다.

1. 15 outer fold의 모든 tier가 actual budget cap을 통과한다.
2. Raw 대비 paired weighted score improvement가 3개 seed 평균 중 적어도 2개에서
   양수다.
3. 15 fold 중 적어도 8개가 Raw보다 양수다.
4. 모든 outer-test record를 정해진 방식으로 집계한 평균 weighted score improvement가
   `>= 0.002`다.

여러 후보가 통과하면 평균 improvement가 가장 큰 후보를 택한다. 동점은 더 큰
minimum seed improvement, 더 낮은 maximum actual cost ratio, candidate 순서
`Raw -> Weighted uplift -> Regret-weighted`로 정한다. 아무 후보도 통과하지 않으면
Raw를 유지한다.

Dev는 family/weight/lambda/safety selection에 쓰지 않는다. final frozen candidate의
one-pass sanity check에서 tier 하나라도 budget cap을 넘거나 technical validity/runtime
check가 실패하면 **재튜닝하지 않고**, 사전에 지정한
`baselines/hash-regex-public.v1.json` frozen Raw artifact로 fallback한다. Dev score가
낮거나 route mix가 다르다는 사실만으로는 fallback하지 않는다. 그렇게 하면 Dev가
다시 tuning set이 된다.

## 7. 무엇을 보고 판단하는가

prediction MAE/RMSE/rank correlation은 diagnosis에 기록하지만 winner 기준은 아니다.
후보별/outer-fold별로 다음을 우선 report한다.

- weighted official routing score와 tier score
- Fast/Balanced/Premium actual cost ratio와 budget pass
- tier별 model count와 route
- seed/fold별 Raw와의 paired difference 및 안정성 gate 결과
- 선택된 gamma/eta, provisional safety, tier-derived lambdas

최종 report는 raw prompt, episode ID, outcome text를 기록하지 않는다. canonical group
digest와 aggregate metric만 기록한다.

## 8. 검증·운영 안전장치

- outer-test outcome을 바꾸어도 inner selection, lambda, safety, fit이 바뀌지 않는
  synthetic test를 둔다.
- Direct uplift control은 Raw와 tolerance 내 동일 prediction/route임을 test한다.
- 모든 candidate는 full final allocator, 특히 Premium fill을 사용했음을 test한다.
- runtime artifact는 기존 schema와 동일한 raw score heads로 읽혀야 하며,
  ID/order permutation invariant regression test와 runtime resource check를 통과해야
  한다.
- evaluator는 Train paths만 받으며 Dev path, runtime resource write, default artifact
  overwrite를 거부한다. full 3x5 execution은 focused test와 one-fold smoke 후 별도
  사용자 승인 때만 실행한다.

## 9. 추적 가능한 근거

- rule/runtime: `docs/CHALLENGE_RULES.md`, `docs/RUNTIME.md`, `docs/SCORING.md`
- baseline: `baselines/hash_regex.py`, `baselines/train_hash_regex.py`,
  `baselines/hash-regex-public.v1.json`,
  `baselines/hash-regex-public-dev-report.v1.json`, `baselines/README.md`
- v3 discipline only: `build/promptbudget-v3/nested-evaluation.json`,
  `docs/superpowers/specs/2026-08-26-promptbudget-v3-nested-evaluation-design.md`
