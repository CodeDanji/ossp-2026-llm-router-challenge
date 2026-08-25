# PromptBudget v2.1 연구 기준선 인수인계서

**상태:** 설계 확정. 다음 대화에서 구현 명세와 구현을 시작한다.

**목적:** v2.0의 결과를 덮어쓰지 않고, 절대값(absolute) 예측 구조를 유지한
`v2.1`을 누수 통제된 연구 기준선으로 만든다. v2.1은 배포 후보가 아니다. 이후
v3가 Light 대비 품질 이득과 추가 비용을 직접 다루는지 검증할 수 있도록 공정한
대조군과 진단 근거를 남긴다.

## 1. 발전 과정과 현재 판단

```text
v1-all-light
  └─ 안전하지만 업그레이드 없음
v2.0 absolute-linear
  └─ 품질/토큰 절대값을 각각 예측 → 비용 gate에서 전 tier fallback
v2.1 absolute-linear research baseline
  └─ 절대값 head는 유지, Train-only 정책 선택·직접 비용 보정·진단을 바로잡음
v3 delta-utility (후속 설계)
  └─ Light 대비 품질 이득과 추가 비용을 직접 모델링
```

v2.1의 목적은 v2를 억지로 통과시키는 것이 아니다. 안전성, 재현성, 원인 분해가
가능한 결과가 1차 성공 조건이다. v1보다 좋아지는 것은 목표이지만 필수는 아니다.
전 tier fallback도 원인이 정량적으로 설명되면 유효한 부정 결과다.

## 2. v2.0에서 확인한 사실

실행에 사용한 Train은 1,760개, Dev는 880개 콘텐츠 그룹이었다. Train/Dev의
canonical content-group 교집합은 0개였다. 따라서 동일 콘텐츠가 두 split에 직접
섞였다는 증거는 없다.

그러나 결과의 연구적 신뢰도는 낮다.

| 관찰 | 수치/증거 | 해석 |
|---|---|---|
| unrestricted v2 정책 | ax31 226, Light 15, Think 639 선택 | 고비용 모델을 거의 항상 선택했다. |
| 비용 | 실제 비용 비율 17.5163 | Fast 1.25, Balanced 2, Premium 4 예산을 모두 크게 초과했다. |
| 안전 동작 | 모든 tier가 `v1-all-light` fallback | 후보가 안전하지 않아 fallback한 것은 올바른 동작이다. |
| 실제 upgrade 이득률 | ax31 17.95%, Think 29.77% | 대부분의 prompt는 Light 대비 실제 이득이 없었다. |
| 예측 upgrade 이득률 | ax31 94.89%, Think 97.73% | 절대 품질 head의 차이가 upgrade 가치를 과도하게 낙관했다. |
| 품질 delta 상관 | ax31 0.0848, Think 0.3519 | 특히 ax31의 Light 대비 이득 신호가 약하다. |
| v2 OOF 목적의 구성 | 토큰 관련 손실 비중 91.68% | 서로 단위가 다른 7개 MSE를 합쳐 라우팅 목적과 불일치했다. |
| 99% 출력-token 잔차 multiplier | Light 16.0, ax31 15.76, Think 12.44 | 긴 꼬리 때문에 request cost upper가 과대해져 후보가 gate를 통과하기 어려웠다. |

Dev에서 unrestricted draft는 안전하지 않았고, 수정된 안전 gate는 모든 tier를
all-light로 고정했다. 이 결과는 “v2가 성공했다”가 아니라 “안전 장치가 위험한
후보를 배포하지 않았다”는 뜻이다.

## 3. v2.0의 문제, 원인, v2.1 해결책

### 3.1 데이터 누수는 세 종류로 분리한다

1. **직접 데이터 중복:** Train/Dev 콘텐츠 그룹 중복은 발견되지 않았다.
2. **후보 선택 누수:** v2.0은 head feature/alpha만 nested-CV로 선택했다. tier
   정책은 `TierSettings(0, 0, 0, 1, 100)`으로 고정되어 정책 탐색이 Train 내부에서
   끝나지 않았다.
3. **평가 적응 위험:** Dev 결과를 이미 관측하고 해석했으므로, Dev를 새 설계의
   최종 독립 시험으로 부를 수 없다. Dev 결과를 보고 v2.1을 다시 조정해서는 안 된다.

v2.1은 2번을 제거하고 3번을 명확히 라벨링한다. 새 데이터를 확보할 수 없으므로
외부 일반화 성능을 증명할 수는 없다.

### 3.2 정책 목표와 학습 목표가 달랐다

v2.0은 모델별 절대 품질과 토큰을 잘 맞추는 head를 골랐지만, 실제 라우팅 질문은
“Light보다 좋아질 가능성이 추가 비용을 정당화하는가”이다. 단위가 다른 raw MSE의
합은 정책의 실제 공식 점수와 예산을 대변하지 못했다.

**v2.1 해결:** 절대값 head는 유지하되, head와 tier 정책의 선택 점수를 inner
validation에서 실제 라우팅 공식 점수로 계산한다. 비용 gate를 먼저 통과한 후보만
one-standard-error 규칙으로 선택한다.

### 3.3 비용 보정 대상이 불완전했다

v2.0은 출력 토큰의 실제/예측 비율로 multiplier를 만들었다. 입력 토큰 요금과
고정비용이 빠져 있고, 전역 99% 값 하나가 긴 꼬리를 모든 prompt에 전파했다.

**v2.1 해결:** 기존 input/output head에서 모델별 예측 화폐 비용을 계산하고,
`실제 화폐 비용 / 예측 화폐 비용`을 Train 내부 OOF에서 보정한다. prompt 길이
bucket이 충분한 표본을 갖지 못하면 전역 값으로 fallback한다.

### 3.4 v2.0을 v3처럼 고치지 않는다

v2.1에는 Light-relative delta 품질 head, 추가 비용 head, upgrade 확률 head를
넣지 않는다. 이들은 v3의 가설이며 v2.1 대조군에 넣으면 비교가 무의미해진다.

## 4. v2.1 고정 설계

### 4.1 데이터 경계

- **Train:** 구조, head, 정책, 비용 보정 multiplier를 선택하고 fit하는 유일한
  outcome-bearing 데이터다.
- **Dev:** 이미 관측된 split이다. 아티팩트를 변경하지 않는 단발
  `previously_observed_dev_confirmation`만 허용한다. 이 결과는 정책 재선택이나
  재보정에 쓰지 않는다.
- **새 locked holdout:** 현재 없다. 따라서 `external_generalization_claim`은 항상
  `false`다. 기존 locked evaluator의 toy smoke 결과를 연구 성능 증거로 쓰지 않는다.

### 4.2 Train-only nested/cross-fit 흐름

1. 콘텐츠 그룹 기준 고정 seed `(137, 271, 811)`와 5 outer fold를 사용한다.
2. 각 outer-train 안에서만 4 inner fold로 후보를 고른다.
3. inner validation의 cost upper multiplier는 해당 inner-fit 데이터의 grouped
   cross-fitted OOF 예측으로만 구한다.
4. 선택된 후보를 outer-train 전체에 fit하고 outer-test에 한 번 적용한다.
5. outer-test outcome은 보고서·CI 전용이다. 최종 후보, 비용 보정, fit에 재사용하지
   않는다.
6. 최종 artifact는 전체 Train에서 동일한 inner 선택 절차를 수행한다. 이후 별도의
   고정 grouped cross-fit OOF로 final cost multiplier를 만들고, 마지막으로 전체
   Train에 head를 fit한다.

### 4.3 사전 등록 후보 그리드

하나의 공통 absolute head를 세 tier가 공유한다. tier별 별도 head는 만들지 않는다.

| 대상 | 후보 |
|---|---|
| common head | sparse feature 수 `64, 256` × ridge alpha `1, 10, 100` |
| Fast | lambda `0.5, 1, 2` × 공통 최소 이득 `0.05, 0.10` × 최대 상대비용 `1.00, 1.25` |
| Balanced | lambda `0.05, 0.10, 0.20` × 최소 이득 `0, 0.05` × 최대 상대비용 `2, 4` |
| Premium | lambda `0, 0.01, 0.05` × 최소 이득 `0, 0.05` × 최대 상대비용 `4, 10` |
| extra safety multiplier | 항상 `1.0` |

각 common-head 후보에서 tier별 최선의 정책을 독립 선택하고, 세 tier의 고정 가중
공식 점수를 합쳐 common head를 선택한다. 이는 additive 공식 점수와 하나의 runtime
head 구조를 함께 만족한다.

### 4.4 직접 비용 보정과 안전 gate

- 예측 화폐 비용은 기존 input/output absolute head와 routing policy 요금표로 계산한다.
- nonconformity는 `actual monetary cost / predicted monetary cost`다.
- 길이 bucket은 문자 수 `<=512`, `513–2048`, `>2048`으로 고정한다.
- 해당 bucket의 Train OOF 표본이 100개 미만이면 전역 multiplier를 사용한다.
- quantile은 Train OOF에서만 계산하며, 후보의 final upper cost에 적용한다.
- 각 후보는 모든 inner validation fold에서 아래 gate를 통과해야 한다.

| Tier | upper cost ratio 조건 |
|---|---|
| Fast | `<= 1.15` (`1.25 - 0.10`) |
| Balanced | `< 2.00` |
| Premium | `< 4.00` |

Gate 통과 후보가 없으면 해당 tier는 명시적으로 `v1-all-light` fallback한다. 이는
정상 연구 결과이지 파이프라인 오류가 아니다.

### 4.5 후보 선택과 결과 판정

- 비용 gate를 통과한 후보만 inner validation의 해당 tier 공식 점수 평균으로 정렬한다.
- 최상 후보의 one-standard-error 이내 후보 중 더 보수적인 정책(낮은 upgrade,
  낮은 최대 상대비용)을 선택한다. 동점은 고정 grid 순서로 결정한다.
- v1-all-light와 official-prompt-heuristic을 모두 비교 기준선으로 보고, 콘텐츠 그룹
  paired-bootstrap 95% CI를 계산한다.
- 보고 라벨은 다음 네 가지다.
  - `integrity_pass`: Train-only 경계, provenance, 검증이 모두 지켜짐
  - `v1_improvement_supported`: paired-bootstrap 95% CI 하한이 v1 대비 0 초과
  - `heuristic_improvement_supported`: CI 하한이 heuristic 대비 0 초과
  - `external_generalization_claim`: 항상 `false`

Dev 확인은 위 라벨을 바꾸지 않으며 `exploratory_confirmation`으로만 기록한다.

### 4.6 아티팩트와 Dev 확인

- 결과는 `build/promptbudget-v2.1/`에만 생성한다.
- 현재 기본 runtime artifact
  `src/promptbudget/resources/artifact.json`은 절대 덮어쓰지 않는다.
- 연구 artifact에는 `research_only: true`, Train/Dev 해시, code/policy hash,
  후보 그리드, fold/seed digest, 선택 근거를 provenance로 담는다.
- Dev 실행은 append-only ledger가 artifact/manifest/Dev digest 조합을 먼저 예약한
  뒤에만 허용한다. 같은 조합의 두 번째 실행은 거부한다.
- Dev 평가 전후 artifact와 manifest hash가 같아야 한다.

## 5. 필수 진단 산출물

원문 prompt와 개별 outcome은 결과 파일에 쓰지 않는다. 집계값과 digest만 저장한다.

1. **integrity/provenance:** split별 콘텐츠 그룹 수와 교집합, 입력·outcome·정책·코드·
   후보 그리드 hash, fold assignment digest.
2. **candidate ledger:** tier별 모든 후보의 fold별 gate 결과, 탈락 사유, 평균 점수,
   표준오차, one-SE 선택 근거.
3. **prediction/calibration:** 모델별 quality error, 실제/예측 및 upper/실제 화폐 비용
   비율의 p50/p90/p99, bucket 표본 수와 global fallback 여부.
4. **routing analysis:** tier별 모델 선택 수, 실제 비용 비율, 선택 upgrade의 실제
   품질 이득/손실률, Light 대비 oracle regret.
5. **comparison:** v1·heuristic 대비 per-tier/weighted 점수 차이와 콘텐츠 그룹
   paired-bootstrap 95% CI.
6. **Dev confirmation:** observed-split 라벨, ledger reservation id, artifact 불변 hash,
   결과. 이 파일은 독립 holdout 보고서가 아니다.

## 6. 구현 시 반드시 검증할 항목

- 콘텐츠 그룹이 outer-train/outer-test 및 inner-fit/validation에 동시에 나타나지 않는다.
- outer-test outcome을 바꿔도 후보 선택, final artifact, final multiplier가 바뀌지 않는다.
- 모든 feature 선택과 residual multiplier가 해당 fit 데이터에서만 유도된다.
- 비용 보정이 output token이 아니라 화폐 비용을 사용한다.
- bucket 표본 부족 시 global fallback하고, 보정 불능이면 non-Light를 거부한다.
- Fast의 0.10 margin 및 Balanced/Premium strict boundary가 정확히 적용된다.
- 정책 점수 기반 선택이 raw MSE 기반 선택과 다른 fixture에서 의도대로 동작한다.
- one-SE tie-break, 결정론적 seed 재실행, v1 fallback, Dev 단발 ledger를 단위 테스트한다.
- 진단 파일이 원문 prompt·개별 outcome을 노출하지 않는지 테스트한다.

Windows에서 기존 전체 테스트가 `fcntl`, `resource`, symlink/FIFO 권한 차이로 실패할 수
있다. v2.1의 targeted test는 별도로 전부 통과해야 하며, 플랫폼 비호환 전체-suite
실패는 코드 실패와 분리해 기록한다.

## 7. v3로 넘길 근거와 경계

v2.1 결과가 v1/heuristic보다 낮거나 fallback 비중이 높다면, 이는 absolute head가
Light-relative upgrade 가치를 충분히 판별하지 못한다는 증거가 된다. 그때 v3는 다음을
별도 설계한다.

- Light 대비 **품질 delta**와 **추가 화폐 비용**을 직접 예측한다.
- Train nested-CV 안에서 delta-utility 정책과 uncertainty를 선택한다.
- v2.1과 같은 비용 gate·Train-only 경계·보고 규약을 유지한다.
- 새 미사용 holdout이 생기기 전에는 v3도 외부 일반화나 배포 승격을 주장하지 않는다.

v2.1에 위 delta 요소를 넣어서는 안 된다. 그래야 v2.1이 v3의 공정한 대조군으로
남는다.

## 8. 다음 대화의 시작 지시

1. 이 문서를 요구사항의 단일 기준으로 읽는다.
2. 별도의 구현 계획 문서를 작성하고 검토 승인을 받은 뒤에만 코드를 수정한다.
3. `v2.0` artifact/report는 보존한다. 기본 runtime artifact와 배포 경로를 바꾸지
   않는다.
4. 구현 후 Train nested-CV와 단 한 번의 Dev confirmation을 실행한다. Dev 결과에
   맞춰 재구현하거나 재실행하지 않는다.
