<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# PromptBudget v2.1 구현 완료 인수인계서

**상태:** 구현·Train 실행·공개 Dev 정책 보정 완료. 다음 대화는 이 문서를 기준으로
결과 분석과 v3 설계를 시작한다.

**코드 기준:** 로컬 `main`의 merge commit `8ec1789` (`merge: PromptBudget v2.1
public Dev calibration`). 원격 push는 하지 않았다.

**범위:** v2.1은 연구용 artifact 생성 및 정책 보정이다. 기본 runtime artifact와
배포 경로는 변경하지 않았고, 숨겨진 평가 성능이나 배포 준비 완료를 주장하지 않는다.

> 이전 `2026-08-25-promptbudget-v2.1-research-handoff.md`는 구현 전 설계 기록이다.
> 특히 Dev를 한 번만 확인해야 한다는 부분은 공식 가이드 재검토 후 정정되었으므로,
> 현재 판단에는 이 문서와 `PROMPTBUDGET_V2_OPERATIONS.md`를 사용한다.

## 1. 한눈에 보는 발전 과정

```text
v1 all-Light
  └─ 비용에는 안전하지만 품질 업그레이드 없음
v2 absolute-linear
  └─ 절대 품질·토큰 예측은 했으나 비용·정책 목표가 어긋나 전 tier fallback
v2.1 absolute-linear research baseline
  └─ Train OOF 화폐비용 보정 + 공개 Dev의 반복 tier 정책 보정
v3 (다음 설계)
  └─ Light 대비 품질 이득과 추가 비용의 delta-utility를 직접 다룰 후보
```

v2.1의 성공은 **안전한 연구 파이프라인과 재현 가능한 공개 Dev 보정이 작동했다**는
의미다. absolute-quality head가 좋은 router라는 결론은 아직 아니다.

## 2. v2에서 무엇이 문제였나

v2는 모델별 절대 품질과 token 사용량을 예측하는 linear head였다. 관측된 실패는
다음과 같았다.

| 문제 | v2 관측 | 해석 |
| --- | --- | --- |
| 과도한 upgrade | unrestricted 정책이 ax31 226, Think 639, Light 15개를 선택 | 절대 품질 차이가 upgrade 가치를 지나치게 낙관했다. |
| 예산 초과 | 실제 비용비율 17.5163 | Fast 1.25, Balanced 2, Premium 4 한도를 모두 크게 초과했다. |
| 안전 fallback | 모든 tier가 all-Light로 귀결 | 안전장치는 올바르게 위험 후보를 막았지만 router 개선은 없었다. |
| 약한 relative signal | 실제 upgrade 이득률: ax31 17.95%, Think 29.77%; 예측 이득률: ax31 94.89%, Think 97.73% | “절대 품질을 잘 맞추는 것”과 “Light보다 올릴 가치가 있는 것”은 다른 문제다. |
| 목적 불일치 | v2 OOF 손실에서 token 관련 항이 91.68% | 단위가 다른 7개 raw MSE 합이 공식 tier 점수·예산 목적을 대변하지 못했다. |
| 과도한 cost upper | output-token 99% residual multiplier: Light 16.0, ax31 15.76, Think 12.44 | 긴 꼬리와 전역 보정이 정책 gate를 지나치게 보수적으로 만들었다. |

Train/Dev canonical content-group 교집합은 0개였다. 직접 콘텐츠 중복은 발견되지
않았지만, 그것만으로 router가 일반화한다고 결론 내릴 수는 없다.

## 3. v2.1에서 채택한 설계와 구현 흐름

### 3.1 Train에서 학습·보정하는 부분

`tools/train_oof.py`는 Train 1,760문항에서 다음을 수행한다.

1. prompt-only dense/hash feature로 세 모델의 absolute quality, output token, Light
   input token head를 fit한다.
2. content-group 기반 3 seed × 5 outer fold와 inner grouped fold를 이용해 common head
   후보를 비교한다. outer-test는 보고 전용이다.
3. Train OOF에서 `actual monetary cost / predicted monetary cost`의 높은 99% 분위수를
   모델·문자수 bucket별로 계산한다. bucket은 short `<=512`, medium `<=2048`, long
   `>2048`이고 표본 100개 미만이면 global multiplier를 쓴다.
4. v2 artifact format에 이 cost calibration을 함께 저장한다. v1 artifact 읽기 호환성은
   유지한다.

`src/promptbudget/policy.py`는 runtime에서 prompt 본문 길이로 bucket을 고르고 이
multiplier를 적용한다. 따라서 artifact 자체는 prompt와 tier만으로 모델을 선택하며,
Dev outcome이나 split/id/순서는 읽지 않는다.

### 3.2 Dev에서 보정하는 부분

`tools/calibrate_policy.py`는 공식 공개 Dev 880문항에서 tier별 정책만 반복 보정한다.
Train에서 학습한 head와 OOF cost calibration은 고정이다.

각 tier에서 다음 576개 조합과 명시적 all-Light 안전 후보, 총 577개를 검사한다.

- `lambda_cost`: 9개 값
- 공통 minimum gain (`ax31`, `axk1-think`에 동일 적용): 4개 값
- `max_relative_cost`: 4개 값
- `safety_multiplier`: `0.25`, `0.5`, `0.75`, `1.0`
- all-Light fallback 1개

공식 scorer의 **실제 Dev 비용비율**로 예산을 판정한다. Fast는 `<=1.15`로 1.25
한도에서 0.10 여유를 요구하고, Balanced/Premium은 각 공식 한도보다 엄격히 작아야
한다. 통과 후보 중 quality point가 최대인 것을 고르고, 동점이면 더 큰 safety
multiplier, 더 낮은 실제 비용을 선호한다.

## 4. 구현 중 발견한 문제와 해결

### 4.1 Dev 사용에 대한 오판과 정정

초기 v2.1 설계는 Dev를 이미 본 데이터라 보고 “단 한 번의 exploratory confirmation만
허용하며 정책 선택에는 쓰지 않는다”고 정했다. 이는 연구 논문의 독립 test 원칙을
이 과제에 과도하게 적용한 **오판**이었다.

공식 가이드 `docs/skt/00_official_challenge_and_submission_guide.md`는 다음을 명시한다.

- 공개 Train 1,760·Dev 880의 prompt, score, token, generation은 학습·검증에 사용 가능
- Train으로 score/비용 예측기를 만들고 Dev에서 tier threshold와 비용 안전계수를 보정
- 진짜 독립 평가는 제출 후 숨겨진 입력이 담당

따라서 지금의 원칙은 다음과 같다.

- **허용:** Dev를 반복해서 `lambda_cost`, gain, relative-cost cap, safety multiplier,
  후보 artifact/정책 선택에 사용
- **금지:** Dev/hidden outcome을 runtime feature, Train target, artifact의 prompt별 lookup,
  split/id/order/source feature, API 후보 답변 비교·재시도로 사용
- **해석:** Dev 최고 결과는 제출 후보 선택 근거이지 외부 일반화 증명은 아니다.

이 정정은 `docs/PROMPTBUDGET_V2_OPERATIONS.md`, v2.1 calibration design, 그리고
`tools/calibrate_policy.py`에 반영되어 있다. 과거 one-time Dev ledger 코드는 남아
있지만 공개 Dev policy calibration의 권장 경로는 아니다.

### 4.2 Train 정책 gate의 분모 오류

첫 full Train 실행은 artifact를 만들었지만 모든 head 후보의 `weighted_score`가 0이고
final tier 선택이 `null`이었다. 원인은 conservative monetary multiplier가 적용된
**예측** 선택비용을 실제 Light 비용으로 나눈 것이었다. multiplier가 1보다 크면
all-Light route도 ratio 1보다 훨씬 커져 모든 후보가 탈락한다.

수정: `tools/train_oof.py`에서 정책 admission ratio를 선택 route의 예측비용 /
동일 행의 예측 all-Light 비용으로 계산했다. 따라서 all-Light ratio는 정확히 1이다.
`test_all_light_predicted_route_has_one_relative_cost_even_when_calibrated`가 이를 고정한다.

### 4.3 Dev 그리드가 safety만 탐색하던 문제

기존 `calibrate_policy.py`는 Dev에서 safety multiplier만 바꾸고 lambda/gain/max cost는
Train artifact 값으로 동결했다. 공식 workflow와 맞지 않았다.

수정: 모든 정책 knob을 Dev grid로 옮겼고, report에 후보 수·선택 settings·공식 비용과
점수를 기록한다. `test_public_dev_grid_varies_every_policy_knob`가 네 knob의 변화를
검증한다.

### 4.4 Fast에 안전 후보가 없던 문제

첫 Dev grid 실행에서 Fast는 모든 non-Light 후보가 예산 여유 조건에 실패했다. 그렇지만
그리드에 all-Light 설정이 없어서 “fallback”으로 기록되었고, 같은 all-Light를 정식
후보와 비교하지 못했다.

수정: 576개 grid에 explicit `v1-all-light` candidate를 추가해 총 577개를 탐색한다.
최종 Fast는 이 후보를 정상 선택했다.

### 4.5 실행 성능 및 배열 호환성

full Train grouped CV가 오래 걸렸다. fold별 feature matrix를 alpha마다 다시 만들지
않고 재사용했고, SVD 한 번으로 여러 ridge alpha를 계산하도록 바꿨다. 또한 NumPy
array에서 `if not predicted`가 예외를 내는 부분을 길이 검사로 고쳤다.

관련 커밋: `9cbf91d`, `7a52aa0`, `fe2da96`.

## 5. 최종 실행 결과

### 5.1 Train

- 입력: Train 1,760문항
- 선택 common head: sparse feature **64**, ridge alpha **1.0**
- inner weighted score:
  - `(64, 1.0)`: `0.6435795455` (선택)
  - `(64, 10.0)`: `0.6402130682`
  - `(64, 100.0)`: `0.6392471591`
  - 256-feature 후보들은 `0.5107670455`~`0.6391477273`
- global monetary multiplier: ax31 `7.0494`, Light `8.9929`, Think `10.8947`
- Train artifact SHA-256: `c904145a57bcc2530ecd47bf4fb55b56a8f75f597539e8a4b5804efccb4a7c33`

### 5.2 공개 Dev 정책 보정

| Tier | 선택 | 실제 비용비율 | 점수 | 선택 분포 |
| --- | --- | ---: | ---: | --- |
| Fast | explicit all-Light | 1.000000 | 0.619318181818 | Light 880 |
| Balanced | λ=0, gain=0, max relative cost=4, safety=1 | 1.84604335938 | 0.694318181818 | ax31 790, Light 87, Think 3 |
| Premium | Balanced와 동일 | 1.84604335938 | 0.694318181818 | ax31 790, Light 87, Think 3 |

Dev calibrated artifact SHA-256:
`16bb7430ba89796a374ee72092694702cdff59b08b0515f59c7830c1671ca346`.

중요한 진단: Balanced/Premium의 conformal/aggregate predicted-upper ratio는 약 52.32/
31.92, Fast all-Light도 약 46.45/25.34로 매우 높다. 이는 Train cost multiplier와
absolute token head가 runtime 비용을 매우 보수적으로 예측한다는 신호다. 공개 Dev의
실제 비용 gate는 통과했지만, 이 불일치는 v3에서 반드시 원인 분석할 핵심 리스크다.

## 6. 산출물·경로·검증 상태

### 코드와 결과 경로

- 코드(main): `C:\Users\alvin\바탕 화면\창프\오픈소스\router-repo`
- 실행 결과 artifact/report: `C:\Users\alvin\바탕 화면\창프\오픈소스\router-repo-promptbudget\build\promptbudget-v2.1`
  - `artifact.json`, `manifest.json`, `train-nested-cv.json`
  - `dev-calibrated-artifact.json`, `dev-calibrated-manifest.json`,
    `dev-policy-calibration.json`
- 기본 runtime artifact(변경 없음):
  `C:\Users\alvin\바탕 화면\창프\오픈소스\router-repo\src\promptbudget\resources\artifact.json`

결과 artifact는 연구 build worktree에 있고 main runtime에는 설치하지 않았다. v3 설계
전에는 이 경계와 결과 파일을 보존한다.

### 검증

다음 focused suite는 **22개 모두 통과**했다.

```console
PYTHONPATH=src python -m unittest \
  tests.promptbudget.test_artifact_policy \
  tests.promptbudget.test_safety \
  tests.promptbudget.test_train_oof_v2 \
  tests.promptbudget.test_calibrate_policy_v2 \
  tests.promptbudget.test_locked_eval -v
```

전체 `unittest discover -s tests -v`는 Windows에서 기존 Linux 전용 의존성(`fcntl`,
`resource`, FIFO/symlink 권한) 및 기존 SPDX 검사 때문에 실패한다. 이는 v2.1 focused
suite 실패가 아니며, 다음 대화에서 v3 로직 실패로 오해하지 않는다.

## 7. v3 분석·설계의 시작점

v3는 새 기능을 바로 구현하지 말고 다음 질문을 먼저 답한다.

1. absolute quality head의 model 간 차이가 실제 Light-relative quality delta와 얼마나
   상관하는가? model별·길이 bucket별로 분석한다.
2. 비용 upper ratio가 25~52까지 커진 주원인이 input token, output token, fixed cost,
   bucket fallback 중 무엇인가? Train OOF 분포(p50/p90/p99)를 분해한다.
3. Balanced/Premium이 동일 정책·분포가 된 이유가 예측 분별력 부족인지, grid의
   제약/동점 규칙 때문인지 확인한다.
4. Fast에서 all-Light가 최선인 것이 실제 품질-비용 경계인지, 현재 absolute utility와
   conservative cost estimator의 산물인지 비교한다.

유력한 v3 가설은 Light 대비 다음을 직접 예측하는 것이다.

- `quality_delta(model, Light)`
- `incremental_monetary_cost(model, Light)`
- upgrade가 양의 실제 이득일 확률 또는 불확실성

그러나 v3도 Train으로 head를 학습하고 공개 Dev에서 정책을 보정하며, runtime에는
prompt와 tier 외 outcome 메타데이터를 넣지 않는다. 새 hidden holdout이 생기기 전에는
외부 일반화·배포 승격을 주장하지 않는다.

## 8. 다음 대화의 권장 순서

1. 이 문서와 Train/Dev JSON을 읽어 v2.1 결과 분석표를 만든다.
2. 위 네 진단 질문의 답과 v3 가설을 분리한다.
3. v3 설계 문서에서 target, feature boundary, Train/Dev 역할, cost safety rule,
   comparison endpoint를 합의한다.
4. 승인 후 TDD로 구현한다. v2.1 artifact, report, default runtime artifact는 변경하지
   않는다.
