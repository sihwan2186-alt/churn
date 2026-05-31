# Phase 5-A: Feature Importance and SHAP Alternatives

마지막 업데이트: 2026-05-27

## 실행 개요

실행 스크립트:

```powershell
.\.venv\Scripts\python.exe phase_5a_interpretability.py
```

분석 대상:

- Variant: `without_billing_zip`
- Model: `LogisticRegression_SMOTE`
- Train data: `X_train_resampled.csv`
- Test data: `X_test.csv`

출력 위치:

- `processed/phase_5a_interpretability/lr_coefficient_importance.csv`
- `processed/phase_5a_interpretability/lr_linear_contribution_importance.csv`
- `processed/phase_5a_interpretability/lr_feature_churn_correlation.csv`
- `processed/phase_5a_interpretability/lr_permutation_vs_coefficient_comparison.csv`
- `processed/phase_5a_interpretability/lr_high_risk_customers.csv`
- `processed/phase_5a_interpretability/lr_local_linear_explanations.csv`
- `processed/phase_5a_interpretability/lr_coefficient_importance.png`
- `processed/phase_5a_interpretability/lr_linear_contribution_importance.png`
- `processed/phase_5a_interpretability/lr_pdp_top_features.png`
- `processed/phase_5a_interpretability/phase_5a_interpretability_summary.json`

## 1. 왜 논문 SHAP 1위와 우리 중요도 1위가 다른가?

논문과 우리 분석은 측정 대상이 다르다.

| 구분 | 논문 | 우리 |
| --- | --- | --- |
| 모델 | EasyEnsemble tree ensemble | LogisticRegression+SVMSMOTE |
| 설명 방법 | SHAP | permutation FI + LR coefficient |
| 1위 신호 | active subscriber rate | AvgMobileRevenue_sqrt |
| 질문 | 개별 예측에 feature가 얼마나 기여했나 | feature를 제거/교란하면 F1이 얼마나 떨어지나 |

따라서 순위 차이는 오류가 아니다. Tree SHAP은 비선형 분기와 임계 효과를 잘 잡고, permutation FI는 전체 test F1 손실을 본다. LR 계수는 다른 모든 feature를 통제했을 때의 조건부 log-odds 효과다.

## 2. 실제 LR 계수 결과

상위 coefficient 기준:

| Rank | Feature | Coefficient | Odds ratio per 1 SD | Direction |
| ---: | --- | ---: | ---: | --- |
| 1 | `AvgFIXRevenue_log` | 1.3722 | 3.9439 | Churn up |
| 2 | `AvgMobileRevenue_sqrt` | 1.3230 | 3.7548 | Churn up |
| 3 | `fixed_revenue_per_subscriber` | -1.2394 | 0.2895 | Churn down |
| 4 | `TotalRevenue_sqrt` | 1.2052 | 3.3374 | Churn up |
| 5 | `AvgFIXRevenue` | -1.1356 | 0.3212 | Churn down |
| 6 | `ARPU_sqrt` | 1.0886 | 2.9700 | Churn up |
| 7 | `revenue_per_active_subscriber` | -0.7513 | 0.4718 | Churn down |
| 8 | `fixed_to_mobile_ratio` | 0.6975 | 2.0088 | Churn up |
| 9 | `CRM_PID_Value_Segment_frequency` | 0.6918 | 1.9973 | Churn up |
| 10 | `revenue_per_subscriber` | -0.6534 | 0.5203 | Churn down |

상위 평균 절대 logit contribution 기준:

| Rank | Feature | Mean abs logit contribution | Coefficient direction |
| ---: | --- | ---: | --- |
| 1 | `AvgMobileRevenue_sqrt` | 1.0722 | Churn up |
| 2 | `TotalRevenue_sqrt` | 0.9773 | Churn up |
| 3 | `ARPU_sqrt` | 0.7344 | Churn up |
| 4 | `CRM_PID_Value_Segment_frequency` | 0.6334 | Churn up |
| 5 | `TotalRevenue` | 0.4310 | Churn down |
| 6 | `revenue_engagement_interaction` | 0.4005 | Churn down |

해석:

- LR에서는 revenue transform 계열이 가장 강하다.
- `AvgMobileRevenue_sqrt`, `TotalRevenue_sqrt`, `ARPU_sqrt`는 coefficient와 contribution 모두 상위다.
- 원시 revenue와 변환 revenue가 동시에 들어가므로 일부 계수 부호가 직관과 반대로 나타난다. 이는 다중공선성 환경의 조건부 효과이며, 단변량 상관 또는 인과효과로 읽으면 안 된다.

## 3. Permutation FI와 coefficient 비교

| Perm Rank | Feature | F1 importance | Coef rank | Coefficient | Direction |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `AvgMobileRevenue_sqrt` | 0.0771 | 2 | 1.3230 | Churn up |
| 2 | `TotalRevenue_sqrt` | 0.0658 | 4 | 1.2052 | Churn up |
| 3 | `revenue_engagement_interaction` | 0.0554 | 13 | -0.5117 | Churn down |
| 4 | `revenue_per_subscriber` | 0.0498 | 10 | -0.6534 | Churn down |
| 5 | `AvgMobileRevenue` | 0.0455 | 16 | -0.4305 | Churn down |
| 6 | `ARPU_sqrt` | 0.0438 | 6 | 1.0886 | Churn up |
| 7 | `Active_subscribers` | 0.0434 | 21 | 0.3769 | Churn up |
| 8 | `CRM_PID_Value_Segment` | 0.0430 | 29 | 0.2001 | Churn up |
| 9 | `CRM_PID_Value_Segment_frequency` | 0.0429 | 9 | 0.6918 | Churn up |
| 10 | `EffectiveSegment_frequency` | 0.0413 | 25 | -0.2768 | Churn down |

핵심:

- `AvgMobileRevenue_sqrt`와 `TotalRevenue_sqrt`는 permutation과 coefficient가 모두 상위라 가장 견고한 신호다.
- `revenue_engagement_interaction`은 permutation 3위지만 coefficient rank는 13위다. 즉 단독 계수 크기보다 모델 성능에 미치는 교란 효과가 더 크다.
- `Active_subscribers`는 permutation 7위지만 coefficient rank 21위다. 활성 가입자 정보는 중요하지만, `active_rate`, `dormant_rate`, `risk_score`, subscriber count 계열과 중복되어 coefficient가 희석된 것으로 해석한다.

## 4. Active Rate가 논문처럼 1위가 아닌 이유

`active_rate` 자체는 coefficient rank가 낮다. 그러나 이것이 활동성 신호가 중요하지 않다는 뜻은 아니다.

이 프로젝트에는 다음처럼 활동성 정보를 중복 표현하는 feature가 많다.

- `Active_subscribers`
- `Not_Active_subscribers`
- `Total_SUBs`
- `active_rate`
- `inactive_rate`
- `dormant_rate`
- `risk_score`
- `dormant_subscribers`
- `revenue_engagement_interaction`

단변량 상관에서는 활동/휴면 계열이 상위에 오른다.

| Feature | Correlation with churn | Correlation rank |
| --- | ---: | ---: |
| `dormant_subscribers` | 0.0872 | 1 |
| `Not_Active_subscribers` | 0.0859 | 2 |
| `Total_SUBs` | 0.0838 | 3 |
| `inactive_revenue_interaction` | 0.0693 | 4 |
| `inactive_rate` | 0.0566 | 8 |
| `active_rate` | -0.0565 | 9 |

해석:

- 논문의 SHAP은 tree ensemble에서 `active_rate`의 비선형 임계 효과를 강조했다.
- 우리 LR에서는 활동성 신호가 여러 feature로 분산되어 coefficient 순위가 낮아졌다.
- 따라서 두 결과는 모순이 아니다. 둘 다 "서비스 참여도 + 매출 패턴"이 핵심이라는 큰 결론은 같다.

## 5. LR에서 SHAP 대안으로 쓸 수 있는 것

선형 모델에서 logit은 다음처럼 정확히 분해된다.

```text
logit(P(churn)) = intercept + sum(coefficient_i * standardized_feature_i)
```

따라서 로컬 설명은 다음 값으로 만들 수 있다.

```text
local contribution_i = coefficient_i * standardized_feature_i
```

이는 probability space의 nonlinear SHAP과 동일하다고 말하면 과장이다. 하지만 **logit space에서는 정확한 additive explanation**이다. 현재 스크립트는 고위험 고객 5명에 대해 이 local contribution을 저장한다.

출력:

- `lr_high_risk_customers.csv`
- `lr_local_linear_explanations.csv`

주의할 점:

- 상위 고위험 고객 5명은 모두 실제 non-churn이었다. 즉 LR의 최고위험 점수는 false positive로 나타났다.
- 이는 모델 설명이 "왜 그렇게 예측했는가"는 잘 보여주지만, "그 예측이 맞았는가"는 별도의 calibration/validation 문제임을 보여준다.

## 6. 논문 SHAP과 우리 대안 비교

| 논문 SHAP 발견 | 우리 대안 분석 | 결론 |
| --- | --- | --- |
| active rate가 핵심 | 활동성 계열은 correlation 상위, permutation에서도 `Active_subscribers` 7위 | 방향성은 대체로 일치하지만 LR coefficient에서는 중복 feature 때문에 희석 |
| geographic billing zone 중요 | Phase 3-B에서 `Billing_ZIP`은 BalancedBagging permutation 1위 | tree ensemble 기준으로 논문과 일치 |
| revenue interaction 중요 | `revenue_engagement_interaction` permutation 3위, contribution 6위 | 일치 |
| 개별 고객 설명 | LR local logit contribution 산출 | LIME 없이도 정확한 선형 local explanation 가능 |
| 비선형 임계 패턴 | PDP plot 생성 | SHAP beeswarm 대안으로 feature-response 곡선 제공 |

## 7. 보고서용 문장

### 방어적 버전

> 본 연구는 Makokha et al. (2026)이 적용한 SHAP/LIME 기반 사후 설명가능성 분석을 동일하게 구현하지는 않았으나, LogisticRegression의 선형 구조를 활용하여 coefficient 기반 global explanation과 coefficient × standardized feature value 기반 local explanation을 제공하였다. 이는 logit space에서 정확한 additive decomposition이므로, 선형 모델에 적합한 SHAP 대안으로 사용할 수 있다.

### 적극적 버전

> 선행 연구가 불투명한 tree ensemble을 SHAP과 LIME으로 사후 설명한 것과 달리, 본 연구의 LogisticRegression 접근은 모델 자체가 방향성 있는 설명을 제공한다. 표준화된 feature에 대한 coefficient는 각 변수가 churn log-odds에 미치는 조건부 효과를 나타내며, 고위험 고객별 예측도 coefficient × feature value로 분해할 수 있다. 이 intrinsic interpretability는 운영자가 모델 판단 근거를 빠르게 검토할 수 있다는 장점을 가진다.

### 가장 권장하는 비교 문장

> Feature importance 비교 결과, 논문의 SHAP 분석과 본 연구의 permutation/coefficient 분석은 순위에서는 차이를 보였으나 핵심 결론에서는 수렴하였다. 논문은 tree ensemble에서 active-rate의 비선형 임계 효과를 1위로 제시한 반면, 본 연구의 LogisticRegression은 revenue transform과 revenue-engagement interaction을 가장 강한 신호로 식별하였다. 이는 측정 방법과 모델 구조의 차이에서 비롯된 자연스러운 결과이며, 두 분석 모두 이탈 예측에서 서비스 참여도와 매출 패턴의 결합이 중요하다는 점을 지지한다.

## 8. 한계

| 한계 | 설명 | 향후 작업 |
| --- | --- | --- |
| probability SHAP은 아님 | LR local contribution은 logit space 설명 | SHAP LinearExplainer 또는 KernelSHAP으로 probability-level 비교 |
| 다중공선성 | 원시/로그/sqrt/ratio feature가 동시에 있어 계수 부호가 불안정할 수 있음 | L1/ElasticNet 또는 group-wise feature selection |
| PDP는 평균 효과 | segment별 이질성을 숨길 수 있음 | segment-conditioned PDP/ICE plot |
| SHAP/LIME 미설치 | 현재 환경에는 `shap`, `lime` 없음 | 패키지 설치 후 직접 비교 가능 |

