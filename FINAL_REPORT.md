# 통신사 고객 이탈 예측 최종 보고서

## 1. 프로젝트 개요

본 프로젝트의 목적은 `Baza customer Telecom v2.csv` 데이터를 활용하여 B2B 통신사 고객의 이탈 여부를 예측하는 것이다. 이 데이터는 고객 세그먼트, 가입자 상태, 매출 정보, 지역 정보 등을 포함하고 있으며, target 변수는 `CHURN`이다.

이탈 고객은 전체 고객 중 약 6.5%에 불과하므로 데이터가 매우 불균형하다. 따라서 단순 accuracy보다 F1, recall, precision, ROC-AUC, PR-AUC, MCC를 함께 사용하여 모델을 평가하였다.

## 2. 데이터 요약

| 항목 | 값 |
| --- | ---: |
| 원본 행 수 | 8,453 |
| 중복 PID 제거 후 행 수 | 8,436 |
| 원본 컬럼 수 | 14 |
| CHURN=No | 7,904 |
| CHURN=Yes | 549 |
| 이탈 비율 | 약 6.5% |

주요 결측 컬럼은 `Suspended_subscribers`, `Not_Active_subscribers`, `CRM_PID_Value_Segment`, `Billing_ZIP`, `ARPU`였다. 결측 자체가 정보일 수 있으므로 결측 flag를 생성했고, imputation과 scaling은 train split에만 fit하여 leakage를 방지하였다.

## 3. 전처리 및 Feature Engineering

전처리 과정에서는 다음 작업을 수행하였다.

1. `PID`, `KA_name` 제거
2. `CHURN`을 `No=0`, `Yes=1`로 변환
3. 중복 PID 제거
4. train/test stratified split
5. 결측 flag 생성
6. train 기준 imputation
7. 가입자 상태 비율 feature 생성
8. 매출 비율 및 가입자당 매출 feature 생성
9. revenue interaction feature 생성
10. log/sqrt revenue transform 생성
11. categorical label encoding 및 frequency encoding
12. SVMSMOTE 기반 resampling

`Billing_ZIP`은 지역 정보가 성능에 미치는 영향을 확인하기 위해 포함 버전과 제외 버전을 모두 실험하였다.

## 4. 모델 실험

실험한 모델은 다음과 같다.

- LogisticRegression + SVMSMOTE
- EasyEnsembleClassifier
- RUSBoostClassifier
- BalancedBaggingClassifier
- RandomForestClassifier + SVMSMOTE
- GradientBoostingClassifier + SVMSMOTE
- HistGradientBoostingClassifier + SVMSMOTE
- CatBoostClassifier
- CatBoost native categorical

CatBoost 실험은 두 방식으로 진행하였다.

- `CatBoost_original_balanced`: 전처리된 encoded/scaled feature 사용
- `CatBoost_native_categorical`: `CRM_PID_Value_Segment`, `EffectiveSegment`, `Billing_ZIP`를 문자열 categorical feature로 직접 사용

## 5. 최종 모델 비교

| 목적 | Variant | Model | Threshold | F1 | Recall | Precision | TP | FP | FN | TN |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 기준 최종 모델 | `without_billing_zip` | `LogisticRegression_SMOTE` | 0.50 | 0.1681 | 0.2661 | 0.1229 | 29 | 207 | 80 | 1372 |
| Recall 중심 운영 모델 | `with_billing_zip` | `BalancedBagging_original` | 0.50 | 0.1526 | 0.5872 | 0.0877 | 64 | 666 | 45 | 913 |
| Recall 극대화 모델 | `with_billing_zip` | `CatBoost_native_categorical` | 0.35 | 0.1310 | 0.8349 | 0.0711 | 91 | 1189 | 18 | 390 |

최종 보고서의 메인 모델은 `without_billing_zip + LogisticRegression_SMOTE`로 선정하였다. 이 모델은 F1이 가장 높고 precision도 recall-heavy 모델보다 안정적이다.

다만 이탈 고객을 최대한 많이 잡는 캠페인 목적이라면 `BalancedBagging_original` 또는 `CatBoost_native_categorical`을 운영 후보로 고려할 수 있다. 이 경우 false positive가 크게 증가하므로 retention campaign 비용을 함께 고려해야 한다.

## 6. Threshold Tuning 결과

기본 threshold 0.5만 사용할 경우 불균형 데이터에서 recall이 제한될 수 있으므로, train 내부 validation split에서 threshold를 0.05부터 0.50까지 탐색하였다. 선택된 threshold는 test set에 한 번만 적용하였다.

| Model | Variant | Selected Threshold | Test F1 | Test Recall | Test Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| `BalancedBagging_original` | `with_billing_zip` | 0.50 | 0.1526 | 0.5872 | 0.0877 |
| `LogisticRegression_SMOTE` | `with_billing_zip` | 0.46 | 0.1507 | 0.3028 | 0.1003 |
| `BalancedBagging_original` | `without_billing_zip` | 0.44 | 0.1353 | 0.7064 | 0.0748 |
| `CatBoost_native_categorical` | `with_billing_zip` | 0.35 | 0.1310 | 0.8349 | 0.0711 |

Threshold 튜닝은 최종 F1 1등 모델을 바꾸지는 못했지만, recall 중심 운영점을 제공하였다.

## 7. Feature Importance 해석

Permutation importance 기준 주요 변수는 다음과 같다.

| 운영 기준 | 주요 feature |
| --- | --- |
| LogisticRegression 메인 모델 | `AvgMobileRevenue_sqrt`, `TotalRevenue_sqrt`, `revenue_engagement_interaction`, `revenue_per_subscriber`, `AvgMobileRevenue` |
| BalancedBagging recall 모델 | `Billing_ZIP`, `revenue_engagement_interaction`, `arpu_risk_interaction`, `ARPU_sqrt`, `AvgMobileRevenue_sqrt` |
| CatBoost native 모델 | `arpu_risk_interaction`, `revenue_engagement_interaction`, `inactive_revenue_interaction`, `ARPU_sqrt`, `TotalRevenue_sqrt` |

해석하면 고객 이탈 예측에는 수익 규모, 가입자 활동성, 비활성 고객 비율, ARPU와 휴면 상태의 상호작용이 중요한 신호로 작용하였다. 일부 ensemble 모델에서는 `Billing_ZIP`도 중요하게 나타났지만, 최종 F1 모델에서는 `Billing_ZIP`을 제외한 variant가 더 좋은 성능을 보였다.

## 8. 오류 분석

`LogisticRegression_SMOTE`는 F1과 precision이 가장 좋지만, recall이 낮아 실제 이탈 고객 중 80명을 놓쳤다. 반면 `BalancedBagging_original`은 recall이 높아 64명의 이탈 고객을 잡았지만, false positive가 666명으로 많았다.

주요 오류 패턴은 다음과 같다.

| 운영 기준 | 모델 | TP | FP | FN | TN | 해석 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 메인 F1 | LogisticRegression_SMOTE | 29 | 207 | 80 | 1372 | precision과 F1은 좋지만 이탈 고객을 많이 놓침 |
| Recall 중심 | BalancedBagging_original | 64 | 666 | 45 | 913 | 이탈 고객을 더 많이 잡지만 false positive가 많음 |
| Recall 극대화 | CatBoost_native_categorical | 91 | 1189 | 18 | 390 | 대부분의 이탈 고객을 잡지만 캠페인 비용이 커질 수 있음 |

따라서 실제 비즈니스 적용 시에는 모델 하나를 무조건 선택하기보다 목적에 따라 운영 기준을 나누는 것이 적절하다.

## 9. 참고 논문과 비교

참고 논문 `Predicting Customer Churn in the Telecommunications Industry using Machine Learning Techniques`에서는 SVMSMOTE와 ensemble 모델을 사용하였다. 논문에서 최종 선택된 모델은 EasyEnsembleClassifier이며, 주요 결과는 다음과 같다.

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC | PR | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EasyEnsemble | 0.664 | 0.533 | 0.077 | 0.382 | 0.129 | 0.551 | 0.079 | 0.034 |
| RUSBoost | 0.803 | 0.527 | 0.086 | 0.209 | 0.121 | 0.588 | 0.084 | 0.036 |
| BalancedBagging | 0.902 | 0.500 | 0.064 | 0.036 | 0.046 | 0.576 | 0.077 | -0.001 |

우리 프로젝트의 최종 F1 모델은 F1 0.1681로 논문 EasyEnsemble의 F1 0.129보다 높다. 단, 실험 split과 feature engineering 구성이 다르므로 직접적인 우열 비교보다는 “비슷한 데이터에서 imbalance-aware 모델과 metric 중심 평가가 중요하다”는 근거로 활용하는 것이 적절하다.

## 10. 최종 결론

최종 모델은 `without_billing_zip + LogisticRegression_SMOTE`로 선정한다. 이 모델은 전체 실험 중 F1이 가장 높고, 불균형 데이터에서 precision과 recall의 균형이 상대적으로 가장 좋다.

운영 관점에서는 목적에 따라 모델 선택이 달라질 수 있다.

- F1과 precision 균형이 중요하면 `LogisticRegression_SMOTE`
- 이탈 고객을 더 많이 잡는 것이 중요하면 `BalancedBagging_original`
- false positive를 감수하고 최대 recall이 필요하면 `CatBoost_native_categorical`

## 11. 한계와 향후 개선 방향

현재 데이터는 정적인 CRM snapshot이므로 이탈 직전의 행동 변화를 충분히 반영하지 못한다. 모델을 계속 바꾸는 것만으로는 F1을 크게 높이기 어렵다.

향후 개선에 필요한 데이터는 다음과 같다.

- 월별 사용량 변화
- 최근 매출 감소 추세
- 결제 실패 이력
- 고객센터 문의 및 불만 기록
- 계약 만료까지 남은 기간
- 요금제 변경 이력
- 최근 n개월 active/inactive subscriber 변화

향후 작업으로는 BalancedBagging 하이퍼파라미터 튜닝을 시도할 수 있지만, 가장 중요한 개선 방향은 시간 기반 고객 행동 feature를 추가하는 것이다.

## 12. 최종 보고서용 문장

본 프로젝트에서는 B2B 통신사 고객 이탈 예측을 위해 불균형 데이터 전처리, SVMSMOTE, 다양한 ensemble 모델, CatBoost native categorical 처리, threshold tuning, 오류 분석을 수행하였다. 실험 결과 F1 기준 최종 모델은 Billing_ZIP을 제외한 `LogisticRegression_SMOTE`이며, recall 중심 운영 시나리오에서는 `BalancedBagging_original` 또는 `CatBoost_native_categorical`이 더 많은 이탈 고객을 탐지할 수 있었다. 다만 전체 precision이 낮고 F1 상승폭이 제한적인 것은 데이터가 정적인 CRM snapshot이며 시간 기반 행동 이력이 부족하기 때문으로 해석된다.

