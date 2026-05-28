# 통신사 B2B 고객 이탈 예측 최종 보고서

## 1. 프로젝트 개요

본 프로젝트의 목표는 `Baza customer Telecom v2.csv`를 사용하여 B2B 통신사 고객의 이탈 여부를 예측하고, 심한 class imbalance 환경에서 모델을 어떻게 평가하고 운영해야 하는지를 분석하는 것이다.

데이터의 이탈 고객 비율은 약 6.5%에 불과하다. 따라서 accuracy는 주요 평가 기준으로 적합하지 않으며, 본 프로젝트에서는 F1, recall, precision, PR-AUC, MCC, confusion matrix, 비용-편익 관점의 순이익을 함께 사용하였다.

최종 주제는 ChurnRadar로 유지한다. 본 보고서는 단순히 높은 점수를 얻는 것보다, 제한적인 정적 CRM snapshot 데이터에서 가능한 전처리, 불균형 처리, 모델 비교, threshold tuning, 오류 분석, 논문 재현, 비즈니스 시나리오 분석을 체계적으로 수행했다는 점을 핵심 성과로 정리한다.

## 2. 데이터 요약

| 항목 | 값 |
| --- | ---: |
| 원본 행 수 | 8,453 |
| 원본 컬럼 수 | 14 |
| PID 중복 제거 후 행 수 | 8,436 |
| 원본 CHURN=Yes | 549 |
| 원본 CHURN=No | 7,904 |
| 원본 이탈 비율 | 6.49% |
| PID 중복 제거 후 CHURN=Yes | 545 |
| PID 중복 제거 후 이탈 비율 | 6.46% |

주요 변수는 고객 가치 세그먼트, 실질 비즈니스 세그먼트, Billing ZIP, 가입자 상태 수, 평균/총 매출, ARPU, KA 담당자명으로 구성된다. `PID`는 식별자이므로 학습 피처에서 제외했고, `KA_name`은 개인정보 및 운영 지속성 이슈가 있어 기본 모델에서는 제외하였다. 별도 연구용 ablation에서는 KA 추상화 피처와 target/frequency encoding 가능성을 검토하였다.

## 3. 전처리 및 Leakage 방지

전처리는 다음 원칙으로 수행하였다.

1. `PID` 기준 중복 제거는 train/test split 이전에 수행
2. `CHURN`은 `No=0`, `Yes=1`로 변환 후 feature matrix에서 제거
3. 80:20 stratified train/test split 적용
4. 결측치 대치, label encoding, scaling은 train split에만 fit
5. SVMSMOTE는 train partition에만 적용
6. test set에는 resampling을 적용하지 않음
7. threshold는 validation split에서 선택하고 test set에는 1회만 적용

결측 처리 방식은 논문 재현을 기준으로 맞추었다. `Not_Active_subscribers`, `Suspended_subscribers`는 0 대체, 범주형 결측은 `Unknown`, `ARPU`와 `Billing_ZIP`은 중앙값 대치를 사용했다. 동시에 결측 자체가 신호일 수 있으므로 일부 실험에서는 missing flag를 추가했다.

## 4. Feature Engineering

논문 기반 feature와 본 프로젝트 확장 feature를 함께 구성하였다.

| 그룹 | 주요 feature |
| --- | --- |
| 원시 매출 | `AvgMobileRevenue`, `AvgFIXRevenue`, `TotalRevenue`, `ARPU` |
| 원시 가입자 수 | `Active_subscribers`, `Not_Active_subscribers`, `Suspended_subscribers`, `Total_SUBs` |
| 비율 feature | `active_rate`, `inactive_rate`, `suspended_rate`, `dormant_rate`, `risk_score` |
| 매출 효율 | `revenue_per_subscriber`, `revenue_per_active_subscriber` |
| 상호작용 | `revenue_engagement_interaction`, `arpu_risk_interaction`, `inactive_revenue_interaction` |
| 변환 feature | log/sqrt revenue transform |
| 범주형 | CRM segment, EffectiveSegment, Billing_ZIP 포함/제외 variant |

논문은 Billing_ZIP을 label encoding으로 포함한 단일 설정만 사용했다. 본 프로젝트는 ZIP 포함, 제외, top-N grouping variant를 구성하여 ZIP의 영향이 모델 계열에 따라 달라지는지 확인했다.

## 5. 논문 재현과 확장 설계

Makokha et al. (2026)은 동일한 B2B 통신사 데이터를 사용하여 SVMSMOTE와 EasyEnsembleClassifier를 중심으로 churn 예측을 수행했다. 논문의 최고 모델은 EasyEnsemble이며 주요 성능은 다음과 같다.

| 기준 | Model | F1 | Recall | Precision | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| 논문 | EasyEnsemble | 0.129 | 0.382 | 0.077 | 0.079 |
| 우리 재현 | EasyEnsemble original with ZIP | 0.128 | 0.587 | 0.072 | 0.085 |

EasyEnsemble 기준으로 F1이 `0.128` 대 `0.129`로 거의 일치하므로, 논문 방법론의 핵심 결과를 재현했다고 볼 수 있다.

다만 본 프로젝트의 `LogisticRegression_SMOTE` hold-out F1 `0.1681`은 논문에서 직접 비교한 모델이 아니므로, 이를 곧바로 “논문보다 우월”이라고 주장하지 않는다. 올바른 해석은 다음과 같다.

> 본 프로젝트는 EasyEnsemble 기준으로 논문 결과를 재현했고, 추가 모델/feature/threshold 실험을 통해 다른 운영 목적에서 경쟁력 있는 대안을 제시했다.

## 6. Hold-Out 모델 결과

주요 운영 후보는 다음과 같다.

| 목적 | Variant | Model | Threshold | F1 | Recall | Precision | TP | FP | FN | TN |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 기준 최종 모델 | without ZIP | LogisticRegression_SMOTE | 0.50 | 0.1681 | 0.2661 | 0.1229 | 29 | 207 | 80 | 1,372 |
| 균형형 recall 후보 | with ZIP | BalancedBagging_original | 0.50 | 0.1526 | 0.5872 | 0.0877 | 64 | 666 | 45 | 913 |
| recall-heavy 후보 | with ZIP | CatBoost_native_categorical | 0.35 | 0.1310 | 0.8349 | 0.0711 | 91 | 1,189 | 18 | 390 |
| recall-heavy 확장 | with ZIP | XGBoost_SMOTE | 0.16 | 0.1253 | 0.9266 | 0.0672 | 101 | 1,402 | 8 | 177 |

F1만 보면 `without_billing_zip + LogisticRegression_SMOTE`가 가장 높다. 그러나 이 모델은 recall이 낮아 실제 이탈 고객 109명 중 29명만 포착한다. 반대로 CatBoost와 XGBoost는 recall이 높지만 false positive가 크게 증가한다. 따라서 모델 선택은 단일 점수보다 운영 목적에 따라 달라져야 한다.

## 7. Cross-Validation 안정성

단일 hold-out 결과만으로 모델 우위를 주장하면 위험하다. 5-fold stratified CV를 추가로 수행한 결과는 다음과 같다.

| Variant | Model | CV F1 mean | CV F1 SD | Recall mean | Precision mean | PR-AUC mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| with ZIP | BalancedBagging_original | 0.1455 | 0.0126 | 0.5248 | 0.0845 | 0.0910 |
| with ZIP | EasyEnsemble_original | 0.1445 | 0.0117 | 0.5835 | 0.0824 | 0.0907 |
| without ZIP | EasyEnsemble_original | 0.1408 | 0.0081 | 0.5835 | 0.0801 | 0.0885 |
| without ZIP | LogisticRegression_SMOTE | 0.1309 | 0.0154 | 0.1743 | 0.1053 | 0.0876 |

중요한 점은 LR의 hold-out F1 `0.1681`이 CV 평균에서는 `0.1309`로 낮아진다는 것이다. 따라서 최종 주장은 “LR이 언제나 논문보다 우월하다”가 아니라 “hold-out F1 기준 LR이 최고였지만, CV 안정성에서는 BalancedBagging/EasyEnsemble 계열이 더 안정적이었다”로 정리한다.

## 8. 차별화 실험 결과

### 8.1 Billing ZIP Ablation

ZIP 포함 여부는 모델 계열에 따라 다르게 작동했다.

- Tree ensemble에서는 Billing_ZIP이 recall과 F1에 도움을 주었다.
- Logistic Regression에서는 ZIP 제외 설정이 최고 F1을 기록했다.
- BalancedBagging permutation importance에서는 Billing_ZIP이 상위 feature로 나타났다.

이는 고카디널리티 지리 변수가 일괄적으로 좋은 feature가 아니라, 모델 구조에 따라 signal과 noise가 다르게 반영됨을 보여준다.

### 8.2 Feature Group Ablation

Feature group ablation 결과, 모델별로 중요한 feature group이 달랐다.

| 모델 | 제거 시 가장 큰 F1 하락 | 해석 |
| --- | --- | --- |
| LogisticRegression_SMOTE | `G_categorical` 제거 시 -0.0875 | segment/frequency encoding이 선형 모델에 중요 |
| BalancedBagging_original | `G_interaction` 제거 시 -0.0181 | tree ensemble에서도 interaction feature가 핵심 |

논문은 feature engineering이 중요하다고 정성적으로 설명했지만, 본 프로젝트는 어떤 feature group이 어느 모델에서 중요한지 정량적으로 확인했다.

### 8.3 CRM Segment Error Analysis

CRM segment bucket별 성능은 다음과 같다.

| Segment bucket | Rows | Positives | Recall | Precision | FN revenue at risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| low value | 784 | 47 | 0.2766 | 0.1413 | 2,550.16 |
| mid value | 500 | 34 | 0.8529 | 0.0960 | 534.32 |
| high value | 404 | 28 | 0.7857 | 0.0655 | 1,441.33 |

모델은 중/고가치 고객에서는 비교적 높은 recall을 보였지만 precision이 낮았다. low-value 고객에서는 오히려 놓치는 이탈자가 많았다. 따라서 전체 평균 성능만 볼 것이 아니라 segment별 실패 패턴을 함께 봐야 한다.

## 9. Interpretability 분석

논문은 EasyEnsemble에 SHAP/LIME을 적용했다. 본 프로젝트는 Logistic Regression의 intrinsic interpretability를 활용하여 계수 기반 해석과 local logit contribution을 계산했다.

LR에서 표준화된 feature의 계수는 feature가 churn log-odds에 미치는 방향과 크기를 직접 제공한다. permutation importance와 계수 분석을 함께 보면 다음 신호가 중요했다.

| 기준 | 주요 feature |
| --- | --- |
| Permutation FI | `AvgMobileRevenue_sqrt`, `TotalRevenue_sqrt`, `revenue_engagement_interaction` |
| LR coefficient | `AvgFIXRevenue_log`, `AvgMobileRevenue_sqrt`, `fixed_revenue_per_subscriber` |
| Local logit contribution | `AvgMobileRevenue_sqrt`, `TotalRevenue_sqrt`, `ARPU_sqrt` |

논문에서 active subscriber rate가 SHAP 1위로 나온 것과 본 프로젝트에서 revenue transform이 상위로 나온 것은 모순이 아니다. 모델 구조와 중요도 측정 방식이 다르기 때문이다. 두 결과 모두 “매출 패턴과 가입자 활동성의 결합이 churn 예측에 중요하다”는 결론에서는 일치한다.

## 10. 비즈니스 임팩트 분석

논문과 동일한 비용-편익 가정을 적용했다.

| 항목 | 값 |
| --- | ---: |
| 이탈 고객 1명 연간 손실 | 5,400 |
| retention 성공률 | 60% |
| TP benefit | 3,240 |
| FP campaign cost | 120 |
| 논문 보고 순이익 | 74,200 |

운영점별 순이익은 다음과 같다.

| 운영점 | TP | FP | 접촉 수 | 순이익 | 논문 대비 | Gross ROI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 논문 EasyEnsemble 역산 | 42 | 503 | 545 | 75,720 | 1.02x | 2.25x |
| LR fixed | 29 | 207 | 236 | 69,120 | 0.93x | 3.78x |
| BalancedBagging fixed | 64 | 666 | 730 | 127,440 | 1.72x | 2.59x |
| CatBoost recall-heavy | 91 | 1,189 | 1,280 | 152,160 | 2.05x | 2.07x |
| XGBoost recall-heavy | 101 | 1,402 | 1,503 | 159,000 | 2.14x | 1.95x |
| BalancedBagging cost-opt | 109 | 1,561 | 1,670 | 165,840 | 2.24x | 1.89x |

운영 시나리오별 권장 모델은 다음과 같다.

| 시나리오 | 권장 운영점 | 순이익 | 해석 |
| --- | --- | ---: | --- |
| 캠페인 예산 500건 이하 | LR fixed | 69,120 | 가장 적은 접촉, 최고 Gross ROI |
| 팀 역량 800건 이하 | BalancedBagging fixed | 127,440 | recall과 비용의 균형 |
| 핵심 3모델 중 매출 보호 우선 | CatBoost recall-heavy | 152,160 | 높은 recall, 높은 접촉 수 |
| XGBoost 포함 확장 | XGBoost recall-heavy | 159,000 | 더 높은 recall, 더 많은 FP |
| 비용 최적 threshold 허용 | BalancedBagging threshold 0.29 | 165,840 | 최대 순이익이나 거의 전체 고객 접촉 |

비용 최적 threshold는 수치상 가장 큰 순이익을 만들지만, 1,688명 중 1,670명에게 캠페인을 보내는 공격적 전략이다. 실제 운영에서는 고객 피로도와 상담 인력 제약을 반드시 고려해야 한다.

## 11. 최종 모델 선택

본 프로젝트의 최종 선택은 하나의 모델이 아니라 목적별 운영 프레임워크다.

| 목적 | 추천 |
| --- | --- |
| 보고서상 F1 기준 최종 모델 | `without_billing_zip + LogisticRegression_SMOTE` |
| CV 안정성과 recall 균형 | `with_billing_zip + BalancedBagging_original` |
| 이탈 고객 최대 포착 | `CatBoost_native_categorical` 또는 `XGBoost_SMOTE` |
| 비용 가정상 최대 순이익 | `BalancedBagging_original`, threshold 0.29 |
| 실제 캠페인 운영 | 예산에 맞춘 top-k 또는 threshold scenario 선택 |

따라서 최종 결론은 “한 모델이 모든 기준에서 최고”가 아니라, “불균형 churn 데이터에서는 목적별 모델 선택이 필요하다”이다.

## 12. 추가 발표용 케이스 스터디

1시간 발표를 위해 추가 비교 실험을 수행했다. 이 실험들은 최종 모델을 바꾸기 위한 것이 아니라, 운영 관점에서 모델을 더 입체적으로 설명하기 위한 케이스 스터디다.

| 추가 케이스 | 핵심 결과 | 발표 활용 |
| --- | --- | --- |
| Top-k budget | top 10%에서는 LR, top 40%에서는 BalancedBagging이 유리 | 예산별 캠페인 전략 |
| Cost threshold sensitivity | 논문 비용 기준에서는 recall 극대화가 유리, FP cost가 커지면 LR이 유리 | 비용 구조가 모델 선택을 바꾼다는 근거 |
| Calibration | raw score는 실제 churn probability를 과대평가, Platt 보정 후 실제 이탈률과 일치 | score와 probability 구분 |
| Segment ROI | low/mid/high value별로 유리한 모델이 다름 | 전체 평균의 한계 설명 |
| Model agreement | 8개 모델 모두가 경고한 고객군의 이탈률은 12.43% | risk tiering 운영안 |

추가 실험에서 가장 중요한 해석은 raw score를 확률로 해석하면 안 된다는 점이다. 실제 test churn rate는 6.46%였지만, raw score 평균은 LR 34.6%, BalancedBagging 46.7%, CatBoost 41.4%, XGBoost 46.9%로 과대평가되어 있었다. Platt calibration 후 평균 score는 실제 churn rate에 가깝게 조정되었다. 따라서 배포 단계에서는 probability calibration이 필수적이다.

또한 모델 합의도 분석에서 8개 모델 모두가 churn으로 판단한 169명의 실제 이탈률은 12.43%였다. 이는 전체 평균 6.46%의 약 1.9배로, 모델 합의도를 캠페인 우선순위 tier로 사용할 수 있음을 보여준다.

## 13. 한계

본 프로젝트의 한계는 다음과 같다.

| 한계 | 설명 | 향후 개선 |
| --- | --- | --- |
| 정적 CRM snapshot | 이탈 직전 행동 변화가 없음 | 월별 사용량/매출 추세 추가 |
| 배포용 확률 보정 미반영 | Phase 6에서 raw/Platt/isotonic 진단은 수행했지만 최종 운영 threshold에는 calibrated probability를 직접 적용하지 않음 | 배포 전 calibrated probability 기반 threshold 재설계 |
| SHAP 직접 비교 미실시 | 논문과 동일한 SHAP plot은 없음 | Tree-SHAP 적용 |
| 비용 가정 고정 | 논문 비용 파라미터에 의존 | 고객별 ARPU 기반 individualized value |
| 단일 데이터셋 | 외부 검증 데이터 없음 | 기간별 hold-out 또는 다른 시장 데이터 |

향후 가장 중요한 개선은 모델을 더 많이 추가하는 것이 아니라, 시간 기반 고객 행동 feature를 확보하는 것이다. 예를 들어 최근 3개월 사용량 변화, 매출 감소 추세, 결제 실패, 고객센터 불만, 계약 만료일, 요금제 변경 이력이 있다면 churn 직전 신호를 훨씬 잘 포착할 수 있다.

## 14. 최종 결론

본 프로젝트는 Makokha et al. (2026)의 동일 데이터셋과 전처리 원칙을 기반으로 EasyEnsembleClassifier 결과를 거의 재현하였다. 또한 ZIP ablation, feature group ablation, CRM segment error analysis, LR 계수 기반 interpretability, 비용-편익 분석을 추가하여 논문이 다루지 않은 운영적 질문을 확장했다.

단일 hold-out에서는 LogisticRegression_SMOTE가 F1 `0.1681`로 최고였지만, 5-fold CV에서는 BalancedBagging과 EasyEnsemble 계열이 더 안정적이었다. 따라서 본 연구의 핵심 주장은 “논문 대비 압도적 성능 우위”가 아니라, “재현 가능한 baseline 위에서 feature 처리와 threshold 전략에 따라 운영 목적별 최적 모델이 달라진다는 점을 실증했다”는 것이다.

최종적으로 ChurnRadar는 낮은 class imbalance 환경에서 accuracy 중심 평가의 한계를 보이고, F1/recall/precision/cost를 함께 고려한 churn prediction 운영 프레임워크를 제시한 프로젝트로 정리할 수 있다.
