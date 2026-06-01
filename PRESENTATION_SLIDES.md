# ChurnRadar 발표 슬라이드 구성안

발표 핵심 흐름은 다음 한 문장으로 잡는다.

이 문서는 `ChurnRadar_Final_Presentation.pptx` 요약 17장 발표본 기준이다. 세부 방어와 긴 발표에는 `ChurnRadar_Detailed_Presentation.pptx` 36장 상세본을 함께 사용한다.

## 발표 시간 배분 (총 60분)

1. **도입 및 문제 정의** (10분): 왜 Churn 예측이 어렵고 중요한가?
2. **방법론 및 실험** (20분): 논문 재현, ZIP/Feature Ablation, 통계적 검정.
3. **비즈니스 및 운영** (20분): ROI 시뮬레이션, n8n/Docker 기반 MLOps 파이프라인.
4. **결론 및 Q&A** (10분): 한계점 인정과 향후 데이터 확보 전략.

> 논문 결과를 재현한 뒤, ZIP ablation, feature group ablation, segment error analysis, interpretability, 비용-편익 분석을 추가하여 운영 목적별 모델 선택 프레임워크를 제시했다.

## Slide 1. 제목

제목:

- ChurnRadar: B2B 통신사 고객 이탈 예측과 운영 시나리오 분석

넣을 내용:

- 데이터: `Baza customer Telecom v2.csv`
- 문제: 약 6.5% 이탈 고객을 조기에 탐지
- 핵심: 불균형 데이터에서 F1, recall, precision, 비용-편익을 함께 고려

발표 멘트:

> 이 프로젝트는 단순히 accuracy가 높은 모델을 만드는 것이 아니라, 매우 적은 이탈 고객을 어떻게 포착하고 실제 retention campaign에서 어떤 기준으로 운영할지 분석한 프로젝트입니다.

## Slide 2. 데이터와 문제 정의

넣을 표:

| 항목 | 값 |
| --- | ---: |
| 원본 데이터 | 8,453 rows x 14 columns |
| PID 중복 제거 후 | 8,436 rows |
| 원본 이탈 고객 | 549 |
| 원본 비이탈 고객 | 7,904 |
| 이탈 비율 | 6.49% |

핵심 메시지:

- 이탈 고객이 매우 적어서 accuracy만 보면 모델을 잘못 평가할 수 있다.
- 주요 평가지표는 F1, recall, precision, PR-AUC, MCC다.
- test set에는 resampling을 적용하지 않고 현실 분포를 유지했다.

발표 멘트:

> 이탈 고객은 전체의 약 6.5%입니다. 그래서 모든 고객을 비이탈로 예측해도 accuracy는 높게 나올 수 있습니다. 이 프로젝트에서는 실제 이탈 고객을 얼마나 잡는지와 false positive 비용을 함께 봤습니다.

## Slide 3. 전처리와 Leakage 방지

넣을 내용:

- PID 기준 중복 제거
- `CHURN` 이진 변환 후 feature에서 제거
- 80:20 stratified split
- train-only imputation, encoding, scaling
- SVMSMOTE는 train에만 적용
- threshold는 validation에서 선택 후 test에 1회 적용

핵심 메시지:

- 모델 성능보다 먼저 leakage 방지가 중요하다.
- 논문 재현과 우리 확장을 같은 split/전처리 원칙 아래 비교했다.

발표 멘트:

> 전처리에서 가장 중요하게 본 것은 leakage 방지입니다. imputation, scaling, label encoding, SVMSMOTE 모두 train 기준으로만 fit했고, test set은 실제 운영 분포처럼 그대로 두었습니다.

## Slide 4. 논문 재현 결과

넣을 표:

| 기준 | Model | F1 | Recall | Precision | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Makokha et al. | EasyEnsemble | 0.129 | 0.382 | 0.077 | 0.079 |
| 우리 재현 | EasyEnsemble original with ZIP | 0.128 | 0.587 | 0.072 | 0.085 |

핵심 메시지:

- EasyEnsemble 기준 F1이 0.128 대 0.129로 거의 일치한다.
- 따라서 논문 baseline은 재현되었다.
- 이후 결과는 논문에 없는 추가 실험으로 분리해 해석한다.

발표 멘트:

> 먼저 논문 모델을 같은 데이터에서 재현했습니다. EasyEnsemble 기준 F1이 논문 0.129와 거의 같은 0.128로 나와서, 이후 확장 실험을 비교할 수 있는 기준선을 확보했습니다.

## Slide 5. Hold-Out 모델 비교

넣을 표:

| 목적 | Model | F1 | Recall | Precision | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 기준 | LogisticRegression_SMOTE | 0.1681 | 0.2661 | 0.1229 | 29 | 207 | 80 |
| 균형형 recall | BalancedBagging_original | 0.1526 | 0.5872 | 0.0877 | 64 | 666 | 45 |
| recall-heavy | CatBoost_native | 0.1310 | 0.8349 | 0.0711 | 91 | 1,189 | 18 |
| recall-heavy 확장 | XGBoost_SMOTE | 0.1242 | 0.9266 | 0.0665 | 101 | 1,417 | 8 |

핵심 메시지:

- F1 기준 최고는 LR이다.
- 이탈 고객을 더 많이 잡으려면 BalancedBagging/CatBoost/XGBoost가 유리하다.
- 단, recall이 오를수록 FP가 크게 증가한다.

발표 멘트:

> 한 모델이 모든 기준에서 최고는 아니었습니다. LR은 F1과 precision이 좋지만 이탈 고객을 많이 놓치고, CatBoost와 XGBoost는 이탈 고객을 많이 잡지만 캠페인 대상이 크게 늘어납니다.

## Slide 6. Cross-Validation 안정성

넣을 표:

| Model | CV F1 mean | CV F1 SD | Recall mean | Precision mean |
| --- | ---: | ---: | ---: | ---: |
| BalancedBagging with ZIP | 0.1455 | 0.0126 | 0.5248 | 0.0845 |
| EasyEnsemble with ZIP | 0.1445 | 0.0117 | 0.5835 | 0.0824 |
| EasyEnsemble without ZIP | 0.1408 | 0.0081 | 0.5835 | 0.0801 |
| LR without ZIP | 0.1309 | 0.0154 | 0.1743 | 0.1053 |

핵심 메시지:

- LR hold-out F1 0.1681은 좋은 단일 split 결과다.
- CV에서는 BalancedBagging/EasyEnsemble이 더 안정적이다.
- 따라서 “논문보다 압도적으로 우월”이 아니라 “운영 목적별 trade-off”로 해석한다.

발표 멘트:

> 단일 test split에서는 LR이 가장 좋았지만, 5-fold CV에서는 그 차이가 완화되었습니다. 그래서 저희는 특정 모델의 일방적 우위가 아니라, 운영 목적별 모델 선택이 핵심이라고 정리했습니다.

## Slide 7. 차별화 실험

넣을 내용:

| 실험 | 핵심 결과 |
| --- | --- |
| Billing ZIP ablation | ZIP은 tree ensemble에는 도움, LR에는 노이즈 가능성 |
| Feature group ablation | LR은 categorical, BalancedBagging은 interaction 제거 시 성능 하락 |
| CRM segment analysis | high/mid value recall은 높지만 precision 낮음 |
| Cost threshold sensitivity | 낮은 threshold가 순이익을 올리지만 접촉 수가 급증 |

핵심 메시지:

- 논문이 단일 파이프라인을 제시했다면, 우리는 어떤 요소가 성능과 운영에 영향을 주는지 분해했다.

발표 멘트:

> 논문은 feature engineering이 중요하다고 설명했지만, 어떤 feature group이 얼마나 중요한지는 정량화하지 않았습니다. 저희는 ZIP, feature group, CRM segment, 비용 threshold를 따로 분리해 분석했습니다.

## Slide 8. Feature Importance와 해석 가능성

넣을 내용:

| 기준 | 주요 feature |
| --- | --- |
| Permutation FI | `AvgMobileRevenue_sqrt`, `TotalRevenue_sqrt`, `revenue_engagement_interaction` |
| LR coefficient | `AvgFIXRevenue_log`, `AvgMobileRevenue_sqrt`, `fixed_revenue_per_subscriber` |
| BalancedBagging FI | `Billing_ZIP`, interaction, ARPU 관련 feature |

핵심 메시지:

- 논문은 SHAP/LIME으로 사후 설명을 수행했다.
- 우리는 LR 계수와 local logit contribution으로 intrinsic interpretability를 확보했다.
- 핵심 신호는 매출 패턴과 가입자 활동성의 결합이다.

발표 멘트:

> 논문에서는 active subscriber rate가 SHAP 1위였고, 저희 LR에서는 revenue transform이 상위에 나왔습니다. 이는 모델 구조와 중요도 측정 방식 차이 때문이며, 두 결과 모두 매출과 활동성의 결합이 중요하다는 점에서는 일치합니다.

## Slide 9. 비즈니스 임팩트

넣을 표:

| 운영점 | 접촉 수 | TP | FP | 순이익 | 논문 대비 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 논문 EasyEnsemble | 545 | 42 | 503 | 75,720 | 1.02x |
| LR fixed | 236 | 29 | 207 | 69,120 | 0.93x |
| BalancedBagging fixed | 730 | 64 | 666 | 127,440 | 1.72x |
| CatBoost recall-heavy | 1,280 | 91 | 1,189 | 152,160 | 2.05x |
| XGBoost recall-heavy | 1,518 | 101 | 1,417 | 157,200 | 2.12x |
| BalancedBagging cost-opt | 1,670 | 109 | 1,561 | 165,840 | 2.24x |

사용 가능 이미지:

- `processed/phase_5b_business_impact/business_impact_bubble.png`
- `processed/phase_5b_business_impact/business_impact_scenarios.png`
- `processed/phase_5b_business_impact/business_impact_dashboard.png`

핵심 메시지:

- 예산 제한이면 LR이 효율적이다.
- 팀 역량 800건이면 BalancedBagging이 균형점이다.
- 매출 보호 극대화라면 CatBoost/XGBoost 또는 비용 최적 threshold가 유리하다.

발표 멘트:

> 논문과 같은 비용 가정을 적용하면, 모델 선택은 더 명확해집니다. 캠페인 예산이 적으면 LR이 좋고, 팀이 800건 정도 처리할 수 있으면 BalancedBagging이 좋고, 매출 보호가 최우선이면 recall-heavy 모델이 더 큰 순이익을 냅니다.

## Slide 10. Top-k 캠페인 예산 전략

넣을 표:

| Top-k | 권장 모델 | 접촉 수 | TP | Recall@k | Precision@k | 순이익 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 5% | EasyEnsemble_with_zip | 84 | 11 | 0.1009 | 0.1310 | 26,880 |
| 10% | LR_no_zip_f1 | 169 | 23 | 0.2110 | 0.1361 | 57,000 |
| 20% | LR_with_zip_recall_constraint | 338 | 33 | 0.3028 | 0.0976 | 70,320 |
| 30% | EasyEnsemble_with_zip | 506 | 46 | 0.4220 | 0.0909 | 93,840 |
| 40% | BalancedBagging_with_zip | 675 | 61 | 0.5596 | 0.0904 | 123,960 |

사용 이미지:

- `processed/phase_6_extended_case_studies/phase6_topk_budget_curves.png`

핵심 메시지:

- 예산이 작으면 LR/EasyEnsemble이 효율적이다.
- 예산이 커질수록 BalancedBagging/CatBoost 계열이 recall을 더 많이 확보한다.
- 실제 운영팀에는 threshold보다 top-k 방식이 더 이해하기 쉽다.

발표 멘트:

> 실제 캠페인에서는 “확률 0.35 이상”보다 “상위 10% 또는 20% 고객만 관리”가 더 자연스럽습니다. 그래서 top-k 실험을 했고, 예산 구간에 따라 최적 모델이 달라지는 것을 확인했습니다.

## Slide 11. 비용 시나리오별 Threshold 민감도

넣을 표:

| 비용 시나리오 | 최적 모델 | Threshold | 순이익 | Recall | Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| 논문 기준 | CatBoost_native_with_zip | 0.08 | 166,080 | 1.0000 | 0.0653 |
| 보수적 캠페인 비용 | LR_no_zip_f1 | 0.53 | 24,480 | 0.2202 | 0.1395 |
| 강한 예산 제약 | CatBoost_balanced_with_zip | 0.74 | 5,760 | 0.0183 | 0.6667 |
| Enterprise value | CatBoost_native_with_zip | 0.08 | 990,120 | 1.0000 | 0.0653 |
| Small business value | BalancedBagging_no_zip | 0.69 | 1,320 | 0.0367 | 0.2353 |

사용 이미지:

- `processed/phase_6_extended_case_studies/phase6_cost_best_paper_baseline.png`

핵심 메시지:

- 논문 비용 구조에서는 FN이 FP보다 훨씬 비싸서 threshold가 낮아진다.
- 캠페인 비용이 커지면 precision 높은 LR 또는 높은 threshold 전략이 유리하다.
- 최적 모델은 데이터만이 아니라 비용 구조에 의해 결정된다.

발표 멘트:

> 비용 가정을 바꾸면 결론이 완전히 바뀝니다. 이탈을 놓치는 비용이 큰 시나리오에서는 recall-heavy 모델이 유리하지만, 캠페인 비용이 커지면 아주 확실한 고객만 접촉하는 높은 threshold 전략이 더 안전합니다.

## Slide 12. Calibration: 점수는 확률이 아니다

넣을 표:

| Case | Method | Brier | ECE | Mean score | 실제 churn rate |
| --- | --- | ---: | ---: | ---: | ---: |
| LR_no_zip_f1 | raw | 0.1557 | 0.2856 | 0.3460 | 0.0646 |
| LR_no_zip_f1 | platt | 0.0602 | 0.0014 | 0.0653 | 0.0646 |
| BalancedBagging | raw | 0.2285 | 0.4022 | 0.4668 | 0.0646 |
| BalancedBagging | platt | 0.0602 | 0.0003 | 0.0648 | 0.0646 |
| XGBoost | raw | 0.2842 | 0.4059 | 0.4689 | 0.0646 |
| XGBoost | platt | 0.0603 | 0.0000 | 0.0646 | 0.0646 |

사용 이미지:

- `processed/phase_6_extended_case_studies/phase6_calibration_comparison.png`

핵심 메시지:

- raw score 평균은 실제 churn rate보다 훨씬 높다.
- score를 확률처럼 쓰려면 calibration이 필요하다.
- 논문이 isotonic calibration을 사용한 이유를 우리 데이터에서도 확인했다.

발표 멘트:

> 모델 score가 0.4라고 해서 실제 이탈 확률이 40%라는 뜻은 아닙니다. 실제 이탈률은 6.46%인데 raw score 평균은 30~40%대로 과대평가되어 있었고, Platt calibration 후에야 실제 이탈률과 맞아졌습니다.

## Slide 13. Segment별 ROI와 실패 패턴

넣을 표:

| Case | Segment | Positives | TP | FP | Recall | Precision | 순이익 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CatBoost_native | low_value | 47 | 38 | 464 | 0.8085 | 0.0757 | 67,440 |
| XGBoost | mid_value | 34 | 34 | 409 | 1.0000 | 0.0767 | 61,080 |
| BalancedBagging | mid_value | 34 | 29 | 273 | 0.8529 | 0.0960 | 61,200 |
| XGBoost | high_value | 28 | 24 | 346 | 0.8571 | 0.0649 | 36,240 |

사용 이미지:

- `processed/phase_6_extended_case_studies/phase6_segment_recall_heatmap.png`

핵심 메시지:

- 모델의 강점은 segment별로 다르다.
- 전체 평균만 보면 어떤 고객군에서 실패하는지 놓친다.
- 고가치 고객군은 precision이 낮아 고객 피로도 관리가 중요하다.

발표 멘트:

> 같은 모델이라도 CRM segment에 따라 성능이 다릅니다. 전체 평균 성능만 보고 운영하면 어떤 고객군에서 false positive가 집중되는지 놓칠 수 있습니다.

## Slide 14. 모델 합의도 기반 Risk Tiering

넣을 표:

| Churn vote count | 고객 수 | 실제 이탈자 | 관측 이탈률 |
| ---: | ---: | ---: | ---: |
| 0 | 91 | 5 | 5.49% |
| 1 | 195 | 6 | 3.08% |
| 3 | 271 | 18 | 6.64% |
| 6 | 355 | 22 | 6.20% |
| 7 | 134 | 13 | 9.70% |
| 8 | 169 | 21 | 12.43% |

사용 이미지:

- `processed/phase_6_extended_case_studies/phase6_model_agreement.png`

핵심 메시지:

- 8개 모델 모두가 위험하다고 본 고객군의 이탈률은 12.43%다.
- 전체 평균 6.46%의 약 1.9배다.
- 모델 합의도를 risk tier로 쓰면 영업팀 우선순위화에 유용하다.

발표 멘트:

> 여러 모델이 동시에 위험하다고 판단한 고객은 실제 이탈률도 높았습니다. 그래서 모델 하나의 예측보다, 모델 합의도를 이용해 A/B/C risk tier를 만드는 방식도 운영적으로 쓸 수 있습니다.

## Slide 15. 통계 검정 보강

넣을 표:

| Case | F1 95% CI | Recall 95% CI | Precision 95% CI |
| --- | ---: | ---: | ---: |
| LR_no_zip_f1 | [0.1155, 0.2229] | [0.1818, 0.3524] | [0.0823, 0.1674] |
| BalancedBagging_with_zip | [0.1200, 0.1862] | [0.4947, 0.6814] | [0.0679, 0.1090] |
| CatBoost_native_with_zip | [0.1072, 0.1557] | [0.7624, 0.9000] | [0.0574, 0.0856] |
| XGBoost_with_zip | [0.1020, 0.1455] | [0.8739, 0.9717] | [0.0541, 0.0789] |

핵심 메시지:

- LR의 F1 point estimate는 높지만 신뢰구간이 넓다.
- BalancedBagging/CatBoost/XGBoost는 recall 중심 운영에서 장점이 일관된다.
- McNemar test는 모델들의 error pattern이 서로 다르다는 보조 근거다.

발표 멘트:

> Bootstrap CI를 보면 LR이 F1 point는 높지만 불확실성이 큽니다. 반면 recall 중심 모델들은 실제 이탈 고객을 더 많이 잡는 특성이 신뢰구간에서도 유지됩니다. 그래서 결론은 한 모델의 절대 우위가 아니라, 서로 다른 오류 패턴과 운영 목적의 선택입니다.

## Slide 16. 한계와 향후 개선

넣을 내용:

| 한계 | 향후 개선 |
| --- | --- |
| 정적 CRM snapshot | 월별 사용량/매출 추세 추가 |
| 배포용 확률 보정 미반영 | calibrated probability 기반 threshold 재설계 |
| SHAP 직접 비교 미실시 | Tree-SHAP 적용 |
| 비용 가정 고정 | 고객별 ARPU 기반 individualized value |
| 외부 검증 없음 | 기간별 hold-out 또는 다른 시장 데이터 |

핵심 메시지:

- 성능 한계는 모델 부족보다 feature 부족에 가깝다.
- churn 직전 행동 변화 데이터가 있으면 개선 가능성이 크다.

발표 멘트:

> 현재 데이터는 정적 CRM snapshot이라 이탈 직전 행동 변화가 부족합니다. 앞으로 성능을 올리려면 모델을 더 추가하는 것보다 월별 사용량 변화, 결제 실패, 불만 기록, 계약 만료 같은 시간 기반 feature가 더 중요합니다.

## Slide 17. 최종 결론

넣을 내용:

- 논문 EasyEnsemble F1 `0.129`를 우리 EasyEnsemble F1 `0.128`로 재현
- hold-out F1 최고는 `LogisticRegression_SMOTE`
- CV 안정성은 `BalancedBagging`/`EasyEnsemble`이 더 좋음
- 비즈니스 비용 기준으로는 운영 목적별 모델이 달라짐
- 최종 주장은 “압도적 성능 우위”가 아니라 “운영 목적별 모델 선택 프레임워크”

발표용 결론 문장:

> 본 프로젝트는 논문 baseline을 재현한 뒤, ZIP ablation, feature group ablation, segment error analysis, interpretability, 비용-편익 분석, calibration, 통계 검정을 추가하여 불균형 churn 데이터에서 모델을 목적별로 선택하는 운영 프레임워크를 제시했다.

## 심층 Q&A 및 방어 메모

주요 질문 대비:

- "왜 F1이 0.16밖에 안 되나요?" -> 데이터의 정적 한계 설명 (Slide 16 연계)
- "Logistic Regression이 최선인가요?" -> 설명 가능성과 통계적 안정성(CV) 근거 제시
- "실제 현업에서 쓰려면?" -> Top-k 전략과 비용 최적화 threshold 제안
- "자동화는 어디까지 했나요?" -> n8n workflow, Docker runner, PSI drift check, pytest 기반 데이터 무결성 검사를 보조 자료로 설명

## 실제 PPT 제작 순서

1. Slide 1-3: 문제 정의와 전처리
2. Slide 4: 논문 재현 성공
3. Slide 5-6: hold-out과 CV 비교
4. Slide 7-8: 우리 차별화 실험과 해석 가능성
5. Slide 9: 비즈니스 임팩트 대시보드
6. Slide 10-14: top-k, 비용, calibration, segment, 모델 합의도
7. Slide 15: bootstrap CI와 McNemar test
8. Slide 16-17: 한계와 최종 결론

이미지 우선순위:

1. `processed/phase_5b_business_impact/business_impact_dashboard.png`
2. `processed/phase_5b_business_impact/business_impact_bubble.png`
3. `processed/phase_5a_interpretability/lr_coefficient_importance.png`
4. `processed/phase_5a_interpretability/lr_linear_contribution_importance.png`
5. `processed/phase_6_extended_case_studies/phase6_topk_budget_curves.png`
6. `processed/phase_6_extended_case_studies/phase6_calibration_comparison.png`
7. `processed/phase_6_extended_case_studies/phase6_model_agreement.png`
8. `presentation_assets/01_model_metric_comparison.png`
