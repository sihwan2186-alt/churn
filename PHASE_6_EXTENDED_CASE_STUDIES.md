# Phase 6: 1시간 발표용 추가 비교 실험 케이스

## 실행 개요

추가 실험의 목적은 단일 성능표를 넘어, 1시간 발표에서 사용할 수 있는 비교 케이스를 충분히 확보하는 것이다. 이미 생성된 train/test 전처리 데이터를 사용하여 다음 6개 관점의 실험을 추가했다.

1. 운영점 8개 모델 비교
2. Top-k 캠페인 예산별 성능
3. 비용 시나리오별 최적 threshold
4. 확률 보정 및 calibration 진단
5. CRM segment별 운영 성능
6. 모델 합의도 및 churn capture overlap

실행 스크립트:

```powershell
.\.venv\Scripts\python.exe phase_6_extended_case_studies.py
```

산출물 폴더:

```text
processed/phase_6_extended_case_studies/
```

## 1. 운영점 8개 비교

| Case | Model | Threshold | F1 | Recall | Precision | TP | FP | FN | 순이익 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR_no_zip_f1 | LogisticRegression_SMOTE | 0.50 | 0.1681 | 0.2661 | 0.1229 | 29 | 207 | 80 | 69,120 |
| LR_with_zip_recall_constraint | LogisticRegression_SMOTE | 0.46 | 0.1507 | 0.3028 | 0.1003 | 33 | 296 | 76 | 71,400 |
| BalancedBagging_with_zip | BalancedBagging_original | 0.50 | 0.1526 | 0.5872 | 0.0877 | 64 | 666 | 45 | 127,440 |
| BalancedBagging_no_zip | BalancedBagging_original | 0.44 | 0.1353 | 0.7064 | 0.0748 | 77 | 952 | 32 | 135,240 |
| EasyEnsemble_with_zip | EasyEnsemble_original | 0.50 | 0.1284 | 0.5872 | 0.0721 | 64 | 824 | 45 | 108,480 |
| CatBoost_native_with_zip | CatBoost_native_categorical | 0.35 | 0.1310 | 0.8349 | 0.0711 | 91 | 1,189 | 18 | 152,160 |
| CatBoost_balanced_with_zip | CatBoost_original_balanced | 0.34 | 0.1317 | 0.7982 | 0.0718 | 87 | 1,125 | 22 | 146,880 |
| XGBoost_with_zip | XGBoost_SMOTE | 0.16 | 0.1253 | 0.9266 | 0.0672 | 101 | 1,402 | 8 | 159,000 |

발표 포인트:

- F1 기준이면 LR이 가장 좋다.
- 순이익 기준이면 recall-heavy 모델이 유리해진다.
- XGBoost는 recall 92.7%로 가장 많은 이탈 고객을 잡지만 FP도 1,402명으로 매우 많다.
- BalancedBagging no-ZIP은 F1은 낮지만 순이익은 with-ZIP보다 높다. ZIP이 항상 이득은 아니라는 추가 근거다.

## 2. Top-k 캠페인 예산 실험

운영 현장에서는 threshold보다 “상위 몇 % 고객에게 캠페인할 것인가”가 더 직관적일 수 있다. 따라서 각 모델의 score로 고객을 정렬한 뒤 top-k%만 캠페인 대상으로 삼았다.

| Top-k | 권장 모델 | 접촉 수 | TP | FP | Recall@k | Precision@k | 순이익 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | EasyEnsemble_with_zip | 84 | 11 | 73 | 0.1009 | 0.1310 | 26,880 |
| 10% | LR_no_zip_f1 | 169 | 23 | 146 | 0.2110 | 0.1361 | 57,000 |
| 15% | LR_no_zip_f1 | 253 | 29 | 224 | 0.2661 | 0.1146 | 67,080 |
| 20% | LR_with_zip_recall_constraint | 338 | 33 | 305 | 0.3028 | 0.0976 | 70,320 |
| 25% | EasyEnsemble_with_zip | 422 | 41 | 381 | 0.3761 | 0.0972 | 87,120 |
| 30% | EasyEnsemble_with_zip | 506 | 46 | 460 | 0.4220 | 0.0909 | 93,840 |
| 40% | BalancedBagging_with_zip | 675 | 61 | 614 | 0.5596 | 0.0904 | 123,960 |
| 50% | CatBoost_balanced_with_zip | 844 | 66 | 778 | 0.6055 | 0.0782 | 120,480 |
| 75% | LR_with_zip_recall_constraint | 1,266 | 91 | 1,175 | 0.8349 | 0.0719 | 153,840 |
| 100% | LR_no_zip_f1 | 1,688 | 109 | 1,579 | 1.0000 | 0.0646 | 163,680 |

발표 포인트:

- 예산이 매우 작으면 LR/EasyEnsemble이 효율적이다.
- 40% 이상으로 예산을 넓히면 BalancedBagging/CatBoost 계열이 이탈 고객을 더 많이 포착한다.
- top-k 방식은 threshold보다 운영팀이 이해하기 쉽다.

주의:

- top 100%는 사실상 전체 고객 캠페인이므로 운영 전략으로는 부적절하다.
- 논문 비용 가정에서는 FP 비용이 낮아 전체 캠페인도 수치상 이익이 나지만, 실제 고객 피로도와 브랜드 비용은 별도로 고려해야 한다.

## 3. 비용 시나리오별 최적 Threshold

모델별 score threshold를 0.01부터 0.99까지 탐색하고, 비용 시나리오별 최대 기대 순이익을 찾았다.

### 3.1 논문 기준 비용 시나리오

| Model case | Threshold | 순이익 | TP | FP | FN | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CatBoost_native_with_zip | 0.08 | 166,080 | 109 | 1,559 | 0 | 1.0000 | 0.0653 |
| BalancedBagging_with_zip | 0.29 | 165,840 | 109 | 1,561 | 0 | 1.0000 | 0.0653 |
| CatBoost_balanced_with_zip | 0.05 | 165,600 | 109 | 1,563 | 0 | 1.0000 | 0.0652 |
| BalancedBagging_no_zip | 0.25 | 164,760 | 109 | 1,570 | 0 | 1.0000 | 0.0649 |
| LR_with_zip_recall_constraint | 0.03 | 164,640 | 108 | 1,544 | 1 | 0.9908 | 0.0654 |
| XGBoost_with_zip | 0.01 | 164,640 | 109 | 1,571 | 0 | 1.0000 | 0.0649 |
| EasyEnsemble_with_zip | 0.33 | 164,520 | 109 | 1,572 | 0 | 1.0000 | 0.0648 |
| LR_no_zip_f1 | 0.03 | 161,040 | 107 | 1,547 | 2 | 0.9817 | 0.0647 |

해석:

- 논문 비용 구조에서는 `TP benefit=3,240`, `FP cost=120`이다.
- FP보다 FN이 훨씬 비싸므로 최적 threshold가 매우 낮아진다.
- 대부분 모델이 recall 1.0 근처로 수렴한다.

### 3.2 보수적 캠페인 비용 시나리오

`FP cost=360`으로 높이면 결과가 완전히 달라진다.

| Model case | Threshold | 순이익 | TP | FP | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LR_no_zip_f1 | 0.53 | 24,480 | 24 | 148 | 0.2202 | 0.1395 |
| XGBoost_with_zip | 0.90 | 12,240 | 8 | 38 | 0.0734 | 0.1739 |
| EasyEnsemble_with_zip | 0.61 | 10,440 | 10 | 61 | 0.0917 | 0.1408 |
| CatBoost_balanced_with_zip | 0.64 | 8,640 | 8 | 48 | 0.0734 | 0.1429 |

해석:

- 캠페인 비용이 커지면 recall-heavy 모델보다 precision이 높은 LR이 유리해진다.
- threshold도 높아져 캠페인 대상이 줄어든다.
- 즉 “최적 모델”은 비용 구조에 강하게 의존한다.

### 3.3 예산 제약이 매우 강한 시나리오

`FP cost=720`에서는 대부분 모델의 기대 순이익이 0에 가깝거나 음수가 된다.

| Model case | Threshold | 순이익 | TP | FP | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CatBoost_balanced_with_zip | 0.74 | 5,760 | 2 | 1 | 0.0183 | 0.6667 |
| BalancedBagging_no_zip | 0.69 | 3,600 | 4 | 13 | 0.0367 | 0.2353 |
| BalancedBagging_with_zip | 0.67 | 360 | 3 | 13 | 0.0275 | 0.1875 |

해석:

- 캠페인 비용이 매우 크면 “많이 잡는 모델”보다 “아주 확실한 고객만 접촉”하는 전략이 유리하다.
- 이 시나리오는 churn model을 전체 자동 캠페인 도구가 아니라 high-confidence prioritization 도구로 써야 함을 보여준다.

## 4. Calibration 실험

raw model score는 calibrated probability가 아니다. 실제 평균 churn rate는 6.46%인데, raw score 평균은 LR 34.6%, BalancedBagging 46.7%, CatBoost 41.4%, XGBoost 46.9%로 과대평가되어 있었다.

| Case | Method | Brier | ECE | Mean score | 실제 churn rate |
| --- | --- | ---: | ---: | ---: | ---: |
| LR_no_zip_f1 | raw | 0.1557 | 0.2856 | 0.3460 | 0.0646 |
| LR_no_zip_f1 | platt | 0.0602 | 0.0014 | 0.0653 | 0.0646 |
| BalancedBagging_with_zip | raw | 0.2285 | 0.4022 | 0.4668 | 0.0646 |
| BalancedBagging_with_zip | platt | 0.0602 | 0.0003 | 0.0648 | 0.0646 |
| CatBoost_native_with_zip | raw | 0.1985 | 0.3501 | 0.4136 | 0.0646 |
| CatBoost_native_with_zip | platt | 0.0603 | 0.0001 | 0.0646 | 0.0646 |
| XGBoost_with_zip | raw | 0.2842 | 0.4059 | 0.4689 | 0.0646 |
| XGBoost_with_zip | platt | 0.0603 | 0.0000 | 0.0646 | 0.0646 |

발표 포인트:

- raw score를 “이탈 확률”로 말하면 안 된다.
- Platt calibration 또는 isotonic calibration을 적용하면 평균 score가 실제 이탈률에 맞춰진다.
- 논문이 isotonic calibration을 수행한 이유가 실증적으로 확인된다.

주의:

- calibration은 probability 해석을 개선하지만 ranking 성능 자체를 크게 올리는 것은 아니다.
- threshold 운영과 probability 운영을 구분해서 설명해야 한다.

## 5. Segment별 운영 성능

CRM segment bucket별로 모델 성능과 순이익을 분해했다.

상위 순이익 case:

| Case | Segment | Positives | TP | FP | Recall | Precision | 순이익 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CatBoost_native_with_zip | low_value | 47 | 38 | 464 | 0.8085 | 0.0757 | 67,440 |
| CatBoost_balanced_with_zip | low_value | 47 | 36 | 435 | 0.7660 | 0.0764 | 64,440 |
| XGBoost_with_zip | low_value | 47 | 43 | 647 | 0.9149 | 0.0623 | 61,680 |
| BalancedBagging_with_zip | mid_value | 34 | 29 | 273 | 0.8529 | 0.0960 | 61,200 |
| XGBoost_with_zip | mid_value | 34 | 34 | 409 | 1.0000 | 0.0767 | 61,080 |
| CatBoost_native_with_zip | mid_value | 34 | 33 | 399 | 0.9706 | 0.0764 | 59,040 |

발표 포인트:

- 모델별 강점이 segment에 따라 다르다.
- low_value에서는 CatBoost 계열이 많은 이탈 고객을 포착했다.
- mid_value에서는 BalancedBagging과 XGBoost가 강했다.
- high_value는 precision이 낮아 고객 피로도 관리가 더 중요하다.

## 6. 모델 합의도와 Capture Overlap

8개 모델 중 몇 개가 동일 고객을 churn으로 판단했는지 vote count를 만들었다.

| Churn vote count | 고객 수 | 실제 이탈자 | 관측 이탈률 |
| ---: | ---: | ---: | ---: |
| 0 | 91 | 5 | 5.49% |
| 1 | 195 | 6 | 3.08% |
| 2 | 137 | 5 | 3.65% |
| 3 | 271 | 18 | 6.64% |
| 4 | 175 | 10 | 5.71% |
| 5 | 161 | 9 | 5.59% |
| 6 | 355 | 22 | 6.20% |
| 7 | 134 | 13 | 9.70% |
| 8 | 169 | 21 | 12.43% |

발표 포인트:

- 모든 모델이 동시에 위험하다고 보는 고객군의 실제 이탈률은 12.43%로 전체 평균 6.46%의 약 1.9배다.
- 모델 합의도는 별도의 ensemble score 또는 priority tier로 활용할 수 있다.
- 다만 vote count 0에서도 실제 이탈자가 5명 있어, 어떤 모델도 모든 churner를 완벽히 잡지는 못한다.

Capture overlap:

| Case | 잡은 이탈자 | 해당 모델만 유일하게 잡은 이탈자 |
| --- | ---: | ---: |
| LR_no_zip_f1 | 29 | 0 |
| BalancedBagging_with_zip | 64 | 0 |
| CatBoost_native_with_zip | 91 | 2 |
| XGBoost_with_zip | 101 | 4 |

해석:

- XGBoost가 가장 많은 churner를 잡고, 유일하게 잡은 churner도 4명 있다.
- CatBoost native도 유일하게 잡은 churner 2명이 있다.
- LR이 잡은 churner는 대부분 다른 recall-heavy 모델도 잡는다.

## 7. 1시간 발표 구성에 추가하는 방법

기존 11장 슬라이드에 아래 5장을 추가하면 1시간 발표 구성이 탄탄해진다.

| 추가 Slide | 제목 | 핵심 메시지 |
| --- | --- | --- |
| 12 | Top-k Budget Strategy | 예산별로 최적 모델이 달라진다 |
| 13 | Cost Scenario Sensitivity | 비용 구조가 바뀌면 threshold와 최적 모델이 바뀐다 |
| 14 | Calibration Matters | raw score는 확률이 아니며 보정이 필요하다 |
| 15 | Segment-specific ROI | 전체 평균보다 segment별 실패/성공 패턴이 중요하다 |
| 16 | Model Agreement Tiering | 여러 모델이 동시에 경고한 고객은 이탈률이 더 높다 |

발표에서 쓸 수 있는 최종 문장:

> 추가 실험 결과, churn 모델의 가치는 단일 F1 점수보다 운영 제약에 따라 달라졌다. 예산이 작으면 top-k LR/EasyEnsemble이 효율적이고, 논문 비용 기준에서는 threshold를 낮춰 recall을 극대화하는 전략이 유리하며, 캠페인 비용이 커지면 precision 높은 LR이 다시 유리해진다. 또한 raw score는 실제 확률보다 크게 과대평가되어 calibration이 필요하고, CRM segment와 모델 합의도를 이용하면 고객 우선순위를 더 세밀하게 나눌 수 있다.

## 8. 산출물

| 파일 | 내용 |
| --- | --- |
| `phase6_model_operating_metrics.csv` | 8개 운영점의 threshold 성능 |
| `phase6_topk_budget_curve.csv` | top-k 예산별 TP/FP/순이익 |
| `phase6_cost_threshold_best_by_model.csv` | 비용 시나리오별 모델 최적 threshold |
| `phase6_cost_threshold_sweep.csv` | 전체 threshold sweep 원본 |
| `phase6_calibration_metrics.csv` | raw/isotonic/platt calibration 비교 |
| `phase6_calibration_bins.csv` | calibration bin 상세 |
| `phase6_segment_operating_metrics.csv` | segment bucket별 성능 |
| `phase6_model_agreement_vote_groups.csv` | 모델 vote count별 이탈률 |
| `phase6_churn_capture_overlap.csv` | 모델별 churner capture overlap |
| `phase6_topk_budget_curves.png` | 발표용 top-k 곡선 |
| `phase6_cost_best_paper_baseline.png` | 논문 비용 기준 최적 threshold 막대그래프 |
| `phase6_calibration_comparison.png` | calibration 비교 그래프 |
| `phase6_segment_recall_heatmap.png` | segment별 recall heatmap |
| `phase6_model_agreement.png` | 모델 합의도 이탈률 그래프 |
