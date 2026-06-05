# Objective Best Model Comparison

목적: 전부 학습된 모델 후보를 F1, recall, F2, precision, PR-AUC, MCC, 비용 순이익 등 목적별로 다시 비교해 각 목적의 최고 모델을 정리한다.

## 학습/비교 범위

- 통합 비교 후보 수: 64개
- 포함 소스: baseline threshold tuning, additional 46 candidate training, recall optimized refit
- 추가 46개 후보는 전부 학습을 시도했고, AdaBoost stump 설정 2개는 base classifier가 random보다 낮다는 오류로 제외되어 44개가 성공했다.
- test set: 1,688명, 실제 이탈자 109명
- 비용 기준: TP benefit 3,240, FP cost 120

## 목적별 최고 모델

| 목적 | 실험 소스 | Variant | Family | Model | Threshold | F1 | F2 | Recall | Precision | PR-AUC | MCC | TP | FP | FN | Contact Rate | Net Value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 중심 | additional_46_candidate_training | with_billing_zip | BalancedBagging | BalancedBagging_tree_depthnone_leaf25 | 0.5100 | 0.1605 | 0.2732 | 0.5138 | 0.0951 | 0.0832 | 0.0909 | 56 | 533 | 53 | 0.3489 | 117,480 |
| Recall 중심 | additional_46_candidate_training | with_billing_zip | GradientBoosting | HistGradientBoosting_balanced_lr0.03 | 0.1900 | 0.1211 | 0.2505 | 0.8716 | 0.0651 | 0.0822 | 0.0051 | 95 | 1365 | 14 | 0.8649 | 144,000 |
| Recall 운영형 | baseline_threshold_tuning | with_billing_zip | CatBoost | CatBoost_original_balanced | 0.3400 | 0.1317 | 0.2640 | 0.7982 | 0.0718 | 0.1018 | 0.0468 | 87 | 1125 | 22 | 0.7180 | 146,880 |
| F2 중심 | baseline_threshold_tuning | with_billing_zip | BalancedBagging | BalancedBagging_original | 0.5000 | 0.1526 | 0.2744 | 0.5872 | 0.0877 | 0.0871 | 0.0820 | 64 | 666 | 45 | 0.4325 | 127,440 |
| Precision 중심 | additional_46_candidate_training | with_billing_zip | LogisticRegression | LogisticRegression_SMOTE_C0.1 | 0.5100 | 0.1358 | 0.1690 | 0.2018 | 0.1023 | 0.0827 | 0.0587 | 22 | 193 | 87 | 0.1274 | 48,120 |
| PR-AUC 중심 | baseline_threshold_tuning | with_billing_zip | CatBoost | CatBoost_original_balanced | 0.3400 | 0.1317 | 0.2640 | 0.7982 | 0.0718 | 0.1018 | 0.0468 | 87 | 1125 | 22 | 0.7180 | 146,880 |
| MCC 중심 | additional_46_candidate_training | with_billing_zip | BalancedBagging | BalancedBagging_tree_depthnone_leaf25 | 0.5100 | 0.1605 | 0.2732 | 0.5138 | 0.0951 | 0.0832 | 0.0909 | 56 | 533 | 53 | 0.3489 | 117,480 |
| 비용 순이익 중심 | baseline_threshold_tuning | with_billing_zip | CatBoost | CatBoost_native_categorical | 0.3500 | 0.1310 | 0.2652 | 0.8349 | 0.0711 | 0.0789 | 0.0470 | 91 | 1189 | 18 | 0.7583 | 152,160 |
| 소규모 캠페인형 | baseline_threshold_tuning | with_billing_zip | LogisticRegression | LogisticRegression_SMOTE | 0.4600 | 0.1507 | 0.2157 | 0.3028 | 0.1003 | 0.0812 | 0.0715 | 33 | 296 | 76 | 0.1949 | 71,400 |

## 해석

- F1 중심 모델은 이탈자 포착과 오탐 사이의 균형이 가장 좋다.
- Recall 중심 모델은 이탈자를 가장 많이 잡지만 FP와 contact rate가 크게 증가한다.
- Recall 운영형은 최소 precision과 최대 접촉률 조건을 둔 현실적인 recall 후보이다.
- PR-AUC 중심 모델은 threshold 선택 전 ranking 품질이 좋은 모델로 해석한다.
- 비용 순이익 중심 모델은 논문 비용 가정에서는 recall을 크게 높이는 모델이 유리하지만, 실제 운영에서는 고객 피로도와 상담 예산 제약을 함께 봐야 한다.
