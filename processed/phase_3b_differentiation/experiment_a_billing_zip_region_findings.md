# Experiment A 추가: Billing_ZIP 지역별 이탈 및 모델 성능

기준: 기존 Experiment A와 동일하게 train/test split을 유지하고, 모델은 전체 학습 데이터로 학습한 뒤 test set을 Billing_ZIP 단위로 나누어 평가했다.
ZIP 단일값은 표본이 작은 값이 많기 때문에 원값별 결과와 ZIP 앞 1자리/2자리 지역 그룹 결과를 함께 저장했다.

## 산출 파일

- `experiment_a_billing_zip_churn_by_value.csv`: 전체 데이터의 Billing_ZIP 원값별 이탈률
- `experiment_a_billing_zip_churn_by_group.csv`: 전체 데이터의 Billing_ZIP 앞 1자리/2자리 그룹별 이탈률
- `experiment_a_billing_zip_model_by_value.csv`: test set의 Billing_ZIP 원값별 F1/recall/precision
- `experiment_a_billing_zip_model_by_group.csv`: test set의 Billing_ZIP 앞 1자리/2자리 그룹별 F1/recall/precision

## 핵심 요약

- 전체 분석 행은 8,436건, 이탈 고객은 545건이다.
- Billing_ZIP 원값은 456개이며, 이 중 427개는 표본 또는 이탈자 수가 작아 탐색용으로만 해석한다.
- 개별 ZIP별로 별도 모델을 학습하면 표본 부족 문제가 크므로, 전체 모델을 학습한 뒤 ZIP별 hold-out 성능을 비교했다.
- 보고서/발표에서는 원값별 극단값보다 `zip_prefix2` 지역 그룹 결과를 중심으로 해석하는 편이 안정적이다.

## 이탈률이 높은 ZIP 원값 그룹

| billing_zip_group | rows | churn_count | churn_rate | support_flag |
| --- | --- | --- | --- | --- |
| 6900 | 23 | 5 | 0.2174 | stable |
| 4800 | 22 | 4 | 0.1818 | stable |
| 4210 | 47 | 6 | 0.1277 | stable |
| 4001 | 40 | 5 | 0.1250 | stable |
| 4850 | 40 | 5 | 0.1250 | stable |
| 4006 | 57 | 6 | 0.1053 | stable |
| 4230 | 244 | 24 | 0.0984 | stable |
| 4600 | 115 | 11 | 0.0957 | stable |
| 4700 | 181 | 17 | 0.0939 | stable |
| 6260 | 43 | 4 | 0.0930 | stable |

## 이탈률이 높은 ZIP 앞 2자리 지역 그룹

| billing_zip_group | rows | churn_count | churn_rate | support_flag |
| --- | --- | --- | --- | --- |
| 69xx | 23 | 5 | 0.2174 | stable |
| 46xx | 155 | 19 | 0.1226 | stable |
| 48xx | 96 | 11 | 0.1146 | stable |
| 47xx | 215 | 24 | 0.1116 | stable |
| 68xx | 58 | 6 | 0.1034 | stable |
| 42xx | 511 | 41 | 0.0802 | stable |
| 66xx | 243 | 19 | 0.0782 | stable |
| 63xx | 370 | 25 | 0.0676 | stable |
| 40xx | 3109 | 209 | 0.0672 | stable |
| 62xx | 119 | 7 | 0.0588 | stable |

## 주요 모델의 ZIP 앞 2자리 그룹별 성능 변화

| variant | model | billing_zip_group | rows | positives | churn_rate | f1 | recall | f1_delta_vs_model_overall | recall_delta_vs_model_overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_billing_zip | BalancedBagging_original | 47xx | 35 | 4 | 0.1143 | 0.3000 | 0.7500 | 0.1474 | 0.1628 |
| with_billing_zip | BalancedBagging_original | 66xx | 48 | 4 | 0.0833 | 0.2143 | 0.7500 | 0.0617 | 0.1628 |
| with_billing_zip | BalancedBagging_original | 64xx | 77 | 4 | 0.0519 | 0.1538 | 0.7500 | 0.0013 | 0.1628 |
| with_billing_zip | BalancedBagging_original | 40xx | 653 | 50 | 0.0766 | 0.1879 | 0.6200 | 0.0353 | 0.0328 |
| with_billing_zip | BalancedBagging_original | 63xx | 73 | 5 | 0.0685 | 0.1765 | 0.6000 | 0.0239 | 0.0128 |
| with_billing_zip | BalancedBagging_original | 41xx | 83 | 5 | 0.0602 | 0.1333 | 0.6000 | -0.0192 | 0.0128 |
| with_billing_zip | BalancedBagging_original | 44xx | 104 | 7 | 0.0673 | 0.1379 | 0.5714 | -0.0146 | -0.0157 |
| with_billing_zip | BalancedBagging_original | 45xx | 44 | 2 | 0.0455 | 0.0909 | 0.5000 | -0.0617 | -0.0872 |
| with_billing_zip | BalancedBagging_original | 61xx | 66 | 2 | 0.0303 | 0.0800 | 0.5000 | -0.0726 | -0.0872 |
| with_billing_zip | BalancedBagging_original | 42xx | 101 | 4 | 0.0396 | 0.0769 | 0.5000 | -0.0756 | -0.0872 |
| with_billing_zip | BalancedBagging_original | 46xx | 33 | 4 | 0.1212 | 0.1176 | 0.2500 | -0.0349 | -0.3372 |
| with_billing_zip | BalancedBagging_original | 60xx | 226 | 8 | 0.0354 | 0.0408 | 0.2500 | -0.1117 | -0.3372 |
| without_billing_zip | LogisticRegression_SMOTE | 64xx | 77 | 4 | 0.0519 | 0.2500 | 0.7500 | 0.0824 | 0.4839 |
| without_billing_zip | LogisticRegression_SMOTE | 66xx | 48 | 4 | 0.0833 | 0.4444 | 0.5000 | 0.2768 | 0.2339 |
| without_billing_zip | LogisticRegression_SMOTE | 47xx | 35 | 4 | 0.1143 | 0.4000 | 0.5000 | 0.2324 | 0.2339 |
| without_billing_zip | LogisticRegression_SMOTE | 45xx | 44 | 2 | 0.0455 | 0.2222 | 0.5000 | 0.0546 | 0.2339 |

## 보고서용 해석 문장

> Billing_ZIP 원값 456개를 모두 나누어 보면 일부 ZIP에서 이탈률이 높게 나타나지만, 다수 값은 표본이 작아 단독 결론으로 쓰기 어렵다. 따라서 본 연구는 전체 모델을 학습한 뒤 Billing_ZIP 원값과 ZIP 앞 1자리/2자리 지역 그룹별 hold-out 성능을 비교했다. 그 결과 지역 정보는 BalancedBagging 계열에서는 일부 지역 그룹의 recall을 끌어올리는 신호로 작동했지만, Logistic Regression에서는 고카디널리티 ZIP 정보가 전체 F1을 낮추는 방향으로 작동했다. 즉 Billing_ZIP은 제거/포함을 단순히 하나로 정할 피처가 아니라, 모델 계열과 지역 그룹 안정성을 함께 고려해야 하는 변수다.
