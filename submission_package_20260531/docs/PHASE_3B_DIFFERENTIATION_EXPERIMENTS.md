# Phase 3-B: Differentiation Experiments

마지막 업데이트: 2026-05-27

## 실행 개요

실행 스크립트:

```powershell
.\.venv\Scripts\python.exe phase_3b_differentiation_experiments.py
```

출력 위치:

- `processed/phase_3b_differentiation/experiment_a_billing_zip_summary.csv`
- `processed/phase_3b_differentiation/experiment_b_feature_group_ablation.csv`
- `processed/phase_3b_differentiation/experiment_c_segment_summary.csv`
- `processed/phase_3b_differentiation/experiment_c_segment_bucket_summary.csv`
- `processed/phase_3b_differentiation/experiment_c_segment_confusion_profiles.csv`
- `processed/phase_3b_differentiation/experiment_c_high_value_submodel.csv`
- `processed/phase_3b_differentiation/experiment_d_cost_threshold_best.csv`
- `processed/phase_3b_differentiation/experiment_d_cost_threshold_sweep.csv`
- `processed/phase_3b_differentiation/threshold_cost_sensitivity.png`

## Experiment A: Billing ZIP Variant

이미 완료된 실험을 Phase 3-B 결과 폴더에 요약했다.

| Variant | Model | F1 | Recall | Precision | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| with ZIP | `BalancedBagging_original` | 0.1526 | 0.5872 | 0.0877 | 0.0871 |
| without ZIP | `BalancedBagging_original` | 0.1397 | 0.5505 | 0.0800 | 0.0798 |
| with ZIP | `LogisticRegression_SMOTE` | 0.1287 | 0.2018 | 0.0944 | 0.0813 |
| without ZIP | `LogisticRegression_SMOTE` | 0.1681 | 0.2661 | 0.1229 | 0.0879 |

해석:

- `Billing_ZIP`은 tree ensemble에서는 recall과 F1을 개선한다.
- LR에서는 ZIP 제거가 더 좋다. 고카디널리티 지역 정보가 선형 경계에서는 noise처럼 작동한 것으로 해석한다.
- permutation importance에서도 `BalancedBagging_original` 기준 `Billing_ZIP`이 F1 importance `0.0135`로 1위다.

보고서 문장:

> 논문은 Billing_ZIP 포함 단일 설정만 제시했지만, 본 연구는 ZIP 포함/제외 ablation을 통해 ZIP 정보의 효과가 모델 계열에 의존함을 보였다. Tree ensemble에서는 지역 정보가 recall을 높였으나, 선형 모델에서는 고카디널리티 인코딩이 F1을 낮췄다.

## Experiment B: Feature Group Ablation

실험 설정:

- `BalancedBagging_original`: with ZIP variant
- `LogisticRegression_SMOTE`: without ZIP variant
- LR은 feature subset마다 `SVMSMOTE`를 다시 fit했다.
- 이미 생성된 train-only 전처리 결과를 사용하므로 scaling/imputation leakage는 추가되지 않는다.

### One-Group-Out 결과

| Model | Dropped group | F1 | Baseline F1 | Delta |
| --- | --- | ---: | ---: | ---: |
| LR_SMOTE | `G_categorical` | 0.0806 | 0.1681 | -0.0875 |
| LR_SMOTE | `G_engineered_rates` | 0.1361 | 0.1681 | -0.0320 |
| LR_SMOTE | `G_transform` | 0.1386 | 0.1681 | -0.0295 |
| LR_SMOTE | `G_interaction` | 0.1392 | 0.1681 | -0.0289 |
| LR_SMOTE | `G_revenue_raw` | 0.1462 | 0.1681 | -0.0219 |
| LR_SMOTE | `G_subscriber_raw` | 0.1479 | 0.1681 | -0.0202 |
| BalancedBagging | `G_interaction` | 0.1344 | 0.1526 | -0.0181 |
| BalancedBagging | `G_categorical` | 0.1382 | 0.1526 | -0.0143 |

핵심 해석:

- LR에서는 `G_categorical` 제거가 가장 치명적이다. CRM segment와 EffectiveSegment frequency가 선형 모델의 기준선을 잡아주는 역할을 한 것으로 보인다.
- BalancedBagging에서는 `G_interaction` 제거가 가장 큰 하락을 만들었다. 이는 논문의 "interaction feature가 engagement와 revenue magnitude를 결합한다"는 주장을 정량적으로 뒷받침한다.
- 기대했던 `G_transform`은 LR에서 중요하지만, 단독으로는 충분하지 않았다. `ONLY_G_transform`의 LR F1은 0.0000으로, 변환 피처는 다른 구조 피처와 결합될 때 의미가 있다.

보고서 문장:

> Feature group ablation 결과, 모델 계열별 기여 구조가 달랐다. LogisticRegression은 categorical segment 정보 제거 시 F1이 0.1681에서 0.0806으로 급락했고, BalancedBagging은 interaction group 제거 시 가장 큰 성능 하락을 보였다. 이는 feature engineering의 효과가 단일 피처 중요도보다 모델 구조와 상호작용한다는 점을 보여준다.

## Experiment C: CRM Segment Analysis

분석 대상:

- Model: `with_billing_zip + BalancedBagging_original`
- Threshold: 0.50

### Segment Bucket 결과

| Bucket | Rows | Positives | Recall | Precision | F1 | PR-AUC | FN revenue at risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| high_value | 404 | 28 | 0.7857 | 0.0655 | 0.1209 | 0.0824 | 1,441.33 |
| mid_value | 500 | 34 | 0.8529 | 0.0960 | 0.1726 | 0.1307 | 534.32 |
| low_value | 784 | 47 | 0.2766 | 0.1413 | 0.1871 | 0.0953 | 2,550.16 |

세부 segment:

- Platinum recall: 0.6364, precision: 0.0654
- Gold recall: 0.8824, precision: 0.0655
- SME recall: 1.0000, precision: 0.1364
- Bronze recall: 0.2826, precision: 0.1461

핵심 해석:

- 예상과 달리 high-value 고객의 recall은 낮지 않았다. 문제는 precision이 매우 낮아 FP가 많다는 점이다.
- low-value 고객은 precision은 상대적으로 높지만 recall이 0.2766으로 낮아 놓치는 이탈자가 많다.
- FN 총매출 위험은 low-value bucket이 가장 크지만, high-value의 FN 1건당 매출 위험이 더 크다.

### High-Value 전용 서브모델

| Model scope | F1 | Recall | Precision | PR-AUC | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global model on high-value test | 0.1209 | 0.7857 | 0.0655 | 0.0824 | 314 | 6 |
| High-value-only model | 0.1325 | 0.3571 | 0.0813 | 0.1341 | 113 | 18 |

해석:

- high-value 전용 모델은 PR-AUC와 precision을 개선하지만 recall을 크게 낮춘다.
- 따라서 고가치 고객 운영에서는 "전용 모델 단독 교체"보다, global high-recall 모델과 high-value precision model을 함께 보는 two-stage review 방식이 더 적합하다.

보고서 문장:

> CRM segment 분석은 단순 전체 F1로는 보이지 않는 운영 리스크를 드러냈다. 고가치 고객군에서는 recall은 높지만 false positive가 과도했고, 저가치 고객군에서는 오히려 recall이 낮았다. 이는 논문의 human oversight 주장을 segment-level 운영 정책으로 구체화하는 결과다.

## Experiment D: Cost-Sensitive Threshold Sensitivity

분석 대상:

- Model: `with_billing_zip + BalancedBagging_original`
- Threshold grid: 0.05 to 0.70
- TP benefit: `FN_cost * retention_rate`, retention rate = 0.60

| Scenario | Cost ratio | Theoretical threshold | Empirical threshold | Net value | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| optimistic_campaign | 180.0 | 0.0092 | 0.29 | 306,330 | 1.0000 | 0.0653 |
| paper_baseline | 45.0 | 0.0357 | 0.29 | 165,840 | 1.0000 | 0.0653 |
| conservative_campaign | 15.0 | 0.1000 | 0.29 | -208,800 | 1.0000 | 0.0653 |
| budget_limited | 7.5 | 0.1818 | 0.51 | -504,360 | 0.5596 | 0.0902 |
| small_business_customers | 10.0 | 0.1429 | 0.51 | -87,480 | 0.5596 | 0.0902 |
| enterprise_customers | 150.0 | 0.0110 | 0.29 | 989,880 | 1.0000 | 0.0653 |

핵심 해석:

- 논문 기준 비용 비율 45:1에서는 threshold 0.29가 최적이며 모든 이탈자를 잡는 쪽이 기대가치가 높다.
- 보수적 캠페인, 예산 제약, 소기업 시나리오에서는 기대가치가 음수다. 즉 이 모델을 그대로 campaign에 쓰면 비용 구조에 따라 손실이 날 수 있다.
- 이론 threshold와 empirical threshold가 크게 다른 이유는 BalancedBagging score가 calibrated probability가 아니기 때문이다. 따라서 threshold는 확률 해석보다 운영 sweep 관점으로 해석해야 한다.

보고서 문장:

> Cost-sensitive threshold 분석 결과, 논문 기준 45:1 비용 가정에서는 낮은 threshold로 recall을 극대화하는 전략이 기대가치를 높였다. 그러나 비용 비율이 낮아지는 보수적 시나리오에서는 기대가치가 음수로 전환되어, 모델 성능뿐 아니라 캠페인 단가와 고객가치 가정이 운영 가능성을 좌우함을 확인했다.

## 최종 요약

| Experiment | 결론 |
| --- | --- |
| A: Billing_ZIP | ZIP은 tree ensemble recall을 높이지만 LR F1은 낮춘다. |
| B: Feature Ablation | LR은 categorical group, BalancedBagging은 interaction group에 가장 민감하다. |
| C: CRM Segment | high-value는 recall보다 precision/FP 문제가 크고, low-value는 recall 문제가 크다. |
| D: Cost Threshold | 논문 비용 가정에서는 high-recall 전략이 유리하지만, 비용 구조가 보수적이면 campaign 가치가 음수다. |

