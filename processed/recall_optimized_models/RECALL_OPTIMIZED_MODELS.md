# Recall-Optimized Churn Models

목적: CatBoost, XGBoost, Logistic Regression을 ChurnRadar의 이탈 고객 포착 목적에 맞게 recall 중심으로 재최적화했다.

선택 규칙: train 내부 validation set에서 `precision >= 0.07`, `predicted_positive_rate <= 0.75`를 만족하는 후보 중 recall을 최우선으로 선택했다. 동률이면 F2와 precision을 사용했다. Test set은 최종 평가에만 사용했다.

## Best Test Results

| Variant | Family | Model | Threshold | Recall | F2 | F1 | Precision | TP | FP | FN | Contact Rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| without_billing_zip | XGBoost | XGBoost_weighted_depth3_lr0.06_n200_spw14.5 | 0.30 | 0.8349 | 0.2517 | 0.1229 | 0.0663 | 91 | 1281 | 18 | 0.8128 |
| with_billing_zip | XGBoost | XGBoost_weighted_depth2_lr0.06_n200_spw20.0 | 0.44 | 0.8165 | 0.2599 | 0.1285 | 0.0697 | 89 | 1187 | 20 | 0.7559 |
| without_billing_zip | CatBoost | CatBoost_weighted_depth4_lr0.03_iter150_spw14.5 | 0.40 | 0.8073 | 0.2561 | 0.1265 | 0.0686 | 88 | 1194 | 21 | 0.7595 |
| without_billing_zip | LogisticRegression | LogisticRegression_original_balanced_C0.03 | 0.43 | 0.7982 | 0.2611 | 0.1299 | 0.0707 | 87 | 1143 | 22 | 0.7287 |
| with_billing_zip | CatBoost | CatBoost_weighted_depth4_lr0.03_iter150_spw20.0 | 0.48 | 0.7798 | 0.2525 | 0.1254 | 0.0682 | 85 | 1162 | 24 | 0.7387 |
| with_billing_zip | LogisticRegression | LogisticRegression_original_balanced_C1.0 | 0.43 | 0.7706 | 0.2589 | 0.1297 | 0.0708 | 84 | 1102 | 25 | 0.7026 |

## Interpretation

- recall 최적화는 이탈 고객을 더 많이 잡는 대신 FP를 크게 늘린다.
- 따라서 이 결과는 F1 대표 모델을 대체하기보다, 이탈 포착 우선 캠페인 후보로 해석한다.
- 실무에서는 contact rate와 상담/혜택 예산을 함께 제한해야 한다.

## CatBoost Native Categorical Comparison

기존 threshold tuning 결과에서 `with_billing_zip + CatBoost_native_categorical`은 threshold `0.35`에서 recall `0.8349`, F1 `0.1310`, precision `0.0711`, TP `91`, FP `1189`, FN `18`을 기록했다. 따라서 CatBoost 계열에서 이탈 포착만 보면 native categorical 운영점이 가장 강한 후보이고, 위의 weighted CatBoost 튜닝 결과는 같은 계열의 보조 비교로 해석한다.
