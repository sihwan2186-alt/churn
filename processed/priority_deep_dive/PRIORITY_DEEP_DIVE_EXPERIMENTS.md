# Priority Deep Dive Experiments

목적: 보고서에서 집중할 3대 실험을 더 깊게 검증했다. 1) 목적별 챔피언 5-fold CV, 2) ZIP 지역별 top-k 캠페인, 3) contact-rate/비용/ARPU proxy 운영 분석.

## 1. 목적별 챔피언 5-fold CV

| case_label | variant | model | f1_mean | f1_sd | recall_mean | recall_sd | precision_mean | contact_rate_mean | net_value_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F2_BalancedBagging_original | with_billing_zip | BalancedBagging_original | 0.1455 | 0.0126 | 0.5248 | 0.0760 | 0.0845 | 0.4004 | 111,120 |
| F1_BalancedBagging_tuned | with_billing_zip | BalancedBagging_tree_depthnone_leaf25 | 0.1437 | 0.0166 | 0.4404 | 0.0632 | 0.0859 | 0.3305 | 94,368 |
| SmallCampaign_LR_SMOTE | with_billing_zip | LogisticRegression_SMOTE | 0.1348 | 0.0285 | 0.2385 | 0.0468 | 0.0940 | 0.1645 | 54,048 |
| Cost_CatBoost_native | with_billing_zip | CatBoost_native_categorical | 0.1271 | 0.0034 | 0.7945 | 0.0201 | 0.0691 | 0.7431 | 140,520 |
| Recall_XGBoost_weighted_no_zip | without_billing_zip | XGBoost_weighted_depth3_lr0.06_n200_spw14.5 | 0.1258 | 0.0039 | 0.8569 | 0.0392 | 0.0679 | 0.8152 | 148,776 |
| Recall_HistGradientBoosting | with_billing_zip | HistGradientBoosting_balanced_lr0.03 | 0.1250 | 0.0051 | 0.8752 | 0.0461 | 0.0673 | 0.8401 | 150,456 |
| Recall_CatBoost_balanced | with_billing_zip | CatBoost_original_balanced | 0.1241 | 0.0042 | 0.7505 | 0.0364 | 0.0676 | 0.7167 | 129,744 |
| Precision_LR_SMOTE_C0.1 | with_billing_zip | LogisticRegression_SMOTE_C0.1 | 0.1174 | 0.0302 | 0.1394 | 0.0300 | 0.1018 | 0.0901 | 32,832 |

## 2. 전역 Top-k/Contact Rate별 최고 운영점

| top_fraction | case_label | selected_count | tp | fp | precision_at_k | recall_at_k | net_value | individualized_net_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | Recall_CatBoost_balanced | 169 | 17 | 152 | 0.1006 | 0.1560 | 36,840 | -15622.1160 |
| 0.2000 | Precision_LR_SMOTE_C0.1 | 338 | 33 | 305 | 0.0976 | 0.3028 | 70,320 | -31622.4060 |
| 0.3000 | F1_BalancedBagging_tuned | 506 | 45 | 461 | 0.0889 | 0.4128 | 90,480 | -49491.1080 |
| 0.4000 | F2_BalancedBagging_original | 675 | 61 | 614 | 0.0904 | 0.5596 | 123,960 | -65747.2020 |
| 0.5000 | Precision_LR_SMOTE_C0.1 | 844 | 68 | 776 | 0.0806 | 0.6239 | 127,200 | -84939.0060 |
| 0.7500 | Precision_LR_SMOTE_C0.1 | 1266 | 92 | 1174 | 0.0727 | 0.8440 | 157,200 | -131363.0100 |

## 3. ZIP 앞 2자리 지역별 Top 30% 캠페인 우수 조합

| Billing_ZIP_prefix2 | case_label | group_rows | group_positives | group_churn_rate | tp | fp | precision_at_k | recall_at_k | net_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40xx | Recall_XGBoost_weighted_no_zip | 653 | 50 | 0.0766 | 24 | 172 | 0.1224 | 0.4800 | 57,120 |
| 40xx | Cost_CatBoost_native | 653 | 50 | 0.0766 | 23 | 173 | 0.1173 | 0.4600 | 53,760 |
| 40xx | Recall_CatBoost_balanced | 653 | 50 | 0.0766 | 23 | 173 | 0.1173 | 0.4600 | 53,760 |
| 40xx | Recall_HistGradientBoosting | 653 | 50 | 0.0766 | 23 | 173 | 0.1173 | 0.4600 | 53,760 |
| 40xx | F1_BalancedBagging_tuned | 653 | 50 | 0.0766 | 22 | 174 | 0.1122 | 0.4400 | 50,400 |
| 40xx | Precision_LR_SMOTE_C0.1 | 653 | 50 | 0.0766 | 22 | 174 | 0.1122 | 0.4400 | 50,400 |
| 40xx | SmallCampaign_LR_SMOTE | 653 | 50 | 0.0766 | 21 | 175 | 0.1071 | 0.4200 | 47,040 |
| 40xx | F2_BalancedBagging_original | 653 | 50 | 0.0766 | 20 | 176 | 0.1020 | 0.4000 | 43,680 |
| 47xx | SmallCampaign_LR_SMOTE | 35 | 4 | 0.1143 | 3 | 7 | 0.3000 | 0.7500 | 8,880 |
| 66xx | Precision_LR_SMOTE_C0.1 | 48 | 4 | 0.0833 | 3 | 11 | 0.2143 | 0.7500 | 8,400 |
| 66xx | SmallCampaign_LR_SMOTE | 48 | 4 | 0.0833 | 3 | 11 | 0.2143 | 0.7500 | 8,400 |
| 64xx | Precision_LR_SMOTE_C0.1 | 77 | 4 | 0.0519 | 3 | 20 | 0.1304 | 0.7500 | 7,320 |

## 4. ZIP 앞 2자리 지역별 threshold oracle 진단

이 표는 test label을 사용해 지역별 최적 threshold를 찾은 진단용 결과다. 운영 적용 전에는 별도 validation으로 다시 선택해야 한다.

| Billing_ZIP_prefix2 | case_label | threshold | group_rows | group_positives | f1 | recall | precision | tp | fp | net_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40xx | F1_BalancedBagging_tuned | 0.3000 | 653 | 50 | 0.1501 | 0.9800 | 0.0813 | 49 | 554 | 92,280 |
| 40xx | Cost_CatBoost_native | 0.0800 | 653 | 50 | 0.1441 | 1.0000 | 0.0776 | 50 | 594 | 90,720 |
| 40xx | F2_BalancedBagging_original | 0.2900 | 653 | 50 | 0.1441 | 1.0000 | 0.0776 | 50 | 594 | 90,720 |
| 40xx | Recall_CatBoost_balanced | 0.0500 | 653 | 50 | 0.1439 | 1.0000 | 0.0775 | 50 | 595 | 90,600 |
| 40xx | Precision_LR_SMOTE_C0.1 | 0.0800 | 653 | 50 | 0.1469 | 0.9800 | 0.0794 | 49 | 568 | 90,600 |
| 40xx | SmallCampaign_LR_SMOTE | 0.0800 | 653 | 50 | 0.1469 | 0.9800 | 0.0794 | 49 | 568 | 90,600 |
| 40xx | Recall_HistGradientBoosting | 0.0100 | 653 | 50 | 0.1429 | 1.0000 | 0.0769 | 50 | 600 | 90,000 |
| 40xx | Recall_XGBoost_weighted_no_zip | 0.0700 | 653 | 50 | 0.1427 | 1.0000 | 0.0768 | 50 | 601 | 89,880 |
| 44xx | F1_BalancedBagging_tuned | 0.3100 | 104 | 7 | 0.1414 | 1.0000 | 0.0761 | 7 | 85 | 12,480 |
| 44xx | F2_BalancedBagging_original | 0.3300 | 104 | 7 | 0.1386 | 1.0000 | 0.0745 | 7 | 87 | 12,240 |
| 47xx | Precision_LR_SMOTE_C0.1 | 0.3400 | 35 | 4 | 0.4444 | 1.0000 | 0.2857 | 4 | 10 | 11,760 |
| 47xx | SmallCampaign_LR_SMOTE | 0.3400 | 35 | 4 | 0.4444 | 1.0000 | 0.2857 | 4 | 10 | 11,760 |
