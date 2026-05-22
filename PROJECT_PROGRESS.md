# ChurnRadar 프로젝트 진행 기록

마지막 업데이트: 2026-05-22

이 문서는 `Baza customer Telecom v2.csv`를 사용한 통신사 고객 이탈 예측 프로젝트에서 지금까지 무엇을 했는지, 왜 했는지, 결과가 어땠는지, 다음에 무엇을 하면 좋은지를 한글 주석 형태로 정리한 기록입니다.

## 다음에 진행할 것

가장 추천하는 다음 작업은 **교수님께 프로젝트 변경 가능 여부를 확인하는 것**입니다.

현재 프로젝트의 F1과 recall이 낮은 이유가 단순한 모델 선택 문제가 아니라 데이터 구조의 한계일 가능성이 크기 때문에, `PROJECT_CHANGE_PROPOSAL.md`에 1차 변경 제안서를 작성했습니다. 이후 2인 팀 프로젝트 상황과 다른 팀원이 조사한 서울 아파트 버블 탐지 데이터 수집 계획을 반영해 `TEAM_PROJECT_SWITCH_REPORT.md`를 추가했습니다. 교수님께 제출할 때는 `TEAM_PROJECT_SWITCH_REPORT.md`를 우선 사용하는 것이 좋습니다.

추천 순서:

1. `TEAM_PROJECT_SWITCH_REPORT.md`를 읽고 교수님께 설명할 문장을 정리하기
2. 프로젝트 변경 승인을 받기
3. 승인되면 서울 아파트 버블 탐지 및 위험 예측 프로젝트로 전환하기
4. 승인이 어렵다면 기존 프로젝트를 “데이터 한계 분석 중심 보고서”로 제출하는 방향으로 유지하기
5. 변경 프로젝트가 확정되면 새 프로젝트 폴더 구조를 만들고, 구별 월별 통합 데이터 생성 파이프라인부터 작성하기

추가 성능 개선을 계속 한다면 다음 후보는 **BalancedBagging 하이퍼파라미터 튜닝**입니다. 다만 기대 상승폭은 크지 않을 가능성이 높으므로, 발표/보고서 정리가 더 효율적인 다음 단계입니다.

## 현재 결론

현재 최종 메인 모델은 `without_billing_zip + LogisticRegression_SMOTE`입니다.

| 기준 | Variant | Model | F1 | Recall | Precision | 비고 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 최종 F1 기준 | `without_billing_zip` | `LogisticRegression_SMOTE` | 0.1681 | 0.2661 | 0.1229 | 최종 보고서의 메인 모델로 적합 |
| Recall 운영 기준 | `with_billing_zip` | `BalancedBagging_original` | 0.1526 | 0.5872 | 0.0877 | 이탈 고객을 더 많이 잡는 캠페인용 후보 |
| Recall 극대화 기준 | `with_billing_zip` | `CatBoost_native_categorical`, threshold 0.35 | 0.1310 | 0.8349 | 0.0711 | 너무 많은 false positive를 감수할 때만 사용 |

주석:

- F1만 보면 LogisticRegression이 가장 좋습니다.
- Recall을 우선하면 BalancedBagging이나 CatBoost가 더 많은 이탈 고객을 잡습니다.
- Precision이 전반적으로 낮은 이유는 churn 비율이 약 6.5%로 매우 불균형하기 때문입니다.
- 현재 데이터에는 고객 행동 이력, 월별 사용량 변화, 결제 실패, 불만/문의 기록 같은 시간 기반 신호가 없어서 성능 상한이 있습니다.

## 데이터 상태

원본 데이터:

- 파일: `Baza customer Telecom v2.csv`
- 원본 shape: 8,453 rows x 14 columns
- 중복 PID 제거 후: 8,436 rows
- target: `CHURN`
- 이탈 고객 수: 549
- 비이탈 고객 수: 7,904
- churn 비율: 약 6.5%

주석:

- churn 고객이 매우 적기 때문에 accuracy만 보면 안 됩니다.
- 예를 들어 모두 “비이탈”이라고 예측해도 accuracy는 높게 나올 수 있습니다.
- 그래서 이 프로젝트에서는 F1, Recall, Precision, ROC-AUC, PR-AUC, MCC를 같이 봅니다.

주요 결측:

| Column | 결측률 | 처리 방식 |
| --- | ---: | --- |
| `Suspended_subscribers` | 약 95.84% | 결측 여부 flag 생성 후 0으로 대체 |
| `Not_Active_subscribers` | 약 49.08% | 결측 여부 flag 생성 후 0으로 대체 |
| `CRM_PID_Value_Segment` | 약 0.06% | `Unknown` 처리 |
| `Billing_ZIP` | 약 0.02% | variant에 따라 사용 또는 제거 |
| `ARPU` | 약 0.01% | `TotalRevenue / Total_SUBs`로 보완 후 train median 사용 |

주석:

- 결측 자체가 정보일 수 있어서 `*_missing` flag를 만들었습니다.
- imputation과 scaling은 train split에만 fit했습니다.
- test 정보가 train으로 새는 leakage를 막기 위한 처리입니다.

## 전처리 파이프라인

주요 파일:

- `preprocess_churn.py`
- `requirements.txt`
- `processed/summary_all.json`
- `processed/model_comparison_billing_zip.csv`

전처리에서 한 일:

1. `PID`, `KA_name` 제거
2. `CHURN`을 `No=0`, `Yes=1`로 변환
3. 중복 PID 제거
4. 결측 flag 생성
5. train/test split
6. train 기준 imputation
7. feature engineering
8. categorical encoding
9. numeric scaling
10. SVMSMOTE로 train set resampling
11. 모델 비교표 생성
12. threshold 튜닝 결과 생성
13. 오류 분석과 feature importance 생성

주석:

- `Billing_ZIP`은 포함 버전과 제외 버전을 둘 다 만들었습니다.
- `with_billing_zip`은 지역 정보를 살리는 실험입니다.
- `without_billing_zip`은 지역 정보가 noise일 가능성을 줄이는 실험입니다.
- 현재 최종 F1은 `without_billing_zip`이 가장 좋습니다.

## 생성된 주요 산출물

| 파일 | 의미 |
| --- | --- |
| `processed/model_comparison_billing_zip.csv` | 모든 모델의 기본 threshold 성능 비교표 |
| `processed/threshold_tuning_best.csv` | validation에서 고른 threshold를 test에 적용한 결과 |
| `processed/threshold_tuning_sweep.csv` | threshold 0.05부터 0.50까지 전체 탐색 결과 |
| `processed/error_analysis_predictions.csv` | test row별 actual, predicted, score, TP/FP/FN/TN |
| `processed/error_analysis_group_summary.csv` | TP/FP/FN/TN 그룹별 평균 feature |
| `processed/feature_importance.csv` | permutation importance 전체 결과 |
| `processed/feature_importance_top.csv` | 운영 기준별 상위 중요 feature |
| `processed/summary_all.json` | 전체 전처리, 모델, threshold, 분석 요약 |

주석:

- 보고서에는 `model_comparison_billing_zip.csv`, `threshold_tuning_best.csv`, `feature_importance_top.csv`를 가장 많이 쓰면 됩니다.
- `error_analysis_predictions.csv`는 행이 많아서 부록이나 세부 분석용입니다.

## 모델 실험 기록

### 1. LogisticRegression + SMOTE

목적:

- 단순하지만 설명 가능한 baseline 모델을 만들기 위해 사용했습니다.
- 불균형 데이터이므로 SVMSMOTE로 minority class를 보강했습니다.

결과:

- Variant: `without_billing_zip`
- F1: 0.1681
- Recall: 0.2661
- Precision: 0.1229
- ROC-AUC: 0.5746
- PR-AUC: 0.0879
- MCC: 0.0956

주석:

- 현재 F1 기준 최종 1등입니다.
- Recall은 낮지만 precision이 다른 recall-heavy 모델보다 낫습니다.
- 보고서의 최종 메인 모델로 가장 적합합니다.

### 2. CatBoost original balanced

목적:

- tabular 데이터에 강한 gradient boosting 계열 모델을 비교하기 위해 추가했습니다.
- `auto_class_weights="Balanced"`로 불균형을 처리했습니다.

결과:

- Best row: `with_billing_zip + CatBoost_original_balanced`
- F1: 0.1353
- Recall: 0.3761
- Precision: 0.0825

주석:

- LogisticRegression보다 recall은 높지만 F1은 낮습니다.
- Precision이 낮아 최종 메인 모델로는 부족합니다.

### 3. 논문 방식 모델 추가

추가한 모델:

- `EasyEnsemble_original`
- `RUSBoost_original`
- `BalancedBagging_original`

목적:

- 참고 논문에서 사용한 imbalance-aware ensemble 모델들과 비교하기 위해 추가했습니다.
- 논문 표와 맞추기 위해 `balanced_accuracy`, `pr_auc`, `mcc`도 추가했습니다.

결과:

- `with_billing_zip + BalancedBagging_original`
- F1: 0.1526
- Recall: 0.5872
- Precision: 0.0877

주석:

- BalancedBagging은 이탈 고객을 많이 잡습니다.
- 그러나 false positive도 많아 precision이 낮습니다.
- retention campaign처럼 “놓치는 것보다 많이 잡는 것”이 중요할 때 후보가 됩니다.

### 4. Threshold 튜닝

목적:

- 기본 threshold 0.5만 쓰면 불균형 문제에서 recall/F1이 제한될 수 있어서 threshold를 따로 탐색했습니다.

방법:

- train 안에서 validation split을 따로 만듦
- threshold 0.05부터 0.50까지 0.01 단위 탐색
- validation F1이 가장 좋은 threshold 선택
- 선택된 threshold를 test에 한 번만 적용

주석:

- test에서 threshold를 고르면 성능이 과장될 수 있습니다.
- 그래서 validation에서 고르고 test는 최종 확인용으로만 사용했습니다.

결과:

| Model | Variant | Threshold | Test F1 | Test Recall | Test Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| `BalancedBagging_original` | `with_billing_zip` | 0.50 | 0.1526 | 0.5872 | 0.0877 |
| `LogisticRegression_SMOTE` | `with_billing_zip` | 0.46 | 0.1507 | 0.3028 | 0.1003 |
| `CatBoost_native_categorical` | `with_billing_zip` | 0.35 | 0.1310 | 0.8349 | 0.0711 |

주석:

- threshold 튜닝으로 최종 F1 1등은 바뀌지 않았습니다.
- 대신 recall 중심 운영점을 만들 수 있었습니다.

### 5. 오류 분석과 feature importance

목적:

- 모델이 어떤 고객을 맞추고 어떤 고객을 놓치는지 보기 위해 수행했습니다.
- 단순히 점수만 보는 것이 아니라, FN과 FP의 특징을 확인하기 위한 작업입니다.

분석 기준:

- `main_f1_baseline`: `without_billing_zip + LogisticRegression_SMOTE`
- `tuned_best_f1`: `with_billing_zip + BalancedBagging_original`
- `recall_heavy`: `with_billing_zip + CatBoost_original_balanced`
- `native_catboost`: `with_billing_zip + CatBoost_native_categorical`

중요 feature 요약:

| 운영 기준 | 중요한 feature |
| --- | --- |
| LogisticRegression | `AvgMobileRevenue_sqrt`, `TotalRevenue_sqrt`, `revenue_engagement_interaction`, `revenue_per_subscriber` |
| BalancedBagging | `Billing_ZIP`, `revenue_engagement_interaction`, `arpu_risk_interaction`, `ARPU_sqrt` |
| CatBoost original | `Billing_ZIP`, `revenue_per_active_subscriber`, `arpu_risk_interaction`, `Total_SUBs` |
| CatBoost native | `arpu_risk_interaction`, `revenue_engagement_interaction`, `inactive_revenue_interaction`, `ARPU_sqrt` |

주석:

- 수익 규모와 가입자 활동 상태가 핵심 신호입니다.
- `Billing_ZIP`은 일부 ensemble 모델에서 중요하지만, 전체 F1 최종 모델에서는 제외한 버전이 더 좋았습니다.
- 이는 지역 정보가 도움이 되는 경우와 noise가 되는 경우가 섞여 있다는 뜻입니다.

### 6. FN 타깃 feature 실험

목적:

- 오류 분석에서 “놓친 churn 고객은 dormant_rate가 높은데 매출이 낮은 경향”이 보여서, 이를 잡기 위한 feature를 추가했습니다.

추가한 feature:

- `low_revenue_high_dormant`
- `small_account_dormant_risk`
- `inactive_to_revenue_ratio`
- `dormant_revenue_pressure`
- `inactive_low_revenue_pressure`
- `suspended_low_revenue_pressure`
- `single_or_small_account_risk`

결과:

- Best F1이 0.1681에서 0.1503으로 하락했습니다.

주석:

- 기대와 달리 성능이 떨어졌습니다.
- 새 feature들이 분리 가능한 신호라기보다 기존 feature와 강하게 겹치는 noise였을 가능성이 큽니다.
- 그래서 기본값에서는 비활성화했습니다.
- 필요하면 아래 명령으로 다시 실험할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe preprocess_churn.py --enable-fn-target-features
```

### 7. CatBoost native categorical 실험

목적:

- CatBoost는 원래 categorical feature를 직접 처리할 때 강하기 때문에, label encoding과 scaling을 거친 입력 대신 문자열 categorical을 그대로 넘기는 실험을 했습니다.

사용한 categorical feature:

- `CRM_PID_Value_Segment`
- `EffectiveSegment`
- `Billing_ZIP`

결과:

| Model | Variant | F1 | Recall | Precision |
| --- | --- | ---: | ---: | ---: |
| `CatBoost_native_categorical` | `without_billing_zip` | 0.1416 | 0.4404 | 0.0844 |
| `CatBoost_original_balanced` | `without_billing_zip` | 0.1288 | 0.4037 | 0.0767 |
| `CatBoost_native_categorical` tuned | `with_billing_zip` | 0.1310 | 0.8349 | 0.0711 |

주석:

- native categorical 방식은 기존 CatBoost보다 좋아졌습니다.
- 하지만 최종 1등인 LogisticRegression F1 0.1681은 넘지 못했습니다.
- recall-heavy 목적에는 쓸 수 있지만 precision이 낮습니다.

## 참고 논문과 비교

논문 PDF: `j.ajnc.20261501.12.pdf`

논문에서 사용한 주요 모델:

- EasyEnsemble
- RUSBoost
- BalancedBagging
- XGBoost
- LightGBM
- CatBoost
- HistGradientBoosting
- MLP
- VotingClassifier
- StackingClassifier

논문에서 성능을 올린 방법:

1. SVMSMOTE로 class imbalance 처리
2. subscriber ratio, revenue interaction, log transform 등 feature engineering
3. accuracy보다 F1, recall, PR-AUC 중심 평가
4. EasyEnsemble을 최종 모델로 선택

논문 주요 결과:

| Model | Acc | Bal Acc | Prec | Recall | F1 | ROC | PR | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EasyEnsemble | 0.664 | 0.533 | 0.077 | 0.382 | 0.129 | 0.551 | 0.079 | 0.034 |
| RUSBoost | 0.803 | 0.527 | 0.086 | 0.209 | 0.121 | 0.588 | 0.084 | 0.036 |
| BalancedBagging | 0.902 | 0.500 | 0.064 | 0.036 | 0.046 | 0.576 | 0.077 | -0.001 |

주석:

- 우리 프로젝트의 LogisticRegression F1 0.1681은 논문 EasyEnsemble F1 0.129보다 높습니다.
- 단, 데이터 split, 전처리, feature 구성 차이가 있으므로 직접적인 우열 비교보다는 참고 비교로 쓰는 것이 안전합니다.
- 논문처럼 recall 중심 운영 모델을 따로 제시하는 방식이 보고서에 좋습니다.

## 현재 한계

현재 데이터는 정적인 CRM snapshot에 가깝습니다.

없는 정보:

- 월별 사용량 변화
- 최근 결제 실패 여부
- 요금제 변경 이력
- 고객센터 문의/불만 이력
- 계약 만료까지 남은 기간
- 최근 n개월 매출 추세
- 실제 이탈 직전 행동 로그

주석:

- 이런 time-based feature가 없으면 churn 직전 신호를 잡기 어렵습니다.
- 그래서 모델만 더 바꿔도 F1이 크게 오르기는 어렵습니다.
- 보고서에서는 이 점을 “데이터 한계와 향후 개선 방향”으로 쓰면 좋습니다.

## 최종 추천 문장

보고서나 발표에는 아래처럼 정리하면 좋습니다.

> 본 프로젝트에서는 통신사 B2B 고객 이탈 예측을 위해 불균형 데이터 전처리, SVMSMOTE, 다양한 ensemble 모델, CatBoost native categorical 처리, threshold tuning, 오류 분석을 수행하였다. 실험 결과 F1 기준 최종 모델은 Billing_ZIP을 제외한 `LogisticRegression_SMOTE`이며, recall 중심 운영 시나리오에서는 `BalancedBagging_original` 또는 `CatBoost_native_categorical`이 더 많은 이탈 고객을 탐지할 수 있었다. 다만 전체 precision이 낮고 F1 상승폭이 제한적인 것은 데이터가 정적인 CRM snapshot이며 시간 기반 행동 이력이 부족하기 때문으로 해석된다.

## 2026-05-22 최종 보고서 정리

요청 내용:

- 지금까지의 모델 실험, 논문 비교, threshold 튜닝, 오류 분석, feature importance 결과를 보고서 형태로 정리했다.

생성한 파일:

- `FINAL_REPORT.md`
- `final_model_summary.csv`

`FINAL_REPORT.md`에 정리한 내용:

- 프로젝트 개요
- 데이터 요약
- 전처리 및 feature engineering
- 모델 실험 목록
- 최종 모델 비교
- threshold tuning 결과
- feature importance 해석
- 오류 분석
- 참고 논문과 비교
- 최종 결론
- 한계와 향후 개선 방향
- 보고서에 바로 넣을 수 있는 최종 문장

`final_model_summary.csv`에 정리한 내용:

- F1 기준 최종 모델: `without_billing_zip + LogisticRegression_SMOTE`
- recall 중심 운영 모델: `with_billing_zip + BalancedBagging_original`
- recall 극대화 모델: `with_billing_zip + CatBoost_native_categorical`
- 논문 기준 참고 모델: `EasyEnsemble`

주석:

- 이 단계부터는 모델 성능을 더 올리는 작업보다 결과를 설득력 있게 전달하는 작업이 더 중요하다.
- 다음 작업은 발표 자료용 슬라이드 구성 또는 결과 시각화가 적합하다.

## 2026-05-22 발표 자료용 시각화와 슬라이드 구성

요청 내용:

- 최종 보고서와 모델 결과를 바탕으로 발표에 바로 사용할 수 있는 슬라이드 구성과 그래프를 정리했다.

생성한 파일:

- `make_presentation_assets.py`
- `PRESENTATION_SLIDES.md`
- `presentation_assets/01_model_metric_comparison.png`
- `presentation_assets/02_confusion_counts.png`
- `presentation_assets/03_precision_recall_tradeoff.png`
- `presentation_assets/04_feature_importance_main.png`
- `presentation_assets/05_paper_comparison.png`

`make_presentation_assets.py`에서 한 일:

- `final_model_summary.csv`를 읽어서 최종 모델, recall 중심 모델, CatBoost recall 극대화 모델, 논문 EasyEnsemble 기준을 한눈에 비교하는 그래프를 만들었다.
- confusion count를 이용해 각 모델이 실제 이탈 고객을 얼마나 잡았는지와 false positive가 얼마나 늘어나는지 시각화했다.
- precision과 recall의 trade-off를 그래프로 표현해서 왜 F1 기준 모델과 recall 기준 모델을 나눠 설명해야 하는지 보여주었다.
- `processed/feature_importance_top.csv`를 이용해 최종 모델의 주요 변수를 막대그래프로 만들었다.
- 논문 표의 EasyEnsemble 결과와 우리 최종 모델 결과를 비교하는 그래프를 만들었다.

`PRESENTATION_SLIDES.md`에서 정리한 내용:

- 총 8장 기준의 발표 흐름을 만들었다.
- 각 슬라이드마다 넣을 핵심 메시지, 사용할 표나 그래프, 발표 멘트를 같이 적었다.
- 논문 비교, 최종 모델 선택 이유, recall 중심 운영 모델, 데이터 한계, 향후 개선 방향을 발표자가 설명하기 쉽게 정리했다.

검증:

```powershell
.\.venv\Scripts\python.exe -m py_compile make_presentation_assets.py
.\.venv\Scripts\python.exe make_presentation_assets.py
```

주석:

- 발표 자료를 새로 만들 때는 `PRESENTATION_SLIDES.md`를 뼈대로 삼고, `presentation_assets/`의 PNG 파일을 슬라이드에 넣으면 된다.
- 실제 PPT 파일이 필요하면 다음 단계에서 이 Markdown 구성안을 PowerPoint 형식으로 옮기면 된다.

## 2026-05-22 프로젝트 변경 제안서 작성

요청 내용:

- 현재 프로젝트의 F1과 recall이 너무 낮아 다른 프로젝트로 변경하고 싶으므로, 교수님을 설득하기 위한 문서를 추가했다.

생성한 파일:

- `PROJECT_CHANGE_PROPOSAL.md`

`PROJECT_CHANGE_PROPOSAL.md`에 정리한 내용:

- 현재 통신사 이탈 예측 프로젝트의 데이터 요약
- 지금까지 수행한 전처리, sampling, 모델 비교, threshold tuning 작업
- 최종 F1 기준 모델과 recall 중심 모델의 성능표
- 왜 현재 프로젝트의 성능이 낮은지에 대한 데이터 구조적 원인
- 추가 모델 실험보다 프로젝트 변경이 더 합리적인 이유
- 대체 프로젝트 후보
  - 온라인 쇼핑 구매 의도 예측
  - 은행 마케팅 성공 예측
  - 온라인 리테일 재구매 또는 고객 이탈 예측
- 교수님께 바로 설명할 수 있는 긴 문장과 짧은 문장
- 변경 승인 후 진행 계획

주석:

- 핵심 설득 논리는 “성능이 낮아서 포기한다”가 아니라 “현재 데이터에는 이탈 예측에 필요한 행동 기반 feature가 부족해서, 같은 데이터에서 모델만 더 바꾸는 것은 학습 효과가 낮다”는 것이다.
- 가장 추천한 변경 후보는 `UCI Online Shoppers Purchasing Intention Dataset` 기반의 온라인 쇼핑 구매 의도 예측 프로젝트다.

## 2026-05-22 팀 프로젝트 전환 보고서 작성

요청 내용:

- 이 프로젝트가 2인 팀 프로젝트이며, 한 명은 현재 통신사 이탈 예측 프로젝트를 마무리하고 한계 분석을 정리하고, 다른 한 명은 서울 아파트 버블 탐지 프로젝트의 데이터 수집 계획을 조사하고 있다는 상황을 반영해 교수님 설득용 보고서를 작성했다.

생성한 파일:

- `TEAM_PROJECT_SWITCH_REPORT.md`

`TEAM_PROJECT_SWITCH_REPORT.md`에 정리한 내용:

- 2인 팀 프로젝트 역할 구조
- 기존 통신사 이탈 예측 프로젝트에서 수행한 전처리와 모델 실험
- F1, recall, precision 성능이 낮게 나온 이유
- 기존 프로젝트를 단순 포기가 아니라 한계 분석까지 마무리했다는 점
- 새 프로젝트인 서울 아파트 버블 탐지 및 위험 예측의 목표
- 다른 팀원이 조사한 데이터 수집 계획
  - 국토부 실거래가
  - 전세 실거래가
  - 한국은행 금리, 대출, M2, BSI
  - KOSIS 소득, CPI, 인구
  - 미분양, 청약 경쟁률
  - BigKinds 뉴스
  - 네이버 검색량
- 새 프로젝트의 target 정의 후보
- target leakage 방지, 시간 기준 train/test split, non-stationarity 대응 방안
- 교수님께 직접 말할 수 있는 긴 설명문과 짧은 설명문

주석:

- 교수님께 제출하거나 설명할 때는 이전의 범용 `PROJECT_CHANGE_PROPOSAL.md`보다 `TEAM_PROJECT_SWITCH_REPORT.md`가 더 적합하다.
- 이제 최종 변경 후보는 온라인 쇼핑 구매 의도 예측이 아니라 **서울 아파트 버블 탐지 및 위험 예측**이다.
- 설득 논리에 “성능 개선에 필요한 시간 기반 고객 행동 데이터를 추가로 확보하기 어려웠기 때문에 기존 churn 프로젝트는 한계 분석으로 마무리하고 전환한다”는 내용을 추가했다.

## 2026-05-22 Churn 데이터 및 모델 선택 방어 문서 작성

요청 내용:

- 프로젝트를 바꾸기 전에 교수님이 기존 churn 데이터와 모델 선택 이유를 질문할 수 있으므로, 데이터 이해도와 모델 사용/미사용 이유를 알기 쉽게 정리한 문서를 만들었다.

생성한 파일:

- `CHURN_DATA_MODEL_DEFENSE.md`

`CHURN_DATA_MODEL_DEFENSE.md`에 정리한 내용:

- `Baza customer Telecom v2.csv` 데이터 개요
- 원본 14개 컬럼의 의미와 처리 방식
- target `CHURN`의 불균형 문제
- 범주형 변수 분포
- 결측치 처리 이유
- train/test split 방식
- `Billing_ZIP` 포함/제외 실험 이유
- feature engineering 목록
- 사용한 모델별 선택 이유
- 최종 모델로 선택하지 않은 모델들의 제외 이유
- XGBoost, LightGBM, SVM, KNN, Naive Bayes, MLP, Voting, Stacking을 우선순위에서 낮춘 이유
- 교수님 예상 질문과 답변
- 최종 설명용 요약 문장

주석:

- 교수님이 “기존 churn 데이터에 대해 충분히 이해했는가?”라고 질문하면 `CHURN_DATA_MODEL_DEFENSE.md`를 기준으로 답하면 된다.
- 핵심 답변은 “여러 모델을 실험했지만, 이 데이터는 정적인 CRM snapshot이라 이탈 예측에 중요한 시간 기반 행동 feature가 부족하다”는 것이다.
- 추가 핵심 답변은 “그 시간 기반 고객 행동 데이터를 따로 찾으려 했지만 확보하기 어려워서, 같은 데이터에서 모델만 더 바꾸는 방식은 한계가 있다”는 것이다.

## 검증 명령

마지막으로 통과한 검증:

```powershell
.\.venv\Scripts\python.exe -m py_compile preprocess_churn.py
.\.venv\Scripts\python.exe preprocess_churn.py
.\.venv\Scripts\python.exe -m py_compile make_presentation_assets.py
.\.venv\Scripts\python.exe make_presentation_assets.py
```

주석:

- 첫 번째 명령은 Python 문법 오류가 없는지 확인합니다.
- 두 번째 명령은 전처리, 모델 비교, threshold 튜닝, 오류 분석, feature importance까지 전체 파이프라인을 다시 실행합니다.
- 세 번째 명령은 발표 자료 생성 스크립트의 Python 문법 오류가 없는지 확인합니다.
- 네 번째 명령은 발표용 PNG 그래프 5개를 다시 생성합니다.

## 앞으로 작업할 때 기록 규칙

앞으로 작업을 추가할 때는 이 문서에 아래 내용을 적습니다.

- 어떤 요청을 처리했는지
- 어떤 파일을 수정했는지
- 어떤 산출물이 갱신됐는지
- 어떤 검증을 실행했는지
- 성능 수치가 좋아졌는지 나빠졌는지
- 좋아지지 않았다면 왜 기본값에서 제외했는지
- 다음 사람이 이어서 볼 때 어떤 판단을 하면 되는지
