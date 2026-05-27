# ChurnRadar 프로젝트 진행 기록

마지막 업데이트: 2026-05-26

이 문서는 `Baza customer Telecom v2.csv`를 사용한 통신사 고객 이탈 예측 프로젝트에서 지금까지 무엇을 했는지, 왜 했는지, 결과가 어땠는지, 다음에 무엇을 하면 좋은지를 한글 주석 형태로 정리한 기록입니다.

## 다음에 진행할 것

현재 결정은 **프로젝트 변경이 아니라 ChurnRadar를 계속 진행하는 것**입니다. 따라서 다음 작업은 새 주제를 찾는 것이 아니라, 이미 만든 전처리/모델링/분석 결과를 최종 보고서와 발표 자료로 완성하는 것입니다.

추천 순서:

1. `FINAL_REPORT.md`를 최종 제출용 문장으로 다듬기
2. `PRESENTATION_SLIDES.md`를 기반으로 실제 PPT 또는 발표용 PDF 만들기
3. `presentation_assets/`의 PNG 5개를 슬라이드에 넣기
4. `CHURN_DATA_MODEL_DEFENSE.md`로 교수님 예상 질문 답변 준비하기
5. `README.md`, `PROJECT_CHANGE_PROPOSAL.md`, `TEAM_PROJECT_SWITCH_REPORT.md`의 현재 유지 결정 내용을 확인하기
6. 최종 제출 전 전처리/모델링과 발표 이미지 생성 명령을 다시 실행해 결과 재현 확인하기

추가 성능 개선을 계속 한다면 다음 후보는 **BalancedBagging 하이퍼파라미터 튜닝**입니다. 다만 기대 상승폭은 크지 않을 가능성이 높으므로, 새 프로젝트 탐색보다 보고서/발표 완성의 우선순위가 더 높습니다.

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

## 2026-05-22 프로젝트 변경 검토 문서 작성

요청 내용:

- 당시에는 현재 프로젝트의 F1과 recall이 낮아 다른 프로젝트로 바꿀 수 있는지 검토하기 위해 문서를 작성했다.

생성한 파일:

- `PROJECT_CHANGE_PROPOSAL.md`

현재 상태:

- 2026-05-26 기준으로 이 문서는 **프로젝트 유지 결정 메모**로 갱신했다.
- 이제 이 파일은 변경을 설득하는 문서가 아니라, ChurnRadar를 계속 진행할 때 어떤 논리로 마무리할지 정리하는 문서다.

핵심 정리:

- 성능이 낮은 이유는 실험 부족이 아니라 심한 target 불균형과 시간 기반 행동 feature 부족이다.
- 하지만 이 한계는 프로젝트 변경 사유가 아니라 최종 보고서에서 설명할 분석 포인트로 사용한다.
- 최종 제출 전략은 `LogisticRegression_SMOTE`를 F1 기준 모델로 두고, `BalancedBagging_original`을 recall 중심 운영 후보로 함께 제시하는 것이다.

## 2026-05-22 팀 프로젝트 방향 검토 문서 작성

요청 내용:

- 당시에는 2인 팀 프로젝트 상황을 반영해 다른 방향 가능성을 설명하는 보고서를 작성했다.

생성한 파일:

- `TEAM_PROJECT_SWITCH_REPORT.md`

현재 상태:

- 2026-05-26 기준으로 이 문서는 **팀 프로젝트 유지 보고서**로 갱신했다.
- 새 프로젝트 데이터 수집 계획이 아니라, 현재 ChurnRadar를 최종 제출까지 어떻게 나눠서 마무리할지 정리한다.

핵심 정리:

- 팀 작업의 우선순위는 새 주제 탐색이 아니라 보고서, 발표 자료, Q&A 준비다.
- 팀원 A는 모델링 수치 검증과 방어 자료를 맡고, 팀원 B는 발표 슬라이드와 시각 자료 정리를 맡는 방식이 적절하다.
- 발표에서는 class imbalance, threshold trade-off, 데이터 feature 한계를 핵심 메시지로 설명한다.

## 2026-05-22 Churn 데이터 및 모델 선택 방어 문서 작성

요청 내용:

- 교수님이 churn 데이터와 모델 선택 이유를 질문할 수 있으므로, 데이터 이해도와 모델 사용/미사용 이유를 알기 쉽게 정리한 문서를 만들었다.

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
- 추가 핵심 답변은 “현재 접근 가능한 데이터만으로는 시간 기반 행동 feature를 추가하기 어렵기 때문에, 최종 보고서에서는 모델별 trade-off와 데이터 한계를 명확히 설명한다”는 것이다.

## 2026-05-26 프로젝트 유지 결정 및 문서 갱신

요청 내용:

- 프로젝트 변경이 아니라 다시 ChurnRadar로 진행하기로 했으므로, 현재 프로젝트를 계속할 때 해야 할 일을 정리하고 Markdown 문서를 갱신했다.

수정한 파일:

- `README.md`
- `PROJECT_CHANGE_PROPOSAL.md`
- `TEAM_PROJECT_SWITCH_REPORT.md`
- `PROJECT_PROGRESS.md`
- `CHURN_DATA_MODEL_DEFENSE.md`
- `FINAL_REPORT.md`
- `PRESENTATION_SLIDES.md`

정리한 방향:

- 최종 주제는 B2B 통신사 고객 이탈 예측으로 유지한다.
- `LogisticRegression_SMOTE`를 F1 기준 최종 모델로 둔다.
- `BalancedBagging_original`과 `CatBoost_native_categorical`은 recall 중심 운영 후보로 설명한다.
- 낮은 성능은 프로젝트 실패가 아니라 class imbalance와 정적 CRM snapshot의 한계로 해석한다.
- 앞으로 할 일은 새 프로젝트 탐색이 아니라 보고서 마감, 발표 자료 제작, 교수님 예상 질문 대비, 결과 재현 확인이다.

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

## 2026-05-27 추가 모델 실험 및 운영 해석 정리

요청 내용:

- 추가 모델 실험을 최대한 진행하고, 현재 결과가 왜 이렇게 나왔는지와 실제 운영에서 어떻게 활용할 수 있는지 정리한다.

생성/수정 파일:

- `additional_model_experiments.py`
- `ADDITIONAL_EXPERIMENTS_AND_OPERATION_SUMMARY.md`
- `processed/additional_experiments/additional_model_results.csv`
- `processed/additional_experiments/additional_top25_summary.csv`
- `processed/additional_experiments/additional_threshold_sweep.csv`
- `processed/additional_experiments/operating_budget_topk.csv`

실험 내용:

- Logistic Regression C값 조정
- class-weight 기반 Logistic Regression
- RidgeClassifier balanced
- LinearSVC balanced
- RandomForest balanced
- ExtraTrees balanced
- GradientBoosting + SMOTE
- HistGradientBoosting balanced
- EasyEnsemble
- RUSBoost
- BalancedBagging 하이퍼파라미터 조합
- CatBoost encoded/native categorical 소규모 튜닝
- validation 기준 threshold 선택
- top-k 캠페인 예산 기준 운영 성능 확인

결과:

| 기준 | Variant | Model | Threshold | F1 | Recall | Precision |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 기존 최종 메인 | `without_billing_zip` | `LogisticRegression_SMOTE` | 0.50 | 0.1681 | 0.2661 | 0.1229 |
| 추가 실험 best | `with_billing_zip` | `BalancedBagging_tree_depthnone_leaf25` | 0.51 | 0.1605 | 0.5138 | 0.0951 |
| 기존 recall 후보 | `with_billing_zip` | `BalancedBagging_original` | 0.50 | 0.1526 | 0.5872 | 0.0877 |

해석:

- 추가 실험에서도 F1 기준 최종 1위는 기존 `LogisticRegression_SMOTE`가 유지된다.
- tuned BalancedBagging은 기존 recall 후보보다 F1과 precision이 개선되어 recall 중심 운영 후보로 더 설득력 있게 제시할 수 있다.
- 모델을 더 많이 바꿔도 성능 상한이 크게 움직이지 않는 이유는 class imbalance와 정적 CRM snapshot의 한계로 해석한다.
- 실제 운영에서는 score 기준 상위 10%, 20%, 30% 고객을 캠페인 대상으로 삼는 top-k 방식이 적절하다.

검증:

```powershell
.\.venv\Scripts\python.exe -m py_compile additional_model_experiments.py
.\.venv\Scripts\python.exe -u additional_model_experiments.py
```

다음 작업:

- `FINAL_REPORT.md`와 `PRESENTATION_SLIDES.md`에 추가 실험 best와 운영 해석을 반영한다.
- 다음 사람이 이어서 볼 때 어떤 판단을 하면 되는지

## 2026-05-27 Phase 3A-5B 통합 및 제출물 갱신

목표:

- 논문 재현 결과, 차별화 실험, CV 안정성, 해석 가능성, 비즈니스 임팩트를 최종 제출물에 반영한다.
- 기존 문서의 오래된 “다음 작업”을 최신 상태에 맞게 정리한다.

완료한 작업:

- `FINAL_REPORT.md`를 최신 Phase 3A-5B 결과 기준으로 다시 작성했다.
- `PRESENTATION_SLIDES.md`를 실제 PPT 제작용 11장 흐름으로 다시 구성했다.
- `README.md`를 최신 모델/문서/산출물 안내 기준으로 갱신했다.
- `final_model_summary.csv`에 LR, BalancedBagging, CatBoost, XGBoost, 비용 최적 운영점, 논문 기준, 논문 재현 행을 통합했다.

현재 핵심 결론:

| 목적 | 추천 운영점 | 핵심 수치 |
| --- | --- | --- |
| 논문 재현 | `EasyEnsemble_original` | F1 0.1284로 논문 0.129 재현 |
| hold-out F1 최고 | `LogisticRegression_SMOTE` | F1 0.1681 |
| CV 안정성 | `BalancedBagging_original` | 5-fold F1 0.1455 ± 0.0126 |
| 핵심 3모델 recall-heavy | `CatBoost_native_categorical` | Recall 0.8349, 순이익 152,160 |
| 확장 recall-heavy | `XGBoost_SMOTE` | Recall 0.9266, 순이익 159,000 |
| 비용 최적 threshold | `BalancedBagging_original`, threshold 0.29 | 순이익 165,840 |

주의해서 써야 할 주장:

- `LR F1=0.1681`과 `CatBoost recall=0.8349`는 같은 모델의 성능이 아니다.
- `LR F1=0.1681`은 hold-out 기준이며, CV 평균은 0.1309로 낮아진다.
- 논문 대비 주장은 “압도적 성능 우위”가 아니라 “논문 baseline 재현 후 운영 목적별 모델 선택 프레임워크 제시”가 가장 안전하다.
- 비용 최적 BalancedBagging threshold 0.29는 수치상 순이익이 가장 높지만, 1,688명 중 1,670명 접촉이므로 실제 운영에서는 고객 피로도와 팀 역량을 함께 고려해야 한다.

남은 작업:

- 실제 PPT 파일 제작
- `processed/phase_5b_business_impact/business_impact_dashboard.png`를 핵심 슬라이드에 삽입
- 발표 전 `CHURN_DATA_MODEL_DEFENSE.md`, `PHASE_4_PAPER_COMPARISON_FRAMEWORK.md`, `PHASE_5B_BUSINESS_IMPACT_ANALYSIS.md`의 예상 질문 문장 확인

## 2026-05-27 Phase 6 추가 비교 실험

목표:

- 1시간 발표에 사용할 수 있도록 단일 모델 성능표를 넘어서는 비교 케이스를 추가한다.
- 운영 예산, 비용 구조, 확률 보정, 고객 segment, 모델 합의도 관점에서 발표 소재를 확장한다.

실행:

```powershell
.\.venv\Scripts\python.exe phase_6_extended_case_studies.py
```

생성 문서와 산출물:

- `PHASE_6_EXTENDED_CASE_STUDIES.md`
- `processed/phase_6_extended_case_studies/phase6_model_operating_metrics.csv`
- `processed/phase_6_extended_case_studies/phase6_topk_budget_curve.csv`
- `processed/phase_6_extended_case_studies/phase6_cost_threshold_best_by_model.csv`
- `processed/phase_6_extended_case_studies/phase6_calibration_metrics.csv`
- `processed/phase_6_extended_case_studies/phase6_segment_operating_metrics.csv`
- `processed/phase_6_extended_case_studies/phase6_model_agreement_vote_groups.csv`
- `processed/phase_6_extended_case_studies/phase6_topk_budget_curves.png`
- `processed/phase_6_extended_case_studies/phase6_calibration_comparison.png`
- `processed/phase_6_extended_case_studies/phase6_model_agreement.png`

추가 실험 핵심 결과:

| 실험 | 핵심 결과 |
| --- | --- |
| 운영점 8개 비교 | XGBoost recall 0.9266, CatBoost native recall 0.8349, LR F1 0.1681 |
| Top-k 예산 | top 10%에서는 LR, top 40%에서는 BalancedBagging이 유리 |
| 비용 threshold | 논문 비용 기준은 낮은 threshold와 recall 극대화가 유리 |
| 보수적 캠페인 비용 | FP cost가 커지면 LR threshold 0.53이 최선 |
| Calibration | raw score는 churn 확률을 과대평가, Platt 보정 후 평균 score가 실제 churn rate와 일치 |
| Segment ROI | low/mid/high value별로 강한 모델이 다름 |
| 모델 합의도 | 8개 모델 모두가 경고한 고객군의 이탈률은 12.43%로 전체 평균의 약 1.9배 |

발표 활용:

- 기존 11장 슬라이드 뒤에 Slide 12-16으로 추가 케이스를 붙이면 1시간 발표 분량을 만들 수 있다.
- `PRESENTATION_SLIDES.md`에 Phase 6 케이스를 이미 반영했다.
