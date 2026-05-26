# 팀 프로젝트 유지 보고서

> 업데이트: 2026-05-26
> 이전에는 팀 프로젝트의 다른 방향 가능성을 설명하기 위한 문서였지만, 현재 결정은 **프로젝트 변경 없이 ChurnRadar를 계속 진행**하는 것입니다. 따라서 이 문서는 팀이 기존 통신사 고객 이탈 예측 프로젝트를 어떻게 마무리할지 정리하는 유지 보고서로 사용합니다.

## 1. 보고서 목적

본 보고서는 최종 팀 프로젝트 주제를 **B2B 통신사 고객 이탈 예측 프로젝트(ChurnRadar)**로 유지하고, 남은 기간 동안 무엇을 보강해 제출할지 정리하기 위해 작성했습니다.

프로젝트를 유지하는 이유는 다음과 같습니다.

- 이미 전처리, 모델 비교, threshold tuning, 오류 분석, feature importance까지 주요 머신러닝 과정을 수행했습니다.
- 이탈 고객 비율이 약 6.5%로 매우 낮아 class imbalance 문제를 명확히 보여줄 수 있습니다.
- F1, recall, precision trade-off를 실제 성능표와 confusion matrix로 설명할 수 있습니다.
- 낮은 성능 자체도 정적 CRM snapshot 데이터의 한계로 해석할 수 있어 보고서의 분석 포인트가 됩니다.

## 2. 프로젝트 목표

`Baza customer Telecom v2.csv` 데이터를 사용해 통신사 B2B 고객의 이탈 여부를 예측합니다.

핵심 질문은 다음과 같습니다.

> 현재 고객의 세그먼트, 가입자 상태, 매출 정보, 지역 정보를 바탕으로 이탈 가능 고객을 사전에 탐지할 수 있는가?

운영 관점에서는 하나의 모델만 고집하지 않고 목적별 모델을 나눕니다.

| 운영 목적 | 사용할 모델 |
| --- | --- |
| F1과 precision 균형 | `without_billing_zip + LogisticRegression_SMOTE` |
| 이탈 고객을 더 많이 잡는 캠페인 | `with_billing_zip + BalancedBagging_original` |
| 최대 recall 실험 | `with_billing_zip + CatBoost_native_categorical`, threshold 0.35 |

## 3. 데이터 요약

| 항목 | 내용 |
| --- | ---: |
| 원본 데이터 크기 | 8,453행 x 14열 |
| 중복 제거 후 데이터 | 8,436행 |
| target | `CHURN` |
| 이탈 고객 수 | 549명 |
| 비이탈 고객 수 | 7,904명 |
| 이탈 비율 | 약 6.5% |

이 데이터는 고객의 정적인 CRM 정보에 가깝습니다. 월별 사용량 변화, 결제 실패, 고객센터 문의/불만, 최근 활동 감소 같은 시간 기반 행동 feature가 부족합니다. 따라서 최종 보고서에서는 이 한계를 모델 성능 해석의 핵심으로 설명합니다.

## 4. 수행한 작업

기존 프로젝트에서 단순 baseline만 수행한 것이 아니라, 다음 작업까지 진행했습니다.

- 중복 고객 제거
- 결측치 처리와 missing flag 생성
- 범주형 변수 encoding
- 수치형 변수 scaling
- revenue 및 subscriber 상태 기반 feature engineering
- `Billing_ZIP` 포함/제외 variant 비교
- SMOTE 및 SVMSMOTE 적용
- Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting 비교
- EasyEnsemble, RUSBoost, BalancedBagging 비교
- CatBoost encoded feature 버전과 native categorical 버전 비교
- threshold tuning
- confusion matrix 기반 오류 분석
- permutation feature importance 분석
- 참고 논문 성능표와 비교
- 발표용 시각화 자료 생성

## 5. 최종 성능

| 기준 | Variant | Model | F1 | Recall | Precision | 해석 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 최종 F1 기준 | `without_billing_zip` | `LogisticRegression_SMOTE` | 0.1681 | 0.2661 | 0.1229 | 최종 보고서 메인 모델 |
| Recall 중심 | `with_billing_zip` | `BalancedBagging_original` | 0.1526 | 0.5872 | 0.0877 | 캠페인용 보조 후보 |
| Recall 극대화 | `with_billing_zip` | `CatBoost_native_categorical`, threshold 0.35 | 0.1310 | 0.8349 | 0.0711 | 오탐이 매우 많음 |
| 참고 논문 | external paper | EasyEnsemble | 0.1290 | 0.3820 | 0.0770 | 참고 기준 |

최종 메인 모델은 F1 기준으로 가장 높은 `LogisticRegression_SMOTE`입니다. 하지만 실제 retention campaign처럼 놓치는 고객을 줄이는 것이 중요하면 `BalancedBagging_original`을 함께 설명합니다.

## 6. 팀이 앞으로 해야 할 일

| 단계 | 작업 | 산출물 |
| --- | --- | --- |
| 1단계 | 최종 보고서 문장 정리 | `FINAL_REPORT.md` |
| 2단계 | 발표 슬라이드 제작 | PPT 또는 발표용 PDF |
| 3단계 | 그래프 삽입 | `presentation_assets/*.png` |
| 4단계 | 교수님 예상 질문 대비 | `CHURN_DATA_MODEL_DEFENSE.md` |
| 5단계 | 결과 재현 확인 | `processed/`, `final_model_summary.csv` |
| 6단계 | 선택적 추가 실험 | BalancedBagging 소규모 튜닝 결과 |

## 7. 역할 분담 제안

| 담당 | 역할 |
| --- | --- |
| 팀원 A | 전처리/모델링 결과 정리, 최종 보고서 수치 검증, 교수님 Q&A 준비 |
| 팀원 B | PPT 제작, 그래프 배치, 발표 멘트 정리, 한계/향후 개선 슬라이드 보강 |
| 공통 | 최종 발표 리허설, 질문 답변 확인, 제출 파일 점검 |

## 8. 발표에서 강조할 메시지

1. 이 프로젝트는 단순 accuracy가 아니라 minority class 탐지 성능을 보는 문제입니다.
2. 이탈 고객이 약 6.5%뿐이라 F1, recall, precision, PR-AUC를 함께 봐야 합니다.
3. 여러 모델을 비교했지만 최종 F1은 Logistic Regression + SVMSMOTE가 가장 높았습니다.
4. recall을 높이면 precision이 크게 떨어져 false positive가 많아지는 trade-off가 발생했습니다.
5. 현재 데이터는 정적 CRM snapshot이라 이탈 직전 행동 변화를 충분히 설명하지 못합니다.
6. 따라서 향후 개선 방향은 모델만 더 바꾸는 것이 아니라 시간 기반 고객 행동 데이터를 추가하는 것입니다.

## 9. 교수님께 드릴 설명문

긴 설명:

> 저희 팀은 최종 프로젝트를 통신사 고객 이탈 예측으로 유지하기로 했습니다. 이 데이터는 이탈 고객 비율이 약 6.5%라 accuracy만으로는 성능을 판단하기 어렵고, recall과 precision 사이의 trade-off가 크게 나타납니다. 그래서 중복 제거, 결측 처리, feature engineering, SVMSMOTE, 여러 모델 비교, threshold tuning, 오류 분석, feature importance까지 수행했습니다. 최종 F1 기준 모델은 `without_billing_zip + LogisticRegression_SMOTE`이며, 이탈 고객을 더 많이 잡는 운영 목적에서는 `BalancedBagging_original`을 보조 후보로 제시했습니다. 성능 수치가 높지는 않지만, 이는 모델을 덜 사용해서가 아니라 현재 데이터가 정적인 CRM snapshot이라 월별 사용량 변화, 결제 실패, 고객 불만 같은 시간 기반 행동 feature가 부족하기 때문으로 해석했습니다. 이 한계를 명확히 설명하는 방향으로 보고서와 발표를 완성하겠습니다.

짧은 설명:

> 프로젝트를 변경하지 않고 ChurnRadar를 계속 진행하겠습니다. 최종 모델은 F1 기준 `LogisticRegression_SMOTE`로 두고, recall 중심 운영 후보로 `BalancedBagging_original`을 함께 제시하겠습니다. 낮은 성능은 class imbalance와 시간 기반 행동 feature 부족이라는 데이터 한계로 해석해 최종 보고서에 반영하겠습니다.

## 10. 결론

본 팀은 최종 프로젝트를 **ChurnRadar: B2B 통신사 고객 이탈 예측**으로 유지합니다.

남은 작업의 우선순위는 새 주제 탐색이 아니라, 이미 생성된 모델 결과와 시각화 자료를 바탕으로 보고서와 발표를 완성하는 것입니다.
