# 프로젝트 유지 결정 메모

> 업데이트: 2026-05-26
> 이전에는 프로젝트 변경 가능성을 검토하기 위해 작성했지만, 현재 결정은 **프로젝트 변경이 아니라 ChurnRadar(통신사 고객 이탈 예측)로 계속 진행**하는 것입니다. 이 파일은 변경 제안서가 아니라, 왜 기존 프로젝트를 유지하고 어떻게 마무리할지 정리한 메모로 사용합니다.

## 1. 결정

최종 프로젝트 주제는 `Baza customer Telecom v2.csv` 기반의 **B2B 통신사 고객 이탈 예측**으로 유지합니다.

이 프로젝트는 성능 수치만 높이는 프로젝트가 아니라, 다음 내용을 보여주는 머신러닝 분석 프로젝트로 정리합니다.

- 심한 class imbalance가 있는 실제형 데이터에서 accuracy가 왜 부족한 지표인지 설명
- 중복 제거, 결측 처리, leakage 방지, categorical encoding, scaling 등 전처리 과정 제시
- revenue, subscriber 상태, dormant/inactive 비율 기반 feature engineering 수행
- Logistic Regression, Random Forest, Gradient Boosting, imbalanced ensemble, CatBoost 비교
- SMOTE/SVMSMOTE와 threshold tuning을 통한 recall/precision trade-off 분석
- confusion matrix, 오류 분석, feature importance를 통한 결과 해석

## 2. 현재 프로젝트 요약

현재 프로젝트는 `Baza customer Telecom v2.csv` 데이터를 사용하여 통신사 B2B 고객의 이탈 여부를 예측하는 것이 목표입니다.

| 항목 | 값 |
| --- | ---: |
| 원본 데이터 | 8,453행 x 14열 |
| 중복 PID 제거 후 | 8,436행 |
| 이탈 고객 | 549명 |
| 비이탈 고객 | 7,904명 |
| 이탈 비율 | 약 6.5% |
| target | `CHURN` |

데이터는 고객의 정적인 CRM snapshot에 가깝습니다. 월별 사용량 변화, 결제 실패, 문의/불만 기록, 최근 활동 감소 같은 시간 기반 행동 정보가 부족하므로 성능 상한이 있습니다. 이 한계는 프로젝트를 버릴 이유가 아니라, 최종 해석에서 반드시 설명해야 할 핵심입니다.

## 3. 이미 수행한 작업

- 중복 고객 제거
- 결측치 처리와 missing flag 생성
- 범주형 변수 label/frequency encoding
- 수치형 변수 scaling
- revenue 관련 파생변수 생성
- 가입자 활동성 비율 feature 생성
- `Billing_ZIP` 포함/제외 버전 비교
- SMOTE 및 SVMSMOTE 적용
- Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting 비교
- BalancedBagging, EasyEnsemble, RUSBoost 비교
- CatBoost encoded feature 버전과 native categorical 버전 비교
- threshold tuning
- confusion matrix 기반 오류 분석
- permutation feature importance 분석
- 참고 논문 성능표와 비교
- 발표용 PNG 그래프 5개 생성

## 4. 현재 주요 성능 결과

| 기준 | Variant | Model | F1 | Recall | Precision | 해석 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 최종 F1 기준 | `without_billing_zip` | `LogisticRegression_SMOTE` | 0.1681 | 0.2661 | 0.1229 | 최종 보고서 메인 모델 |
| Recall 중심 | `with_billing_zip` | `BalancedBagging_original` | 0.1526 | 0.5872 | 0.0877 | 이탈 고객을 더 많이 잡는 캠페인용 후보 |
| Recall 극대화 | `with_billing_zip` | `CatBoost_native_categorical`, threshold 0.35 | 0.1310 | 0.8349 | 0.0711 | 오탐을 크게 감수할 때만 의미 있음 |
| 논문 기준 | external paper | EasyEnsemble | 0.1290 | 0.3820 | 0.0770 | 참고 논문에서도 F1이 높지 않음 |

최종 보고서에서는 `without_billing_zip + LogisticRegression_SMOTE`를 메인 모델로 둡니다. 다만 운영 목적이 “이탈 고객을 최대한 많이 찾아 retention campaign을 보내는 것”이라면 `BalancedBagging_original`을 보조 후보로 제시합니다.

## 5. 앞으로 해야 할 일

### 5.1 보고서 정리

- `FINAL_REPORT.md`를 최종 제출용 말투로 정리
- 낮은 F1을 숨기지 말고 class imbalance와 데이터 feature 한계로 해석
- 최종 모델을 하나만 말하지 말고 F1 기준 모델과 recall 운영 후보를 구분
- 논문 비교는 “우리 모델이 절대적으로 좋다”가 아니라 “유사 데이터에서도 어려운 문제”라는 근거로 사용

### 5.2 발표 자료 완성

- `PRESENTATION_SLIDES.md`의 8장 구조를 PPT로 옮기기
- `presentation_assets/`의 그래프 5개 삽입
- Slide 4~6에서 F1, recall, precision trade-off를 집중 설명
- Slide 8에서 한계와 향후 개선 방향을 결론으로 정리

### 5.3 교수님 질문 대비

`CHURN_DATA_MODEL_DEFENSE.md`를 기준으로 아래 질문에 답할 준비를 합니다.

- 왜 accuracy가 아니라 F1, recall, precision을 봤는가?
- 왜 Logistic Regression이 최종 모델인가?
- 왜 recall이 높은 CatBoost를 최종 모델로 쓰지 않았는가?
- 왜 `Billing_ZIP` 포함/제외를 둘 다 실험했는가?
- 왜 성능이 낮은데도 프로젝트를 유지하는가?
- 이 프로젝트에서 배운 점은 무엇인가?

### 5.4 선택적 추가 실험

시간이 남을 때만 아래 실험을 작게 추가합니다. 새 프로젝트로 갈아타는 작업은 하지 않습니다.

- BalancedBagging의 estimator 수, sampling strategy, base estimator를 소규모 grid로 비교
- threshold별 예상 campaign 비용을 가정해 precision/recall trade-off를 비용 관점으로 해석
- 최종 모델 coefficient 또는 permutation importance 표를 발표용으로 더 깔끔하게 정리

## 6. 교수님께 설명할 문장

긴 설명:

> 저희 팀은 최종 주제를 바꾸지 않고 통신사 고객 이탈 예측 프로젝트를 계속 진행하기로 했습니다. 이 데이터는 이탈 고객 비율이 약 6.5%로 매우 불균형하고, 고객의 월별 사용량 변화나 결제 실패, 불만 기록 같은 시간 기반 행동 feature가 부족합니다. 그래서 F1과 precision/recall 수치가 높게 나오지는 않았지만, 이 점을 프로젝트의 한계이자 분석 포인트로 정리했습니다. 중복 제거, 결측 처리, SVMSMOTE, 여러 불균형 대응 모델, CatBoost, threshold tuning, 오류 분석, feature importance까지 수행했고, 최종적으로 F1 기준 모델은 `without_billing_zip + LogisticRegression_SMOTE`로 선정했습니다. 또한 recall을 더 중시하는 운영 상황에서는 `BalancedBagging_original`을 보조 후보로 제시했습니다.

짧은 설명:

> 프로젝트를 변경하지 않고 기존 ChurnRadar를 유지하겠습니다. 낮은 성능은 단순 실패가 아니라, 심한 class imbalance와 시간 기반 행동 feature 부족에서 나온 결과로 해석하고, 전처리-모델 비교-threshold tuning-오류 분석까지 수행한 과정 중심으로 최종 보고서를 정리하겠습니다.

## 7. 최종 결론

이 파일의 결론은 “프로젝트 변경”이 아닙니다.

최종 방향은 다음과 같습니다.

- 주제: B2B 통신사 고객 이탈 예측
- 메인 모델: `without_billing_zip + LogisticRegression_SMOTE`
- 보조 운영 후보: `with_billing_zip + BalancedBagging_original`
- 핵심 메시지: 불균형 churn 데이터에서는 accuracy보다 F1, recall, precision, PR-AUC가 중요하며, recall을 높이면 precision이 급격히 낮아지는 trade-off가 발생한다.
- 한계: 정적 CRM snapshot이라 시간 기반 행동 feature가 부족하다.
- 제출 전략: 낮은 성능을 감추는 대신, 그 원인과 모델별 운영 trade-off를 명확히 설명한다.
