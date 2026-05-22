# 프로젝트 변경 제안서

> 업데이트: 이 문서는 1차 변경 제안서입니다. 이후 2인 팀 프로젝트 상황과 다른 팀원이 조사한 서울 아파트 버블 탐지 데이터 수집 계획을 반영하여 `TEAM_PROJECT_SWITCH_REPORT.md`를 추가 작성했습니다. 교수님께 제출하거나 설명할 때는 `TEAM_PROJECT_SWITCH_REPORT.md`를 우선 사용하는 것이 좋습니다.

## 1. 제안 목적

현재 진행 중인 통신사 고객 이탈 예측 프로젝트는 여러 모델과 전처리 방법을 적용했음에도 F1과 recall 성능이 충분히 개선되지 않았습니다. 단순히 모델 선택의 문제가 아니라, 현재 데이터가 가진 구조적 한계 때문에 추가 실험을 계속해도 의미 있는 성능 향상과 해석을 얻기 어렵다고 판단했습니다.

따라서 본 팀은 기존 프로젝트를 중단하거나 축소하고, 더 명확한 target과 풍부한 feature를 가진 새로운 분류 예측 프로젝트로 변경하고자 합니다.

## 2. 현재 프로젝트 요약

현재 프로젝트는 `Baza customer Telecom v2.csv` 데이터를 사용하여 통신사 B2B 고객의 이탈 여부를 예측하는 것이 목표였습니다.

데이터 요약:

- 전체 원본 데이터: 8,453행 x 14열
- 중복 PID 제거 후 데이터: 8,436행
- 이탈 고객 수: 549명
- 비이탈 고객 수: 7,904명
- 이탈 비율: 약 6.5%
- target: `CHURN`

현재 데이터는 고객의 정적인 CRM 정보에 가깝습니다. 즉, 고객의 월별 사용량 변화, 결제 실패 이력, 문의/불만 기록, 최근 활동 감소, 서비스 이용 패턴 변화 같은 시간 기반 행동 정보가 부족합니다.

## 3. 지금까지 수행한 개선 작업

성능을 올리기 위해 아래 작업들을 이미 수행했습니다.

- 중복 고객 제거
- 결측치 처리
- 범주형 변수 인코딩
- 수치형 변수 스케일링
- revenue 관련 파생변수 생성
- `Billing_ZIP` 포함/제외 버전 비교
- SMOTE 적용
- SVMSMOTE 적용
- Logistic Regression, Random Forest, Balanced Random Forest, Balanced Bagging, Easy Ensemble, RUSBoost 비교
- CatBoost native categorical 처리
- threshold tuning 적용
- confusion matrix 기반 오류 분석
- feature importance 분석
- 논문 성능표와 비교

즉, 성능이 낮은 원인을 단순히 “모델을 적게 실험했기 때문”으로 보기 어렵습니다. 이미 불균형 데이터에 자주 쓰이는 여러 방법을 적용했지만, 성능 상승폭이 제한적이었습니다.

## 4. 현재 주요 성능 결과

| 기준 | Variant | Model | F1 | Recall | Precision | 해석 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 최종 F1 기준 | `without_billing_zip` | `LogisticRegression_SMOTE` | 0.1681 | 0.2661 | 0.1229 | F1 기준으로 가장 안정적 |
| Recall 중심 | `with_billing_zip` | `BalancedBagging_original` | 0.1526 | 0.5872 | 0.0877 | 이탈 고객은 더 많이 잡지만 오탐이 많음 |
| Recall 극대화 | `with_billing_zip` | `CatBoost_native_categorical`, threshold 0.35 | 0.1310 | 0.8349 | 0.0711 | recall은 높지만 precision이 너무 낮음 |
| 논문 기준 | - | EasyEnsemble | 0.1290 | 0.3820 | 0.0770 | 참고 논문에서도 F1이 높지 않음 |

결과 해석:

- F1 기준 최종 모델도 0.1681 수준으로 낮습니다.
- recall을 높이면 precision이 크게 떨어져 false positive가 매우 많아집니다.
- CatBoost처럼 recall을 0.8349까지 올릴 수는 있지만, precision이 0.0711이라 실제 운영 관점에서는 너무 많은 정상 고객을 이탈 고객으로 잘못 분류합니다.
- 참고 논문의 EasyEnsemble 결과도 F1 0.129, recall 0.382로 낮아, 이 데이터 계열 자체가 높은 성능을 내기 어려운 문제일 가능성이 있습니다.

## 5. 현재 프로젝트를 계속 진행하기 어려운 이유

### 5.1 target 불균형이 매우 심함

이탈 고객 비율이 약 6.5%라서 모델이 다수 클래스인 비이탈 고객에 치우치기 쉽습니다. SMOTE, SVMSMOTE, balanced ensemble을 적용해도 precision과 recall의 균형을 맞추기 어려웠습니다.

### 5.2 이탈을 설명할 핵심 행동 변수가 부족함

고객 이탈은 보통 아래와 같은 변화 신호가 중요합니다.

- 최근 사용량 감소
- 월별 요금 변화
- 납부 지연 또는 결제 실패
- 고객센터 문의 증가
- 불만 접수
- 약정 만료 시점
- 서비스 이용 빈도 감소
- 경쟁사 이동 가능성

하지만 현재 데이터에는 이런 시간 기반 또는 행동 기반 변수가 거의 없습니다. 그래서 모델이 이탈의 원인을 충분히 학습하기 어렵습니다.

### 5.3 추가 모델 실험의 기대효과가 낮음

이미 여러 알고리즘과 sampling 방법을 비교했습니다. 이후에도 XGBoost, LightGBM, 추가 hyperparameter tuning을 할 수는 있지만, 현재 feature 자체가 약하기 때문에 성능 향상 폭은 제한적일 가능성이 큽니다.

### 5.4 보고서 설득력이 약해질 위험이 있음

현재 프로젝트를 계속하면 최종 보고서의 핵심이 “왜 성능이 낮은가”에 집중될 가능성이 큽니다. 물론 이것도 의미 있는 분석이지만, 수업 프로젝트의 목적이 모델 비교, feature engineering, 성능 개선, 결과 해석이라면 더 적합한 데이터셋으로 변경하는 것이 학습 성과를 더 잘 보여줄 수 있습니다.

## 6. 프로젝트 변경의 필요성

프로젝트를 바꾸려는 이유는 단순히 점수가 낮기 때문이 아닙니다.

현재 프로젝트의 문제는 “모델을 잘못 골랐다”가 아니라 “데이터가 예측 문제를 충분히 설명하지 못한다”는 점입니다. 따라서 같은 데이터로 계속 모델만 바꾸는 것은 분석적으로 효율이 낮습니다.

새 프로젝트로 변경하면 다음을 더 잘 보여줄 수 있습니다.

- 데이터 전처리 과정
- feature engineering 효과
- 모델별 성능 비교
- F1, recall, precision의 trade-off 해석
- confusion matrix 기반 오류 분석
- 실제 문제 상황에 맞는 모델 선택
- 성능 개선 전후 비교

## 7. 대체 프로젝트 후보

### 후보 1. 온라인 쇼핑 구매 의도 예측

추천도: 높음

데이터 후보:

- UCI Online Shoppers Purchasing Intention Dataset
- 데이터 크기: 12,330개 세션
- target: `Revenue`
- 문제 유형: 구매 여부 binary classification
- feature 예시: 페이지 방문 수, 페이지 체류 시간, bounce rate, exit rate, page value, 방문자 유형, 주말 여부, 월 정보 등

장점:

- 현재 프로젝트와 달리 사용자의 행동 기반 feature가 포함되어 있습니다.
- 구매 여부라는 target이 명확합니다.
- 분류 모델, threshold tuning, feature importance, confusion matrix 분석을 모두 적용하기 좋습니다.
- “이탈 예측”에서 “구매 의도 예측”으로 바뀌지만, 고객 행동 예측이라는 큰 주제는 유지할 수 있습니다.

주의할 점:

- `PageValues` 같은 일부 변수는 target과 강하게 연결될 수 있으므로, leakage 가능성을 검토해야 합니다.

### 후보 2. 은행 마케팅 성공 예측

추천도: 높음

데이터 후보:

- UCI Bank Marketing Dataset
- 데이터 크기: 약 45,211개
- target: `y`
- 문제 유형: 정기예금 가입 여부 binary classification
- feature 예시: 나이, 직업, 결혼 여부, 교육 수준, 잔고, 대출 여부, 연락 방식, 캠페인 접촉 횟수, 이전 캠페인 결과 등

장점:

- 데이터 수가 현재 프로젝트보다 많습니다.
- 범주형 변수와 수치형 변수가 모두 있어 전처리 과정을 보여주기 좋습니다.
- 마케팅 캠페인이라는 비즈니스 목적이 명확합니다.
- 현재 프로젝트에서 사용한 Logistic Regression, ensemble, CatBoost 등을 그대로 비교할 수 있습니다.

주의할 점:

- `duration` 변수는 실제 예측 시점에서는 알기 어려운 변수일 수 있으므로, 포함/제외 버전을 비교하는 것이 좋습니다.

### 후보 3. 온라인 리테일 재구매 또는 고객 이탈 예측

추천도: 중간

데이터 후보:

- UCI Online Retail II Dataset
- 데이터 기간: 2009-12-01부터 2011-12-09까지의 거래 기록
- 문제 유형: 고객 재구매 예측, 고객 이탈 예측, 매출 예측 등으로 설계 가능

장점:

- 거래 날짜가 있어 시간 기반 feature engineering을 할 수 있습니다.
- RFM feature를 만들 수 있습니다.
  - Recency: 최근 구매일
  - Frequency: 구매 빈도
  - Monetary: 구매 금액
- 현재 프로젝트의 가장 큰 한계였던 “시간 기반 행동 정보 부족”을 해결할 수 있습니다.

주의할 점:

- target을 직접 정의해야 하므로 전처리 난이도가 더 높습니다.
- 시간 기준 train/test split 설계가 필요합니다.

## 8. 최종 추천 변경 방향

1차 검토 단계에서 가장 현실적인 변경안은 **온라인 쇼핑 구매 의도 예측 프로젝트**였습니다.

이유:

- 데이터 크기가 적당합니다.
- target이 명확합니다.
- 행동 기반 feature가 포함되어 있습니다.
- 현재 프로젝트에서 이미 만든 전처리, 모델 비교, threshold tuning, feature importance 분석 흐름을 재사용할 수 있습니다.
- F1과 recall을 개선하는 실험을 보여주기 더 적합합니다.
- 교수님께 “기존 프로젝트의 한계를 분석한 뒤, 더 적절한 데이터셋으로 변경한다”는 논리로 설명하기 좋습니다.

다만 이후 팀 프로젝트 상황을 반영했을 때, 다른 팀원이 이미 서울 아파트 버블 탐지 프로젝트의 데이터 수집 계획을 조사했으므로 최종 전환 후보는 **서울 아파트 버블 탐지 및 위험 예측 프로젝트**로 변경했습니다. 최종 제출용 논리는 `TEAM_PROJECT_SWITCH_REPORT.md`에 정리했습니다.

## 9. 교수님께 드릴 설명 문장

교수님께는 아래와 같이 설명할 수 있습니다.

> 기존 통신사 고객 이탈 예측 프로젝트에서 Logistic Regression, BalancedBagging, EasyEnsemble, RUSBoost, CatBoost, SMOTE, SVMSMOTE, threshold tuning 등을 적용해 보았습니다. 하지만 최종 F1이 0.1681 수준이고, recall을 높이면 precision이 0.07~0.09 수준으로 떨어져 실제 예측 모델로 설명하기 어려웠습니다. 이 문제는 모델 선택보다 데이터 자체의 한계가 더 크다고 판단했습니다. 현재 데이터는 고객의 정적인 CRM 정보 중심이고, 이탈 예측에 중요한 월별 사용량 변화, 결제 실패, 불만 기록, 최근 활동 변화 같은 시간 기반 행동 feature가 부족합니다. 따라서 같은 데이터에서 모델만 더 바꾸는 것보다, 행동 feature와 명확한 target이 있는 새로운 데이터셋으로 프로젝트를 변경하는 것이 수업 목표인 전처리, feature engineering, 모델 비교, 성능 개선, 결과 해석을 더 잘 보여줄 수 있다고 생각합니다.

짧게 말할 경우:

> 기존 프로젝트는 여러 모델을 적용해도 F1과 recall 개선 폭이 작았고, 원인이 데이터의 구조적 한계라고 판단했습니다. 그래서 더 명확한 target과 행동 기반 feature가 있는 데이터셋으로 변경하여 모델 성능 개선과 해석을 더 잘 보여주는 방향으로 진행하고 싶습니다.

## 10. 변경 후 진행 계획

프로젝트 변경이 승인되면 아래 순서로 진행합니다.

1. 새 데이터셋 선정 및 다운로드
2. target 분포 확인
3. 결측치, 이상치, 중복 데이터 확인
4. train/test split 설계
5. baseline 모델 생성
6. Logistic Regression, Random Forest, Gradient Boosting, CatBoost 비교
7. class imbalance가 있으면 SMOTE 또는 class_weight 적용
8. threshold tuning으로 recall과 precision trade-off 분석
9. feature importance 또는 SHAP 기반 해석
10. 최종 보고서와 발표 자료 작성

## 11. 결론

현재 통신사 이탈 예측 프로젝트는 많은 전처리와 모델 개선을 수행했음에도 성능 향상이 제한적이었습니다. 그 이유는 모델 자체보다 데이터의 target 불균형과 feature 부족에 있습니다.

따라서 프로젝트를 변경하는 것은 단순히 낮은 성능을 피하기 위한 선택이 아니라, 더 적절한 데이터로 머신러닝 프로젝트의 핵심 과정을 제대로 보여주기 위한 합리적인 결정입니다.

1차 후보로는 **온라인 쇼핑 구매 의도 예측** 또는 **은행 마케팅 성공 예측**을 검토했습니다. 이후 팀원 조사 내용을 반영한 최종 후보는 **서울 아파트 버블 탐지 및 위험 예측**이며, 최종 설득 보고서는 `TEAM_PROJECT_SWITCH_REPORT.md`를 기준으로 합니다.

## 참고 데이터 출처

- UCI Online Shoppers Purchasing Intention Dataset: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- UCI Bank Marketing Dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing
- UCI Online Retail II Dataset: https://archive.ics.uci.edu/dataset/502/online+retail+ii
