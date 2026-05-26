# 연구 발표 자료: 논문 대비 이탈 예측 개선 분석 [김시환]

## 발표 핵심 메시지

논문은 B2B 통신 고객 이탈 예측을 위해 SVMSMOTE와 EasyEnsembleClassifier를 중심으로 통합 전처리 파이프라인을 구성했다. 우리는 같은 문제를 더 설명 가능한 연구 흐름으로 확장하기 위해 원본을 보존하고, `PID`와 `KA_name`을 제외한 뒤 컬럼별/값별 CSV를 별도 데이터로 분리하여 각 변수의 역할, 이탈률, 단일 변수 성능 한계를 먼저 확인했다.

핵심 결론은 단일 컬럼만으로는 F1이 0.15 근처에서 한계가 있으며, 최종 성능은 결측 flag, 이상치 flag, 활성 비율, 지역/세그먼트 교차 신호, 임계값 튜닝을 결합해야 올라간다는 점이다.

## 논문은 이렇게, 우리는 이렇게

| 비교항목 | 논문 방식 | 우리 방식 | 진행 이유 |
| --- | --- | --- | --- |
| 데이터 규모 | 8,454 unique business accounts | 원본 8,453행을 기준으로 분석하고 PID 중복/충돌을 별도 데이터 품질 이슈로 제시 | 동일 PID에 Yes/No가 섞인 사례가 있어 단순 unique 처리보다 중복 기준을 명확히 해야 함 |
| PID | 중복 제거 키로 사용 | 원본은 보존하고 모델/분리 CSV에서는 제외 | 식별자는 예측 신호가 아니라 leakage 위험이 있으므로 학습 변수에서 제외 |
| KA_name | 범주형 feature로 포함 후 label encoding | 요청 기준에 따라 모델 데이터에서 제외 | 담당자명은 조직 변화에 취약하고 운영상 민감한 변수라 고객 행동 중심 모델을 우선 구성 |
| 결측 처리 | 비활성/정지 가입자 결측은 0, CRM 결측은 Unknown, ZIP/ARPU는 median | 값 대체와 별도로 missing flag/exists flag를 보존 | 정지 가입자 수처럼 결측 자체가 이탈률 차이를 보이는 MNAR 신호일 수 있음 |
| 범주형 처리 | Billing_ZIP 등 고카디널리티도 label encoding 중심 | 값별 이탈률 CSV, CatBoost native categorical, leakage-safe target encoding 후보로 분리 | 지역별 이탈률 차이가 크고 ordinal label 값 자체에는 순서 의미가 없기 때문 |
| 로그 변환 | 매출 변수 중심 log transform | 매출뿐 아니라 가입자 수 변수도 왜도/이상치를 보고 log1p 후보로 처리 | 가입자 수 변수도 강한 우측 왜곡과 이상치 이탈률 상승이 관찰됨 |
| 피처 엔지니어링 | active subscriber rate, interaction terms 등 22개 feature | active_rate, missing flag, outlier flag, CRM x Segment, zip risk 등을 추가 후보로 제안 | 단일 변수 상관은 약하지만 구간/교차/결측에서 이탈 신호가 강하게 나타남 |
| 모델 선택 | EasyEnsembleClassifier + SVMSMOTE | LogisticRegression_SMOTE, BalancedBagging, CatBoost, soft ensemble 비교 | 불균형 데이터라 F1형/Recall형 운영 목적별 모델을 분리해 비교해야 함 |

## 성능 비교

| 구분 | 데이터/피처 조건 | 모델 | 임계값 | F1 | Recall | Precision | PR-AUC | MCC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 논문 기준 best | external_paper | EasyEnsembleClassifier + SVMSMOTE | not_reported | 0.1290 | 0.3820 | 0.0770 | 0.0790 | 0.0340 |
| 우리 단일 목적 F1 best | without_billing_zip | LogisticRegression_SMOTE | 0.5000 | 0.1681 | 0.2661 | 0.1229 | 0.0879 | 0.0956 |
| 우리 Recall 캠페인형 best | with_billing_zip | BalancedBagging_original | 0.5000 | 0.1526 | 0.5872 | 0.0877 | 0.0871 | 0.0820 |
| 우리 soft ensemble best | with_billing_zip | soft_avg_lr_bagging_catboost_native | 0.4700 | 0.1480 | 0.4679 | 0.0879 | 0.0861 | 0.0688 |

## 컬럼별 CSV 역할과 단일 컬럼 모델 한계

| 한글 컬럼명 | CSV 역할 | 최고위험 값/구간 | 최고위험 이탈률(%) | 단일 컬럼 best model | 단일 컬럼 F1 |
| --- | --- | --- | --- | --- | --- |
| 전체 가입자 수 | 고객의 전체 계약 규모와 대형 계정 이탈 위험을 확인한다. | (14.0, 235.0] | 9.5206 | DecisionTree_balanced | 0.1498 |
| 평균 모바일 매출 | 모바일 매출 규모가 이탈과 어떤 관계인지 확인한다. | (221.67, 499.83] | 8.9508 | LogisticRegression_balanced | 0.1441 |
| 활성 가입자 수 | 서비스를 실제 사용하는 가입자 규모를 확인한다. | (11.0, 110.0] | 9.4518 | LogisticRegression_balanced | 0.1426 |
| 총 매출 | 고객 총 매출 규모와 이탈 위험을 확인한다. | (222.33, 499.83] | 8.8270 | LogisticRegression_balanced | 0.1416 |
| 실질 비즈니스 세그먼트 | 실제 사업 규모/유형별 이탈 위험을 확인한다. | VSE | 8.8579 | DecisionTree_balanced | 0.1346 |
| CRM 고객가치 등급 | CRM이 판단한 고객가치 등급별 이탈 위험을 확인한다. | SME | 8.8028 | LogisticRegression_balanced | 0.1276 |
| 청구 우편번호 | 지역별 이탈 위험과 논문 SHAP의 geographic billing zone 신호를 검증한다. | 4701 | 25.0000 | DecisionTree_balanced | 0.1276 |
| 비활성 가입자 수 | 가입 후 미활성 상태인 고객의 이탈 위험을 확인한다. | (5.0, 214.0] | 8.1492 | DecisionTree_balanced | 0.1259 |
| 가입자당 평균 매출 | 가입자당 수익성과 이탈 위험을 확인한다. | (21.83, 29.95] | 7.1132 | DecisionTree_balanced | 0.1232 |
| 평균 유선 매출 | 유선/번들 이용 여부가 이탈 방어 효과를 갖는지 확인한다. | (-0.001, 480.5] | 6.4947 | DecisionTree_balanced | 0.1230 |
| 정지 가입자 수 | 정지 가입자 존재 여부와 결측 자체의 위험 신호를 확인한다. | (0.999, 22.0] | 8.8068 | Dummy_stratified | 0.0617 |

## 복수 모델 앙상블 실험

| 데이터/피처 조건 | 모델 | 선택 임계값 | F1 | Recall | Precision | PR-AUC | MCC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| with_billing_zip | soft_avg_lr_bagging_catboost_native | 0.4700 | 0.1480 | 0.4679 | 0.0879 | 0.0861 | 0.0688 |
| with_billing_zip | soft_avg_lr_balancedbagging_catboost | 0.4900 | 0.1434 | 0.3486 | 0.0903 | 0.0888 | 0.0602 |
| with_billing_zip | soft_avg_lr_balancedbagging | 0.5000 | 0.1379 | 0.2936 | 0.0901 | 0.0832 | 0.0537 |
| without_billing_zip | soft_avg_lr_bagging_catboost_native | 0.4500 | 0.1376 | 0.5321 | 0.0790 | 0.0871 | 0.0516 |
| without_billing_zip | soft_avg_lr_balancedbagging | 0.4900 | 0.1336 | 0.3211 | 0.0843 | 0.0858 | 0.0459 |
| without_billing_zip | soft_avg_lr_balancedbagging_catboost | 0.5000 | 0.1285 | 0.2936 | 0.0823 | 0.0862 | 0.0394 |

앙상블은 여러 모델의 예측 확률을 평균냈다. 검증셋에서 threshold를 고른 뒤 테스트셋에서 평가했으므로, 테스트셋 threshold 직접 최적화보다 보수적인 비교다.

## 왜 이런 방식으로 진행했는가

- 원본 CSV를 유지했다: 재현성과 감사 가능성을 위해 원본 데이터는 수정하지 않았다.
- 컬럼별 CSV를 만들었다: 통합 전처리를 바로 하면 어떤 컬럼이 어떤 신호를 주는지 설명하기 어렵다.
- `PID`를 제외했다: 고객 식별자는 예측 가능한 행동 신호가 아니라 leakage 위험이 있다.
- `KA_name`을 제외했다: 담당자명은 조직 개편에 취약하고 개인/운영 민감도가 높아 고객 행동 중심 모델을 우선했다.
- 결측 flag를 보존했다: 정지/비활성 가입자 결측은 단순 결측이 아니라 운영 기록의 부재라는 신호일 수 있다.
- 수치형을 구간화해 이탈률을 봤다: Pearson 상관은 낮지만 특정 구간과 이상치에서 이탈률이 상승한다.
- 여러 모델을 비교했다: 불균형 데이터에서는 정확도보다 F1, Recall, Precision, PR-AUC를 함께 봐야 한다.

## 성능을 더 올리는 방안

1. `CRM_PID_Value_Segment x EffectiveSegment` 교차 target encoding을 train fold 내부에서만 계산한다.
2. `Billing_ZIP`은 최소 표본수 기준 smoothing target encoding 또는 CatBoost ordered statistics로 처리한다.
3. `Suspended_subscribers_exists`, `Not_Active_subscribers_missing`, `has_fix_revenue`, `mobile_only` 같은 flag를 최종 모델에 명시적으로 넣는다.
4. `TotalRevenue`와 `AvgMobileRevenue`는 상관이 매우 높으므로 둘을 동시에 넣기보다 비율/차이/대표 변수로 정리한다.
5. 가입자 수와 매출 극단값은 제거하지 말고 outlier flag로 보존한다.
6. 단일 holdout보다 Stratified K-Fold 반복 검증으로 성능 신뢰구간을 제시한다.
7. 최종 운영 목적을 F1형과 Recall형으로 분리한다. 캠페인 대상 누락이 치명적이면 Recall형 모델을 별도 채택한다.
8. threshold를 고정 0.5로 두지 말고 retention 예산과 상담 가능 인원에 맞춰 조정한다.

## 발표 흐름 제안

1. 문제 정의: B2B 통신 고객 이탈은 적은 수의 이탈자라도 매출 영향이 크다.
2. 논문 요약: SVMSMOTE + EasyEnsemble, F1 0.129, SHAP 주요 변수.
3. 데이터 재점검: PID 중복, 결측, 지역/세그먼트 이탈률 차이.
4. 우리 방식: 원본 보존, PID/KA 제외, 컬럼별 CSV 분리, 한글 사전화.
5. 컬럼별 역할: 각 CSV가 무슨 질문에 답하는지 설명.
6. 모델 비교: 단일 컬럼 한계와 통합 모델 개선.
7. 성능 한계와 개선안: target encoding, 교차 피처, threshold, K-Fold.