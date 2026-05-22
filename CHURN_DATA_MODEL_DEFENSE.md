# Churn 데이터 및 모델 선택 방어 정리

## 1. 문서 목적

이 문서는 교수님께 프로젝트 변경을 말씀드리기 전에, 기존 통신사 고객 이탈 예측 프로젝트를 충분히 이해하고 실험했다는 것을 설명하기 위해 작성했습니다.

핵심 목적은 다음과 같습니다.

- 이 데이터가 어떤 데이터인지 알기 쉽게 설명한다.
- 각 컬럼이 무엇을 의미하고 어떻게 처리했는지 정리한다.
- 왜 해당 모델들을 사용했는지 설명한다.
- 왜 일부 모델은 최종 모델로 쓰지 않았거나 추가로 사용하지 않았는지 설명한다.
- F1, recall, precision이 낮은 이유가 단순한 실험 부족이 아니라 데이터 구조적 한계라는 점을 정리한다.

## 2. 데이터 개요

사용한 데이터는 `Baza customer Telecom v2.csv`입니다. 통신사 B2B 고객 단위의 이탈 여부를 예측하는 데이터입니다.

| 항목 | 내용 |
| --- | --- |
| 원본 행 수 | 8,453행 |
| 원본 컬럼 수 | 14개 |
| 중복 PID 제거 후 행 수 | 8,436행 |
| target | `CHURN` |
| 이탈 고객 | 549명 |
| 비이탈 고객 | 7,904명 |
| 이탈 비율 | 약 6.5% |
| 문제 유형 | binary classification |

중요한 점은 target이 매우 불균형하다는 것입니다. 전체 고객 중 이탈 고객이 약 6.5%뿐이므로, 모델이 대부분을 비이탈로 예측해도 accuracy는 높게 나올 수 있습니다. 그래서 accuracy만 보면 안 되고, F1, recall, precision, PR-AUC, MCC를 함께 봐야 합니다.

## 3. 원본 컬럼 설명

| 컬럼명 | 의미 | 처리 방법 | 판단 이유 |
| --- | --- | --- | --- |
| `PID` | 고객 식별자 | 중복 제거 후 제거 | 식별자는 예측 일반화에 도움 되지 않고 leakage 위험이 있음 |
| `CRM_PID_Value_Segment` | 고객 가치 등급 | `Sliver`를 `Silver`로 수정, 결측은 `Unknown`, label/frequency encoding | 고객 가치 등급은 이탈과 관련 가능성이 있음 |
| `EffectiveSegment` | 고객 사업/규모 세그먼트 | 결측은 `Unknown`, label/frequency encoding | SOHO, VSE, SME 등 고객군 차이를 반영 |
| `Billing_ZIP` | 청구 우편번호 | 포함 버전과 제외 버전 모두 실험 | 지역 정보가 도움이 될 수도 있지만 noise가 될 수도 있음 |
| `KA_name` | 계정/관리자명으로 추정되는 식별성 변수 | 제거 | 특정 담당자나 이름 정보는 일반화가 어렵고 식별자 성격이 강함 |
| `Active_subscribers` | 활성 가입자 수 | 수치형 feature로 사용 | 고객 규모와 이용 상태 반영 |
| `Not_Active_subscribers` | 비활성 가입자 수 | 결측 flag 생성 후 0으로 대체 | 결측이 “없음”에 가까울 가능성이 있고, 결측 여부 자체도 정보일 수 있음 |
| `Suspended_subscribers` | 정지 가입자 수 | 결측 flag 생성 후 0으로 대체 | 결측률이 매우 높아 flag로 보존 |
| `Total_SUBs` | 전체 가입자 수 | 수치형 feature로 사용 | 고객 규모를 나타냄 |
| `AvgMobileRevenue` | 평균 모바일 매출 | 원본, log, sqrt 파생변수 생성 | 매출 규모와 분포 왜도를 반영 |
| `AvgFIXRevenue` | 평균 유선 매출 | 원본, log, sqrt 파생변수 생성 | 유선/모바일 매출 구조 반영 |
| `TotalRevenue` | 총 매출 | 원본, log, sqrt 파생변수 생성 | 고객 가치와 이탈 위험의 핵심 변수 |
| `ARPU` | 가입자당 평균 매출 | `TotalRevenue / Total_SUBs`로 일부 보정 후 중앙값 대체 | 고객 수익성을 나타냄 |
| `CHURN` | 이탈 여부 | `No=0`, `Yes=1`로 변환 | 예측 target |

## 4. 범주형 변수 분포

### 4.1 `CRM_PID_Value_Segment`

| 값 | 건수 |
| --- | ---: |
| Bronze | 3,820 |
| Silver | 2,039 |
| Gold | 1,453 |
| Platinum | 537 |
| SME | 284 |
| Iron | 246 |
| SE | 41 |
| Lead | 27 |
| 결측 | 5 |
| Sliver | 1 |

`Sliver`는 오타로 판단하여 `Silver`로 통합했습니다.

### 4.2 `EffectiveSegment`

| 값 | 건수 |
| --- | ---: |
| SOHO | 6,301 |
| VSE | 1,795 |
| SME | 284 |
| SE | 42 |
| Other | 29 |
| LE | 2 |

SOHO와 VSE에 데이터가 많이 몰려 있어, 세그먼트만으로 이탈을 강하게 구분하기는 어렵습니다.

## 5. 결측치와 데이터 품질

| 컬럼 | 결측률 | 처리 |
| --- | ---: | --- |
| `Suspended_subscribers` | 약 95.84% | 결측 여부 flag 생성 후 0 대체 |
| `Not_Active_subscribers` | 약 49.08% | 결측 여부 flag 생성 후 0 대체 |
| `CRM_PID_Value_Segment` | 약 0.06% | `Unknown` 대체 |
| `Billing_ZIP` | 약 0.02% | 포함 버전에서는 중앙값 대체 |
| `ARPU` | 약 0.01% | `TotalRevenue / Total_SUBs`로 보정 후 중앙값 대체 |

결측치를 단순 삭제하지 않은 이유는 이탈 고객이 549명뿐이라 행을 삭제하면 minority class가 더 줄어들기 때문입니다. 특히 `Suspended_subscribers`와 `Not_Active_subscribers`는 결측률이 높지만, 값이 없다는 사실 자체가 “해당 상태의 가입자가 없다”는 의미일 가능성이 있어 0으로 대체하고 missing flag를 함께 만들었습니다.

## 6. 전처리 방식

### 6.1 중복 제거

`PID` 기준 중복 17건을 제거했습니다. 같은 고객이 중복으로 들어가면 train/test에 같은 고객 정보가 섞일 수 있어 성능이 과장될 수 있기 때문입니다.

### 6.2 target 변환

`CHURN`은 문자열이므로 아래와 같이 변환했습니다.

```text
No  -> 0
Yes -> 1
```

### 6.3 train/test split

전체 데이터를 80:20으로 나누었고, `stratify=y`를 사용했습니다.

| 구분 | 비이탈 0 | 이탈 1 |
| --- | ---: | ---: |
| train | 6,312 | 436 |
| test | 1,579 | 109 |

stratify를 사용한 이유는 이탈 고객 비율이 낮기 때문에 random split만 하면 train/test의 이탈 비율이 흔들릴 수 있기 때문입니다.

### 6.4 `Billing_ZIP` 포함/제외 버전

`Billing_ZIP`은 지역 정보를 담을 수 있지만, 우편번호는 고유값이 많고 일반화가 어려운 변수일 수 있습니다. 그래서 아래 두 버전을 모두 만들었습니다.

| variant | 설명 |
| --- | --- |
| `with_billing_zip` | `Billing_ZIP` 포함 |
| `without_billing_zip` | `Billing_ZIP` 제외 |

최종 F1 기준으로는 `without_billing_zip + LogisticRegression_SMOTE`가 가장 좋았습니다. 즉, 지역 정보가 일부 모델에서는 도움을 주지만, 전체적으로는 noise로 작동하는 경우도 있다고 해석했습니다.

### 6.5 범주형 인코딩

`CRM_PID_Value_Segment`, `EffectiveSegment`, `Billing_ZIP`은 label encoding과 frequency encoding을 함께 적용했습니다.

이유:

- label encoding은 tree 계열 모델이 사용할 수 있는 숫자 형태로 바꾸기 위해 사용했습니다.
- frequency encoding은 각 범주의 등장 빈도 자체가 고객군의 대표성을 나타낼 수 있기 때문에 추가했습니다.
- encoding 기준은 train set에서만 만들고 test에는 train 기준을 적용했습니다.

### 6.6 스케일링

Logistic Regression과 거리/선형 기반 모델의 안정성을 위해 수치형 변수에 `StandardScaler`를 적용했습니다. scaler는 train set에만 fit하고 test에는 transform만 적용했습니다.

### 6.7 불균형 처리

이탈 고객이 약 6.5%뿐이므로 SVMSMOTE를 사용해 train set의 minority class를 보강했습니다.

| 구분 | 비이탈 0 | 이탈 1 |
| --- | ---: | ---: |
| 원래 train | 6,312 | 436 |
| SVMSMOTE 후 train | 6,312 | 3,668 |

SMOTE 계열은 test set에는 절대 적용하지 않았습니다. test set은 현실 데이터 분포를 유지해야 하기 때문입니다.

## 7. Feature engineering

원본 변수만으로는 신호가 약하다고 판단하여, 도메인 의미가 있는 파생변수를 만들었습니다.

| feature 그룹 | 예시 | 의미 |
| --- | --- | --- |
| 가입자 상태 비율 | `active_rate`, `inactive_rate`, `suspended_rate`, `dormant_rate` | 전체 가입자 중 활성/비활성/정지 비중 |
| 매출 비율 | `mobile_revenue_ratio`, `fix_revenue_ratio` | 모바일/유선 매출 구성 |
| 가입자당 매출 | `revenue_per_subscriber`, `revenue_per_active_subscriber` | 고객 규모 대비 매출 |
| 상호작용 | `revenue_engagement_interaction`, `arpu_risk_interaction` | 매출과 활동 상태의 결합 효과 |
| 계정 규모 flag | `multi_subscriber`, `large_account` | 소형/대형 고객 구분 |
| 매출 구조 flag | `mobile_only`, `fixed_only`, `revenue_zero` | 서비스 이용 형태 |
| 분포 보정 | `AvgMobileRevenue_log`, `TotalRevenue_sqrt`, `ARPU_sqrt` | 매출 변수의 왜도 완화 |

중요 feature 분석 결과, 최종 모델에서는 `AvgMobileRevenue_sqrt`, `TotalRevenue_sqrt`, `revenue_engagement_interaction`, `revenue_per_subscriber`, `AvgMobileRevenue` 등이 중요했습니다. 즉, 이 데이터에서는 고객의 수익 규모와 이용 상태가 가장 강한 신호였습니다.

## 8. 사용한 모델과 사용 이유

| 모델 | 사용 이유 | 최종 판단 |
| --- | --- | --- |
| Logistic Regression + SVMSMOTE | 설명 가능한 baseline이 필요하고, 불균형 데이터를 보정한 뒤 선형 경계가 어느 정도 작동하는지 확인하기 위해 사용 | F1 0.1681로 최종 F1 기준 1등 |
| Random Forest + SVMSMOTE | 비선형 관계와 변수 상호작용을 잡기 위해 사용 | recall과 F1이 낮아 최종 제외 |
| Gradient Boosting + SVMSMOTE | boosting 계열이 tabular data에서 강하므로 비교 | F1이 낮아 최종 제외 |
| HistGradientBoosting + SVMSMOTE | 빠른 gradient boosting 방식으로 추가 비교 | accuracy는 높지만 churn을 거의 못 잡아 제외 |
| EasyEnsemble | 불균형 데이터에 특화된 ensemble이고 참고 논문에서 사용했기 때문에 비교 | recall은 높지만 precision이 낮아 최종 제외 |
| RUSBoost | undersampling과 boosting을 결합한 imbalance-aware 모델이라 비교 | F1과 recall 모두 낮아 제외 |
| BalancedBagging | 불균형 데이터에서 minority class 탐지를 늘리기 위해 사용 | recall 0.5872로 캠페인용 후보, 하지만 precision 낮음 |
| CatBoost original balanced | 범주형/수치형 tabular data에 강하고 class weight를 자동 반영할 수 있어 사용 | recall은 개선되지만 F1이 최종 모델보다 낮음 |
| CatBoost native categorical | CatBoost의 장점인 categorical 직접 처리를 확인하기 위해 사용 | recall-heavy 목적에는 가능하지만 precision이 너무 낮음 |

## 9. 최종 주요 성능

| 기준 | Variant | Model | F1 | Recall | Precision | 해석 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| F1 기준 최종 | `without_billing_zip` | `LogisticRegression_SMOTE` | 0.1681 | 0.2661 | 0.1229 | 최종 보고서용 메인 모델 |
| Recall 중심 운영 | `with_billing_zip` | `BalancedBagging_original` | 0.1526 | 0.5872 | 0.0877 | 이탈 고객을 더 많이 잡는 캠페인용 후보 |
| Recall 극대화 | `with_billing_zip` | `CatBoost_native_categorical`, threshold 0.35 | 0.1310 | 0.8349 | 0.0711 | 놓치는 이탈 고객은 적지만 오탐이 너무 많음 |
| 논문 참고 | 외부 논문 | EasyEnsemble | 0.1290 | 0.3820 | 0.0770 | 논문에서도 F1이 높지 않음 |

정리하면, 최종 모델은 F1 기준으로는 Logistic Regression이 가장 낫고, recall을 많이 올리고 싶으면 BalancedBagging이나 CatBoost를 사용할 수 있습니다. 하지만 recall을 높일수록 precision이 크게 떨어지는 문제가 있었습니다.

## 10. 왜 F1과 recall이 낮은가

### 10.1 이탈 고객 수가 너무 적음

test set 기준 이탈 고객은 109명뿐입니다. 모델이 이 중 일부를 더 맞히거나 놓치는 것만으로 성능 지표가 크게 흔들립니다.

### 10.2 현재 feature가 이탈 원인을 직접 설명하지 못함

이탈 예측에서 중요한 정보는 보통 다음과 같습니다.

- 최근 몇 개월 사용량 감소
- 요금 납부 실패
- 고객센터 불만/문의 증가
- 약정 종료 시점
- 경쟁사 이동 가능성
- 서비스 품질 문제
- 최근 매출 감소 추세

하지만 현재 데이터는 한 시점의 고객 상태와 매출 요약값에 가깝습니다. 즉, “이 고객이 최근에 이탈하려는 행동을 보였는지”를 알 수 있는 시간이력 변수가 부족합니다.

### 10.3 recall을 올리면 precision이 무너짐

CatBoost threshold 0.35에서는 recall이 0.8349까지 올라갔지만 precision은 0.0711입니다. 이는 이탈 고객을 많이 잡는 대신 정상 고객도 너무 많이 이탈로 예측한다는 뜻입니다.

### 10.4 논문 결과도 높지 않음

참고 논문에서도 EasyEnsemble의 F1은 0.129, recall은 0.382였습니다. 우리 최종 F1 0.1681은 논문보다 높지만, 절대적인 성능은 여전히 낮습니다. 이 점은 해당 유형의 telecom churn 데이터 자체가 높은 분류 성능을 내기 어렵다는 근거로 볼 수 있습니다.

### 10.5 시간 기반 고객 행동 데이터를 추가로 확보하기 어려움

현재 churn 데이터의 성능을 의미 있게 올리려면 아래와 같은 시간 기반 고객 행동 데이터가 필요합니다.

- 월별 사용량 변화
- 최근 매출 감소 추세
- 결제 실패 또는 납부 지연 이력
- 고객센터 문의 및 불만 기록
- 약정 만료 시점
- 최근 서비스 이용 빈도 변화
- 회선 해지, 회선 변경, 상품 변경 이력

하지만 현재 접근 가능한 범위에서는 이런 데이터를 추가로 확보하기 어려웠습니다. 그래서 기존 데이터에 모델만 더 추가하는 방식보다는, 현재 churn 프로젝트를 한계 분석까지 마무리하고 시간 기반 feature engineering이 가능한 새 프로젝트로 전환하는 것이 더 합리적이라고 판단했습니다.

## 11. 최종 모델을 Logistic Regression으로 선택한 이유

F1 기준 최종 모델은 `without_billing_zip + LogisticRegression_SMOTE`입니다.

선택 이유:

1. 전체 실험 중 F1이 가장 높았습니다.
2. precision이 recall-heavy 모델보다 상대적으로 낫습니다.
3. 설명 가능성이 좋습니다.
4. feature importance와 coefficient 해석이 가능합니다.
5. 복잡한 모델보다 오히려 generalization이 안정적이었습니다.
6. `Billing_ZIP`을 제외해 지역 noise의 영향을 줄인 버전에서 성능이 더 좋았습니다.

교수님께 설명할 문장:

> 최종 모델은 가장 복잡한 모델이 아니라 F1 기준으로 가장 안정적인 Logistic Regression + SVMSMOTE를 선택했습니다. 이 데이터에서는 비선형 모델이나 ensemble 모델이 recall은 높였지만 precision이 크게 떨어져 실제 운영 관점에서는 오탐이 많았습니다. 따라서 보고서의 메인 모델은 F1이 가장 높은 Logistic Regression으로 두고, recall 중심 운영 후보로 BalancedBagging과 CatBoost를 별도로 제시했습니다.

## 12. 다른 모델을 최종으로 쓰지 않은 이유

| 모델 | 쓰지 않은 이유 |
| --- | --- |
| Random Forest | 비선형 관계를 잡을 수 있지만 이 데이터에서는 F1 0.0930 수준으로 낮았고, 이탈 고객 탐지가 약했습니다. |
| Gradient Boosting | F1 0.0670 수준으로 낮았고, recall도 0.0550에 그쳐 최종 후보가 되기 어려웠습니다. |
| HistGradientBoosting | accuracy는 높지만 recall이 매우 낮았습니다. 불균형 데이터에서 다수 클래스에 치우친 결과로 판단했습니다. |
| RUSBoost | 불균형 데이터용 모델이지만 F1 0.0749, recall 0.0642로 성능이 낮았습니다. |
| EasyEnsemble | 참고 논문과 비교하기 위해 사용했지만 precision이 낮아 최종 모델로는 부적합했습니다. |
| BalancedBagging | recall은 높지만 false positive가 많아 precision이 0.0877에 그쳤습니다. 최종 메인 모델이 아니라 캠페인용 후보로만 남겼습니다. |
| CatBoost | tabular data에 강하지만 이 데이터에서는 F1 기준으로 Logistic Regression을 넘지 못했습니다. threshold를 낮추면 recall은 높지만 precision이 크게 떨어졌습니다. |
| XGBoost / LightGBM | 참고 논문 후보로 검토했지만, 이미 GradientBoosting, HistGradientBoosting, CatBoost 계열에서 비슷한 boosting 실험을 했고 성능 한계가 명확했습니다. 추가 실험을 해도 feature 자체가 부족해 큰 개선 가능성이 낮다고 판단했습니다. |
| SVM | minority class가 적고 feature engineering 후 차원이 늘어난 상태에서 계산 비용과 해석성이 좋지 않습니다. 또한 SVMSMOTE로 이미 SVM 기반 sampling을 사용했기 때문에 우선순위를 낮췄습니다. |
| KNN | 고차원 feature와 스케일링된 매출/비율 변수에서 거리 기반 판단이 불안정할 수 있고, 불균형 데이터에 약해 우선순위를 낮췄습니다. |
| Naive Bayes | feature 간 독립 가정이 강한데, 이 데이터는 `TotalRevenue`, `ARPU`, 가입자 수, 파생 매출 변수들이 강하게 연결되어 있어 가정이 맞지 않습니다. |
| MLP | 데이터 수와 minority class 수가 충분하지 않고, 해석 가능성이 떨어집니다. 성능 개선보다 overfitting 위험이 크다고 판단했습니다. |
| Voting / Stacking | base model들의 성능이 전반적으로 낮은 상황에서는 앙상블을 쌓아도 실질 개선 가능성이 낮고, 보고서 설명 복잡도만 커질 수 있어 최종 단계에서 제외했습니다. |

## 13. 교수님 예상 질문과 답변

### Q1. 왜 accuracy가 높은 모델을 선택하지 않았나요?

이 데이터는 이탈 고객이 약 6.5%뿐이라, 대부분을 비이탈로 예측해도 accuracy가 높게 나옵니다. 그래서 accuracy보다 이탈 고객을 얼마나 잘 잡는지 보는 recall, precision, F1, PR-AUC가 더 중요합니다.

### Q2. 왜 recall이 높은 CatBoost를 최종 모델로 선택하지 않았나요?

CatBoost threshold 0.35는 recall이 0.8349로 높지만 precision이 0.0711입니다. 즉, 이탈 고객은 많이 잡지만 정상 고객도 매우 많이 이탈로 잘못 예측합니다. 그래서 최종 메인 모델로는 F1이 더 높은 Logistic Regression을 선택했고, CatBoost는 recall 극대화 운영 후보로만 제시했습니다.

### Q3. 왜 `Billing_ZIP`을 빼기도 했나요?

`Billing_ZIP`은 지역 정보를 담을 수 있지만 우편번호는 고유값이 많고, 특정 지역 패턴에 과적합될 수 있습니다. 그래서 포함/제외를 모두 실험했습니다. 최종 F1 기준으로는 제외 버전이 더 좋아서 메인 모델에서는 제외했습니다.

### Q4. 결측치가 많은 `Suspended_subscribers`를 왜 삭제하지 않았나요?

결측률은 높지만, 이 컬럼은 “정지 가입자가 없음”을 의미하는 빈칸일 가능성이 있습니다. 또한 이탈 고객 수가 적어 행을 삭제하면 minority class가 더 줄어듭니다. 그래서 결측 여부 flag를 만들고 값은 0으로 대체했습니다.

### Q5. SMOTE를 test에도 적용했나요?

아닙니다. SMOTE는 train set에만 적용했습니다. test set은 실제 데이터 분포를 유지해야 하므로 resampling하지 않았습니다.

### Q6. 이 프로젝트를 바꾸려는 이유는 실험을 덜 해서인가요?

아닙니다. Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting, EasyEnsemble, RUSBoost, BalancedBagging, CatBoost, threshold tuning, feature engineering까지 수행했습니다. 그럼에도 성능이 제한적이었고, 원인은 모델보다 데이터 feature의 한계라고 판단했습니다.

### Q7. 그래도 기존 프로젝트에서 배운 점은 무엇인가요?

불균형 데이터에서는 accuracy가 의미 없을 수 있고, recall을 높이면 precision이 급격히 낮아지는 trade-off가 발생한다는 것을 확인했습니다. 또한 고객 이탈 예측에는 단순한 정적 정보보다 시간 기반 행동 데이터가 매우 중요하다는 결론을 얻었습니다.

## 14. 교수님께 말할 최종 요약

> 기존 통신사 이탈 예측 프로젝트는 단순히 성능이 낮아서 포기하려는 것이 아닙니다. 원본 데이터 구조를 확인하고, 결측치 처리, 중복 제거, 파생변수 생성, SVMSMOTE, 여러 불균형 대응 모델, CatBoost native categorical 처리, threshold tuning까지 수행했습니다. 그 결과 F1 기준 최종 모델은 `without_billing_zip + LogisticRegression_SMOTE`였고 F1은 0.1681, recall은 0.2661이었습니다. recall을 높이는 모델도 있었지만 precision이 0.07~0.09 수준으로 낮아 실제 모델로 설명하기 어려웠습니다. 분석 결과 이 데이터는 정적인 CRM snapshot 중심이라 이탈 예측에 중요한 사용량 변화, 결제 실패, 불만 기록, 약정 종료 같은 시간 기반 행동 feature가 부족합니다. 따라서 같은 데이터에서 모델만 더 바꾸는 것보다, 시간 기반 feature를 만들 수 있는 다른 프로젝트로 전환하는 것이 더 타당하다고 판단했습니다.

추가로 짧게 덧붙일 문장:

> 성능 개선을 위해 필요한 시간 기반 고객 행동 데이터를 추가로 찾으려 했지만 확보하기 어려웠기 때문에, 기존 churn 데이터는 한계 분석으로 마무리하고 새 프로젝트로 진행하려고 합니다.

## 15. 최종 결론

이 churn 데이터는 다음과 같이 정리할 수 있습니다.

- 데이터는 B2B 통신사 고객 단위의 정적 CRM snapshot입니다.
- target인 churn은 약 6.5%로 매우 불균형합니다.
- 주요 신호는 매출 규모, 가입자 활동 상태, 고객 세그먼트입니다.
- 다양한 모델과 불균형 처리 방법을 적용했지만 F1과 recall 개선이 제한적이었습니다.
- 성능 한계의 핵심 원인은 모델 부족이 아니라 시간 기반 행동 feature 부족입니다.
- 성능 개선에 필요한 시간 기반 고객 행동 데이터를 추가로 확보하기 어려웠습니다.
- 따라서 기존 프로젝트는 한계 분석까지 마무리하고, 더 적합한 데이터 구조를 가진 새 프로젝트로 변경하는 것이 합리적입니다.
