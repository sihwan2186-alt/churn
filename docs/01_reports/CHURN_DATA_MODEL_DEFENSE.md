# Churn 데이터 및 모델 선택 방어 정리

## 1. 문서 목적

이 문서는 ChurnRadar 프로젝트를 최종 주제로 유지하면서, 통신사 고객 이탈 예측 데이터와 모델 선택을 충분히 이해하고 실험했다는 것을 설명하기 위해 작성했습니다.

핵심 목적은 다음과 같습니다.

- 이 데이터가 어떤 데이터인지 알기 쉽게 설명한다.
- 각 컬럼이 무엇을 의미하고 어떻게 처리했는지 정리한다.
- 왜 해당 모델들을 사용했는지 설명한다.
- 왜 일부 모델은 최종 모델로 쓰지 않았거나 보조 후보로만 남겼는지 설명한다.
- F1, recall, precision이 낮은 이유가 단순한 실험 부족이 아니라 데이터 구조적 한계라는 점을 정리한다.
- 성능이 낮아도 프로젝트를 유지할 때 어떤 분석 포인트를 강조해야 하는지 정리한다.

### 1.1 한눈에 보는 방어 포인트

| 질문 방향 | 핵심 답변 |
| --- | --- |
| 왜 accuracy를 기준으로 안 봤는가 | 이탈 고객이 약 6.5%뿐인 불균형 데이터라 accuracy는 다수 class인 비이탈 예측에 의해 과대평가될 수 있습니다. |
| 왜 Logistic Regression을 최종 모델로 선택했는가 | F1이 가장 높고, precision/recall 균형이 상대적으로 안정적이며, coefficient 기반 설명이 가능합니다. |
| 왜 recall이 높은 모델을 메인으로 쓰지 않았는가 | recall을 높이면 이탈 고객은 더 많이 잡지만 FP가 늘어 precision이 크게 낮아졌기 때문입니다. |
| 왜 성능이 낮은가 | 모델 부족보다 정적 CRM snapshot 데이터의 한계가 큽니다. 사용량 변화, 결제 실패, 불만 기록 같은 시간 기반 행동 feature가 없습니다. |
| 프로젝트의 의의는 무엇인가 | 불균형 데이터에서 지표 선택, threshold trade-off, feature engineering, 모델별 운영 목적 구분을 보여주는 분석입니다. |

### 1.2 읽는 순서

시간이 부족하면 `2. 데이터 개요`, `6. 전처리 방식`, `8. 사용한 모델과 사용 이유`, `9. 최종 주요 성능`, `13. 교수님 예상 질문과 답변`, `14. 교수님께 말할 최종 요약` 순서로 보면 됩니다.

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

binary classification은 정답 class가 두 개인 분류 문제를 의미합니다. 이 프로젝트에서는 고객을 `CHURN=1` 이탈 고객 또는 `CHURN=0` 비이탈 고객 중 하나로 예측하므로 binary classification입니다.

| class | 의미 | 모델의 판단 |
| --- | --- | --- |
| `CHURN=1` | 이탈 고객 | 이탈 가능성이 있다고 예측한 고객 |
| `CHURN=0` | 비이탈 고객 | 이탈 가능성이 낮다고 예측한 고객 |

이 데이터는 tabular data입니다. tabular data는 엑셀이나 CSV처럼 행과 열로 이루어진 표 형태의 데이터를 말합니다. 여기서는 한 행이 한 고객을 나타내고, 각 열은 고객 세그먼트, 가입자 수, 매출, 이탈 여부 같은 고객 속성을 나타냅니다.

중요한 점은 target이 매우 불균형하다는 것입니다. 전체 고객 중 이탈 고객이 약 6.5%뿐이므로, 모델이 대부분을 비이탈로 예측해도 accuracy는 높게 나올 수 있습니다. 그래서 accuracy만 보면 안 되고, F1, recall, precision, PR-AUC, MCC를 함께 봐야 합니다.

minority class는 데이터에서 표본 수가 상대적으로 적은 class를 의미합니다. 반대로 표본 수가 많은 class는 majority class라고 합니다.

| 구분 | 이 프로젝트의 class | 건수 | 의미 |
| --- | --- | ---: | --- |
| minority class | `CHURN=1` | 549명 | 실제 이탈 고객입니다. 수가 적지만 예측에서 가장 중요하게 찾아야 하는 대상입니다. |
| majority class | `CHURN=0` | 7,904명 | 비이탈 고객입니다. 수가 많아 모델이 이 class 위주로 학습하기 쉽습니다. |

이 프로젝트에서는 이탈 고객이 전체의 약 6.5%뿐이므로 모델이 아무 생각 없이 대부분을 비이탈로 예측해도 accuracy가 높아 보일 수 있습니다. 하지만 실제 목적은 “이탈할 고객을 미리 찾는 것”이기 때문에 minority class인 이탈 고객을 얼마나 잘 탐지하는지가 핵심입니다. 그래서 SVMSMOTE로 minority class를 보강했고, 평가에서도 recall, precision, F1, PR-AUC, MCC를 함께 확인했습니다.

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
| `AvgMobileRevenue` | 평균 모바일 매출 | 원본값 유지 + log/sqrt 파생변수 생성 | 매출 규모와 분포 왜도를 반영 |
| `AvgFIXRevenue` | 평균 유선 매출 | 원본값 유지 + log/sqrt 파생변수 생성 | 유선/모바일 매출 구조 반영 |
| `TotalRevenue` | 총 매출 | 원본값 유지 + log/sqrt 파생변수 생성 | 고객 가치와 이탈 위험의 핵심 변수 |
| `ARPU` | 가입자당 평균 매출 | `TotalRevenue / Total_SUBs`로 일부 보정 후 중앙값 대체, log/sqrt 파생변수 생성 | 고객 수익성을 나타냄 |
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

`EffectiveSegment`는 고객의 사업 규모나 유형을 나타내는 세그먼트입니다.

| 세그먼트 | 의미 | 해석 |
| --- | --- | --- |
| SOHO | Small Office / Home Office | 개인사업자 또는 아주 작은 사무실 규모의 고객군으로 해석할 수 있습니다. |
| VSE | Very Small Enterprise | 매우 작은 기업 고객군을 의미합니다. SOHO보다는 조직 형태가 있지만 규모는 작은 고객입니다. |
| SME | Small and Medium-sized Enterprise | 중소기업 고객군을 의미합니다. SOHO/VSE보다 상대적으로 사업 규모가 큰 고객입니다. |
| SE | Small Enterprise | 소기업 고객군으로 해석할 수 있습니다. |
| LE | Large Enterprise | 대기업 고객군으로 해석할 수 있습니다. |

SOHO와 VSE에 데이터가 많이 몰려 있어, 세그먼트만으로 이탈을 강하게 구분하기는 어렵습니다.

## 5. 결측치와 데이터 품질

결측률은 전체 행 중 특정 컬럼의 값이 비어 있는 비율입니다. 예를 들어 결측률이 50%라면 해당 컬럼의 절반 정도가 비어 있다는 뜻입니다. 결측률이 높다고 해서 무조건 컬럼을 삭제하는 것은 아니며, 값이 비어 있다는 사실 자체가 의미를 가질 수 있는지 먼저 판단해야 합니다.

| 컬럼 | 결측률 | 처리 |
| --- | ---: | --- |
| `Suspended_subscribers` | 약 95.84% | 결측 여부 flag 생성 후 0 대체 |
| `Not_Active_subscribers` | 약 49.08% | 결측 여부 flag 생성 후 0 대체 |
| `CRM_PID_Value_Segment` | 약 0.06% | `Unknown` 대체 |
| `Billing_ZIP` | 약 0.02% | 포함 버전에서는 중앙값 대체 |
| `ARPU` | 약 0.01% | `TotalRevenue / Total_SUBs`로 보정 후 중앙값 대체 |

결측치를 단순 삭제하지 않은 이유는 이탈 고객이 549명뿐이라 행을 삭제하면 minority class인 이탈 고객 표본이 더 줄어들기 때문입니다. 특히 `Suspended_subscribers`와 `Not_Active_subscribers`는 결측률이 높지만, 값이 없다는 사실 자체가 “해당 상태의 가입자가 없다”는 의미일 가능성이 있어 0으로 대체하고 missing flag를 함께 만들었습니다.

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

80:20으로 나눈 이유는 학습에 사용할 데이터를 충분히 확보하면서도, 학습에 쓰지 않은 별도 test set으로 일반화 성능을 확인하기 위해서입니다. 특히 이 데이터는 전체 표본 수가 크지 않고 이탈 고객 수가 549명뿐이므로 test set을 너무 크게 잡으면 학습할 minority class가 더 줄어들 수 있습니다.

`stratify=y`를 사용한 이유는 이탈 고객 비율이 낮기 때문에 random split만 하면 train/test의 이탈 비율이 흔들릴 수 있기 때문입니다. stratify를 적용하면 train과 test 모두에서 이탈 고객 비율이 원본 데이터와 비슷하게 유지되어, 모델이 학습한 분포와 평가하는 분포가 크게 달라지는 문제를 줄일 수 있습니다. 따라서 이 방식은 불균형 데이터에서 더 공정하고 안정적인 성능 평가를 하기 위한 절차입니다.

### 6.4 `Billing_ZIP` 포함/제외 버전

`Billing_ZIP`은 지역 정보를 담을 수 있지만, 우편번호는 고유값이 많고 일반화가 어려운 변수일 수 있습니다. 그래서 아래 두 버전을 모두 만들었습니다.

| variant | 설명 |
| --- | --- |
| `with_billing_zip` | `Billing_ZIP` 포함 |
| `without_billing_zip` | `Billing_ZIP` 제외 |

최종 F1 기준으로는 `without_billing_zip + LogisticRegression_SMOTE`가 가장 좋았습니다. 즉, 지역 정보가 일부 모델에서는 도움을 주지만, 전체적으로는 noise로 작동하는 경우도 있다고 해석했습니다.

### 6.5 범주형 인코딩

`CRM_PID_Value_Segment`, `EffectiveSegment`, `Billing_ZIP`은 label encoding과 frequency encoding을 함께 적용했습니다.

범주형 변수는 `SOHO`, `Bronze`, `Silver`처럼 문자나 그룹 이름으로 된 변수입니다. 대부분의 머신러닝 모델은 숫자 입력을 사용하므로, 이런 값을 숫자 feature로 변환해야 합니다.

| 인코딩 방식 | 의미 | 예시 | 사용 이유 |
| --- | --- | --- | --- |
| Label encoding | 각 범주에 정수 번호를 붙이는 방식 | `Bronze=0`, `Silver=1`, `Gold=2`처럼 변환 | 문자형 범주를 모델이 사용할 수 있는 숫자 형태로 바꾸기 위해 사용했습니다. |
| Frequency encoding | 각 범주가 train set에서 얼마나 자주 등장했는지를 값으로 넣는 방식 | `SOHO`가 70% 등장하면 `0.70`처럼 변환 | 범주의 등장 빈도 자체가 고객군의 대표성이나 규모 정보를 담을 수 있어 추가했습니다. |

두 방식을 함께 쓴 이유는 label encoding이 범주의 구분 정보를 제공하고, frequency encoding이 범주의 빈도 정보를 제공하기 때문입니다. 예를 들어 `EffectiveSegment=SOHO`라는 사실뿐 아니라 SOHO가 데이터에서 매우 자주 등장하는 대표 고객군이라는 정보도 모델에 전달할 수 있습니다.

encoding 기준은 train set에서만 만들고 test에는 train 기준을 적용했습니다. test set까지 보고 encoding 기준을 만들면 평가 데이터 정보가 학습 과정에 섞이는 data leakage가 발생할 수 있기 때문입니다.

### 6.6 스케일링

Logistic Regression과 거리/선형 기반 모델의 안정성을 위해 수치형 변수에 `StandardScaler`를 적용했습니다. scaler는 train set에만 fit하고 test에는 transform만 적용했습니다.

### 6.7 불균형 처리

이탈 고객이 약 6.5%뿐이므로 SVMSMOTE를 사용해 train set의 minority class를 보강했습니다. 즉, 학습 데이터에서 수가 적은 `CHURN=1` 이탈 고객 class의 학습 신호를 늘려 모델이 비이탈 고객만 예측하는 방향으로 치우치지 않도록 했습니다.

SVMSMOTE를 사용한 이유는 단순히 이탈 고객 데이터를 복사하는 것이 아니라, SVM이 찾은 결정 경계 주변의 minority class 샘플을 기준으로 합성 데이터를 만들기 때문입니다. 이탈/비이탈이 애매하게 갈리는 경계 근처의 이탈 고객 패턴을 더 학습하게 하여, 모델이 희소한 이탈 class를 조금 더 잘 인식하도록 만드는 목적이 있습니다.

| 방법 | 의미 | 사용 이유 |
| --- | --- | --- |
| 단순 oversampling | 기존 이탈 고객 행을 그대로 복사 | 데이터 수는 늘지만 같은 표본이 반복되어 overfitting 위험이 있음 |
| SMOTE | 가까운 minority class 샘플 사이를 보간해 합성 표본 생성 | 단순 복사보다 낫지만, 경계와 상관없는 합성 표본도 생길 수 있음 |
| SVMSMOTE | SVM 기반으로 class 경계 주변의 minority class를 중심으로 합성 표본 생성 | 이탈/비이탈 구분이 어려운 영역을 보강해 불균형 데이터에서 이탈 탐지 학습을 돕기 위해 사용 |

| 구분 | 비이탈 0 | 이탈 1 |
| --- | ---: | ---: |
| 원래 train | 6,312 | 436 |
| SVMSMOTE 후 train | 6,312 | 3,668 |

SMOTE 계열은 test set에는 절대 적용하지 않았습니다. test set은 현실 데이터 분포를 유지해야 하기 때문입니다.

## 7. Feature engineering

원본 변수만으로는 신호가 약하다고 판단하여, 도메인 의미가 있는 파생변수를 만들었습니다.

### 7.1 수치형 feature와 log/sqrt 파생변수

수치형 feature는 고객의 가입자 수, 매출, ARPU처럼 숫자로 크기와 양을 표현하는 변수입니다. 이 프로젝트에서는 `Active_subscribers`, `Not_Active_subscribers`, `Suspended_subscribers`, `Total_SUBs`, `AvgMobileRevenue`, `AvgFIXRevenue`, `TotalRevenue`, `ARPU` 등이 핵심 수치형 feature입니다.

매출 계열 변수는 고객마다 값 차이가 크고 일부 큰 값이 평균을 끌어올리는 우측으로 긴 분포를 가질 수 있습니다. 그래서 원본값만 사용하지 않고 아래처럼 원본, log, sqrt 형태를 함께 만들었습니다.

| 형태 | 생성 방식 | 의미 |
| --- | --- | --- |
| 원본 feature | 기존 값을 그대로 사용 | 실제 매출/가입자 규모 자체를 보존합니다. |
| log 파생변수 | `log1p(x) = log(1 + x)` 적용 | 큰 값을 압축해 이상치 영향을 줄이고, 0 값도 안전하게 변환합니다. |
| sqrt 파생변수 | `sqrt(x)` 적용 | 원본보다 완만하게 값을 줄여 왜도를 낮추되, log보다 원래 크기 정보를 더 남깁니다. |

예를 들어 `TotalRevenue`에 대해 `TotalRevenue`, `TotalRevenue_log`, `TotalRevenue_sqrt`를 함께 사용하면 모델이 “총매출의 절대 규모”와 “큰 매출값을 완화한 패턴”을 동시에 학습할 수 있습니다. 실제 구현에서는 `AvgMobileRevenue`, `AvgFIXRevenue`, `TotalRevenue`, `ARPU` 네 개 매출 계열 변수에 log/sqrt 파생변수를 생성했습니다.

| feature 그룹 | 예시 | 의미 |
| --- | --- | --- |
| 가입자 상태 비율 | `active_rate`, `inactive_rate`, `suspended_rate`, `dormant_rate` | 전체 가입자 중 활성/비활성/정지 비중 |
| 매출 비율 | `mobile_revenue_ratio`, `fix_revenue_ratio` | 모바일/유선 매출 구성 |
| 가입자당 매출 | `revenue_per_subscriber`, `revenue_per_active_subscriber` | 고객 규모 대비 매출 |
| 상호작용 | `revenue_engagement_interaction`, `arpu_risk_interaction` | 매출과 활동 상태의 결합 효과 |
| 계정 규모 flag | `multi_subscriber`, `large_account` | 소형/대형 고객 구분 |
| 매출 구조 flag | `mobile_only`, `fixed_only`, `revenue_zero` | 서비스 이용 형태 |
| 분포 보정 | `AvgMobileRevenue_log`, `TotalRevenue_sqrt`, `ARPU_sqrt` | 매출 변수의 왜도 완화 |

### 7.2 총 feature 개수와 도출 과정

메인 파이프라인에서 `Billing_ZIP`을 포함하면 최종 feature는 55개이고, `Billing_ZIP`을 제외하면 52개입니다. 3개 차이는 `Billing_ZIP`, `Billing_ZIP_missing`, `Billing_ZIP_frequency`가 빠지기 때문입니다.

feature를 55개까지 늘린 이유는 단순히 변수 수를 늘리기 위해서가 아니라, 원본 CRM snapshot이 이탈 원인을 직접 보여주지 못하기 때문에 고객 상태, 매출 구조, 활동성, 결측 여부, 범주 대표성을 모델이 더 잘 볼 수 있도록 변환했기 때문입니다.

| feature 묶음 | 개수 | 예시 | 도출 이유 |
| --- | ---: | --- | --- |
| 기본 입력 feature | 11 | `CRM_PID_Value_Segment`, `EffectiveSegment`, `Billing_ZIP`, 가입자 수, 매출, `ARPU` | 원본 데이터의 고객 속성과 매출 규모를 보존하기 위해 사용했습니다. |
| 결측 flag | 4 | `Not_Active_subscribers_missing`, `Suspended_subscribers_missing`, `ARPU_missing`, `Billing_ZIP_missing` | 값이 비어 있다는 사실 자체가 고객 상태 정보일 수 있어 보존했습니다. |
| 가입자 상태/비율 feature | 5 | `dormant_subscribers`, `active_rate`, `inactive_rate`, `suspended_rate`, `dormant_rate` | 단순 가입자 수보다 전체 가입자 대비 활성/비활성 비율이 이탈 신호를 더 잘 보여줄 수 있습니다. |
| 매출 비율/가입자당 매출 | 8 | `mobile_revenue_ratio`, `fixed_to_mobile_ratio`, `revenue_per_subscriber` | 고객 규모가 달라도 매출 구조와 가입자당 수익성을 비교할 수 있게 만들었습니다. |
| 위험도/상호작용/균형 feature | 6 | `risk_score`, `revenue_engagement_interaction`, `arpu_risk_interaction`, `revenue_balance` | 매출과 활동성이 따로가 아니라 결합될 때 이탈 위험을 설명할 수 있어 추가했습니다. |
| binary flag | 10 | `has_inactive`, `has_suspended`, `multi_subscriber`, `large_account`, `mobile_only`, `revenue_zero` | 고객을 명확한 유형으로 나누어 모델이 조건을 쉽게 학습하도록 했습니다. |
| log/sqrt 변환 feature | 8 | `AvgMobileRevenue_log`, `TotalRevenue_sqrt`, `ARPU_sqrt` | 매출 변수의 큰 값과 왜도 영향을 줄이고 안정적인 패턴을 학습하기 위해 만들었습니다. |
| frequency encoding feature | 3 | `CRM_PID_Value_Segment_frequency`, `EffectiveSegment_frequency`, `Billing_ZIP_frequency` | 범주의 등장 빈도 자체가 고객군의 대표성 정보를 담을 수 있어 추가했습니다. |

따라서 총 feature 수는 `11 + 4 + 5 + 8 + 6 + 10 + 8 + 3 = 55개`입니다. 최종 메인 모델에서는 `Billing_ZIP` 관련 3개를 제외한 52개 feature를 사용했습니다.

교수님께 설명할 때는 다음처럼 말하면 됩니다.

> 원본 14개 컬럼에서 target과 식별성 변수를 제외한 뒤, 고객 상태 비율, 매출 구조, 가입자당 매출, 결측 flag, log/sqrt 변환, 범주 빈도 인코딩을 추가해 ZIP 포함 기준 55개 feature를 만들었습니다. 이탈 예측에 필요한 시간 기반 행동 데이터가 부족했기 때문에, 현재 snapshot 안에서 고객 활동성과 수익 구조를 최대한 명시적으로 펼친 것입니다.

### 7.3 Feature importance와 coefficient

feature importance는 모델 성능에 각 feature가 얼마나 중요한 역할을 했는지 보는 지표입니다. 이 프로젝트에서는 특히 permutation feature importance를 사용했습니다. 이는 특정 feature 값을 섞어서 정보가 사라지게 만든 뒤, 모델 성능이 얼마나 떨어지는지 확인하는 방식입니다. 어떤 feature를 섞었을 때 성능이 많이 떨어지면, 그 feature가 예측에 중요하다고 해석할 수 있습니다.

coefficient는 Logistic Regression 같은 선형 모델에서 각 feature에 붙는 가중치입니다. Logistic Regression은 아래처럼 feature들의 가중합으로 이탈 확률을 계산합니다.

```text
logit(P(churn)) = intercept + coefficient_1 * feature_1 + coefficient_2 * feature_2 + ...
```

coefficient가 양수이면 해당 feature 값이 커질수록 이탈 가능성을 높이는 방향으로 작용하고, 음수이면 이탈 가능성을 낮추는 방향으로 작용합니다. coefficient의 절댓값이 클수록 모델이 그 feature를 더 강하게 반영한다고 볼 수 있습니다. 이 프로젝트에서는 수치형 feature를 `StandardScaler`로 표준화했기 때문에, coefficient는 “해당 feature가 1 표준편차 증가할 때 이탈 log-odds가 얼마나 변하는지”로 해석할 수 있습니다.

| 구분 | 무엇을 보는가 | 해석할 때 주의점 |
| --- | --- | --- |
| Feature importance | 해당 feature가 모델 성능에 얼마나 기여했는지 | 방향성은 알려주지 않습니다. 중요한 feature라도 이탈을 높이는지 낮추는지는 별도로 봐야 합니다. |
| Coefficient | feature가 이탈 확률을 높이는 방향인지 낮추는 방향인지 | 상관관계가 강한 feature들이 함께 있으면 계수 부호가 직관과 다를 수 있습니다. 인과관계로 해석하면 안 됩니다. |

따라서 feature importance는 “무엇이 중요한가”를 보는 데 사용했고, coefficient는 “그 feature가 이탈 가능성을 높이는 방향인지 낮추는 방향인지”를 설명하는 데 사용했습니다.

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

### 8.1 모델별 역할과 장단점

아래 내용은 각 모델을 왜 실험했는지, 어떤 상황에서 쓰면 좋은지, 그리고 이번 프로젝트에서 어떤 기능을 확인했는지를 정리한 것입니다.

| 모델 | 언제 쓰면 좋은가 | 장점 | 단점/주의점 | 이번 프로젝트에서 확인한 기능 |
| --- | --- | --- | --- | --- |
| Logistic Regression + SVMSMOTE | 설명 가능한 baseline이 필요하고, 변수와 이탈 사이의 관계를 비교적 단순한 선형 경계로 확인하고 싶을 때 | 빠르고 안정적이며 coefficient로 어떤 feature가 이탈 확률을 높이거나 낮추는지 설명하기 쉽습니다. | feature 관계가 복잡한 비선형 구조이면 한계가 있습니다. 파생변수와 스케일링 품질에 영향을 많이 받습니다. | SVMSMOTE로 이탈 class를 보강한 뒤, 표준화된 feature의 선형 결합으로 이탈 확률을 계산했습니다. 최종 F1 기준 가장 안정적인 메인 모델 역할을 했습니다. |
| Random Forest + SVMSMOTE | 여러 feature 사이의 비선형 관계와 상호작용을 자동으로 잡고 싶을 때 | 여러 decision tree를 묶어 예측하므로 단일 tree보다 과적합을 줄이고, feature importance를 확인할 수 있습니다. | minority class가 매우 적으면 다수 class 위주로 판단할 수 있고, 확률 보정이 약할 수 있습니다. | 여러 tree의 voting으로 이탈을 예측해 보았지만, 이 데이터에서는 recall과 F1이 낮아 최종 모델로는 부적합했습니다. |
| Gradient Boosting + SVMSMOTE | tabular data에서 순차적으로 오차를 줄이는 boosting 모델을 비교하고 싶을 때 | 이전 tree가 틀린 부분을 다음 tree가 보완하므로 복잡한 패턴을 학습할 수 있습니다. | hyperparameter에 민감하고, 불균형 데이터에서는 minority class보다 majority class 오차를 줄이는 방향으로 치우칠 수 있습니다. | boosting 방식이 이탈 탐지 성능을 개선하는지 확인했지만, F1이 낮아 제외했습니다. |
| HistGradientBoosting + SVMSMOTE | gradient boosting을 더 빠르게 학습시키고 싶거나 데이터가 클 때 | feature 값을 구간으로 나누는 histogram 방식이라 일반 gradient boosting보다 빠르게 학습할 수 있습니다. | accuracy는 높아도 minority class를 거의 못 잡는 경우가 생길 수 있습니다. | 빠른 boosting 대안으로 실험했지만, churn을 거의 잡지 못해 제외했습니다. |
| EasyEnsemble | class imbalance가 심하고, 이탈 고객을 더 많이 잡는 recall 중심 후보가 필요할 때 | majority class를 여러 번 나누어 balanced subset을 만들고 ensemble하므로 minority class 탐지에 유리합니다. | 정상 고객을 이탈로 잘못 예측하는 FP가 늘어 precision이 낮아질 수 있습니다. | 참고 논문 모델과 비교하고, 불균형 특화 ensemble이 recall을 높이는지 확인했습니다. |
| RUSBoost | undersampling과 boosting을 함께 사용해 불균형 데이터에 대응하고 싶을 때 | 학습 과정에서 majority class를 줄이면서 boosting을 수행해 minority class에 더 집중할 수 있습니다. | majority class 정보를 일부 버리므로 데이터 손실이 있고, noise에 민감할 수 있습니다. | imbalance-aware boosting 후보로 실험했지만 F1과 recall 모두 낮아 제외했습니다. |
| BalancedBagging | 이탈 고객을 많이 잡는 캠페인 운영 후보가 필요하고, precision보다 recall을 더 중시할 때 | 각 base model이 balanced sample로 학습해 minority class 탐지율을 높일 수 있습니다. | recall은 높아질 수 있지만 FP가 늘어 precision이 낮아지고, 설명력은 Logistic Regression보다 약합니다. | recall 0.5872로 이탈 고객을 많이 잡는 후보가 되었지만, precision이 낮아 메인 모델이 아니라 캠페인용 후보로 남겼습니다. |
| CatBoost original balanced | 범주형/수치형이 섞인 tabular data에서 강한 boosting 모델을 쓰고 싶을 때 | categorical feature와 비선형 관계에 강하고, class weight를 통해 불균형을 반영할 수 있습니다. | 모델이 복잡해 설명이 어렵고, threshold를 낮추면 FP가 급격히 늘 수 있습니다. | class imbalance를 반영한 CatBoost가 recall을 개선하는지 확인했지만, F1 기준으로 Logistic Regression을 넘지 못했습니다. |
| CatBoost native categorical | label/frequency encoding 없이 CatBoost가 범주형 변수를 직접 처리하는 효과를 확인하고 싶을 때 | 범주형 변수를 직접 다룰 수 있어 encoding 과정에서 정보 손실을 줄일 수 있고, 범주형 상호작용을 잘 잡을 수 있습니다. | recall-heavy 설정에서는 이탈 고객을 많이 잡지만 정상 고객도 많이 이탈로 예측해 precision이 낮아질 수 있습니다. | `CRM_PID_Value_Segment`, `EffectiveSegment` 같은 categorical feature를 CatBoost 방식으로 직접 처리했을 때의 성능을 확인했습니다. recall 극대화 후보로는 가능했지만 메인 모델로는 부적합했습니다. |

정리하면, Logistic Regression은 설명 가능성과 F1 균형이 좋아 최종 메인 모델로 선택했고, BalancedBagging과 CatBoost는 이탈 고객을 더 많이 찾는 recall 중심 운영 후보로 해석했습니다. 반면 tree/boosting 계열 모델들은 복잡한 패턴을 잡을 수 있다는 장점이 있었지만, 이 데이터에서는 precision과 F1이 충분히 개선되지 않았습니다.

### 8.2 실험 A~G의 목적

실험 A~G는 단순히 모델을 많이 돌린 것이 아니라, 최종 결론을 방어하기 위해 서로 다른 질문을 검증한 실험입니다. 핵심은 “왜 이 모델을 선택했는가”, “어떤 feature가 실제로 의미 있었는가”, “운영에서는 어떤 기준으로 써야 하는가”를 확인하는 것입니다.

| 실험 | 무엇을 확인하려 했는가 | 결과 해석 |
| --- | --- | --- |
| 실험 A: Billing ZIP Ablation | `Billing_ZIP`을 포함하는 것이 도움이 되는지, 아니면 noise가 되는지 확인 | Tree ensemble에서는 ZIP이 recall/F1을 높였지만, Logistic Regression에서는 ZIP 제거 버전이 더 좋았습니다. 즉 ZIP 효과는 모델 계열에 따라 달랐습니다. |
| 실험 B: Feature Group Ablation | 어떤 feature group이 성능에 가장 크게 기여하는지 확인 | LR은 categorical group 제거 시 F1이 크게 하락했고, BalancedBagging은 interaction group 제거 시 하락이 컸습니다. feature 중요도는 모델 구조와 함께 해석해야 합니다. |
| 실험 C: CRM Segment Error Analysis | Bronze/Silver/Gold/Platinum/SME 등 고객 세그먼트별 오류 패턴이 다른지 확인 | 전체 성능 하나만 보면 세그먼트별 리스크를 놓칠 수 있습니다. 고가치 고객은 FP/precision 문제가 크고, 일부 저가치 고객군은 recall 문제가 나타났습니다. |
| 실험 D: Cost Threshold Sweep | FP 비용과 TP 이익이 달라질 때 최적 threshold와 모델이 바뀌는지 확인 | 캠페인 비용이 낮으면 recall-heavy 전략이 유리하고, 비용이 높아지면 precision 높은 모델이나 보수적 threshold가 유리합니다. |
| 실험 E: Paper Ablation Variants | 논문형 core feature와 확장 feature 조합을 비교 | 논문 재현용 feature만 보는 것이 아니라, ZIP grouping, log/sqrt, interaction, KA 추상화 등 feature 설계의 효과를 비교했습니다. |
| 실험 F: 추가 모델 & Soft Ensemble | 추가 모델이나 soft ensemble이 단일 모델보다 나은지 확인 | 복잡한 모델과 앙상블이 항상 F1을 올리지는 않았습니다. F1 목적에서는 단일 모델이 더 단순하고 안정적인 경우가 있었습니다. |
| 실험 G: 통계 검증 | hold-out 점수 차이가 우연인지, 모델 오류 패턴이 실제로 다른지 확인 | Bootstrap CI와 McNemar test로 모델별 trade-off가 단순 점수 차이가 아니라 서로 다른 오류 패턴임을 보조 검증했습니다. |

발표에서는 이렇게 정리할 수 있습니다.

> 실험 A~G는 모델 점수표를 늘리기 위한 실험이 아니라, ZIP 정보의 효과, feature group의 기여, 세그먼트별 오류, 비용 구조별 threshold, 논문형 feature 재현, 앙상블 효과, 통계적 안정성을 각각 확인하기 위한 검증 실험입니다. 이를 통해 최종 모델 선택이 단일 F1 점수만이 아니라 데이터 구조와 운영 목적을 함께 고려한 결정임을 설명할 수 있습니다.

## 9. 최종 주요 성능

| 기준 | Variant | Model | F1 | Recall | Precision | 해석 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| F1 기준 최종 | `without_billing_zip` | `LogisticRegression_SMOTE` | 0.1681 | 0.2661 | 0.1229 | 최종 보고서용 메인 모델 |
| Recall 중심 운영 | `with_billing_zip` | `BalancedBagging_original` | 0.1526 | 0.5872 | 0.0877 | 이탈 고객을 더 많이 잡는 캠페인용 후보 |
| Recall 극대화 | `with_billing_zip` | `CatBoost_native_categorical`, threshold 0.35 | 0.1310 | 0.8349 | 0.0711 | 놓치는 이탈 고객은 적지만 오탐이 너무 많음 |
| 논문 참고 | 외부 논문 | EasyEnsemble | 0.1290 | 0.3820 | 0.0770 | 논문에서도 F1이 높지 않음 |

정리하면, 최종 모델은 F1 기준으로는 Logistic Regression이 가장 낫고, recall을 많이 올리고 싶으면 BalancedBagging이나 CatBoost를 사용할 수 있습니다. 하지만 recall을 높일수록 precision이 크게 떨어지는 문제가 있었습니다.

### 9.1 주요 성능지표 해석

이 프로젝트에서는 이탈 고객이 전체의 약 6.5%뿐이기 때문에 accuracy만으로 모델을 판단하기 어렵습니다. 그래서 아래 지표들을 함께 사용해 “이탈 고객을 얼마나 잘 찾는지”와 “오탐이 얼마나 많은지”를 같이 확인했습니다.

| 지표 | 의미 | 이 프로젝트에서 보는 이유 |
| --- | --- | --- |
| F1 | precision과 recall의 조화평균입니다. 이탈 고객을 맞히는 능력과 오탐을 줄이는 능력 사이의 균형을 보여줍니다. | 이탈 고객을 어느 정도 잡으면서도 정상 고객을 너무 많이 이탈로 잘못 예측하지 않는 모델을 고르기 위해 사용했습니다. |
| Recall | 실제 이탈 고객 중 모델이 이탈이라고 맞힌 비율입니다. | 이탈 고객을 놓치지 않는 능력을 봅니다. recall이 높으면 더 많은 이탈 고객을 캠페인 대상으로 잡을 수 있습니다. |
| Precision | 모델이 이탈이라고 예측한 고객 중 실제 이탈 고객의 비율입니다. 수식으로는 `TP / (TP + FP)`입니다. | 오탐을 얼마나 줄였는지 봅니다. 예를 들어 모델이 100명을 이탈 고객이라고 예측했는데 실제 이탈 고객이 10명이라면 precision은 10%입니다. precision이 낮으면 정상 고객에게도 불필요한 이탈 방지 캠페인을 많이 하게 됩니다. |
| PR-AUC | threshold를 바꿨을 때 precision과 recall의 관계를 전체적으로 요약한 면적 지표입니다. | 이탈 고객처럼 positive class가 적은 불균형 데이터에서는 ROC-AUC보다 실제 탐지 품질을 더 잘 보여줄 수 있습니다. |
| MCC | TP, TN, FP, FN을 모두 반영하는 상관계수형 지표입니다. 값은 -1부터 1까지이며, 1에 가까울수록 좋은 분류입니다. | class imbalance 상황에서도 한쪽 class만 잘 맞히는 모델을 과대평가하지 않고 전체 confusion matrix의 균형을 확인하기 위해 사용했습니다. |

### 9.2 Threshold와 Confusion Matrix 용어

threshold는 모델이 예측한 이탈 확률을 실제 class로 바꾸는 기준값입니다. 예를 들어 threshold가 0.5라면 모델이 어떤 고객의 이탈 확률을 0.5 이상으로 예측할 때 `CHURN=1`로 판단하고, 0.5 미만이면 `CHURN=0`으로 판단합니다. threshold를 낮추면 더 많은 고객을 이탈로 예측하므로 recall은 올라가기 쉽지만 FP가 늘어 precision이 낮아질 수 있습니다.

이 프로젝트에서는 이탈 고객 `CHURN=1`을 positive, 비이탈 고객 `CHURN=0`을 negative로 두고 해석했습니다.

| 용어 | 의미 | 이탈 예측에서의 해석 |
| --- | --- | --- |
| TP | True Positive | 실제 이탈 고객을 이탈이라고 맞힌 경우입니다. |
| TN | True Negative | 실제 비이탈 고객을 비이탈이라고 맞힌 경우입니다. |
| FP | False Positive | 실제 비이탈 고객을 이탈이라고 잘못 예측한 경우입니다. 오탐이며, 불필요한 이탈 방지 캠페인 비용으로 이어질 수 있습니다. |
| FN | False Negative | 실제 이탈 고객을 비이탈이라고 잘못 예측한 경우입니다. 미탐이며, 실제 이탈 고객을 놓치는 문제입니다. |

따라서 recall은 `TP / (TP + FN)`으로 실제 이탈 고객을 얼마나 놓치지 않았는지 보고, precision은 `TP / (TP + FP)`로 이탈이라고 예측한 고객 중 실제 이탈 고객이 얼마나 되는지 봅니다.

### 9.3 Top-k 캠페인 전략

Top-k 캠페인은 모델이 예측한 이탈 위험 점수 순서대로 고객을 정렬한 뒤, 상위 k% 고객만 캠페인 대상으로 선택하는 방식입니다. 예를 들어 test 고객 1,688명 중 top 10% 캠페인을 한다면, 이탈 위험 점수가 가장 높은 약 169명만 상담, 할인, 유지 캠페인 대상으로 잡습니다.

threshold 방식은 “이탈 확률이 0.35 이상이면 연락한다”처럼 확률 기준을 정하는 방식입니다. 반면 Top-k 방식은 “예산상 상위 10% 고객에게만 연락한다”처럼 운영 가능한 접촉 수를 기준으로 대상을 정하는 방식입니다.

| 방식 | 의미 | 장점 | 주의점 |
| --- | --- | --- | --- |
| Threshold | 예측 확률이 특정 기준 이상인 고객을 선택 | 모델 기준이 명확하고 재현하기 쉽습니다. | 모델 score가 실제 확률로 잘 보정되어 있지 않으면 기준값 해석이 어려울 수 있습니다. |
| Top-k | 위험 점수 상위 k% 고객을 선택 | 예산, 상담 인력, 캠페인 가능 인원에 맞춰 운영하기 쉽습니다. | k를 너무 크게 잡으면 정상 고객까지 많이 접촉해 FP 비용과 고객 피로도가 커질 수 있습니다. |

Top-k 실험에서는 각 모델의 score로 고객을 정렬한 뒤, top 5%, 10%, 20%, 40%처럼 접촉 비율을 바꾸어 TP, FP, Recall@k, Precision@k, 순이익을 비교했습니다. 이 방식은 실제 현업에서 “몇 명에게 연락할 수 있는가”라는 예산 제약과 직접 연결되기 때문에, 단순 threshold보다 운영팀이 이해하기 쉽습니다.

예시 해석:

- 예산이 작으면 top 10%처럼 소수 고객만 선별해야 하므로 precision이 상대적으로 중요한 모델이 유리합니다.
- 예산이 넓어지면 top 30~40%까지 접촉할 수 있어 recall이 높은 BalancedBagging/CatBoost 계열이 더 많은 이탈 고객을 잡을 수 있습니다.
- top 100%는 모든 고객에게 연락하는 것과 같으므로 실제 운영 전략으로는 부적절합니다.

교수님께 설명할 문장:

> Top-k 캠페인은 모델이 위험하다고 본 고객을 순위화한 뒤, 예산에 맞춰 상위 k%만 관리하는 방식입니다. 실제 운영에서는 확률 threshold보다 “이번 달에 상위 10% 고객만 연락하자” 같은 방식이 더 직관적이기 때문에, top-k 실험을 통해 예산 규모별로 어떤 모델이 유리한지 확인했습니다.

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

하지만 현재 접근 가능한 범위에서는 이런 데이터를 추가로 확보하기 어려웠습니다. 따라서 최종 보고서에서는 이 점을 프로젝트의 핵심 한계로 명확히 설명하고, 현재 데이터 안에서 가능한 전처리, 모델 비교, threshold tuning, 오류 분석을 충실히 수행했다는 점을 강조합니다.

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

이 섹션은 발표 중 바로 답변할 수 있도록 질문을 주제별로 정리했습니다. 답변은 길게 외우기보다, 각 질문의 핵심 논리를 이해하고 말하는 방식으로 준비하면 됩니다.

### 13.1 데이터와 전처리 질문

#### Q1. 이 문제를 왜 binary classification이라고 하나요?

정답 class가 `CHURN=1` 이탈과 `CHURN=0` 비이탈 두 가지이기 때문입니다. 모델은 각 고객이 이탈할지, 이탈하지 않을지를 두 class 중 하나로 예측합니다.

#### Q2. minority class가 무엇이고, 왜 중요한가요?

minority class는 데이터에서 수가 적은 class입니다. 이 프로젝트에서는 이탈 고객 `CHURN=1`이 549명, 약 6.5%뿐이므로 minority class입니다. 실제 목적은 이탈 고객을 찾는 것이기 때문에, 수가 적더라도 이 class를 얼마나 잘 탐지하는지가 핵심입니다.

#### Q3. 왜 train/test를 80:20으로 나누었나요?

학습에 사용할 데이터를 충분히 확보하면서도, 학습에 쓰지 않은 별도 test set으로 일반화 성능을 확인하기 위해서입니다. 이탈 고객 수가 적기 때문에 test set을 너무 크게 잡으면 train set의 minority class가 더 줄어들 수 있어 80:20 비율을 사용했습니다.

#### Q4. 왜 `stratify=y`를 사용했나요?

이탈 고객 비율이 낮기 때문에 random split만 하면 train/test의 이탈 비율이 달라질 수 있습니다. `stratify=y`를 사용하면 train과 test 모두 원본과 비슷한 이탈 비율을 유지하므로, 불균형 데이터에서 더 안정적인 평가가 가능합니다.

#### Q5. 결측치가 많은 `Suspended_subscribers`를 왜 삭제하지 않았나요?

결측률은 높지만, 이 컬럼의 빈칸은 “정지 가입자가 없음”을 의미할 가능성이 있습니다. 또한 행을 삭제하면 이미 적은 이탈 고객 표본이 더 줄어듭니다. 그래서 값을 0으로 대체하고, 결측 여부를 나타내는 flag를 추가했습니다.

#### Q6. label encoding과 frequency encoding을 왜 같이 사용했나요?

label encoding은 범주를 숫자로 구분하기 위한 방식이고, frequency encoding은 해당 범주가 얼마나 자주 등장하는지를 반영하는 방식입니다. 두 방식을 함께 사용하면 `SOHO`라는 범주 자체의 정보와, SOHO가 데이터에서 많이 등장하는 대표 고객군이라는 정보를 모두 전달할 수 있습니다.

#### Q7. `Billing_ZIP`은 왜 포함/제외를 모두 실험했나요?

우편번호는 지역 정보를 담을 수 있지만 고유값이 많고, 특정 지역 패턴에 과적합될 위험도 있습니다. 그래서 포함 버전과 제외 버전을 모두 실험했습니다. 최종 F1 기준으로는 제외 버전이 더 좋아서 메인 모델에서는 제외했습니다.

#### Q8. log/sqrt 파생변수는 왜 만들었나요?

매출 변수는 고객마다 차이가 크고 우측으로 긴 분포를 가질 수 있습니다. log와 sqrt 변환은 큰 값을 압축해 이상치 영향을 줄이고, 원본값이 가진 규모 정보와 변환값이 가진 안정적인 패턴을 함께 학습하게 해줍니다.

#### Q9. 총 feature 55개는 어떻게 나온 건가요?

원본 14개 컬럼에서 target인 `CHURN`, 식별자인 `PID`, 기본 모델에서 제외한 `KA_name`을 제외한 뒤, 고객 상태 비율, 매출 비율, 가입자당 매출, 상호작용 feature, binary flag, 결측 flag, log/sqrt 변환, frequency encoding을 추가했습니다. `Billing_ZIP` 포함 기준으로 55개이고, 최종 메인 모델처럼 ZIP을 제외하면 `Billing_ZIP`, `Billing_ZIP_missing`, `Billing_ZIP_frequency`가 빠져 52개가 됩니다.

### 13.2 모델과 성능지표 질문

#### Q10. 왜 accuracy가 높은 모델을 선택하지 않았나요?

이 데이터는 이탈 고객이 약 6.5%뿐이라, 대부분을 비이탈로 예측해도 accuracy가 높게 나올 수 있습니다. 따라서 accuracy보다 이탈 고객을 얼마나 잘 찾는지 보는 recall, precision, F1, PR-AUC와 전체 confusion matrix 균형을 보는 MCC가 더 중요합니다.

#### Q11. 왜 SVMSMOTE를 사용했나요?

이탈 고객이 너무 적어 모델이 비이탈 class 위주로 학습할 수 있기 때문입니다. SVMSMOTE는 단순 복사가 아니라 SVM 결정 경계 주변의 minority class 샘플을 기준으로 합성 데이터를 만들어, 이탈/비이탈 구분이 어려운 영역을 더 학습하게 합니다.

#### Q12. SMOTE 계열을 test set에도 적용했나요?

아닙니다. resampling은 train set에만 적용했습니다. test set까지 resampling하면 실제 운영 환경의 데이터 분포가 왜곡되어 평가가 과장될 수 있습니다. test set은 현실 분포를 유지한 상태로 평가해야 합니다.

#### Q13. 왜 Logistic Regression을 최종 모델로 선택했나요?

최종 F1이 가장 높았고, recall-heavy 모델보다 precision이 상대적으로 안정적이었습니다. 또한 coefficient를 통해 어떤 feature가 이탈 가능성을 높이거나 낮추는지 설명할 수 있어 방어와 해석에 유리했습니다.

#### Q14. 왜 recall이 높은 CatBoost를 최종 모델로 선택하지 않았나요?

CatBoost threshold 0.35는 recall이 0.8349로 높지만 precision이 0.0711입니다. 이탈 고객은 많이 잡지만 정상 고객도 매우 많이 이탈로 잘못 예측합니다. 그래서 메인 모델은 F1이 더 높은 Logistic Regression으로 두고, CatBoost는 recall 극대화 운영 후보로만 제시했습니다.

#### Q15. BalancedBagging은 왜 최종 모델이 아니라 후보인가요?

BalancedBagging은 recall 0.5872로 이탈 고객을 더 많이 잡지만 precision이 0.0877로 낮습니다. 캠페인 대상자를 넓게 잡는 운영 목적에는 쓸 수 있지만, 보고서의 대표 모델로는 오탐 부담이 커서 보조 후보로 두었습니다.

#### Q16. recall을 올리면 왜 precision이 떨어지나요?

threshold를 낮추면 더 많은 고객을 이탈로 예측합니다. 이때 실제 이탈 고객을 더 많이 잡아 recall은 올라가지만, 정상 고객까지 이탈로 잘못 예측하는 FP도 늘어 precision이 떨어집니다. 이 프로젝트에서도 이 trade-off가 뚜렷하게 나타났습니다.

#### Q17. threshold tuning은 왜 했나요?

기본 threshold 0.5만 사용하면 불균형 데이터에서 이탈 고객을 거의 못 잡을 수 있습니다. threshold를 조정하면 recall과 precision의 균형을 바꿀 수 있으므로, 운영 목적에 맞는 의사결정 기준을 찾기 위해 threshold tuning을 수행했습니다.

#### Q18. feature importance와 coefficient는 어떻게 다르게 해석하나요?

feature importance는 어떤 feature가 모델 성능에 많이 기여했는지를 보여줍니다. coefficient는 Logistic Regression에서 해당 feature가 이탈 가능성을 높이는 방향인지 낮추는 방향인지를 보여줍니다. 즉, feature importance는 “무엇이 중요한가”, coefficient는 “어떤 방향으로 영향을 주는가”를 보는 데 사용했습니다.

#### Q19. 실험 A~G는 각각 무엇을 검증한 건가요?

실험 A~G는 단순한 추가 실험이 아니라 최종 결론을 방어하기 위한 검증입니다. A는 ZIP 포함 여부, B는 feature group 기여도, C는 세그먼트별 오류, D는 비용별 threshold, E는 논문형/확장 feature variant, F는 추가 모델과 soft ensemble, G는 bootstrap과 McNemar 기반 통계 검증을 확인했습니다.

#### Q20. Top-k 캠페인은 무엇인가요?

Top-k 캠페인은 모델 점수가 높은 고객부터 순위를 매긴 뒤, 예산에 맞춰 상위 k% 고객만 캠페인 대상으로 고르는 방식입니다. 예를 들어 top 10%는 위험 점수가 가장 높은 10% 고객에게만 연락하는 전략입니다. 실제 운영에서는 “확률 0.35 이상”보다 “이번 달 상위 10%만 관리”가 더 직관적이기 때문에 top-k 분석을 추가했습니다.

### 13.3 한계와 방어 질문

#### Q21. F1이 낮은데 모델이 의미 있다고 볼 수 있나요?

절대적인 성능은 높지 않지만, 이 프로젝트의 의미는 높은 점수 자체보다 불균형 churn 데이터에서 어떤 지표로 평가하고 어떤 trade-off가 발생하는지 분석한 데 있습니다. 여러 모델과 전처리, feature engineering, threshold tuning을 수행했음에도 성능이 제한적이었기 때문에, 원인이 모델 부족보다 데이터 feature 한계에 있다는 결론을 제시할 수 있습니다.

#### Q22. 성능 한계의 가장 큰 원인은 무엇인가요?

현재 데이터가 정적 CRM snapshot에 가깝기 때문입니다. 고객 이탈은 최근 사용량 감소, 결제 실패, 불만 증가, 약정 종료 같은 시간 기반 행동 변화와 관련이 큰데, 현재 데이터에는 이런 시계열 행동 feature가 없습니다.

#### Q23. 성능을 올리려면 어떤 데이터가 추가로 필요하나요?

월별 사용량 변화, 최근 매출 감소 추세, 납부 지연 이력, 고객센터 문의/불만 기록, 약정 만료 시점, 상품 변경 또는 해지 이력 같은 시간 기반 행동 데이터가 필요합니다. 이런 feature가 있어야 이탈 직전의 행동 패턴을 더 직접적으로 학습할 수 있습니다.

#### Q24. 왜 복잡한 모델보다 Logistic Regression이 더 좋았나요?

현재 feature가 이탈 원인을 직접적으로 설명하지 못하는 상황에서는 복잡한 모델이 추가 패턴을 찾기보다 noise를 학습할 수 있습니다. Logistic Regression은 파생변수로 펼친 신호를 단순하고 안정적으로 반영했고, 이 데이터에서는 그 균형이 F1 기준으로 가장 좋았습니다.

#### Q25. 이 프로젝트를 유지하는 이유는 무엇인가요?

이 프로젝트는 단순히 높은 성능만 보여주는 프로젝트가 아니라, 불균형 데이터에서 모델을 어떻게 평가하고 해석하는지 보여주는 프로젝트입니다. 성능 한계를 숨기지 않고 class imbalance, feature 한계, threshold trade-off를 근거로 설명하는 것이 오히려 더 설득력 있는 분석 방향입니다.

#### Q26. 이 프로젝트에서 배운 점은 무엇인가요?

불균형 데이터에서는 accuracy가 의미 없을 수 있고, recall을 높이면 precision이 급격히 낮아지는 trade-off가 발생한다는 점을 확인했습니다. 또한 고객 이탈 예측에는 단순한 정적 정보보다 시간 기반 행동 데이터가 중요하다는 결론을 얻었습니다.

## 14. 교수님께 말할 최종 요약

> 저희는 최종 프로젝트를 통신사 고객 이탈 예측으로 유지하기로 했습니다. 원본 데이터 구조를 확인하고, 결측치 처리, 중복 제거, 파생변수 생성, SVMSMOTE, 여러 불균형 대응 모델, CatBoost native categorical 처리, threshold tuning까지 수행했습니다. 그 결과 F1 기준 최종 모델은 `without_billing_zip + LogisticRegression_SMOTE`였고 F1은 0.1681, recall은 0.2661이었습니다. recall을 높이는 모델도 있었지만 precision이 0.07~0.09 수준으로 낮아 false positive가 많이 증가했습니다. 분석 결과 이 데이터는 정적인 CRM snapshot 중심이라 이탈 예측에 중요한 사용량 변화, 결제 실패, 불만 기록, 약정 종료 같은 시간 기반 행동 feature가 부족합니다. 따라서 최종 보고서에서는 모델별 trade-off와 데이터 한계를 명확히 설명하고, 목적에 따라 F1 기준 모델과 recall 중심 운영 후보를 구분해 제시하겠습니다.

추가로 짧게 덧붙일 문장:

> 성능 수치가 높지는 않지만, 이 프로젝트는 심한 class imbalance에서 어떤 지표를 보고 모델을 선택해야 하는지 보여주는 분석으로 정리하겠습니다.

## 15. 최종 결론

이 churn 데이터는 다음과 같이 정리할 수 있습니다.

- 데이터는 B2B 통신사 고객 단위의 정적 CRM snapshot입니다.
- target인 churn은 약 6.5%로 매우 불균형합니다.
- 주요 신호는 매출 규모, 가입자 활동 상태, 고객 세그먼트입니다.
- 다양한 모델과 불균형 처리 방법을 적용했지만 F1과 recall 개선이 제한적이었습니다.
- 성능 한계의 핵심 원인은 모델 부족이 아니라 시간 기반 행동 feature 부족입니다.
- 성능 개선에 필요한 시간 기반 고객 행동 데이터를 추가로 확보하기 어려웠습니다.
- 따라서 최종 프로젝트는 ChurnRadar로 유지하되, 보고서에서는 데이터 한계, 모델별 trade-off, 운영 목적별 모델 선택을 핵심 결론으로 제시합니다.
