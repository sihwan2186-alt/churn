# 오늘 작업: 데이터 이해, 컬럼 분리, 논문 비교 확인 [김시환]

## 1. 생성 결과

원본 `Baza customer Telecom v2.csv`는 수정하지 않았다.

새 결과물 위치:

- `processed/column_split_datasets/00_base_without_pid_ka.csv`
- `processed/column_split_datasets/01_column_churn_pairs/`
- `processed/column_split_datasets/02_category_value_subsets/`
- `processed/column_split_datasets/03_profiles/`
- `processed/column_split_datasets/04_column_preprocessed/`
- `processed/column_split_datasets/05_single_column_model_screening.csv`
- `processed/column_split_datasets/05_single_column_best_models.csv`

모델용/분리용 데이터에서는 `PID`, `KA_name`을 제외했다.

## 2. 컬럼별 분리 방식

각 원본 피처를 `피처 + CHURN` 형태로 분리했다.

예시:

- `CRM_PID_Value_Segment_churn.csv`
- `EffectiveSegment_churn.csv`
- `Billing_ZIP_churn.csv`
- `Active_subscribers_churn.csv`
- `TotalRevenue_churn.csv`

범주형 값별 데이터도 별도 생성했다.

예시:

- `CRM_PID_Value_Segment/Bronze_churn.csv`
- `CRM_PID_Value_Segment/Gold_churn.csv`
- `EffectiveSegment/SOHO_churn.csv`
- `EffectiveSegment/VSE_churn.csv`
- `Billing_ZIP/zip_4000_churn.csv`

## 2-1. 컬럼별 Yes/No 이탈률 확인 CSV

각 컬럼마다 `Yes`, `No`, 전체 수, 이탈률을 바로 볼 수 있는 파일을 추가 생성했다.

위치:

- `processed/column_split_datasets/06_yes_no_rate_by_column/`
- `processed/column_split_datasets/03_profiles/all_column_value_yes_no_rate_summary.csv`
- `processed/column_split_datasets/03_profiles/category_value_yes_no_rate_summary.csv`
- `processed/column_split_datasets/03_profiles/numeric_bins_yes_no_rate_summary.csv`
- `processed/column_split_datasets/03_profiles/column_yes_no_rate_summary.csv`

예시 파일:

- `CRM_PID_Value_Segment_yes_no_rate.csv`: Bronze, Gold, Platinum 등급별 Yes/No/이탈률
- `EffectiveSegment_yes_no_rate.csv`: SOHO, VSE, SME 등 세그먼트별 Yes/No/이탈률
- `Total_SUBs_yes_no_rate.csv`: 전체 가입자 수 구간별 Yes/No/이탈률
- `Billing_ZIP_yes_no_rate.csv`: 우편번호별 Yes/No/이탈률

## 2-2. 한글 컬럼명 및 간단 설명 추가

원본 `Baza customer Telecom v2.csv`는 수정하지 않았다.

사람이 보기 쉬운 한글명/설명 파일만 새로 추가했다.

위치:

- `processed/column_split_datasets/00_data_dictionary_korean.csv`
- `processed/column_split_datasets/00_data_dictionary_korean.md`
- `processed/column_split_datasets/07_korean_readable_summaries/`
- `processed/column_split_datasets/07_korean_readable_summaries/yes_no_rate_by_column_ko/`

예시:

- `CRM_PID_Value_Segment` → `CRM 고객가치 등급`
- `EffectiveSegment` → `실질 비즈니스 세그먼트`
- `Billing_ZIP` → `청구 우편번호`
- `Active_subscribers` → `활성 가입자 수`
- `Not_Active_subscribers` → `비활성 가입자 수`
- `Suspended_subscribers` → `정지 가입자 수`
- `Total_SUBs` → `전체 가입자 수`
- `AvgMobileRevenue` → `평균 모바일 매출`
- `AvgFIXRevenue` → `평균 유선 매출`
- `TotalRevenue` → `총 매출`
- `ARPU` → `가입자당 평균 매출`
- `CHURN` → `이탈 여부`

## 3. 각각의 데이터별 전처리 방향

### 범주형 데이터

대상:

- `CRM_PID_Value_Segment`
- `EffectiveSegment`
- `Billing_ZIP`

전처리:

- 결측 여부 flag
- frequency encoding
- label encoding
- 최종 모델에서는 CatBoost native categorical 또는 leakage-safe target encoding 권장

주의:

- `Billing_ZIP`은 고유값이 456개라 one-hot만 쓰면 차원이 커진다.
- 이탈율 기반 target encoding은 반드시 train fold 안에서만 계산해야 한다.

### 수치형 데이터

대상:

- `Active_subscribers`
- `Not_Active_subscribers`
- `Suspended_subscribers`
- `Total_SUBs`
- `AvgMobileRevenue`
- `AvgFIXRevenue`
- `TotalRevenue`
- `ARPU`

전처리:

- 결측 여부 flag
- median imputation
- `log1p`
- z-score
- IQR outlier flag

주의:

- 가입자 수 변수도 강한 우측 왜곡이 있어 논문보다 넓게 로그 변환을 적용할 필요가 있다.
- `Suspended_subscribers`, `Not_Active_subscribers`는 결측 자체가 신호일 수 있으므로 단순 0 대체만 하면 정보가 손실될 수 있다.

## 4. 단일 컬럼 모델 스크리닝 결과

단일 컬럼만 사용한 빠른 성능 한계 확인 결과, 가장 좋은 신호는 아래 순서였다.

| 컬럼 | 단일 컬럼 기준 best model | F1 | Recall | Precision | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| `Total_SUBs` | DecisionTree_balanced | 0.1498 | 0.3909 | 0.0927 | 0.0781 |
| `AvgMobileRevenue` | LogisticRegression_balanced | 0.1441 | 0.4545 | 0.0856 | 0.0837 |
| `Active_subscribers` | LogisticRegression_balanced | 0.1426 | 0.4000 | 0.0868 | 0.0829 |
| `TotalRevenue` | LogisticRegression_balanced | 0.1416 | 0.4455 | 0.0842 | 0.0843 |
| `EffectiveSegment` | DecisionTree_balanced | 0.1346 | 0.3273 | 0.0847 | 0.0714 |
| `CRM_PID_Value_Segment` | LogisticRegression_balanced | 0.1276 | 0.5818 | 0.0717 | 0.0696 |
| `Billing_ZIP` | DecisionTree_balanced | 0.1276 | 0.7000 | 0.0702 | 0.0822 |
| `Not_Active_subscribers` | DecisionTree_balanced | 0.1259 | 0.2455 | 0.0846 | 0.0727 |

해석:

- 단일 컬럼만으로는 F1이 0.15 근처에서 멈춘다.
- 즉 최종 모델은 단일 변수보다 `active_rate`, `inactive_ratio`, `CRM x Segment`, `zip risk`, `outlier flag`, `missing flag` 같은 교차/파생 피처가 필요하다.
- 단일 컬럼 기준으로는 가입자 수와 매출 규모가 가장 강하지만, 이는 그대로 쓰기보다 활용률과 이상치 flag로 바꾸는 편이 더 좋다.

## 5. 논문 확인 결과

로컬 논문 파일:

- `j.ajnc.20261501.12.pdf`

논문 핵심:

- 데이터: 8,454 unique business accounts, 14 raw attributes
- 최종 feature set: 22 variables
- 이탈 비율: 약 6.5%, 불균형비 14.3:1
- resampling: SVMSMOTE 선택
- 최종 best model: EasyEnsembleClassifier
- 논문 성능: F1 0.129, Recall 약 0.382
- 주요 설명 변수: active subscriber rate, geographic billing zone, engineered interaction terms

논문 전처리:

- `PID` 기준 중복 제거
- `Not_Active_subscribers`, `Suspended_subscribers` 결측은 0 대체
- CRM 고객가치 등급 결측은 Unknown 처리
- `Billing_ZIP`, `ARPU` 결측은 median imputation
- 범주형 변수는 label encoding
- revenue 계열 변수에 로그 변환
- 수치형 변수는 z-score scaling
- train/test는 80:20 stratified split

## 6. 논문 대비 우리 방향

차별화 포인트:

- 논문은 `PID` 중복 제거에 사용했고, `KA_name`은 범주형 feature로 포함했다. 이번 작업에서는 요청대로 `PID`, `KA_name`을 모델 데이터에서 제외한다.
- 논문은 `Billing_ZIP`을 label encoding으로 처리했지만, 우리는 zip별 이탈율과 고카디널리티 특성을 따로 확인하고 target encoding/CatBoost를 검토한다.
- 논문은 가입자 수 변수 로그 변환을 강조하지 않았지만, 우리 EDA에서는 가입자 수 변수도 강한 우측 왜곡을 보였다.
- 논문은 결측값을 대부분 대체했지만, 우리는 결측 여부 자체를 별도 feature로 보존한다.
- 논문은 CRM 등급과 세그먼트의 교차 위험을 별도로 강조하지 않았지만, 우리는 `CRM_PID_Value_Segment x EffectiveSegment`를 핵심 교차 피처로 본다.

## 7. 다음 작업

1. 분리된 컬럼별 CSV를 기준으로 각 컬럼의 전처리 타당성 검토
2. `PID`, `KA_name` 제외 조건을 고정한 최종 feature set 구성
3. 논문 방식 재현 모델과 우리 방식 모델을 같은 split에서 비교
4. 성능 한계 분석: 단일 컬럼, 논문 feature set, 우리 파생 feature set을 단계별 비교
5. 최종 보고서에 논문 대비 개선점 정리
