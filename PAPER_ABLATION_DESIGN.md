# Paper Reproduction vs Extension Ablation

마지막 업데이트: 2026-05-27

## 핵심 보정

사용자가 정리한 A안/B안 설계를 실제 `Baza customer Telecom v2.csv` 기준으로 검증했다.

- 원본 CSV 크기: `8,453 x 14`
- 컬럼 14개에는 `CHURN` target이 포함된다.
- `PID` 중복 17건 제거 후: `8,436 x 14`
- target 분포: `No=7,891`, `Yes=545`
- stratified 80:20 split 후 test rows: `1,688`

논문 설명의 `14 raw -> 22 final`은 현재 CSV만으로는 그대로 재현할 수 없다. 현재 CSV에서 `PID`와 `CHURN`을 제외하면 실제 모델 입력 raw column은 12개다.

따라서 이 프로젝트에서 재현 가능한 논문형 core는 다음처럼 정의했다.

| 구성 | 피처 수 |
| --- | ---: |
| ZIP 포함 core: 12 raw + 8 engineered | 20 |
| ZIP 제외 core: 11 raw + 8 engineered | 19 |
| ZIP 포함 확장: 20 + 5 interaction/rank | 25 |
| ZIP 제외 확장: 19 + 5 interaction/rank | 24 |
| ZIP 포함 core + KA 추상화 | 21 |
| ZIP top-50 확장 + KA 추상화 | 26 |
| ZIP top-50 확장 + KA 연구용 target encoding | 29 |

## 생성 스크립트

```powershell
.\.venv\Scripts\python.exe paper_ablation_variants.py
```

출력 위치:

- `processed/paper_ablation_variants/variant_summary.csv`
- `processed/paper_ablation_variants/summary_all.json`
- `processed/paper_ablation_variants/<variant>/X_train.csv`
- `processed/paper_ablation_variants/<variant>/X_test.csv`
- `processed/paper_ablation_variants/<variant>/X_train_resampled.csv`
- `processed/paper_ablation_variants/<variant>/summary.json`
- `processed/paper_ablation_variants/<variant>/feature_columns.json`

## Variant 목록

| Variant | 의미 | Feature count |
| --- | --- | ---: |
| `paper_core_zip_log` | 논문형 core, ZIP label encoding, revenue log | 20 |
| `paper_core_no_zip_log` | 논문형 core, ZIP 제외 | 19 |
| `paper_core_zip_top50_log` | ZIP top-50 + Other grouping | 20 |
| `extended_zip_log_sqrt_interactions` | ZIP 포함, revenue log + subscriber sqrt + 확장 interaction | 25 |
| `extended_no_zip_log_sqrt_interactions` | ZIP 제외, 확장 interaction | 24 |
| `extended_zip_top50_log_sqrt_interactions` | ZIP top-50 grouping + 확장 interaction | 25 |
| `paper_core_zip_log_ka_abstract` | 논문형 core에서 `KA_name` 실명 제거, KA 추상 피처 2개 추가 | 21 |
| `extended_zip_top50_log_sqrt_ka_abstract` | 확장 variant에서 `KA_name` 실명 제거, KA 추상 피처 2개 추가 | 26 |
| `extended_zip_top50_log_sqrt_ka_research` | 연구용 KA target/frequency encoding 5개 추가 | 29 |

## Core 피처

Raw input:

- `CRM_PID_Value_Segment`
- `EffectiveSegment`
- `KA_name`
- `Billing_ZIP` 또는 ZIP 제외 variant에서는 제거
- `Active_subscribers`
- `Not_Active_subscribers`
- `Suspended_subscribers`
- `Total_SUBs`
- `AvgMobileRevenue`
- `AvgFIXRevenue`
- `TotalRevenue`
- `ARPU`

Engineered:

- `active_rate`
- `inactive_rate`
- `suspended_rate`
- `risk_score`
- `mobile_revenue_ratio`
- `fixed_revenue_ratio`
- `revenue_per_subscriber`
- `revenue_x_active_rate`

## 확장 피처

- `revenue_x_risk`
- `inactive_x_fixed_ratio`
- `suspended_x_mobile_ratio`
- `arpu_per_active`
- `total_rev_rank_by_segment`

`total_rev_rank_by_segment`는 train split 안에서 segment별 `TotalRevenue` empirical percentile 기준을 fit하고, test에는 train 기준만 적용한다.

## KA_name 처리 전략

실제 데이터에서 `KA_name`은 코드형 8개와 실명형 4개로 나뉜다.

- 코드형: `AD`, `AD?`, `DI`, `MT`, `RJ`, `VM`, `VT`, `VU`
- 실명형: 위 코드형이 아닌 담당자명
- raw 기준 이탈율: 코드형 `6.21%`, 실명형 `7.09%`
- Platinum/Gold/SME에서는 실명형 KA의 이탈율이 더 높게 나타난다.

그래서 실험 축을 3단계로 분리했다.

| KA mode | 최종 feature에서 raw `KA_name` 포함 | 추가 피처 | 용도 |
| --- | --- | --- | --- |
| `label` | 포함 | 없음 | 논문 재현 |
| `abstract` | 제거 | `KA_is_code_type`, `KA_type_x_premium` | 배포 친화/윤리 위험 낮음 |
| `research_full` | 제거 | abstract 2개 + `KA_churn_rate_encoded`, `KA_customer_count`, `KA_avg_portfolio_revenue` | 연구용 성능 검증 |

`research_full`의 `KA_churn_rate_encoded`는 target-dependent feature이므로 반드시 train split 또는 CV fold 안에서만 fit해야 한다. 현재 스크립트는 training row에는 leave-one-out KA 이탈율을 넣고, test에는 train의 `KA_name -> CHURN mean`만 매핑한다. test의 미지 KA는 train 전체 이탈율로 대체한다.

## Leakage 방지 구현

- PID 중복 제거는 split 전에 수행한다.
- `CHURN`과 `PID`는 feature matrix에서 제거한다.
- `Not_Active_subscribers`, `Suspended_subscribers`는 train/test 모두 0 대체하되 target을 쓰지 않는다.
- `ARPU`, `Billing_ZIP` median은 train에서만 계산해 test에 적용한다.
- label encoder는 train에서만 fit하고 test unseen category는 `Unknown`으로 보낸다.
- ZIP top-N grouping은 train frequency 기준으로 top 50을 정한다.
- KA target/frequency/portfolio 집계는 해당 variant에서 train 기준으로만 계산하며, train row의 KA target encoding은 leave-one-out으로 자기 label을 제외한다.
- `StandardScaler`는 train에서만 fit한다.
- `SVMSMOTE`는 train에만 fit/apply한다.
