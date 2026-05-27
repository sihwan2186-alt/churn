# Phase 5-B: 비즈니스 임팩트 완전 정량 분석

## 1. 비용-편익 파라미터

본 분석은 Makokha et al. (2026)의 비용-편익 프레임워크를 그대로 적용하여 논문과 직접 비교 가능한 비즈니스 임팩트를 계산하였다.

| 항목 | 값 | 의미 |
| --- | ---: | --- |
| 연간 이탈 손실 | 5,400 | 이탈 고객 1명당 연간 매출 손실 |
| Retention 성공률 | 60% | 캠페인으로 회수 가능한 손실 비율 |
| TP benefit | 3,240 | `5,400 × 0.60` |
| FP campaign cost | 120 | 비이탈 고객에게 캠페인을 집행한 비용 |
| 논문 보고 순이익 | 74,200 | Makokha et al.의 EasyEnsemble 기준값 |

주의할 점은 원본 CSV의 `ARPU`는 월평균 약 25 수준이며, 이탈 고객의 단순 연환산 ARPU도 약 300 수준이라는 점이다. 따라서 본 보고서의 5,400 단위는 CSV의 원시 `ARPU × 12`를 그대로 대체한 값이 아니라, 논문이 사용한 비즈니스 비용 가정을 동일 적용한 비교용 파라미터로 해석해야 한다.

실제 데이터 프로파일:

| 데이터 | 집단 | 계정 수 | 평균 ARPU | 중앙 ARPU | 평균 TotalRevenue |
| --- | --- | ---: | ---: | ---: | ---: |
| 원본 | 전체 | 8,453 | 24.44 | 19.32 | 148.83 |
| 원본 | 이탈 | 549 | 25.00 | 19.23 | 173.81 |
| PID 중복 제거 | 전체 | 8,436 | 24.44 | 19.30 | 148.66 |
| PID 중복 제거 | 이탈 | 545 | 24.97 | 19.20 | 173.96 |

## 2. 공식

논문과 동일하게 baseline 대비 추가 순이익을 다음 공식으로 계산하였다.

```text
Net Benefit = TP × 3,240 - FP × 120
```

여기서 FN은 “모델이 없었을 때도 잃는 고객”으로 baseline 손실에 이미 포함되므로, 추가 순이익 공식에서는 직접 차감하지 않는다.

논문 수치를 recall/precision에서 역산하면 `TP=42`, `FP=503`, `FN=68`이다.

```text
42 × 3,240 - 503 × 120 = 75,720
```

논문 보고값은 74,200으로 약 1,520 차이가 있는데, 이는 논문에 공개된 precision/recall이 반올림된 값이어서 TP/FP를 역산할 때 생기는 차이로 해석하는 것이 가장 자연스럽다. 비교 기준선은 논문 보고값 74,200을 그대로 사용하였다.

또한 본 문서의 ROI는 두 가지로 구분한다.

| 지표 | 공식 | 비고 |
| --- | --- | --- |
| Gross ROI | `TP benefit / FP cost` | 사용자 초안의 `LR 3.78x`와 같은 정의 |
| Net ROI | `Net Benefit / FP cost` | 순이익 기준의 더 보수적 ROI |

## 3. 모델별 순이익 비교

| 운영점 | TP | FP | FN | 접촉 수 | Recall | 순이익 | 논문 대비 | Gross ROI | Net ROI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 논문 EasyEnsemble | 42 | 503 | 68 | 545 | 38.2% | 75,720 | 1.02x | 2.25x | 1.25x |
| LR fixed | 29 | 207 | 80 | 236 | 26.6% | 69,120 | 0.93x | 3.78x | 2.78x |
| BalancedBagging fixed | 64 | 666 | 45 | 730 | 58.7% | 127,440 | 1.72x | 2.59x | 1.59x |
| CatBoost recall-heavy | 91 | 1,189 | 18 | 1,280 | 83.5% | 152,160 | 2.05x | 2.07x | 1.07x |
| XGBoost recall-heavy | 101 | 1,402 | 8 | 1,503 | 92.7% | 159,000 | 2.14x | 1.95x | 0.95x |
| BalancedBagging cost-opt | 109 | 1,561 | 0 | 1,670 | 100.0% | 165,840 | 2.24x | 1.89x | 0.89x |

핵심 해석은 두 층으로 나누는 것이 좋다.

첫째, 사용자가 지정한 핵심 3개 운영점(LR, BalancedBagging, CatBoost)만 비교하면 CatBoost가 152,160 단위로 최대 순이익을 달성한다. 이는 논문 보고값 74,200의 약 2.05배다.

둘째, Phase 3-A에서 추가한 XGBoost와 Phase 3-B의 비용 최적 threshold까지 포함하면, 최종 최대 순이익은 BalancedBagging threshold 0.29의 165,840 단위다. 이 운영점은 test churner 109명을 모두 포착하지만, 전체 1,688명 중 1,670명에게 캠페인을 보내는 매우 공격적인 전략이므로 실제 운영에서는 팀 역량과 고객 피로도를 함께 고려해야 한다.

## 4. 운영 시나리오별 권장 모델

| 시나리오 | 제약/목표 | 권장 운영점 | 접촉 수 | 순이익 | 논문 대비 |
| --- | --- | --- | ---: | ---: | ---: |
| 예산 제한 | 최대 500건 접촉 | LR fixed | 236 | 69,120 | 0.93x |
| 팀 역량 균형 | 최대 800건 접촉 | BalancedBagging fixed | 730 | 127,440 | 1.72x |
| 핵심 3모델 무제한 | LR/BalancedBagging/CatBoost 중 최대 순이익 | CatBoost recall-heavy | 1,280 | 152,160 | 2.05x |
| 확장 모델 포함 | XGBoost까지 포함 | XGBoost recall-heavy | 1,503 | 159,000 | 2.14x |
| 비용 최적 threshold | paper baseline 비용 기준 순이익 최대화 | BalancedBagging cost-opt | 1,670 | 165,840 | 2.24x |

예산이 엄격하면 LR이 가장 실용적이다. 논문보다 절대 순이익은 5,080 낮지만, 접촉 수가 236건으로 논문 EasyEnsemble 545건의 43% 수준이며 Gross ROI는 3.78x로 가장 높다.

팀이 약 800건까지 처리할 수 있다면 BalancedBagging fixed가 가장 좋은 절충점이다. 접촉 수 730건으로 운영 가능 범위에 들어오면서 논문 대비 72% 높은 순이익을 제공한다.

매출 보호가 최우선이고 접촉 비용/고객 피로도가 상대적으로 덜 중요하다면 CatBoost 또는 XGBoost 같은 recall-heavy 운영점이 더 적합하다. 비용 최적 threshold까지 허용하면 BalancedBagging threshold 0.29가 가장 큰 순이익을 보이지만, 실무에서는 캠페인 대상이 거의 전체 test set으로 확장되는 점을 반드시 명시해야 한다.

## 5. Break-Even 분석

손익분기 조건은 다음과 같다.

```text
FP_max = TP × (3,240 / 120) = TP × 27
```

| 운영점 | TP | 실제 FP | 최대 허용 FP | 여유 FP | 순이익 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 논문 EasyEnsemble | 42 | 503 | 1,134 | +631 | 75,720 |
| LR fixed | 29 | 207 | 783 | +576 | 69,120 |
| BalancedBagging fixed | 64 | 666 | 1,728 | +1,062 | 127,440 |
| CatBoost recall-heavy | 91 | 1,189 | 2,457 | +1,268 | 152,160 |
| XGBoost recall-heavy | 101 | 1,402 | 2,727 | +1,325 | 159,000 |
| BalancedBagging cost-opt | 109 | 1,561 | 2,943 | +1,382 | 165,840 |

모든 운영점이 break-even을 크게 상회한다. 이는 논문 비용 가정에서 FP 비용 120이 TP benefit 3,240에 비해 매우 작기 때문이다. 즉 이 비용 구조에서는 precision을 희생하더라도 recall을 높이는 전략이 순이익 관점에서 유리해진다.

## 6. 최종 보고서용 서술

본 연구에서 Makokha et al. (2026)과 동일한 비용-편익 분석 프레임워크(연간 이탈 손실 5,400 단위, retention 성공률 60%, FP 캠페인 비용 120 단위)를 적용한 결과, 모델 운영 목적에 따라 서로 다른 최적점이 도출되었다. 캠페인 예산이 500건 이하로 제한된 환경에서는 Logistic Regression이 236건의 접촉만으로 69,120 단위의 순이익과 3.78배의 Gross ROI를 달성하여 가장 효율적인 운영안으로 나타났다. 반면 약 800건의 캠페인 처리가 가능한 균형형 운영에서는 BalancedBagging이 127,440 단위의 순이익으로 논문 기준값 대비 1.72배의 개선을 보였다. 매출 보호를 최우선으로 하는 경우, 핵심 3개 운영점 중 CatBoost가 152,160 단위의 순이익을 달성하여 논문 보고값 74,200 대비 약 2.05배의 비즈니스 임팩트를 제공하였다. 더 나아가 비용 민감 threshold 최적화까지 허용하면 BalancedBagging threshold 0.29가 165,840 단위의 최대 순이익을 보였으나, 이는 1,670건의 대규모 캠페인 접촉을 전제로 하므로 실제 배포에서는 고객 피로도와 운영 자원 제약을 함께 고려해야 한다.

## 7. 산출물

| 파일 | 내용 |
| --- | --- |
| `processed/phase_5b_business_impact/arpu_financial_profile.csv` | 원본/PID 중복 제거 데이터의 ARPU 및 매출 프로파일 |
| `processed/phase_5b_business_impact/business_impact_operating_points.csv` | 모델 운영점별 TP/FP/FN, 순이익, ROI, break-even 지표 |
| `processed/phase_5b_business_impact/business_impact_scenarios.csv` | 예산/팀역량/무제한/비용최적 시나리오별 권장 모델 |
| `processed/phase_5b_business_impact/business_impact_break_even.csv` | 운영점별 최대 허용 FP 및 안전 여유 |
| `processed/phase_5b_business_impact/business_impact_bubble.png` | TP-FP 기반 비즈니스 임팩트 버블 차트 |
| `processed/phase_5b_business_impact/business_impact_scenarios.png` | 운영 시나리오별 권장 모델 막대 차트 |
| `processed/phase_5b_business_impact/business_impact_dashboard.png` | 발표용 1페이지 비즈니스 임팩트 대시보드 |
| `processed/phase_5b_business_impact/phase_5b_business_impact_summary.json` | 핵심 수치 요약 JSON |
