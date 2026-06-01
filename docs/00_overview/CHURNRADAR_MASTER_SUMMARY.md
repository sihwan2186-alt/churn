# ChurnRadar 통합 정리 문서

생성일: 2026-06-01

이 문서는 작업 폴더 안의 Markdown 문서를 중복 없이 통합한 마스터 요약이다. 루트 문서, `archive/`, `processed/`의 보조 문서, `submission_package_20260531/`의 제출 패키지 안내를 함께 반영했다. `submission_package_20260531/docs/` 안의 문서들은 루트 문서와 내용이 동일한 제출용 복사본이므로, 본문에는 한 번만 병합했다.

현재 파일 위치는 2026-06-01 정리 이후 `README.md`와 `docs/README.md`를 기준으로 본다. 최종 발표 파일은 `deliverables/presentations/ChurnRadar_Detailed_Presentation_Polished.pptx`이고, 발표 대본은 `docs/03_presentation/PRESENTATION_SCRIPT_POLISHED.md`이다.

## 1. 전체 결론

ChurnRadar는 `Baza customer Telecom v2.csv` 기반의 B2B 통신사 고객 이탈 예측 프로젝트다. 최종 메시지는 단일 최고 모델을 찾는 것이 아니라, 심한 class imbalance 환경에서 F1, recall, precision, 비용, 캠페인 예산에 따라 모델과 threshold를 다르게 선택해야 한다는 운영 프레임워크를 제시하는 것이다.

핵심 결론:

- 참고 논문 Makokha et al. (2026)의 EasyEnsemble baseline F1 `0.129`를 우리 EasyEnsemble F1 `0.128` 수준으로 재현했다.
- hold-out F1 기준 최고 모델은 `without_billing_zip + LogisticRegression_SMOTE`, F1 `0.1681`이다.
- 5-fold CV 안정성 기준으로는 `with_billing_zip + BalancedBagging_original` 및 EasyEnsemble 계열이 더 안정적이다.
- 이탈 고객을 최대한 많이 잡는 목적에서는 CatBoost/XGBoost recall-heavy 운영점이 유리하지만 false positive가 크게 늘어난다.
- 비용-편익 관점에서는 캠페인 비용, 고객 가치, 상담 가능 인원에 따라 최적 모델과 threshold가 달라진다.
- 데이터는 정적 CRM snapshot이라 월별 사용량 변화, 결제 실패, 계약 만료, VOC 같은 시간 기반 feature가 부족하다. 성능 한계의 핵심 원인은 모델 부족보다 feature 한계에 가깝다.

## 2. 데이터와 문제 정의

| 항목 | 값 |
| --- | ---: |
| 원본 파일 | `Baza customer Telecom v2.csv` |
| 원본 크기 | 8,453 rows x 14 columns |
| PID 중복 제거 후 | 8,436 rows |
| 원본 CHURN=Yes | 549 |
| 원본 CHURN=No | 7,904 |
| 이탈 비율 | 약 6.5% |
| 문제 유형 | binary classification |

이탈 고객이 약 6.5%뿐이므로 accuracy는 핵심 지표가 아니다. 모델이 대부분을 비이탈로 예측해도 accuracy가 높게 나올 수 있기 때문에 F1, recall, precision, PR-AUC, MCC, confusion matrix, 비용 기준 순이익을 함께 봐야 한다.

주요 컬럼:

| 컬럼 | 의미 | 처리 방향 |
| --- | --- | --- |
| `PID` | 고객 식별자 | 중복 제거 후 학습 feature에서 제외 |
| `KA_name` | 담당 키 어카운트 매니저 | 기본 모델에서는 제외, 연구용 KA 추상화 variant만 별도 검토 |
| `CRM_PID_Value_Segment` | 고객 가치 등급 | 결측 `Unknown`, 오타 `Sliver`는 `Silver`로 통합 |
| `EffectiveSegment` | 실질 비즈니스 세그먼트 | SOHO/VSE/SME 등 고객군 차이 반영 |
| `Billing_ZIP` | 청구 우편번호 | 포함/제외/top-N grouping variant 비교 |
| 가입자 수/매출 변수 | 활동 상태와 수익성 | log/sqrt, ratio, interaction feature 생성 |
| `CHURN` | 이탈 여부 | `No=0`, `Yes=1` 변환 후 target으로 사용 |

## 3. 전처리와 Leakage 방지

전처리 원칙:

1. `PID` 기준 중복 제거는 train/test split 전에 수행한다.
2. `CHURN`은 target으로 분리하고 feature matrix에서 제거한다.
3. 80:20 stratified train/test split을 사용한다.
4. 결측치 대치, label/frequency encoding, scaling은 train split에만 fit한다.
5. SVMSMOTE는 train partition에만 적용하고 test set에는 적용하지 않는다.
6. threshold는 validation split에서 선택하고 test set에는 1회만 적용한다.
7. target/frequency encoding이 필요한 KA 연구용 variant는 train fold 내부에서만 계산한다.

결측 처리:

| 컬럼 | 결측률 | 처리 |
| --- | ---: | --- |
| `Suspended_subscribers` | 약 95.84% | missing flag 생성 후 0 대체 |
| `Not_Active_subscribers` | 약 49.08% | missing flag 생성 후 0 대체 |
| `CRM_PID_Value_Segment` | 약 0.06% | `Unknown` 대체 |
| `Billing_ZIP` | 약 0.02% | 포함 variant에서 train median 대체 |
| `ARPU` | 약 0.01% | `TotalRevenue / Total_SUBs` 보정 후 train median 대체 |

## 4. Feature Engineering

논문 기반 feature와 프로젝트 확장 feature를 함께 구성했다.

| 그룹 | 예시 |
| --- | --- |
| 원시 매출 | `AvgMobileRevenue`, `AvgFIXRevenue`, `TotalRevenue`, `ARPU` |
| 원시 가입자 수 | `Active_subscribers`, `Not_Active_subscribers`, `Suspended_subscribers`, `Total_SUBs` |
| 비율 feature | `active_rate`, `inactive_rate`, `suspended_rate`, `dormant_rate`, `risk_score` |
| 매출 효율 | `revenue_per_subscriber`, `revenue_per_active_subscriber` |
| 상호작용 | `revenue_engagement_interaction`, `arpu_risk_interaction`, `inactive_revenue_interaction` |
| 변환 feature | log/sqrt revenue transform |
| 범주형 | CRM segment, EffectiveSegment, Billing_ZIP 포함/제외/top-N variant |

단일 컬럼 모델 스크리닝에서는 `Total_SUBs`, `AvgMobileRevenue`, `Active_subscribers`, `TotalRevenue` 등이 상대적으로 강했지만, 단일 컬럼만으로는 F1이 약 0.15 근처에서 멈췄다. 따라서 최종 모델은 결측 flag, 이상치 flag, 활동률, 매출 변환, segment/ZIP/interaction feature를 함께 사용해야 한다.

## 5. 논문 재현과 비교 원칙

비교는 세 층으로 분리한다.

| 비교 층 | 의미 |
| --- | --- |
| 재현 비교 | 논문 best 모델인 EasyEnsemble끼리 비교 |
| 추가 발견 비교 | 논문에 없던 LR_SMOTE, ZIP ablation, cost threshold 등을 별도 기여로 제시 |
| 운영 지점 비교 | F1-best, recall-heavy, cost-sensitive threshold는 목적이 다르므로 직접 우열로만 해석하지 않음 |

논문과 직접 비교 가능한 핵심 결과:

| 기준 | Model | F1 | Recall | Precision | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| 논문 | EasyEnsemble | 0.129 | 0.382 | 0.077 | 0.079 |
| 우리 재현 | EasyEnsemble original with ZIP | 0.128 | 0.587 | 0.072 | 0.085 |

올바른 표현:

> 본 프로젝트는 EasyEnsemble 기준으로 논문 결과를 재현했고, 추가 모델/feature/threshold 실험을 통해 다른 운영 목적에서 경쟁력 있는 대안을 제시했다.

피해야 할 표현:

- 우리 모델이 논문보다 압도적으로 좋다.
- F1 `0.1681`과 recall `0.9266`을 같은 모델의 동시 성능처럼 말한다.
- cost-opt threshold가 실제 운영에서도 무조건 최선이다.
- raw score를 실제 이탈 확률처럼 말한다.

## 6. 핵심 모델 결과

| 목적 | Variant | Model | Threshold | F1 | Recall | Precision | 순이익 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| F1 기준 최종 | `without_billing_zip` | `LogisticRegression_SMOTE` | 0.50 | 0.1681 | 0.2661 | 0.1229 | 69,120 |
| 균형형 운영 | `with_billing_zip` | `BalancedBagging_original` | 0.50 | 0.1526 | 0.5872 | 0.0877 | 127,440 |
| 핵심 3모델 recall-heavy | `with_billing_zip` | `CatBoost_native_categorical` | 0.35 | 0.1310 | 0.8349 | 0.0711 | 152,160 |
| 확장 recall-heavy | `with_billing_zip` | `XGBoost_SMOTE` | 0.16 | 0.1242 | 0.9266 | 0.0665 | 157,200 |
| 비용 최적 threshold | `with_billing_zip` | `BalancedBagging_original` | 0.29 | 0.1225 | 1.0000 | 0.0653 | 165,840 |
| 논문 기준 | external | EasyEnsemble | 0.35 | 0.1290 | 0.3820 | 0.0770 | 74,200 |

5-fold CV 요약:

| Variant | Model | CV F1 mean | CV F1 SD | Recall mean | Precision mean |
| --- | --- | ---: | ---: | ---: | ---: |
| with ZIP | `BalancedBagging_original` | 0.1455 | 0.0126 | 0.5248 | 0.0845 |
| with ZIP | `EasyEnsemble_original` | 0.1445 | 0.0117 | 0.5835 | 0.0824 |
| without ZIP | `EasyEnsemble_original` | 0.1408 | 0.0081 | 0.5835 | 0.0801 |
| without ZIP | `LogisticRegression_SMOTE` | 0.1309 | 0.0154 | 0.1743 | 0.1053 |

해석:

- LR은 hold-out F1과 precision이 높고 설명 가능성이 좋다.
- BalancedBagging/EasyEnsemble은 CV 안정성과 recall 균형이 좋다.
- CatBoost/XGBoost는 이탈 고객 포착률이 높지만 FP가 크다.
- 최종 선택은 하나의 모델이 아니라 목적별 운영점 선택이다.

## 7. 차별화 실험

Phase 3-B 핵심:

| 실험 | 결론 |
| --- | --- |
| Billing ZIP ablation | ZIP은 tree ensemble recall/F1을 높였지만 LR에서는 고카디널리티 noise처럼 작동해 F1을 낮췄다. |
| Feature group ablation | LR은 categorical group 제거 시 F1이 크게 하락했고, BalancedBagging은 interaction group 제거에 가장 민감했다. |
| CRM segment analysis | high-value는 recall보다 FP/precision 문제가 크고, low-value는 recall 문제가 컸다. |
| Cost threshold sensitivity | 논문 비용 가정에서는 high-recall 전략이 유리하지만, 비용 구조가 보수적이면 캠페인 가치가 음수로 바뀔 수 있다. |

주요 수치:

- LR에서 `G_categorical` 제거 시 F1 `0.1681 -> 0.0806`
- BalancedBagging에서 `G_interaction` 제거 시 F1 `0.1526 -> 0.1344`
- high-value 전용 모델은 precision/PR-AUC를 개선하지만 recall이 크게 낮아져 two-stage review 방식이 더 적합하다.

## 8. Interpretability

논문은 EasyEnsemble에 SHAP/LIME을 적용했다. 본 프로젝트는 최종 F1 모델인 Logistic Regression의 intrinsic interpretability를 활용해 coefficient, permutation importance, local logit contribution을 계산했다.

상위 해석 신호:

| 기준 | 주요 feature |
| --- | --- |
| Permutation FI | `AvgMobileRevenue_sqrt`, `TotalRevenue_sqrt`, `revenue_engagement_interaction` |
| LR coefficient | `AvgFIXRevenue_log`, `AvgMobileRevenue_sqrt`, `fixed_revenue_per_subscriber` |
| Local contribution | `AvgMobileRevenue_sqrt`, `TotalRevenue_sqrt`, `ARPU_sqrt` |
| 단변량 correlation | `dormant_subscribers`, `Not_Active_subscribers`, `Total_SUBs` |

논문 SHAP에서 active subscriber rate가 1위였고, 우리 LR에서는 revenue transform이 상위로 나온 것은 모순이 아니다. 모델 구조와 중요도 측정 방식이 다르기 때문이다. 두 결과는 모두 서비스 참여도와 매출 패턴의 결합이 이탈 예측에 중요하다는 결론으로 수렴한다.

## 9. 비즈니스 임팩트

논문과 동일한 비용-편익 가정을 사용했다.

| 항목 | 값 |
| --- | ---: |
| 이탈 고객 1명 연간 손실 | 5,400 |
| Retention 성공률 | 60% |
| TP benefit | 3,240 |
| FP campaign cost | 120 |
| 순이익 공식 | `TP x 3,240 - FP x 120` |

운영 시나리오:

| 시나리오 | 권장 운영점 | 접촉 수 | 순이익 | 해석 |
| --- | --- | ---: | ---: | --- |
| 예산 500건 이하 | LR fixed | 236 | 69,120 | 가장 적은 접촉과 높은 Gross ROI |
| 팀 역량 800건 이하 | BalancedBagging fixed | 730 | 127,440 | recall과 비용의 균형 |
| 매출 보호 우선 | CatBoost recall-heavy | 1,280 | 152,160 | 높은 recall, 높은 접촉 수 |
| XGBoost 포함 확장 | XGBoost recall-heavy | 1,518 | 157,200 | 더 높은 recall, 더 많은 FP |
| 비용 최적 threshold | BalancedBagging threshold 0.29 | 1,670 | 165,840 | 최대 순이익이나 거의 전체 고객 접촉 |

중요한 해석:

- 논문 비용 구조에서는 FP 비용이 낮아 recall 극대화 전략이 순이익상 유리하게 보인다.
- 실제 운영에서는 고객 피로도, 상담 인력, 브랜드 비용을 같이 고려해야 한다.
- 비용 구조가 보수적으로 바뀌면 precision이 높은 LR 또는 높은 threshold 전략이 더 안전해진다.

## 10. 추가 케이스 스터디

Phase 6은 1시간 발표를 위해 운영 관점의 비교 케이스를 확장했다.

| 케이스 | 핵심 결과 |
| --- | --- |
| Top-k budget | 예산이 작으면 LR/EasyEnsemble, 40% 이상에서는 BalancedBagging이 유리 |
| Cost scenario sensitivity | 비용 구조가 바뀌면 최적 threshold와 모델이 바뀜 |
| Calibration | raw score는 실제 churn probability를 과대평가하며 Platt/isotonic 보정 필요 |
| Segment ROI | 모델 강점과 실패 패턴이 low/mid/high value segment별로 다름 |
| Model agreement | 8개 모델 모두 경고한 고객군의 실제 이탈률은 12.43%, 전체 평균의 약 1.9배 |

Calibration 핵심:

- 실제 test churn rate는 6.46%다.
- raw score 평균은 LR 34.6%, BalancedBagging 46.7%, CatBoost 41.4%, XGBoost 46.9%로 과대평가되어 있었다.
- Platt calibration 후 평균 score는 실제 churn rate에 가깝게 조정되었다.

## 11. 추가 후보 실험과 통계 검정

Phase 7:

- 추가 실험 best였던 `BalancedBagging_tree_depthnone_leaf25`는 hold-out F1 `0.1605`였지만 CV 평균은 `0.1418`로 기존 `BalancedBagging_original` `0.1455`보다 낮았다.
- paper/KA ablation 최고 F1은 `paper_core_zip_log_ka_abstract + BalancedBagging_original`의 `0.1561`로 기존 최종 F1 `0.1681`을 넘지 못했다.
- 최종 모델이나 핵심 결론을 바꿀 근거는 없었다.

Phase 8:

| Case | F1 point | F1 95% CI | Recall point | Recall 95% CI | Precision point | Precision 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LR_no_zip_f1 | 0.1681 | [0.1155, 0.2229] | 0.2661 | [0.1818, 0.3524] | 0.1229 | [0.0823, 0.1674] |
| BalancedBagging_with_zip | 0.1526 | [0.1200, 0.1862] | 0.5872 | [0.4947, 0.6814] | 0.0877 | [0.0679, 0.1090] |
| CatBoost_native_with_zip | 0.1310 | [0.1072, 0.1557] | 0.8349 | [0.7624, 0.9000] | 0.0711 | [0.0574, 0.0856] |
| XGBoost_with_zip | 0.1242 | [0.1020, 0.1455] | 0.9266 | [0.8739, 0.9717] | 0.0665 | [0.0541, 0.0789] |

McNemar paired test는 주요 모델들의 error pattern이 서로 다름을 보였지만, class imbalance 문제에서는 McNemar만으로 최종 모델을 고르면 안 된다. 보조 근거로만 사용한다.

## 12. 발표 자료와 방어 전략

발표 자료는 요약 17장 PPT와 상세 36장 PPT로 구성되어 있다. 요약본은 제한 시간 발표용이고, 상세본은 데이터 사용/제외/수정, 전처리, 논문 비교, 실험, 하이퍼파라미터 튜닝, 컬럼별 해석을 모두 설명하는 방어/긴 발표용이다.

| 파일 | 용도 |
| --- | --- |
| `deliverables/presentations/ChurnRadar_Final_Presentation.pptx` | 요약 발표본 17장 |
| `deliverables/presentations/ChurnRadar_Detailed_Presentation.pptx` | 상세 발표본 36장 |

| 구간 | 슬라이드 | 핵심 |
| --- | --- | --- |
| 문제 정의 | 1-3 | 데이터 불균형, leakage 방지, 평가 지표 |
| 논문 재현/모델 비교 | 4-6 | EasyEnsemble 재현, hold-out과 CV 차이 |
| 차별화/해석 | 7-8 | ZIP, feature group, segment, interpretability |
| 운영 의사결정 | 9-14 | 비용, top-k, calibration, segment, risk tier |
| 통계/한계/결론 | 15-17 | bootstrap CI, 데이터 한계, 최종 메시지 |

발표 핵심 문장:

> 이 프로젝트의 결론은 단일 최고 모델이 아니라, 불균형 churn 데이터에서 운영 목적별로 모델과 threshold를 선택하는 프레임워크입니다.

예상 질문 방어:

- F1이 낮은 이유: 이탈률 6.5%의 강한 불균형과 정적 CRM snapshot 한계.
- 최종 모델: 하나만 고르면 hold-out F1 기준 LR이지만, 운영에서는 목적별 선택이 맞음.
- XGBoost를 최종으로 밀지 않는 이유: recall은 높지만 FP가 1,417명으로 커 비용과 고객 피로도를 고려해야 함.
- TreeSHAP 미구현: LR 계수, permutation importance, local contribution, segment 분석으로 설명 가능성을 일관되게 확보.

## 13. 자동화와 MLOps

구현된 자동화:

| 구성 | 내용 |
| --- | --- |
| n8n workflow | `n8n_automation/churnradar_n8n_workflow.json` |
| Docker runner | `n8n_automation/churn_runner.py`, `Dockerfile.runner`, `docker-compose.yml` |
| 실행 순서 | `/health` -> `/run/full-reproduction` -> `/summary` |
| 보안 | `/summary`, `/run/*` endpoint에 `X-API-KEY` 헤더 적용 |
| 드리프트 | `monitor_drift.py`의 PSI 기반 drift check |
| 테스트 | `tests/test_data_integrity.py` |

운영 고도화 로드맵:

- 새 데이터에서 F1이 0.14 아래로 떨어지거나 PSI가 0.2를 넘으면 재학습 검토.
- 최근 6개월 또는 12개월 rolling window 기반 재학습 권장.
- 원본 데이터와 대량 `processed/` 산출물은 Git보다 DVC 또는 외부 스토리지로 관리.
- 실제 예측 API에서는 요청자, 예측 점수, 모델 버전, 이후 실제 이탈 여부를 로그로 남겨야 한다.

## 14. 파일 구조와 역할

핵심 제출/발표 파일:

| 파일 | 역할 |
| --- | --- |
| `README.md` | 프로젝트 현재 결론과 주요 문서 안내 |
| `docs/01_reports/FINAL_REPORT.md` | 제출용 최종 보고서 본문 |
| `docs/03_presentation/PRESENTATION_SLIDES.md` | 요약 17장 발표 흐름과 멘트 |
| `deliverables/presentations/ChurnRadar_Final_Presentation.pptx` | 요약 발표 PowerPoint 17장 |
| `deliverables/presentations/ChurnRadar_Detailed_Presentation.pptx` | 상세 발표 PowerPoint 36장 |
| `docs/01_reports/CHURN_DATA_MODEL_DEFENSE.md` | 교수님 예상 질문 방어 자료 |
| `PROJECT_STATUS_CURRENT.md` | 최신 진행 상태 |
| `NEXT_ACTIONS.md` | 제출 전 필수 확인과 향후 보강 |
| `PROJECT_FILE_SUMMARY.md` | 전체 파일과 산출물 역할 요약 |

주요 스크립트:

| 스크립트 | 역할 |
| --- | --- |
| `preprocess_churn.py` | 전처리, 모델 비교, threshold tuning, 오류 분석, feature importance |
| `paper_ablation_variants.py` | 논문형 core/확장/KA/ZIP ablation variant 생성 |
| `phase_3b_differentiation_experiments.py` | ZIP, feature group, segment, cost threshold 실험 |
| `phase_4_cross_validation.py` | 5-fold CV 안정성 비교 |
| `phase_5a_interpretability.py` | LR 계수, contribution, PDP 분석 |
| `phase_5b_business_impact.py` | 비용-편익, ROI, dashboard 생성 |
| `phase_6_extended_case_studies.py` | top-k, calibration, segment, model agreement |
| `phase_7_next_experiments.py` | 추가 후보와 paper-ablation benchmark |
| `phase_8_statistical_validation.py` | bootstrap CI와 McNemar test |
| `make_final_ppt.py` | 요약 17장 PPT 생성 |
| `make_detailed_ppt.py` | 상세 36장 PPT 생성 |
| `monitor_drift.py` | PSI 기반 데이터 drift 점검 |

중요 산출물 폴더:

| 폴더 | 내용 |
| --- | --- |
| `processed/model_a_with_billing_zip/` | ZIP 포함 기본 전처리 split |
| `processed/model_b_without_billing_zip/` | ZIP 제외 기본 전처리 split |
| `processed/paper_ablation_variants/` | 논문형/확장/KA/ZIP variant split |
| `processed/phase_3b_differentiation/` | 차별화 실험 결과 |
| `processed/phase_4_paper_comparison/` | CV fold/result summary |
| `processed/phase_5a_interpretability/` | 계수/기여도/PDP/해석 가능성 이미지 |
| `processed/phase_5b_business_impact/` | ROI, break-even, dashboard |
| `processed/phase_6_extended_case_studies/` | top-k, calibration, segment, agreement |
| `processed/phase_8_statistical_validation/` | bootstrap CI와 McNemar 결과 |
| `submission_package_20260531/` | 제출용으로 필요한 파일만 선별한 패키지 |

## 15. 중복 및 불필요 파일 정리 결과

이번 정리에서 실제 삭제한 파일:

- `.pytest_cache/`
- 루트 및 하위 폴더의 `__pycache__/` 7개
- `runner.log`

PowerPoint를 열어둔 상태에서는 `~$ChurnRadar_Detailed_Presentation.pptx` 같은 Office 임시 잠금 파일이 보일 수 있다. 이 파일은 실제 발표 자료가 아니며 PowerPoint를 닫으면 삭제 대상이다.

삭제하지 않은 중복:

| 항목 | 판단 |
| --- | --- |
| `submission_package_20260531/docs/`의 루트 문서 복사본 | 제출 패키지 재현성을 위해 보존. 통합 문서에서는 한 번만 반영 |
| `submission_package_20260531/scripts/`, `processed_selected/`, `presentation_assets/` 복사본 | 제출 패키지 구성 목적이 있어 보존 |
| 여러 variant의 `y_train.csv`, `y_test.csv`, `y_train_resampled.csv` | 같은 split을 공유해 내용이 같지만 각 variant 재현성을 위해 보존 |
| `.venv/` | 로컬 실행 환경이며 `requirements.txt`로 재생성 가능하지만, 현재 테스트/실행에 필요할 수 있어 보존 |
| `processed/column_split_datasets/07_korean_readable_summaries/00_data_dictionary_korean.md` | 원본 사전과 동일하지만 한글 요약 폴더의 사용자용 복사본이므로 보존 |

정리 기준:

- 실행 캐시와 로그처럼 재생성 가능하고 프로젝트 의미가 없는 파일만 삭제했다.
- 제출 패키지, variant별 데이터, 발표 자산처럼 중복이더라도 목적이 있는 복사본은 삭제하지 않았다.
- 중복 Markdown 내용은 이 문서에서 병합해 반복을 제거했다.

## 16. 남은 확인 사항

제출 전 필수:

1. `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
2. `.\.venv\Scripts\python.exe -m pytest -q tests\test_data_integrity.py`
3. `deliverables/presentations/ChurnRadar_Final_Presentation.pptx`와 `deliverables/presentations/ChurnRadar_Detailed_Presentation.pptx`를 열어 학교 양식, 팀원 이름, 제목, 표/그래프 잘림 여부 확인
4. n8n UI에서 `ChurnRadar Docker Reproduction Pipeline` 수동 실행 확인
5. 원본 CSV 제출 가능 여부를 교수님/제출 지침에 맞춰 결정

하면 좋은 보강:

- `docs/01_reports/CHURN_DATA_MODEL_DEFENSE.md` 기준으로 예상 질문 10분 리허설
- 보고서 결론과 PPT 결론 문장 일치 확인
- n8n API key를 기본값이 아닌 긴 랜덤 문자열로 교체
- 새 데이터가 생기면 `monitor_drift.py --current 새파일.csv` 실행

더 하지 않아도 되는 일:

- 새 주제로 프로젝트를 바꾸는 일
- 모델을 무작정 더 추가하는 일
- F1과 recall 수치를 서로 다른 운영점에서 가져와 한 모델 성능처럼 합쳐 말하는 일
- 비용 최적 threshold를 실제 운영에서도 항상 최선이라고 주장하는 일

## 17. 반영한 Markdown 문서

중복 제거 후 본문에 반영한 주요 문서:

| 문서 | 반영 내용 |
| --- | --- |
| `README.md` | 현재 결론, 핵심 모델, 주요 문서/산출물 |
| `docs/01_reports/FINAL_REPORT.md` | 최종 보고서 구조와 핵심 수치 |
| `PROJECT_STATUS_CURRENT.md` | 최신 진행 현황 |
| `PROJECT_FILE_SUMMARY.md` | 파일 역할과 산출물 구조 |
| `PROJECT_PROGRESS.md` | 날짜별 진행 흐름과 검증 기록 |
| `docs/01_reports/CHURN_DATA_MODEL_DEFENSE.md` | 데이터/모델 방어 논리 |
| `PHASE_3A_REPRODUCTION_EXPERIMENTS.md` | 논문 재현 |
| `PHASE_3B_DIFFERENTIATION_EXPERIMENTS.md` | ZIP/feature/segment/cost 실험 |
| `PHASE_4_PAPER_COMPARISON_FRAMEWORK.md` | 논문 비교 원칙과 CV |
| `PHASE_5A_FEATURE_IMPORTANCE_AND_SHAP_ALTERNATIVES.md` | 해석 가능성 |
| `PHASE_5B_BUSINESS_IMPACT_ANALYSIS.md` | 비용-편익 |
| `PHASE_6_EXTENDED_CASE_STUDIES.md` | top-k, calibration, segment, agreement |
| `PHASE_7_NEXT_EXPERIMENTS.md` | 추가 후보 점검 |
| `PHASE_8_STATISTICAL_VALIDATION.md` | bootstrap CI, McNemar |
| `PAPER_ABLATION_DESIGN.md` | paper-core/KA/ZIP variant 설계 |
| `TODAY_COLUMN_SPLIT_AND_PAPER_CHECK_KIM_SIHWAN.md` | 컬럼 분리와 논문 확인 |
| `RESEARCH_PRESENTATION_MATERIAL_KIM_SIHWAN.md` | 초기 연구 발표 흐름 |
| `docs/03_presentation/PRESENTATION_SLIDES.md` | 17장 요약 발표 구성 |
| `PRESENTATION_REHEARSAL.md` | 발표 리허설 노트 |
| `PRESENTATION_REHEARSAL_RUN_20260531.md` | 리허설 1회차 기록 |
| `PRESENTATION_CUE_CARD.md` | 발표 직전 큐카드 |
| `NEXT_ACTIONS.md` | 제출 전/후 액션 |
| `docs/04_operations/MLOPS_ROADMAP.md` | 운영 배포 로드맵 |
| `docs/04_operations/N8N_DOCKER_WORKFLOW_GUIDE.md` | n8n Docker 실행 가이드 |
| `archive/*.md` | 프로젝트 유지 결정, 팀 보고, 추가 실험 기록 |
| `processed/column_split_datasets/*.md` | 컬럼 사전과 단일 컬럼 분석 |
| `processed/research_presentation/slide_outline.md` | 초기 슬라이드 흐름 |
| `submission_package_20260531/SUBMISSION_PACKAGE_README.md` | 제출 패키지 구성 |

## 18. 최종 한 문장

ChurnRadar는 Makokha et al. (2026)의 baseline을 재현한 뒤, feature 처리, threshold, 비용 구조, segment, calibration, 통계 검정을 통해 불균형 churn 데이터에서 모델을 목적별로 선택하는 운영 프레임워크를 제시한 프로젝트다.
