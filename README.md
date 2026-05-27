# ChurnRadar

`Baza customer Telecom v2.csv`를 사용한 B2B 통신사 고객 이탈 예측 프로젝트입니다.

## 현재 결론

프로젝트는 다른 주제로 변경하지 않고 ChurnRadar로 유지합니다. 핵심 성과는 논문 baseline 재현과, 운영 목적별 모델 선택 프레임워크 제시입니다.

- 논문 EasyEnsemble F1 `0.129`를 우리 EasyEnsemble F1 `0.128`로 재현
- hold-out F1 최고: `without_billing_zip + LogisticRegression_SMOTE`
- CV 안정성: `with_billing_zip + BalancedBagging_original` 및 EasyEnsemble 계열이 더 안정적
- 비용-편익 기준: 예산/팀 역량/매출 보호 목표에 따라 최적 모델이 달라짐
- 1시간 발표용 추가 케이스: top-k 예산, 비용 threshold, calibration, segment ROI, 모델 합의도 분석 완료

## 핵심 모델 요약

| 목적 | Variant | Model | Threshold | F1 | Recall | Precision | 순이익 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| F1 기준 최종 | `without_billing_zip` | `LogisticRegression_SMOTE` | 0.50 | 0.1681 | 0.2661 | 0.1229 | 69,120 |
| 균형형 운영 | `with_billing_zip` | `BalancedBagging_original` | 0.50 | 0.1526 | 0.5872 | 0.0877 | 127,440 |
| 핵심 3모델 recall-heavy | `with_billing_zip` | `CatBoost_native_categorical` | 0.35 | 0.1310 | 0.8349 | 0.0711 | 152,160 |
| 확장 recall-heavy | `with_billing_zip` | `XGBoost_SMOTE` | 0.16 | 0.1253 | 0.9266 | 0.0672 | 159,000 |
| 비용 최적 threshold | `with_billing_zip` | `BalancedBagging_original` | 0.29 | 0.1225 | 1.0000 | 0.0653 | 165,840 |
| 논문 기준 | external | EasyEnsemble | 0.35 | 0.1290 | 0.3820 | 0.0770 | 74,200 |

## 지금 남은 일

1. `PRESENTATION_SLIDES.md`를 기반으로 실제 PPT 제작
2. 발표에 `processed/phase_5b_business_impact/business_impact_dashboard.png` 삽입
3. 1시간 발표에서는 `PHASE_6_EXTENDED_CASE_STUDIES.md`의 추가 케이스 5장을 뒤에 배치
4. 교수님 질문 대비용으로 `CHURN_DATA_MODEL_DEFENSE.md`와 Phase 4/5/6 문서 확인
5. 최종 제출 전 주요 스크립트 문법 확인

```powershell
.\.venv\Scripts\python.exe -m py_compile preprocess_churn.py
.\.venv\Scripts\python.exe -m py_compile phase_3b_differentiation_experiments.py
.\.venv\Scripts\python.exe -m py_compile phase_4_cross_validation.py
.\.venv\Scripts\python.exe -m py_compile phase_5a_interpretability.py
.\.venv\Scripts\python.exe -m py_compile phase_5b_business_impact.py
```

## 주요 문서

| 파일 | 역할 |
| --- | --- |
| `FINAL_REPORT.md` | 최신 Phase 3-5 결과가 통합된 최종 보고서 |
| `PRESENTATION_SLIDES.md` | 실제 PPT 제작용 슬라이드 구성안 |
| `PHASE_3A_REPRODUCTION_EXPERIMENTS.md` | 논문 재현 및 XGBoost/EasyEnsemble 보강 |
| `PHASE_3B_DIFFERENTIATION_EXPERIMENTS.md` | ZIP, feature ablation, segment, cost threshold 실험 |
| `PHASE_4_PAPER_COMPARISON_FRAMEWORK.md` | 논문 비교 프레임워크와 CV 안정성 분석 |
| `PHASE_5A_FEATURE_IMPORTANCE_AND_SHAP_ALTERNATIVES.md` | LR 계수 기반 설명가능성 분석 |
| `PHASE_5B_BUSINESS_IMPACT_ANALYSIS.md` | 비용-편익 및 운영 시나리오 분석 |
| `PHASE_6_EXTENDED_CASE_STUDIES.md` | 1시간 발표용 추가 비교 실험 케이스 |
| `CHURN_DATA_MODEL_DEFENSE.md` | 예상 질문 방어 자료 |

## 주요 산출물 폴더

| 폴더 | 내용 |
| --- | --- |
| `processed/model_a_with_billing_zip/` | ZIP 포함 전처리 데이터 |
| `processed/model_b_without_billing_zip/` | ZIP 제외 전처리 데이터 |
| `processed/phase_3b_differentiation/` | 차별화 실험 결과 |
| `processed/phase_4_paper_comparison/` | 5-fold CV 결과 |
| `processed/phase_5a_interpretability/` | 계수/기여도/해석 가능성 결과 |
| `processed/phase_5b_business_impact/` | 비용-편익 표와 발표용 이미지 |
| `processed/phase_6_extended_case_studies/` | top-k, calibration, segment, 모델 합의도 추가 실험 |

## 최종 메시지

이 프로젝트의 결론은 “한 모델이 모든 기준에서 최고”가 아니라, “불균형 churn 데이터에서는 F1, recall, precision, 비용, 캠페인 역량에 따라 운영 모델을 다르게 선택해야 한다”입니다.
