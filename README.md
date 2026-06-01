# ChurnRadar

`Baza customer Telecom v2.csv` 기반 B2B 통신사 고객 이탈 예측 프로젝트입니다.

## 먼저 볼 것

| 목적 | 파일 |
| --- | --- |
| 최종 발표 PPT | `deliverables/presentations/ChurnRadar_Detailed_Presentation_Polished.pptx` |
| 발표 대본 | `docs/03_presentation/PRESENTATION_SCRIPT_POLISHED.md` |
| 최종 보고서 | `docs/01_reports/FINAL_REPORT.md` |
| 예상 질문 방어 자료 | `docs/01_reports/CHURN_DATA_MODEL_DEFENSE.md` |
| 핵심 모델 요약표 | `final_model_summary.csv` |
| 제출용 묶음 | `submission_package_20260531/` |

## 핵심 결론

시계열 데이터를 찾을 수 없어 정적 CRM 스냅샷 기준으로 진행했습니다. 이 조건에서 ChurnRadar의 결론은 단일 최고 모델을 고르는 것이 아니라, F1, recall, precision, 비용, 예산, 캠페인 운영 역량에 따라 모델과 threshold를 다르게 선택해야 한다는 것입니다.

- 논문 EasyEnsemble baseline F1 `0.129`를 재현값 F1 `0.128`로 확인
- Hold-out F1 최고: `without_billing_zip + LogisticRegression_SMOTE`
- CV 안정성: `with_billing_zip + BalancedBagging_original` 및 EasyEnsemble 계열이 더 안정적
- Recall 중심 운영: CatBoost/XGBoost 계열은 더 많은 이탈자를 잡지만 FP가 커짐
- 비용/예산 기준: top-k, cost threshold, segment에 따라 최적 운영점이 달라짐

## 폴더 구조

| 위치 | 내용 |
| --- | --- |
| `deliverables/presentations/` | 사람이 바로 열어볼 최종 PPT 3종 |
| `docs/00_overview/` | 프로젝트 상태, 파일 요약, 전체 진행 기록 |
| `docs/01_reports/` | 최종 보고서와 질문 대응 문서 |
| `docs/02_experiments/` | Phase별 실험 문서와 ablation 설계 |
| `docs/03_presentation/` | 발표 구성안, 리허설, cue card, 최종 대본 |
| `docs/04_operations/` | MLOps와 n8n Docker 자동화 문서 |
| `docs/05_references/` | 참고 논문과 보조 PDF |
| `processed/` | 전처리/실험/그래프 산출물 전체 |
| `presentation_assets/` | PPT에 사용한 주요 이미지 |
| `n8n_automation/` | Docker runner와 n8n workflow |
| `tools/presentation/` | 발표 PPT 문구/대본 정리 도구 |
| `submission_package_20260531/` | 제출용으로 추려 둔 패키지 |

원본 CSV와 핵심 Python 실행 스크립트는 여러 코드가 루트 기준 경로를 사용하므로 루트에 둡니다. 대신 문서와 발표 산출물은 하위 폴더로 분리했습니다.

## 재현 실행

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q tests\test_data_integrity.py
```

주요 산출물 재생성:

```powershell
.\.venv\Scripts\python.exe -u preprocess_churn.py
.\.venv\Scripts\python.exe -u phase_3b_differentiation_experiments.py
.\.venv\Scripts\python.exe -u phase_4_cross_validation.py
.\.venv\Scripts\python.exe -u phase_5a_interpretability.py
.\.venv\Scripts\python.exe -u phase_5b_business_impact.py
.\.venv\Scripts\python.exe -u phase_6_extended_case_studies.py
.\.venv\Scripts\python.exe -u phase_7_next_experiments.py
.\.venv\Scripts\python.exe -u phase_8_statistical_validation.py
.\.venv\Scripts\python.exe -u make_presentation_assets.py
.\.venv\Scripts\python.exe -u make_final_ppt.py
.\.venv\Scripts\python.exe -u make_detailed_ppt.py
.\.venv\Scripts\python.exe -u tools\presentation\polish_churnradar_presentation.py
```

생성된 발표 PPT는 `deliverables/presentations/`에 저장됩니다.
