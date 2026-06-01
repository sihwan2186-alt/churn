# ChurnRadar 파일 구조 요약

마지막 업데이트: 2026-06-01

## 한 줄 요약

이 프로젝트는 `Baza customer Telecom v2.csv` 기반 B2B 통신사 고객 이탈 예측 프로젝트입니다. 시계열 데이터를 찾을 수 없어 정적 CRM 스냅샷 기준으로 진행했으며, 최종 결론은 단일 최고 모델이 아니라 운영 목적별 모델/threshold 선택 프레임워크입니다.

## 처음 보는 사람이 열 파일

| 목적 | 파일 |
| --- | --- |
| 최종 발표 PPT | `deliverables/presentations/ChurnRadar_Detailed_Presentation_Polished.pptx` |
| 발표 대본 | `docs/03_presentation/PRESENTATION_SCRIPT_POLISHED.md` |
| 최종 보고서 | `docs/01_reports/FINAL_REPORT.md` |
| 질문 대비 | `docs/01_reports/CHURN_DATA_MODEL_DEFENSE.md` |
| 전체 요약 | `docs/00_overview/CHURNRADAR_MASTER_SUMMARY.md` |
| 핵심 모델 표 | `final_model_summary.csv` |

## 최상위 폴더 역할

| 위치 | 내용 |
| --- | --- |
| `deliverables/presentations/` | 최종 PPT와 발표용 PPT 원본 |
| `docs/` | 사람이 읽는 모든 Markdown/PDF 문서 |
| `processed/` | 전처리 데이터, 실험 결과 CSV/JSON/PNG |
| `presentation_assets/` | PPT에 들어가는 핵심 이미지 |
| `n8n_automation/` | Docker runner와 n8n workflow |
| `submission_package_20260531/` | 제출용으로 추린 패키지 |
| `tools/` | 발표 정리 등 보조 도구 |
| `tests/` | 데이터 무결성 테스트 |
| `archive/` | 이전 의사결정과 보조 기록 |

## 루트에 남겨 둔 파일

| 파일/패턴 | 이유 |
| --- | --- |
| `Baza customer Telecom v2.csv` | 핵심 스크립트들이 루트 기준으로 원본 CSV를 참조 |
| `preprocess_churn.py`, `phase_*.py` | 재현 실행 진입점 |
| `make_final_ppt.py`, `make_detailed_ppt.py` | PPT 자동 생성 진입점 |
| `requirements.txt`, `pytest.ini` | 실행 환경 설정 |
| `final_model_summary.csv` | README와 발표에서 바로 확인하는 핵심 요약표 |

## 문서 폴더

| 폴더 | 내용 |
| --- | --- |
| `docs/00_overview/` | 프로젝트 상태, 전체 요약, 진행 기록, 다음 작업 |
| `docs/01_reports/` | 최종 보고서와 모델/데이터 방어 문서 |
| `docs/02_experiments/` | Phase별 실험, 논문 비교, ablation, 통계 검정 |
| `docs/03_presentation/` | 발표 구성, 리허설, cue card, 최종 대본 |
| `docs/04_operations/` | MLOps, n8n, Docker 운영 문서 |
| `docs/05_references/` | 참고 논문과 보조 PDF |

## 주요 실행 명령

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_data_integrity.py
.\.venv\Scripts\python.exe -u make_final_ppt.py
.\.venv\Scripts\python.exe -u make_detailed_ppt.py
.\.venv\Scripts\python.exe -u tools\presentation\polish_churnradar_presentation.py
```

PPT 생성 결과는 `deliverables/presentations/`에 저장됩니다.
