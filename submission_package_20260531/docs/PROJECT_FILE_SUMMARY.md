# ChurnRadar 전체 파일 요약

마지막 업데이트: 2026-06-01

이 문서는 제출 패키지만이 아니라 원본 프로젝트 루트 전체를 기준으로 정리한 파일 요약이다.

## 1. 현재 상태 한 줄 요약

이 프로젝트는 `Baza customer Telecom v2.csv` 기반 B2B 통신사 고객 이탈 예측 프로젝트이며, 최종 결론은 “한 모델이 모든 기준에서 최고”가 아니라 “F1, recall, precision, 비용, 운영 예산에 따라 모델과 threshold를 다르게 선택해야 한다”이다.

최종 발표 파일:

- `ChurnRadar_Final_Presentation.pptx` - 요약 발표본 17장
- `ChurnRadar_Detailed_Presentation.pptx` - 상세 발표본 36장

핵심 제출/발표 문서:

- `FINAL_REPORT.md`
- `PRESENTATION_SLIDES.md`
- `CHURN_DATA_MODEL_DEFENSE.md`
- `MLOPS_ROADMAP.md`
- `README.md`

## 2. 파일 규모

`.venv`, `.git`, `__pycache__`, PowerPoint 임시 잠금 파일(`~$*.pptx`)을 제외한 현재 산출물 유형은 다음과 같다.

| 확장자 | 개수 | 의미 |
| --- | ---: | --- |
| `.csv` | 697 | 전처리 데이터, 모델 결과, 실험별 요약표 |
| `.md` | 53 | 보고서, 발표 구성, phase별 실험 문서, archive 문서, 제출 패키지 문서, 통합 요약 |
| `.json` | 39 | 실험 summary, feature column metadata, n8n workflow JSON |
| `.py` | 35 | 전처리, 실험, 분석, PPT 생성, n8n runner, 데이터 무결성 테스트 스크립트 |
| `.png` | 33 | 발표용 그래프와 dashboard 이미지 |
| `.txt` | 4 | requirements 계열 의존성 파일 |
| `.pptx` | 4 | 루트/제출 패키지의 요약 17장 PPT와 상세 36장 PPT |
| `.pdf` | 2 | 참고 논문/자료 |
| `.joblib` | 2 | 전처리/모델 artifact |
| `.yml` | 2 | n8n Docker Compose 구성 |
| `.runner` | 2 | Docker runner 이미지 정의 |
| `.ini` | 2 | pytest 설정 |
| `.dockerignore` | 1 | Docker build context 제외 규칙 |
| `.gitignore` | 1 | Git 추적 제외 규칙 |

## 3. 최상위 핵심 파일

| 파일 | 역할 |
| --- | --- |
| `README.md` | 프로젝트 현재 결론, 주요 문서/산출물 안내 |
| `FINAL_REPORT.md` | 제출용 최종 보고서 본문 |
| `PRESENTATION_SLIDES.md` | 요약 17장 발표 흐름과 멘트 |
| `ChurnRadar_Final_Presentation.pptx` | 실제 생성된 요약 발표 자료 17장 |
| `ChurnRadar_Detailed_Presentation.pptx` | 데이터/전처리/실험/논문 비교를 세세히 담은 상세 발표 자료 36장 |
| `final_model_summary.csv` | 핵심 운영점 요약표 |
| `requirements.txt` | 실행 의존성 목록 |
| `PROJECT_PROGRESS.md` | 날짜별 작업 이력과 검증 기록 |
| `PROJECT_FILE_SUMMARY.md` | 현재 파일 구조 요약 문서 |
| `N8N_DOCKER_WORKFLOW_GUIDE.md` | n8n Docker import와 실행 명령어 |
| `MLOPS_ROADMAP.md` | 운영 배포를 가정한 모니터링, 재학습, 보안, DVC 로드맵 |

## 4. Python 스크립트 역할

| 스크립트 | 역할 |
| --- | --- |
| `preprocess_churn.py` | 핵심 전처리, 모델 비교, threshold tuning, 오류 분석, feature importance |
| `additional_model_experiments.py` | 추가 모델/하이퍼파라미터 후보 탐색 |
| `create_column_split_datasets.py` | 컬럼별/값별 데이터 이해용 CSV 생성 |
| `make_presentation_assets.py` | 발표용 기본 PNG 5개 생성 |
| `make_research_presentation_materials.py` | 초기 연구 발표 자료와 role summary 생성 |
| `paper_ablation_variants.py` | 논문형 core/확장/KA/ZIP ablation variant 전처리 생성 |
| `phase_3b_differentiation_experiments.py` | ZIP, feature group, segment, cost threshold 차별화 실험 |
| `phase_4_cross_validation.py` | 5-fold CV 안정성 비교 |
| `phase_5a_interpretability.py` | LR 계수, contribution, PDP 기반 해석 가능성 분석 |
| `phase_5b_business_impact.py` | 비용-편익, ROI, break-even, business dashboard 생성 |
| `phase_6_extended_case_studies.py` | top-k, calibration, segment ROI, 모델 합의도 추가 케이스 |
| `phase_7_next_experiments.py` | 추가 진행 후보 점검: tuned BalancedBagging CV, paper-ablation benchmark |
| `phase_8_statistical_validation.py` | bootstrap CI와 McNemar paired test |
| `make_final_ppt.py` | 요약 17장 PowerPoint 자동 생성 |
| `make_detailed_ppt.py` | 프로젝트 루트에서 상세 36장 PowerPoint 자동 생성 |
| `n8n_automation/churn_runner.py` | n8n에서 HTTP로 호출하는 Docker runner |
| `tests/test_data_integrity.py` | 원본 데이터 파일 존재, 필수 컬럼, target 값, 빈 데이터 여부 검사 |

## 5. Phase별 문서 역할

| 문서 | 핵심 내용 |
| --- | --- |
| `PHASE_3A_REPRODUCTION_EXPERIMENTS.md` | 논문 EasyEnsemble baseline 재현 |
| `PHASE_3B_DIFFERENTIATION_EXPERIMENTS.md` | ZIP/feature/segment/cost threshold 분석 |
| `PHASE_4_PAPER_COMPARISON_FRAMEWORK.md` | 논문과 우리 프로젝트의 공정 비교 원칙 및 CV |
| `PHASE_5A_FEATURE_IMPORTANCE_AND_SHAP_ALTERNATIVES.md` | SHAP 대안으로 LR 계수/기여도 해석 |
| `PHASE_5B_BUSINESS_IMPACT_ANALYSIS.md` | 비용-편익과 운영 시나리오 |
| `PHASE_6_EXTENDED_CASE_STUDIES.md` | 1시간 발표용 추가 비교 실험 |
| `PHASE_7_NEXT_EXPERIMENTS.md` | 남은 추가 실험 후보를 실제로 검증한 결과 |
| `PHASE_8_STATISTICAL_VALIDATION.md` | bootstrap CI와 McNemar 검정 결과 |

## 6. processed 폴더 구조

| 폴더 | 내용 |
| --- | --- |
| `processed/model_a_with_billing_zip/` | ZIP 포함 기본 전처리 split과 artifact |
| `processed/model_b_without_billing_zip/` | ZIP 제외 기본 전처리 split과 artifact |
| `processed/additional_experiments/` | 추가 모델 탐색 결과 |
| `processed/column_split_datasets/` | 컬럼별/값별 데이터 이해용 CSV와 한글 요약 |
| `processed/paper_ablation_variants/` | 논문형 core/확장/KA/ZIP variant별 split |
| `processed/phase_3b_differentiation/` | 차별화 실험 결과 |
| `processed/phase_4_paper_comparison/` | CV fold/result summary |
| `processed/phase_5a_interpretability/` | 계수/기여도/PDP/해석 가능성 이미지 |
| `processed/phase_5b_business_impact/` | ROI, break-even, business dashboard |
| `processed/phase_6_extended_case_studies/` | top-k, calibration, segment, model agreement |
| `processed/phase_7_next_experiments/` | 추가 후보 점검 결과 |
| `processed/phase_8_statistical_validation/` | bootstrap CI와 McNemar 결과 |
| `processed/research_presentation/` | 초기 연구 발표용 보조 CSV와 slide outline |

## 7. 발표 이미지와 PPT

| 위치 | 내용 |
| --- | --- |
| `presentation_assets/01_model_metric_comparison.png` | 핵심 모델 metric 비교 |
| `presentation_assets/02_confusion_counts.png` | TP/FP/FN/TN 비교 |
| `presentation_assets/03_precision_recall_tradeoff.png` | precision-recall trade-off |
| `presentation_assets/04_feature_importance_main.png` | 주요 feature importance |
| `presentation_assets/05_paper_comparison.png` | 논문 baseline 대비 비교 |
| `processed/phase_5b_business_impact/business_impact_dashboard.png` | 비즈니스 임팩트 dashboard |
| `processed/phase_6_extended_case_studies/*.png` | top-k, cost, calibration, segment, agreement 그래프 |
| `ChurnRadar_Final_Presentation.pptx` | 위 이미지와 표를 통합한 요약 PPT 17장 |
| `ChurnRadar_Detailed_Presentation.pptx` | 데이터 사용/제외/수정, 전처리, 논문 비교, 실험, 튜닝, 컬럼별 해석을 확장한 상세 PPT 36장 |

## 8. n8n 자동화 구성

| 파일 | 내용 |
| --- | --- |
| `n8n_automation/churnradar_n8n_workflow.json` | n8n에서 import 가능한 workflow JSON |
| `n8n_automation/churn_runner.py` | 프로젝트 스크립트를 실행하는 HTTP runner |
| `n8n_automation/requirements.runner.txt` | runner Docker 이미지 의존성 고정 버전 |
| `n8n_automation/Dockerfile.runner` | Python runner 이미지 |
| `n8n_automation/docker-compose.yml` | runner와 전용 n8n 컨테이너 구성 |
| `.dockerignore` | Docker build context에서 대량 산출물 제외 |
| `N8N_DOCKER_WORKFLOW_GUIDE.md` | Docker 실행, import, 테스트 명령어 |

현재 검증 결과:

- n8n workflow JSON 문법 검사 통과
- n8n 2.12.3 임시 컨테이너 CLI import 통과
- 기존 `edurisk-n8n` 컨테이너 workflow import 통과
- `churnradar-runner` build 및 `/health` 통과
- runner의 `POST /run/ppt` 통과, 요약 PPT 17장 확인
- `make_detailed_ppt.py` 실행 통과, 상세 PPT 36장 확인
- workflow는 중복 실행을 줄이기 위해 `Health -> Full Reproduction -> Summary` 구조로 정리
- runner의 `GET /summary`는 `.venv`를 순회하지 않도록 최적화했고 파일 카운트 반환을 확인

## 9. Archive 폴더

`archive/`에는 최종 제출의 중심 문서는 아니지만, 의사결정 흐름 보존에 의미 있는 문서를 보관했다.

| 파일 | 내용 |
| --- | --- |
| `archive/PROJECT_CHANGE_PROPOSAL.md` | 프로젝트 변경 검토에서 유지 결정으로 바뀐 기록 |
| `archive/TEAM_PROJECT_SWITCH_REPORT.md` | 팀 프로젝트 유지 보고서 |
| `archive/ADDITIONAL_EXPERIMENTS_AND_OPERATION_SUMMARY.md` | Phase 3A-8 이전 추가 모델 탐색 기록 |

## 10. 중복 정리 결과

삭제한 정확 중복 파일:

- `processed/research_presentation/research_presentation_material.md`
- `processed/column_split_datasets/03_profiles/category_value_churn_summary.csv`
- `processed/column_split_datasets/03_profiles/numeric_bins_churn_summary.csv`

수정한 생성 스크립트:

- `make_research_presentation_materials.py`: 상위 발표 문서만 생성하도록 변경
- `create_column_split_datasets.py`: `yes_no_rate` 이름의 summary만 생성하도록 변경

삭제하지 않은 중복:

- 여러 variant의 `y_train.csv`, `y_test.csv`, `y_train_resampled.csv`

이 파일들은 같은 split을 공유해서 내용은 같지만, 각 variant 폴더의 재현성을 위해 남겨두는 것이 안전하다.

## 11. 최종 재현 점검 결과

아래 체인을 순서대로 재실행했고 모두 정상 종료했다.

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
```

상세 발표본은 별도 생성 스크립트로 추가 생성했다.

```powershell
.\.venv\Scripts\python.exe -u make_detailed_ppt.py
```

검증 메모:

- Python 문법 검사: 기존 핵심 재현 스크립트 16개와 `make_detailed_ppt.py` 통과
- 데이터 무결성 테스트: `python -m pytest -q tests/test_data_integrity.py` 4개 통과
- 주요 dependency import: 통과
- PPT slide count: 요약본 17장, 상세본 36장 확인
- `phase_5b_business_impact.py`에서 matplotlib `tight_layout` warning이 한 번 출력되었지만 산출물 생성에는 실패가 없었다.

## 12. 최종 발표/보고서에서 쓰면 좋은 핵심 문장

> 본 프로젝트는 Makokha et al. (2026)의 EasyEnsemble baseline을 F1 0.128로 재현했고, 이후 ZIP ablation, feature group ablation, segment analysis, cost-sensitive threshold, calibration, model agreement, bootstrap CI를 통해 불균형 churn 데이터에서 운영 목적별 모델 선택 프레임워크를 제시했다. 최종 F1 기준은 LogisticRegression_SMOTE가 높았지만, CV 안정성과 recall 중심 운영에서는 BalancedBagging/EasyEnsemble 및 CatBoost/XGBoost 계열이 다른 장점을 보여주었다. 따라서 결론은 단일 모델의 절대 우위가 아니라, 캠페인 예산과 비용 구조에 맞춘 운영점 선택이다.
