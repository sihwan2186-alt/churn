# ChurnRadar Submission Package

생성일: 2026-05-31

## 포함된 핵심 파일

| 위치 | 내용 |
| --- | --- |
| `ChurnRadar_Final_Presentation.pptx` | 최종 발표 PPT, 17장 |
| `final_model_summary.csv` | 최종 운영점 요약표 |
| `requirements.txt` | 로컬 실행 의존성 |
| `docs/` | 최종 보고서, 발표 구성, 방어 문서, Phase별 실험 문서 |
| `scripts/` | 전처리, 실험, 분석, PPT 생성, drift check Python 스크립트 |
| `presentation_assets/` | PPT에 사용한 주요 이미지 |
| `processed_selected/` | 핵심 요약 CSV/JSON/PNG 산출물 |
| `n8n_automation/` | Docker runner와 n8n workflow |
| `tests/` | 데이터 무결성 pytest |

## 의도적으로 제외한 것

- 원본 `Baza customer Telecom v2.csv`: 개인정보/원본 데이터 제출 제한 가능성 때문에 제외
- 전체 `processed/`: 대량 중간 산출물과 train/test split 전체는 제외하고 핵심 요약만 포함
- `.venv`, Docker volume, cache, log 파일

원본 데이터 제출이 허용된 경우에만 `Baza customer Telecom v2.csv`를 별도로 추가한다.

## 빠른 확인 명령

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_data_integrity.py
.\.venv\Scripts\python.exe monitor_drift.py
```

## 발표 핵심 문장

본 프로젝트는 Makokha et al. (2026)의 EasyEnsemble baseline을 재현한 뒤, ZIP ablation, feature group ablation, segment analysis, cost threshold, top-k budget, calibration, bootstrap CI를 통해 불균형 churn 데이터에서 운영 목적별 모델 선택 프레임워크를 제시했다.
