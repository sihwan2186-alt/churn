# ChurnRadar 앞으로 해야 할 일

마지막 업데이트: 2026-06-01

## 제출 전 반드시 할 일

1. 가상환경 의존성 동기화

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q tests\test_data_integrity.py
```

1. 최종 PPT 2종 육안 확인

- `deliverables/presentations/ChurnRadar_Detailed_Presentation_Polished.pptx`를 최종 발표용으로 우선 확인한다.
- `deliverables/presentations/ChurnRadar_Final_Presentation.pptx`는 요약 발표본 17장이다. 학교 양식, 팀원 이름, 제출용 제목, 표/그래프 잘림 여부를 확인한다.
- `deliverables/presentations/ChurnRadar_Detailed_Presentation.pptx`는 상세 발표본 36장 원본이다.
- 특히 두 PPT 전체에서 숫자와 결론 문장이 서로 충돌하지 않는지 확인한다.

1. n8n 수동 실행 확인

```powershell
docker compose -f .\n8n_automation\docker-compose.yml up -d --build churn-runner
```

n8n UI에서 `ChurnRadar Docker Reproduction Pipeline`을 열고 `Execute Workflow`를 한 번 실행한다.

1. 최종 제출 패키지 결정

- 반드시 포함: `README.md`, `docs/01_reports/FINAL_REPORT.md`, `docs/03_presentation/PRESENTATION_SCRIPT_POLISHED.md`, `deliverables/presentations/ChurnRadar_Detailed_Presentation_Polished.pptx`, 핵심 Python 스크립트, 핵심 CSV 요약표
- 상황에 따라 포함: `processed/` 전체 산출물, `n8n_automation/`, `archive/`
- 개인정보/원본 데이터 제출 제한이 있으면 `Baza customer Telecom v2.csv` 포함 여부를 교수님 지침에 맞춘다.

## 하면 좋은 보강

| 우선순위 | 작업 | 이유 |
| --- | --- | --- |
| 높음 | `docs/01_reports/CHURN_DATA_MODEL_DEFENSE.md`로 예상 질문 10분 리허설 | 낮은 precision과 class imbalance 질문에 대비 |
| 높음 | `docs/01_reports/FINAL_REPORT.md`의 결론 문장을 PPT 결론과 동일하게 맞춤 | 보고서와 발표 간 메시지 충돌 방지 |
| 중간 | n8n API 키를 기본값이 아닌 긴 값으로 교체 | 로컬 외 환경에서 실행할 때 안전함 |
| 중간 | 새 데이터가 생기면 `monitor_drift.py --current 새파일.csv` 실행 | 분포 변화가 있으면 재학습 필요 여부 판단 가능 |
| 낮음 | DVC 또는 외부 스토리지로 데이터/산출물 버전 관리 | 제출 이후 운영형 프로젝트로 확장할 때 유용 |

## 더 하지 않아도 되는 일

- 새 주제로 프로젝트를 바꾸는 일
- 모델을 무작정 더 추가하는 일
- F1 0.1681과 recall 0.9266을 같은 모델 성능처럼 합쳐 말하는 일
- 비용 최적 threshold를 실제 운영에서도 무조건 최선이라고 주장하는 일

## 최종 발표 핵심 문장

이 프로젝트는 시계열 데이터를 찾을 수 없어 정적 CRM 스냅샷 기준으로 진행했고, 참고 논문의 불균형 churn 예측 baseline을 재현한 뒤 ZIP 포함 여부, threshold, 비용 구조, 캠페인 예산, segment, calibration, 통계 검정을 추가해 **운영 목적별 모델 선택 프레임워크**를 제시한 프로젝트로 정리하면 가장 안전하다.
