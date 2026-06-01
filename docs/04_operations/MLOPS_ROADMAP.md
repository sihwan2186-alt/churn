# ChurnRadar MLOps 및 유지보수 로드맵

마지막 업데이트: 2026-05-30

본 프로젝트는 현재 분석/발표용 결과물은 완성 단계이며, 실제 운영 배포를 가정하면 아래 순서로 고도화하는 것이 적절하다.

## 1. 데이터 드리프트 모니터링

구현 완료:

- `monitor_drift.py`에서 PSI(Population Stability Index) 기반 드리프트 점검을 수행한다.
- 기본 점검 변수는 `TotalRevenue`, `Active_subscribers`, `ARPU`이다.
- n8n runner의 `POST /run/monitor-drift` endpoint에서도 실행할 수 있다.

실행 예시:

```powershell
.\.venv\Scripts\python.exe monitor_drift.py --current "Baza customer Telecom v2.csv"
.\.venv\Scripts\python.exe monitor_drift.py --current "new_customers.csv" --output processed\drift_report.json
```

## 2. 모델 재학습 전략

권장 기준:

- 새 데이터에서 F1이 0.14 아래로 떨어지거나 PSI가 0.2를 넘는 feature가 나오면 재학습을 검토한다.
- 데이터가 계속 쌓이는 운영 환경에서는 최근 6개월 또는 12개월 rolling window 학습을 권장한다.
- 재학습 후에는 hold-out F1만 보지 말고 recall, precision, PR-AUC, 비용 기준 순이익을 함께 비교한다.

## 3. API 보안 및 로깅

구현 완료:

- `n8n_automation/churn_runner.py`는 `/summary`와 `/run/*` endpoint에서 `X-API-KEY` 헤더를 확인한다.
- `runner.log`에 script 실행 결과와 timeout을 기록한다.
- `docker-compose.yml`에서 `CHURN_RUNNER_API_KEY`를 환경 변수로 전달한다.

추가 권장:

- 로컬 기본 키 `churn-radar-secret-2026`은 제출/데모용이고, 서버 배포 시에는 긴 랜덤 문자열로 교체한다.
- 실제 예측 API를 만들 경우 요청자, 예측 점수, 사용 모델 버전, 이후 실제 이탈 여부를 DB에 기록한다.

## 4. 데이터 및 모델 버전 관리

권장:

- 원본 데이터와 대량 `processed/` 산출물은 Git보다 DVC 또는 외부 스토리지로 관리한다.
- 모델 artifact는 모델명, threshold, 학습 데이터 버전, 생성일을 함께 기록한다.
- 최종 보고서에 들어간 수치와 재현 pipeline에서 생성되는 수치가 일치하는지 release tag 단위로 보존한다.
