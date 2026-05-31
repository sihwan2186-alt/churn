# ChurnRadar 현재 진행 현황

마지막 업데이트: 2026-05-30

## 한 줄 결론

현재 프로젝트는 B2B 통신사 고객 이탈 예측 주제로 최종 발표/보고서 단계까지 진행되었고, 핵심 결론은 단일 최고 모델보다 **F1, Recall, 비용, 캠페인 예산에 맞춘 운영점 선택**이 중요하다는 것이다.

## 완료된 범위

| 영역 | 현재 상태 |
| --- | --- |
| 데이터 이해 | 원본 `Baza customer Telecom v2.csv` 구조, 결측, target 불균형, 주요 feature 해석 정리 완료 |
| 전처리 | PID/KA 제거, 중복 PID 처리, 결측 flag, train 기준 imputation/scaling, feature engineering 완료 |
| 모델링 | Logistic Regression, CatBoost, XGBoost, EasyEnsemble, RUSBoost, BalancedBagging 등 비교 완료 |
| 논문 재현 | 참고 논문 EasyEnsemble F1 0.129를 우리 EasyEnsemble F1 0.128 수준으로 재현 |
| 추가 실험 | ZIP ablation, feature group ablation, segment 분석, threshold/cost/top-k/calibration/model agreement 완료 |
| 통계 검정 | bootstrap confidence interval, McNemar paired test 완료 |
| 발표 자료 | `ChurnRadar_Final_Presentation.pptx` 17장 생성 완료 |
| 자동화 | n8n workflow JSON, Docker runner, runner API, 로그, API 키 인증 구성 완료 |
| 품질 점검 | 데이터 무결성 테스트와 드리프트 점검 스크립트 추가 |

## 현재 추천 모델

| 목적 | 추천 운영점 | 핵심 수치 | 사용 이유 |
| --- | --- | ---: | --- |
| F1 기준 최종 | `without_billing_zip + LogisticRegression_SMOTE` | F1 0.1681 | 최종 보고서의 메인 모델로 가장 설명하기 좋음 |
| 안정적 recall 운영 | `with_billing_zip + BalancedBagging_original` | Recall 0.5872 | 이탈 고객을 더 많이 잡는 캠페인 후보 |
| recall 극대화 | `with_billing_zip + XGBoost_SMOTE`, threshold 0.16 | Recall 0.9266 | false positive 비용을 감수할 때만 적합 |
| 비용 최적 | `with_billing_zip + BalancedBagging_original`, threshold 0.29 | 순이익 165,840 | 논문형 비용 가정에서는 가장 높은 순이익 |

## 이번 점검에서 수정한 문제

- 루트의 중복 테스트 파일 `test_data_integrity.py`를 제거하고 `tests/test_data_integrity.py`만 남김
- 오타 중복 문서 `N8N_DOCKLL_WORKFLOW_GUIDE.md` 제거
- n8n runner에 API 키 인증이 추가되어 있었지만 workflow 요청 헤더가 빠져 있던 문제 수정
- Docker compose에 `CHURN_RUNNER_API_KEY` 환경 변수 연결
- `monitor_drift.py`의 JSON 중복 출력과 취약한 PSI 계산 로직 정리
- 문서의 드리프트 스크립트 경로를 실제 위치인 `monitor_drift.py` 기준으로 정리

## 방금 통과한 검증

```powershell
$files = rg --files -g "*.py" -g "!.venv/**" -g "!__pycache__/**"
.\.venv\Scripts\python.exe -m py_compile @files
.\.venv\Scripts\python.exe tests\test_data_integrity.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe monitor_drift.py
.\.venv\Scripts\python.exe -m json.tool .\n8n_automation\churnradar_n8n_workflow.json
docker compose -f .\n8n_automation\docker-compose.yml config
```

## 남아 있는 확인 사항

- 전체 재현 체인 전체 재실행은 시간이 오래 걸려 이번 빠른 점검에서는 수행하지 않았다.
- n8n UI에서 실제 `Execute Workflow` 버튼 클릭 검증은 별도 확인이 필요하다.
- PPT 파일은 생성과 slide count는 확인되어 있지만, 학교 양식/이름/제출 포맷은 사람이 한 번 열어서 최종 확인해야 한다.
