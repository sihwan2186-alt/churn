# ChurnRadar

`Baza customer Telecom v2.csv`를 사용한 B2B 통신사 고객 이탈 예측 프로젝트입니다.

## 현재 결정

프로젝트를 다른 주제로 변경하지 않고, **ChurnRadar(통신사 고객 이탈 예측)**로 계속 진행합니다. 최종 발표와 보고서는 낮은 성능을 실패로 설명하는 것이 아니라, 심한 class imbalance와 정적 CRM snapshot 데이터에서 어떤 전처리, 모델 비교, threshold tuning, 오류 분석을 수행했는지 보여주는 방향으로 정리합니다.

## 최종 모델 요약

| 목적 | Variant | Model | F1 | Recall | Precision |
| --- | --- | --- | ---: | ---: | ---: |
| 최종 F1 기준 | `without_billing_zip` | `LogisticRegression_SMOTE` | 0.1681 | 0.2661 | 0.1229 |
| Recall 중심 운영 | `with_billing_zip` | `BalancedBagging_original` | 0.1526 | 0.5872 | 0.0877 |
| Recall 극대화 | `with_billing_zip` | `CatBoost_native_categorical`, threshold 0.35 | 0.1310 | 0.8349 | 0.0711 |

## 지금 해야 할 일

1. `FINAL_REPORT.md`를 최종 제출용 본문으로 다듬기
2. `PRESENTATION_SLIDES.md`를 기반으로 실제 PPT 만들기
3. `presentation_assets/`의 PNG 5개를 슬라이드에 삽입하기
4. `CHURN_DATA_MODEL_DEFENSE.md`로 교수님 예상 질문 답변 준비하기
5. 시간이 남으면 BalancedBagging 하이퍼파라미터 튜닝을 소규모로만 추가 실험하기
6. 최종 제출 전 아래 명령으로 결과 재현 확인하기

```powershell
.\.venv\Scripts\python.exe -m py_compile preprocess_churn.py
.\.venv\Scripts\python.exe preprocess_churn.py
.\.venv\Scripts\python.exe -m py_compile make_presentation_assets.py
.\.venv\Scripts\python.exe make_presentation_assets.py
```

## 주요 문서

| 파일 | 역할 |
| --- | --- |
| `PROJECT_PROGRESS.md` | 전체 진행 기록과 남은 작업 목록 |
| `FINAL_REPORT.md` | 최종 보고서 본문 |
| `PRESENTATION_SLIDES.md` | 발표 슬라이드 구성안 |
| `CHURN_DATA_MODEL_DEFENSE.md` | 데이터, 모델 선택, 예상 질문 방어 자료 |
| `PROJECT_CHANGE_PROPOSAL.md` | 변경 제안서가 아니라, 현재는 프로젝트 유지 결정 메모 |
| `TEAM_PROJECT_SWITCH_REPORT.md` | 과거 방향 검토 문서였으나, 현재는 팀 프로젝트 유지 보고서 |
