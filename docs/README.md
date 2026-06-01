# 문서 인덱스

프로젝트 문서는 아래 순서대로 보면 가장 빠릅니다.

## 추천 열람 순서

1. `00_overview/PROJECT_STATUS_CURRENT.md` - 현재 완료 범위와 남은 일
2. `01_reports/FINAL_REPORT.md` - 제출용 최종 보고서
3. `03_presentation/PRESENTATION_SCRIPT_POLISHED.md` - 실제 발표 대본
4. `01_reports/CHURN_DATA_MODEL_DEFENSE.md` - 교수님 질문 대비
5. `02_experiments/PHASE_4_PAPER_COMPARISON_FRAMEWORK.md` - 논문 비교와 CV 안정성
6. `02_experiments/PHASE_8_STATISTICAL_VALIDATION.md` - bootstrap CI와 McNemar 검정

## 폴더별 역할

| 폴더 | 역할 |
| --- | --- |
| `00_overview/` | 전체 요약, 파일 구조, 진행 기록, 다음 작업 |
| `01_reports/` | 최종 보고서와 모델/데이터 방어 논리 |
| `02_experiments/` | Phase별 재현, ablation, CV, 비즈니스 영향, 통계 검정 |
| `03_presentation/` | 발표 슬라이드 구성, cue card, 리허설, 최종 대본 |
| `04_operations/` | n8n, Docker, MLOps, 재학습/모니터링 문서 |
| `05_references/` | 참고 논문과 보조 PDF |

발표에서 반드시 언급할 맥락: 시계열 데이터를 찾을 수 없어 정적 CRM 스냅샷 기준으로 진행했고, 따라서 결론은 단일 최고 모델보다 운영 목적별 모델 선택 프레임워크에 있다.
