# ChurnRadar 발표 리허설 노트

## 60분 진행안

| 구간 | 시간 | 슬라이드 | 핵심 |
| --- | ---: | --- | --- |
| 도입 | 10분 | 1-3 | 데이터 불균형, leakage 방지, 평가 지표 |
| 논문 재현/모델 비교 | 15분 | 4-8 | EasyEnsemble 재현, hold-out과 CV 차이, feature 해석 |
| 운영 분석 | 20분 | 9-14 | 비용, top-k, calibration, segment, 모델 합의도 |
| 신뢰도/한계/결론 | 10분 | 15-17 | bootstrap CI, 한계, 최종 메시지 |
| Q&A | 5분 | 보조 문서 | 낮은 F1, 모델 선택, 실제 운영 질문 방어 |

## 슬라이드별 한 문장

1. ChurnRadar는 낮은 이탈 비율의 B2B 통신 고객을 운영 관점에서 예측하는 프로젝트다.
2. 이탈 고객이 약 6.5%라 accuracy보다 F1, recall, precision, PR-AUC가 중요하다.
3. train-only 전처리와 test 분포 보존으로 leakage를 막았다.
4. 논문 EasyEnsemble F1 0.129를 우리 EasyEnsemble F1 0.128로 재현했다.
5. hold-out F1은 LR이 높지만 recall-heavy 모델은 더 많은 이탈 고객을 잡는다.
6. CV에서는 BalancedBagging/EasyEnsemble 계열이 더 안정적이었다.
7. ZIP, feature group, segment, cost threshold를 분해해 논문보다 운영 질문을 확장했다.
8. LR 계수와 permutation FI는 매출 패턴과 가입자 활동성 결합이 중요함을 보여준다.
9. 비용 가정을 넣으면 예산과 팀 역량에 따라 권장 모델이 달라진다.
10. 실제 캠페인은 threshold보다 top-k 예산 전략으로 설명하는 편이 자연스럽다.
11. 비용 구조가 바뀌면 최적 threshold와 최적 모델도 바뀐다.
12. raw score는 확률이 아니므로 calibration 없이 확률처럼 말하면 안 된다.
13. 전체 평균만 보면 segment별 실패 패턴을 놓친다.
14. 여러 모델이 동시에 경고한 고객군은 실제 이탈률이 더 높아 risk tier로 쓸 수 있다.
15. bootstrap CI와 McNemar test는 모델 간 trade-off가 서로 다른 오류 패턴임을 보강한다.
16. 성능 상한의 핵심은 모델 부족보다 정적 CRM snapshot의 feature 한계다.
17. 최종 결론은 단일 최고 모델이 아니라 운영 목적별 모델 선택 프레임워크다.

## 반드시 피할 말

- "우리 모델이 논문보다 압도적으로 좋다."
- "F1 0.1681과 recall 0.9266을 동시에 달성했다."
- "cost-opt threshold가 실제 운영에서도 무조건 최선이다."
- "모델 score 0.4는 이탈 확률 40%다."

## Q&A 방어 문장

**왜 F1이 낮나요?**

이탈 고객이 약 6.5%뿐이고, 데이터가 월별 사용량 변화나 계약 만료 같은 이탈 직전 행동을 담지 못한 정적 CRM snapshot이기 때문입니다. 참고 논문의 EasyEnsemble도 F1 0.129였고, 우리는 이 baseline을 재현한 뒤 운영 목적별 trade-off를 확장했습니다.

**왜 Logistic Regression을 메인으로 두나요?**

hold-out F1과 precision이 가장 높고 계수 기반 설명이 가능하기 때문입니다. 다만 CV 안정성과 recall 중심 운영에서는 BalancedBagging/EasyEnsemble이 더 적합하므로, 하나의 최종 모델이 아니라 목적별 모델을 나눠 제시했습니다.

**실제 현업에서는 어떻게 쓰나요?**

확률 threshold만 고정하기보다 상담 가능 인원에 맞춰 top-k 고객을 뽑는 방식이 자연스럽습니다. 캠페인 비용이 낮으면 recall-heavy 전략이 유리하고, 비용이 높으면 precision이 높은 LR 또는 높은 threshold 전략이 더 안전합니다.

**자동화는 어디까지 했나요?**

n8n workflow가 Docker runner의 `/health`, `/run/full-reproduction`, `/summary`를 순서대로 호출합니다. 데이터 무결성 테스트, PSI drift check, PPT 재생성까지 연결되어 재현성과 운영 확장성을 보여줄 수 있습니다.
