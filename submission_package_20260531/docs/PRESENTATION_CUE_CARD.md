# ChurnRadar 발표 직전 큐카드

## 첫 문장

ChurnRadar는 B2B 통신 고객 이탈을 예측하는 데서 끝나지 않고, 실제 retention campaign에서 예산과 비용에 따라 어떤 모델을 써야 하는지까지 정리한 프로젝트입니다.

## 꼭 외울 숫자

| 항목 | 숫자 |
| --- | ---: |
| 원본 데이터 | 8,453 rows, 14 columns |
| PID 중복 제거 후 | 8,436 rows |
| 이탈률 | 약 6.5% |
| 논문 EasyEnsemble F1 | 0.129 |
| 재현 EasyEnsemble F1 | 0.128 |
| LR hold-out F1 | 0.1681 |
| BalancedBagging CV F1 mean | 0.1455 |
| CatBoost recall | 0.8349 |
| XGBoost recall | 0.9266 |
| XGBoost FP | 1,417 |
| BalancedBagging cost-opt net benefit | 165,840 |
| PPT 장수 | 요약 17장 / 상세 36장 |

## 핵심 주장

단일 최고 모델을 찾은 것이 아니라, 운영 목적별 모델 선택 프레임워크를 만든 것이다.

## 3개 방어 문장

1. F1이 낮은 이유는 이탈률 6.5%의 강한 불균형과 정적 CRM snapshot 한계 때문이다.
2. LR은 hold-out F1 기준으로 좋지만, CV 안정성과 recall 목적에서는 BalancedBagging/EasyEnsemble/CatBoost 계열도 중요하다.
3. XGBoost는 recall이 높지만 FP가 크므로 비용과 고객 피로도를 함께 봐야 한다.

## 말하면 안 되는 문장

- 논문보다 우리 모델이 무조건 좋다.
- XGBoost가 최종 모델이다.
- score 0.4는 이탈 확률 40%다.
- cost-opt threshold는 실제 운영에서도 항상 최선이다.

## 마지막 문장

따라서 ChurnRadar의 결론은 높은 점수 하나가 아니라, 예산, 비용, recall 목표, 고객 segment에 따라 모델과 threshold를 다르게 선택하는 운영 프레임워크입니다.
