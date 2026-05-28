# 추가 모델 실험 및 운영 해석 정리

마지막 업데이트: 2026-05-27

## 1. 목적

기존 최종 모델인 `without_billing_zip + LogisticRegression_SMOTE`를 유지해도 되는지 확인하기 위해 추가 모델 실험을 진행했다. 목표는 단순히 점수를 조금 올리는 것만이 아니라, 현재 결과가 왜 이렇게 나왔는지와 실제 운영에서 어떤 방식으로 활용할 수 있는지를 정리하는 것이다.

## 2. 추가 실험 범위

새 스크립트:

- `additional_model_experiments.py`

생성 산출물:

- `processed/additional_experiments/additional_model_results.csv`
- `processed/additional_experiments/additional_top25_summary.csv`
- `processed/additional_experiments/additional_threshold_sweep.csv`
- `processed/additional_experiments/operating_budget_topk.csv`

실험한 모델/방식:

- Logistic Regression C값 조정
- class-weight 기반 Logistic Regression
- RidgeClassifier balanced
- LinearSVC balanced
- RandomForest balanced
- ExtraTrees balanced
- GradientBoosting + SMOTE
- HistGradientBoosting balanced
- EasyEnsemble
- RUSBoost
- BalancedBagging 하이퍼파라미터 조합
- CatBoost encoded feature 튜닝
- CatBoost native categorical 튜닝
- validation 기준 threshold 선택
- top-k 캠페인 예산 기준 운영 성능 확인

총 46개 조합을 확인했고, 이 중 44개 조합이 정상 평가되었다. AdaBoost stump 기반 조합 2개는 base classifier가 random보다 나쁘다는 오류로 제외했다.

## 3. 추가 실험 최고 결과

validation에서 threshold를 고른 뒤 test에 적용한 기준으로는 아래 조합이 추가 실험 중 가장 좋았다.

| 기준 | Variant | Model | Threshold | F1 | Recall | Precision | TP | FP | FN | TN |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 추가 실험 best | `with_billing_zip` | `BalancedBagging_tree_depthnone_leaf25` | 0.51 | 0.1605 | 0.5138 | 0.0951 | 56 | 533 | 53 | 1046 |

기존 메인 모델과 비교하면 다음과 같다.

| 구분 | Variant | Model | F1 | Recall | Precision |
| --- | --- | --- | ---: | ---: | ---: |
| 기존 최종 메인 | `without_billing_zip` | `LogisticRegression_SMOTE` | 0.1681 | 0.2661 | 0.1229 |
| 추가 실험 best | `with_billing_zip` | `BalancedBagging_tree_depthnone_leaf25` | 0.1605 | 0.5138 | 0.0951 |
| 기존 recall 후보 | `with_billing_zip` | `BalancedBagging_original` | 0.1526 | 0.5872 | 0.0877 |

결론적으로 추가 실험에서도 F1 기준 최종 1위는 기존 `LogisticRegression_SMOTE`가 유지된다. 다만 BalancedBagging 튜닝 모델은 기존 recall 후보보다 precision과 F1이 개선되어, 운영 후보로 더 설득력 있게 제시할 수 있다.

## 4. 상위 후보 해석

| 순위 | Variant | Model | F1 | Recall | Precision | 해석 |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | `with_billing_zip` | `BalancedBagging_tree_depthnone_leaf25` | 0.1605 | 0.5138 | 0.0951 | recall을 절반 이상 유지하면서 기존 BalancedBagging보다 F1 개선 |
| 2 | `with_billing_zip` | `BalancedBagging_tree_depthnone_leaf10` | 0.1554 | 0.4954 | 0.0922 | recall과 precision 균형이 비슷한 운영 후보 |
| 3 | `with_billing_zip` | `BalancedBagging_tree_depth5_leaf25` | 0.1441 | 0.5963 | 0.0820 | recall은 높지만 오탐이 많음 |
| 4 | `without_billing_zip` | `EasyEnsemble_n10` | 0.1431 | 0.3578 | 0.0894 | 논문 계열 모델의 보조 비교 후보 |
| 5 | `with_billing_zip` | `LogisticRegression_SMOTE_C0.1` | 0.1358 | 0.2018 | 0.1023 | precision은 비교적 높지만 recall이 낮음 |

## 5. 왜 기존 메인 모델을 넘기 어려웠는가

추가 실험 결과를 보면 모델을 더 많이 바꿔도 성능이 크게 상승하지 않았다. 이유는 다음과 같이 해석할 수 있다.

1. target이 매우 불균형하다.
   전체 이탈 비율이 약 6.5%라서 이탈 고객을 조금만 더 많이 잡아도 false positive가 급격히 늘어난다.

2. 현재 데이터는 정적 CRM snapshot이다.
   월별 사용량 감소, 결제 실패, 고객센터 문의, 계약 만료까지 남은 기간 같은 이탈 직전 행동 신호가 없다.

3. feature 간 분리력이 강하지 않다.
   PR-AUC가 전반적으로 낮고, 좋은 모델도 precision이 0.10 안팎에 머문다. 이는 모델 문제가 아니라 feature가 minority class를 뚜렷하게 분리하지 못한다는 신호다.

4. `Billing_ZIP`은 양면성이 있다.
   BalancedBagging 계열에서는 recall 개선에 도움이 되지만, 최종 F1 기준에서는 `without_billing_zip` Logistic Regression이 더 안정적이다. 지역 정보가 signal과 noise를 동시에 갖는 것으로 볼 수 있다.

5. threshold 선택이 매우 민감하다.
   일부 모델은 test set에서 사후적으로 threshold를 고르면 F1이 올라가지만, validation에서 고른 threshold를 test에 적용하면 상승폭이 제한된다. 따라서 보고서에서는 test threshold 직접 최적화 결과를 최종 성능으로 쓰면 안 된다.

## 6. 무엇을 배웠는가

- 단일 accuracy가 아니라 F1, recall, precision, PR-AUC, MCC를 함께 봐야 한다.
- 최종 제출용 메인 모델은 F1 기준으로 `LogisticRegression_SMOTE`가 가장 타당하다.
- 캠페인 운영 후보는 `BalancedBagging_tree_depthnone_leaf25`가 기존 BalancedBagging보다 설명하기 좋다.
- CatBoost와 EasyEnsemble은 recall을 높일 수 있지만 precision 손실이 커서 메인 모델로 쓰기는 어렵다.
- 성능 한계는 실험 부족보다 데이터 구조의 한계에 가깝다.
- 모델 하나를 고정하기보다 “F1형 모델”과 “recall형 운영 모델”을 나누는 설명이 가장 설득력 있다.

## 7. 실제 운영 활용 방식

운영에서는 모델이 예측한 score를 기준으로 위험 고객을 정렬한 뒤, 상담/혜택 제공 예산에 맞춰 상위 k%만 캠페인 대상으로 삼을 수 있다.

예시 결과:

| Model | 상위 비율 | 대상 고객 수 | 포착 이탈 고객 | Precision@k | Recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BalancedBagging_tree_depthnone_leaf25` | 10% | 169 | 16 / 109 | 0.0947 | 0.1468 |
| `BalancedBagging_tree_depthnone_leaf25` | 20% | 338 | 30 / 109 | 0.0888 | 0.2752 |
| `BalancedBagging_tree_depthnone_leaf25` | 30% | 506 | 45 / 109 | 0.0889 | 0.4128 |
| `CatBoost_encoded_depth4_lr0.03` | 10% | 169 | 18 / 109 | 0.1065 | 0.1651 |
| `without_billing_zip + LogisticRegression_SMOTE_C0.1` | 10% | 169 | 24 / 109 | 0.1420 | 0.2202 |

운영 해석:

- 상담 인력이 적으면 상위 10%만 대상으로 삼아 precision을 우선한다.
- 예산이 조금 더 있으면 상위 20~30%까지 확대해 recall을 높인다.
- retention 혜택 비용이 낮다면 BalancedBagging처럼 recall이 높은 모델을 쓸 수 있다.
- 고비용 혜택이라면 Logistic Regression처럼 precision이 상대적으로 나은 모델이 안전하다.

## 8. 최종 보고서에 넣을 결론 문장

추가 모델 실험 결과, BalancedBagging의 하이퍼파라미터를 조정하면 recall 중심 운영 후보의 F1을 0.1605까지 개선할 수 있었지만, 전체 F1 기준 최종 모델인 `without_billing_zip + LogisticRegression_SMOTE`의 F1 0.1681을 넘지는 못했다. 이는 모델 선택만으로 해결되는 문제가 아니라, 이탈 고객 비율이 약 6.5%로 매우 낮고 데이터가 정적인 CRM snapshot이라 이탈 직전 행동 변화를 충분히 담지 못하기 때문으로 해석된다. 따라서 본 프로젝트에서는 `LogisticRegression_SMOTE`를 최종 F1 모델로 제시하고, 운영 목적상 더 많은 이탈 고객을 잡아야 하는 경우에는 tuned BalancedBagging을 recall 중심 후보로 함께 제시한다.

## 9. 다음 우선순위

1. `FINAL_REPORT.md`에 추가 실험 결과와 위 결론 문장을 반영한다.
2. PPT에는 기존 메인 모델과 추가 실험 best BalancedBagging을 함께 비교한다.
3. 발표에서는 “새 모델을 많이 돌렸지만 성능 상한이 크게 움직이지 않았다”는 점을 데이터 한계의 근거로 사용한다.
4. 교수님 질문 대비 시 `processed/additional_experiments/` 산출물을 근거 자료로 제시한다.
