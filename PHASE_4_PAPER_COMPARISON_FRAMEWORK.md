# Phase 4: Paper Comparison Framework

마지막 업데이트: 2026-05-27

## 1. 비교 원칙

Makokha et al. (2026)과의 비교는 세 층으로 분리한다.

1. **재현 비교**: 논문 best 모델인 EasyEnsemble끼리 비교한다.
2. **추가 발견 비교**: 논문에 없던 `LogisticRegression_SMOTE`, ZIP ablation, cost threshold 등을 별도 기여로 제시한다.
3. **운영 지점 비교**: F1-best, recall-heavy, cost-sensitive threshold는 서로 다른 목적이므로 같은 표 안에서 직접 우열로만 해석하지 않는다.

이 원칙을 적용하면 핵심 메시지는 다음이다.

> 본 프로젝트는 EasyEnsemble 기준으로 논문 결과를 거의 재현했고, 추가 모델/feature/threshold 실험을 통해 다른 운영 목적에서 경쟁력 있는 대안을 제시했다. 다만 단일 hold-out 최고점인 `LR_SMOTE F1=0.1681`은 5-fold CV에서 평균 `0.1309`로 낮아져, “논문보다 우월”이 아니라 “추가 최적화 가능성”으로 해석한다.

## 2. 방법론 비교표

| 항목 | Makokha et al. (2026) | 우리 프로젝트 | 차이 및 시사점 |
| --- | --- | --- | --- |
| 데이터셋 | 불가리아 통신사 B2B, 약 8,454개 계정, 14 raw features | 동일 원천 CSV. 로컬 파일 기준 8,453행, PID 중복 17건 제거 후 8,436행 | 동일 원천이라 비교 가능. 행 수 표기는 로컬 CSV 기준으로 명시 |
| 결측 처리 | subscriber 필드 0 대체, 범주형 Unknown, ARPU/ZIP 중앙값 | 동일 전략 | 방법론 재현 |
| Feature Engineering | 14 raw -> 22 final로 설명 | 메인 파이프라인은 ZIP 포함 55개, ZIP 제외 52개. 별도 paper-aligned ablation은 실제 CSV 기준 20/19개 core 생성 | 논문 22개는 현재 CSV의 미상 raw 2개 때문에 완전 동일 재현 불가. 대신 core/extended variant를 분리 |
| Billing_ZIP | 포함 후 label encoding | 포함/제외/top-N variant 비교 | 추가 기여: ZIP 효과가 모델 계열에 의존함을 실증 |
| 인코딩 | Label encoding, CatBoost native categorical | Label encoding, CatBoost native categorical | 대체로 일치 |
| 스케일링 | Z-score, train 기준 fit | Z-score, train 기준 fit | 일치 |
| 불균형 처리 | SVMSMOTE 등 비교 후 채택 | SVMSMOTE train-only 적용 | 일치 |
| 모델 범위 | EasyEnsemble, RUSBoost, XGBoost, LightGBM, CatBoost, HistGB, BalancedBagging, MLP, Voting, Stacking 등 | CatBoost, EasyEnsemble, RUSBoost, BalancedBagging, GradientBoosting, HistGB, RandomForest, XGBoost, LR_SMOTE 등 | 논문은 breadth, 우리는 ZIP/threshold/segment/feature ablation의 depth |
| Threshold | F1 최대화 + recall 제약, 단일 운영 threshold | validation F1 + `recall >= 0.30`, 추가 cost-sensitive sweep | 추가 기여: business scenario별 threshold 민감도 |
| 교차검증 | 5-fold Stratified CV, EasyEnsemble F1 `0.121 ± 0.018` | Phase 4에서 5-fold CV 실행. key model 4개 비교 | 공정 비교 보강 완료 |
| 확률 보정 | Isotonic calibration | 미적용 | 한계. cost threshold 결과도 probability가 아니라 score sweep으로 해석해야 함 |
| 평가 지표 | Acc, Bal.Acc, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC | 동일 지표 대부분 보고. `mcc`는 CSV에 포함 | MCC 미보고 한계는 해소됨 |
| 비용 분석 | 단일 비용 가정 기반 revenue protection | Phase 3-B에서 6개 비용 scenario sensitivity 구현 | 단일 추정은 아직 논문보다 약하지만 민감도 분석은 확장 |
| Explainability | SHAP, LIME, permutation FI | Permutation FI + feature group ablation. SHAP/LIME 미구현 | 설명가능성은 부분 구현 |
| 배포 | FastAPI/dashboard | 미구현 | 운영 배포는 향후 과제 |

## 3. Hold-Out 성능 비교

| 기준 | 모델/운영 지점 | F1 | Recall | Precision | PR-AUC | MCC | 해석 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 논문 best | EasyEnsemble | 0.129 | 0.382 | 0.077 | 0.079 | 0.034 | 논문 기준점 |
| 재현 비교 | with ZIP `EasyEnsemble_original` | 0.1284 | 0.5872 | 0.0721 | 0.0845 | 0.0321 | F1 거의 동일, recall은 더 높고 precision은 낮음 |
| 우리 F1-best hold-out | without ZIP `LogisticRegression_SMOTE` | 0.1681 | 0.2661 | 0.1229 | 0.0879 | 0.0956 | precision/F1은 높지만 recall 30% 미만 |
| 우리 business-balanced | with ZIP `BalancedBagging_original` | 0.1526 | 0.5872 | 0.0877 | 0.0871 | 0.0820 | recall 제약과 F1 균형에서 가장 실무적 |
| 우리 recall-heavy | with ZIP `CatBoost_native_categorical`, threshold 0.35 | 0.1310 | 0.8349 | 0.0711 | 0.0789 | 확인 필요 | 고재현율 운영점. FP 비용 큼 |

해석:

- 논문과의 직접 재현 비교는 `EasyEnsemble_original` 기준으로 해야 한다. 이 경우 F1 `0.1284`로 논문 `0.129`와 사실상 일치한다.
- `LR_SMOTE F1=0.1681`은 논문 모델보다 우수한 결과가 아니라, 논문에 없던 추가 모델 조합의 hold-out 발견이다.
- `Recall=0.8349`는 고재현율 threshold 운영점이며, `F1=0.1681`과 같은 모델/threshold에서 나온 수치가 아니다. 보고서에서 반드시 분리한다.

## 4. 5-Fold CV 비교

실행:

```powershell
.\.venv\Scripts\python.exe phase_4_cross_validation.py
```

출력:

- `processed/phase_4_paper_comparison/phase_4_cv_fold_results.csv`
- `processed/phase_4_paper_comparison/phase_4_cv_summary.csv`
- `processed/phase_4_paper_comparison/phase_4_cv_summary.json`

| Variant | Model | CV F1 mean ± SD | 95% CI of mean | Recall mean | Precision mean | PR-AUC mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| with ZIP | `BalancedBagging_original` | 0.1455 ± 0.0126 | [0.1345, 0.1565] | 0.5248 | 0.0845 | 0.0910 |
| with ZIP | `EasyEnsemble_original` | 0.1445 ± 0.0117 | [0.1342, 0.1547] | 0.5835 | 0.0824 | 0.0907 |
| without ZIP | `EasyEnsemble_original` | 0.1408 ± 0.0081 | [0.1337, 0.1480] | 0.5835 | 0.0801 | 0.0885 |
| without ZIP | `LogisticRegression_SMOTE` | 0.1309 ± 0.0154 | [0.1174, 0.1443] | 0.1743 | 0.1053 | 0.0876 |

논문 CV:

- EasyEnsemble CV F1 `0.121 ± 0.018`
- mean의 95% CI 근사: `0.121 ± 1.96*(0.018/sqrt(5)) = [0.105, 0.137]`

해석:

- 우리 `EasyEnsemble`/`BalancedBagging` CV 평균은 논문 CV 평균보다 높다.
- 그러나 CI가 일부 겹치므로, paired CV나 동일 split 기반 비교 없이는 통계적 우월성을 강하게 주장하지 않는다.
- `LR_SMOTE`는 hold-out F1 `0.1681`이었지만 CV 평균은 `0.1309`로 낮아졌다. 이는 단일 split 최고점의 낙관성을 보여준다.

보고서 문장:

> 5-fold CV 결과, 단일 hold-out에서 최고 F1을 보인 LogisticRegression+SVMSMOTE는 평균 F1 0.1309로 낮아져 split 민감성이 확인되었다. 반면 with-ZIP BalancedBagging과 EasyEnsemble은 각각 0.1455±0.0126, 0.1445±0.0117로 더 안정적인 성능을 보였다. 따라서 본 연구의 최종 주장은 “단일 모델의 압도적 우위”보다 “논문 재현 및 운영 목적별 대안 제시”로 정리하는 것이 타당하다.

## 5. 기여 비교표

| 기여 차원 | Makokha et al. (2026) | 우리 프로젝트 |
| --- | --- | --- |
| 재현성 | 원 논문 결과 제시 | EasyEnsemble hold-out F1 `0.1284`로 논문 `0.129` 재현 |
| 모델 추가 | 광범위 모델 비교 | LR_SMOTE, BalancedBagging, XGBoost 추가 확인. LR hold-out F1 높지만 CV에서는 약함 |
| ZIP 변수 실험 | ZIP 포함 단일 처리 | ZIP 포함/제외/top-N ablation으로 모델 의존성 확인 |
| Feature 기여도 | SHAP/LIME 중심 | Feature group ablation으로 categorical/interaction group 기여 정량화 |
| Segment 운영 분석 | CRM segment를 feature로 사용 | CRM segment별 recall/precision/FN revenue risk 분석 |
| Threshold | 단일 threshold | recall 제약 threshold + cost-sensitive scenario sweep |
| 비용 분석 | 단일 비용 가정 | 6개 비용 scenario 민감도 분석. 보수적 비용 구조에서는 campaign 가치 음수 |
| Explainability | SHAP + LIME + permutation FI | Permutation FI + ablation. SHAP/LIME은 미구현 |
| 배포 | FastAPI/dashboard | 미구현 |

## 6. 성능 차이 원인 분석

### 원인 1: 알고리즘과 feature representation

`LogisticRegression_SMOTE`가 hold-out에서 강했던 이유는 feature engineering이 비선형 관계를 ratio/log/sqrt/interaction 형태로 미리 펼쳐주었기 때문으로 해석할 수 있다. 선형 모델은 SVMSMOTE의 경계 주변 합성 샘플을 비교적 단순한 결정 경계로 활용한다.

다만 CV에서는 이 장점이 안정적으로 유지되지 않았다. 따라서 LR 결과는 “흥미로운 추가 발견”이지 최종적으로 논문보다 우월한 주장은 아니다.

### 원인 2: ZIP의 모델 의존 효과

`Billing_ZIP`은 tree ensemble에서는 지역 신호로 작동해 BalancedBagging F1/recall을 높였다. 반면 LR에서는 ZIP 제거가 더 좋았다. 이는 고카디널리티 label encoding이 선형 모델에는 noise로 작동할 수 있음을 보여준다.

### 원인 3: threshold 목적 차이

F1-best, recall-heavy, cost-sensitive threshold는 서로 다른 목적이다.

- F1-best: `LR_SMOTE`, F1 0.1681, recall 0.2661
- recall-constrained balanced: `BalancedBagging`, F1 0.1526, recall 0.5872
- recall-heavy: `CatBoost_native`, recall 0.8349, precision 0.0711
- cost-heavy: threshold 0.29에서 recall 1.0이 가능하지만 precision 0.0653

따라서 “우리 recall 0.8349와 F1 0.1681을 동시에 달성”으로 쓰면 안 된다.

### 원인 4: split variance

Test set의 churn positive는 109명이다. TP 몇 명 차이로 F1이 크게 변동한다. 실제로 LR hold-out F1은 0.1681이지만 CV 평균은 0.1309다. 단일 split 결과는 반드시 CV와 함께 해석한다.

## 7. 최종 보고서용 문장

### 재현성 문장

> 본 연구는 Makokha et al. (2026)의 동일 데이터셋과 전처리 원칙을 기반으로 EasyEnsembleClassifier를 재현한 결과, hold-out F1-score 0.1284를 얻어 논문 보고값 0.129와 사실상 일치하는 결과를 확인하였다. 이는 전처리 및 불균형 처리 파이프라인의 재현가능성을 뒷받침한다.

### 추가 기여 문장

> 선행 연구가 Billing_ZIP을 포함한 단일 설정만 제시한 것과 달리, 본 연구는 ZIP 포함/제외 및 top-N grouping variant를 구성하여 고카디널리티 지리 변수가 모델 계열별로 상이하게 작동함을 보였다. Tree ensemble에서는 ZIP 정보가 recall과 F1을 개선했으나, LogisticRegression에서는 ZIP 제외 설정이 더 높은 F1을 달성하였다.

### 성능 비교 문장

> 단일 hold-out test에서는 without-ZIP LogisticRegression+SVMSMOTE가 F1-score 0.1681을 달성하여 논문 EasyEnsemble 기준값 0.129를 상회하였다. 그러나 5-fold CV에서는 해당 모델의 평균 F1이 0.1309로 낮아졌고, with-ZIP BalancedBagging과 EasyEnsemble이 각각 0.1455±0.0126, 0.1445±0.0117로 더 안정적이었다. 따라서 본 결과는 특정 모델의 일방적 우위라기보다, 운영 목적과 feature 처리 전략에 따라 최적 모델이 달라짐을 보여주는 증거로 해석한다.

### 운영 문장

> 비용 민감 threshold 분석 결과, 논문 기준 비용 비율 45:1에서는 recall을 극대화하는 낮은 threshold가 기대가치를 높였으나, 캠페인 비용이 커지는 보수적 시나리오에서는 기대가치가 음수로 전환되었다. 이는 churn 모델의 운영 성과가 모델 F1뿐 아니라 고객가치, 캠페인 비용, retention 성공률 가정에 크게 의존함을 시사한다.

## 8. 한계 및 향후 연구

| 한계 | 현재 상태 | 향후 작업 |
| --- | --- | --- |
| 확률 보정 | 미적용 | Isotonic/Platt calibration, Brier score, ECE 보고 |
| SHAP/LIME | 미구현 | Tree-SHAP으로 ZIP, interaction, segment 효과 시각화 |
| 배포 | 미구현 | Streamlit 또는 FastAPI scoring prototype |
| 통계 검정 | 독립 CV 요약만 있음 | 동일 fold paired 비교, bootstrap CI, McNemar/paired t-test |
| 비용 분석 | fixed scenario sensitivity | 고객별 ARPU 기반 individualized net value 분석 |
| 데이터 한계 | 정적 CRM snapshot | 기간별 사용량 변화, 결제 실패, 계약 만료 등 temporal feature 추가 |

## 9. 최종 결론

> 본 프로젝트는 논문 결과를 재현하는 데 성공했고, ZIP ablation, feature group ablation, CRM segment 분석, cost-sensitive threshold sensitivity를 통해 논문이 다루지 않은 운영적 질문을 확장했다. 단일 hold-out 최고 성능은 논문보다 높았지만 CV에서는 그 차이가 완화되었으므로, 최종 주장은 “논문 대비 압도적 성능 우위”가 아니라 “재현 가능한 baseline 위에서 운영 목적별 모델 선택 프레임워크를 제시했다”로 정리하는 것이 가장 탄탄하다.

