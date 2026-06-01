# Phase 8: 통계 검정 보강

마지막 업데이트: 2026-05-28

## 실행 목적

단일 hold-out 성능표만으로 모델 차이를 해석하면 과장될 수 있으므로, Phase 6에서 저장한 test-set 예측값을 이용해 통계적 신뢰도를 보강했다.

실행 스크립트:

```powershell
.\.venv\Scripts\python.exe phase_8_statistical_validation.py
```

출력 위치:

- `processed/phase_8_statistical_validation/bootstrap_metric_ci.csv`
- `processed/phase_8_statistical_validation/bootstrap_pairwise_metric_differences.csv`
- `processed/phase_8_statistical_validation/mcnemar_paired_tests.csv`
- `processed/phase_8_statistical_validation/phase_8_statistical_validation_summary.json`

## 1. Bootstrap 95% CI

Test rows 1,688개를 5,000회 bootstrap resampling하여 주요 운영점의 F1, recall, precision 신뢰구간을 계산했다.

| Case | F1 point | F1 95% CI | Recall point | Recall 95% CI | Precision point | Precision 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LR_no_zip_f1 | 0.1681 | [0.1155, 0.2229] | 0.2661 | [0.1818, 0.3524] | 0.1229 | [0.0823, 0.1674] |
| BalancedBagging_with_zip | 0.1526 | [0.1200, 0.1862] | 0.5872 | [0.4947, 0.6814] | 0.0877 | [0.0679, 0.1090] |
| EasyEnsemble_with_zip | 0.1284 | [0.1004, 0.1563] | 0.5872 | [0.4952, 0.6742] | 0.0721 | [0.0557, 0.0893] |
| CatBoost_native_with_zip | 0.1310 | [0.1072, 0.1557] | 0.8349 | [0.7624, 0.9000] | 0.0711 | [0.0574, 0.0856] |
| XGBoost_with_zip | 0.1242 | [0.1020, 0.1455] | 0.9266 | [0.8739, 0.9717] | 0.0665 | [0.0541, 0.0789] |

해석:

- LR의 F1 point는 가장 높지만 CI가 넓다. test positive가 109명뿐이므로 split uncertainty가 크다.
- BalancedBagging은 LR보다 recall CI가 확실히 높고 precision CI는 낮다.
- XGBoost는 recall CI가 가장 높지만 precision은 가장 낮은 축에 있다.
- 따라서 “한 모델이 통계적으로 모든 면에서 우월”이 아니라 “운영 목적별 차이가 안정적으로 관찰됨”으로 해석하는 것이 안전하다.

## 2. McNemar Paired Test

McNemar test는 두 모델의 전체 정오분류 패턴이 다른지 확인한다. 여기서는 continuity-corrected chi-square approximation을 사용했다.

| Left | Right | Left only correct | Right only correct | p-value | 해석 |
| --- | --- | ---: | ---: | ---: | --- |
| LR_no_zip_f1 | BalancedBagging_with_zip | 489 | 65 | 3.26e-72 | error pattern이 다름 |
| LR_no_zip_f1 | EasyEnsemble_with_zip | 629 | 47 | 1.32e-110 | error pattern이 다름 |
| BalancedBagging_with_zip | EasyEnsemble_with_zip | 189 | 31 | 3.50e-26 | error pattern이 다름 |
| CatBoost_native_with_zip | XGBoost_with_zip | 268 | 65 | 1.76e-28 | error pattern이 다름 |
| BalancedBagging_with_zip | CatBoost_native_with_zip | 571 | 75 | 1.77e-84 | error pattern이 다름 |

주의:

- McNemar는 전체 accuracy 관점의 paired correctness test다.
- 이 프로젝트는 class imbalance가 심하므로 McNemar만으로 최종 모델을 고르면 안 된다.
- 대신 “모델들이 같은 실수를 하는 것이 아니라 서로 다른 운영 trade-off를 가진다”는 보조 근거로 쓰는 것이 적절하다.

## 보고서용 문장

> Hold-out test set에 대해 5,000회 bootstrap 신뢰구간을 계산한 결과, LR의 F1 point estimate는 0.1681로 가장 높았으나 95% CI가 [0.1155, 0.2229]로 넓었다. 반면 BalancedBagging은 recall 0.5872 [0.4947, 0.6814], XGBoost는 recall 0.9266 [0.8759, 0.9717]로 recall 중심 운영에서 일관된 장점을 보였다. 또한 McNemar paired test에서 주요 모델 쌍의 error pattern이 유의하게 달라, 모델 간 차이는 단순 점수 차이가 아니라 서로 다른 false positive/false negative trade-off로 해석하는 것이 타당하다.
