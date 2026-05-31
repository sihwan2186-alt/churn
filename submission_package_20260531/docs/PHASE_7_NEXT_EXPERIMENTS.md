# Phase 7: 추가 진행 후보 실험 점검

마지막 업데이트: 2026-05-28

## 실행 목적

전체 파일을 확인한 결과, 이미 Phase 6까지 모델 비교, 논문 재현, CV, feature ablation, 비용 분석, calibration, segment 분석이 완료되어 있었다. 새 모델을 무작정 더 추가하는 것보다 다음 두 가지 빈칸을 확인하는 것이 가장 가치 있다고 판단했다.

1. 추가 실험 best였던 `BalancedBagging_tree_depthnone_leaf25`가 5-fold CV에서도 안정적인지 확인
2. 이미 생성된 `processed/paper_ablation_variants/`의 paper-core, ZIP top-N, KA 추상화 variant를 실제 모델 성능으로 벤치마크

실행 스크립트:

```powershell
.\.venv\Scripts\python.exe phase_7_next_experiments.py
```

출력 위치:

- `processed/phase_7_next_experiments/tuned_candidate_cv_summary.csv`
- `processed/phase_7_next_experiments/tuned_candidate_cv_fold_results.csv`
- `processed/phase_7_next_experiments/paper_ablation_benchmark.csv`
- `processed/phase_7_next_experiments/paper_ablation_best_by_variant.csv`
- `processed/phase_7_next_experiments/paper_ablation_top30.csv`
- `processed/phase_7_next_experiments/phase_7_next_experiments_summary.json`

## 1. Tuned BalancedBagging CV 검증

추가 모델 실험에서 hold-out 기준 가장 좋았던 `BalancedBagging_tree_depthnone_leaf25`를 기존 Phase 4 CV 후보와 같은 방식으로 검증했다. 전처리, 인코딩, 스케일링, SVMSMOTE는 각 fold의 train partition 안에서만 fit했다. 이 CV는 기존 Phase 4와 맞추기 위해 threshold 0.50을 사용했다.

| Variant | Model | CV F1 mean | CV F1 SD | Recall mean | Precision mean | PR-AUC mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| with ZIP | `BalancedBagging_original` | 0.1455 | 0.0126 | 0.5248 | 0.0845 | 0.0910 |
| with ZIP | `EasyEnsemble_original` | 0.1445 | 0.0117 | 0.5835 | 0.0824 | 0.0907 |
| with ZIP | `BalancedBagging_tree_depthnone_leaf25` | 0.1418 | 0.0215 | 0.4679 | 0.0836 | 0.0916 |
| without ZIP | `LogisticRegression_SMOTE` | 0.1309 | 0.0154 | 0.1743 | 0.1053 | 0.0876 |

해석:

- tuned BalancedBagging은 hold-out에서는 F1 0.1605였지만, 5-fold CV 평균은 0.1418로 기존 `BalancedBagging_original` 0.1455보다 낮았다.
- 표준편차도 0.0215로 더 커서 안정성은 오히려 약했다.
- 따라서 `BalancedBagging_tree_depthnone_leaf25`를 최종 운영 후보로 교체할 근거는 부족하다.
- 최종 보고서의 CV 안정성 주장은 기존처럼 `BalancedBagging_original` 또는 `EasyEnsemble_original` 중심으로 두는 것이 안전하다.

## 2. Paper Ablation Variant 벤치마크

기존 `paper_ablation_variants.py`는 paper-core, ZIP top-50, KA abstract/research variant를 생성했지만, 이 variant들을 실제 모델별로 일괄 평가하는 표는 없었다. Phase 7에서는 각 variant에 대해 `LogisticRegression_SMOTE`, `BalancedBagging_original`, `BalancedBagging_tree_depthnone_leaf25`, `EasyEnsemble_original`을 평가했다. Threshold는 train 내부 validation split에서 선택하고 test에 1회 적용했다.

상위 결과:

| Variant | Model | Threshold | F1 | Recall | Precision | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `paper_core_zip_log_ka_abstract` | `BalancedBagging_original` | 0.53 | 0.1561 | 0.4495 | 0.0944 | 0.0955 |
| `extended_zip_top50_log_sqrt_interactions` | `BalancedBagging_tree_depthnone_leaf25` | 0.55 | 0.1545 | 0.3303 | 0.1008 | 0.1004 |
| `extended_zip_log_sqrt_interactions` | `BalancedBagging_original` | 0.55 | 0.1509 | 0.3578 | 0.0956 | 0.1053 |
| `paper_core_zip_log` | `BalancedBagging_tree_depthnone_leaf25` | 0.54 | 0.1460 | 0.3394 | 0.0930 | 0.1052 |
| `extended_no_zip_log_sqrt_interactions` | `BalancedBagging_original` | 0.55 | 0.1453 | 0.3486 | 0.0918 | 0.0985 |
| `extended_zip_top50_log_sqrt_ka_research` | `LogisticRegression_SMOTE` | 0.44 | 0.1412 | 0.3394 | 0.0892 | 0.0850 |

해석:

- paper-ablation 계열의 최고 F1은 0.1561로 기존 최종 hold-out F1 0.1681을 넘지 못했다.
- KA 실명 대신 `KA_is_code_type`, `KA_type_x_premium`만 사용하는 `paper_core_zip_log_ka_abstract`가 가장 높게 나와, 배포 친화적인 KA 추상화 feature는 발표 보조 근거로 쓸 수 있다.
- `extended_zip_top50_log_sqrt_interactions`는 F1 0.1545, precision 0.1008로 비교적 균형이 좋지만, 기존 최종 모델 교체 수준은 아니다.
- 연구용 KA target/frequency encoding은 최고 F1 0.1412로 기대보다 낮았다. target-dependent feature를 최종 주장에 강하게 쓰지 않는 것이 안전하다.

## 결론

Phase 7 실험 결과, 최종 모델이나 핵심 결론을 바꿀 필요는 없다.

- F1 기준 최종 모델: `without_billing_zip + LogisticRegression_SMOTE` 유지
- CV 안정성 모델: `with_billing_zip + BalancedBagging_original` / `EasyEnsemble_original` 유지
- 추가 후보 `BalancedBagging_tree_depthnone_leaf25`: hold-out 개선은 있었지만 CV 안정성 부족
- paper/KA ablation: 최종 성능 개선보다는 발표에서 "KA 실명 제거 후에도 경쟁력 있는 추상화 variant를 검토했다"는 보조 근거로 활용

따라서 여기서 더 진행한다면 모델 성능 개선보다 실제 PPT 제작, 발표 멘트 정리, 교수님 질문 대비가 우선이다. 실험을 하나 더 한다면 모델 추가가 아니라 paired bootstrap CI 또는 McNemar test처럼 결과 신뢰도를 보강하는 통계 검정이 가장 적절하다.
