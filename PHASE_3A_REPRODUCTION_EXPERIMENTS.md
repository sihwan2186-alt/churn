# Phase 3-A: Paper Model Reproduction Experiments

마지막 업데이트: 2026-05-27

## 1. 공정한 비교 결론

`F1 0.1681 > 논문 0.129`는 직접 비교가 아니다.

- 논문 best: `EasyEnsemble`, F1 `0.129`
- 우리 재현: `EasyEnsemble_original`, F1 `0.1284` with ZIP, `0.1268` without ZIP
- 우리 추가 발견: `LogisticRegression_SMOTE`, F1 `0.1681` without ZIP

따라서 보고서 문장은 다음처럼 분리한다.

> 본 연구의 EasyEnsemble F1(0.128)은 논문 원본(0.129)과 ±0.001 수준에서 일치하여 구현 재현성을 확인하였다. LogisticRegression+SVMSMOTE의 F1(0.168)은 논문에서 실험하지 않은 조합으로 추가적 발견이다.

## 2. 추가 구현 사항

`preprocess_churn.py`에 다음을 추가했다.

- `XGBoost_SMOTE`
- `EasyEnsemble_n50_original`
- threshold 선택 기준: validation F1 최대화 + `recall >= 0.30` 제약
- 실행 옵션: `--skip-error-analysis`

`requirements.txt`에는 `xgboost`를 추가했다.

## 3. 주요 모델 결과

| Variant | Model | F1 | Recall | Precision | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| without ZIP | `LogisticRegression_SMOTE` | 0.1681 | 0.2661 | 0.1229 | 0.0879 |
| with ZIP | `BalancedBagging_original` | 0.1526 | 0.5872 | 0.0877 | 0.0871 |
| without ZIP | `EasyEnsemble_n50_original` | 0.1371 | 0.6239 | 0.0770 | 0.0835 |
| with ZIP | `EasyEnsemble_n50_original` | 0.1316 | 0.5963 | 0.0739 | 0.0884 |
| with ZIP | `EasyEnsemble_original` | 0.1284 | 0.5872 | 0.0721 | 0.0845 |
| without ZIP | `EasyEnsemble_original` | 0.1268 | 0.5505 | 0.0717 | 0.0817 |
| with ZIP | `XGBoost_SMOTE` | 0.1268 | 0.6147 | 0.0707 | 0.0791 |
| without ZIP | `XGBoost_SMOTE` | 0.1211 | 0.6972 | 0.0663 | 0.0790 |

해석:

- EasyEnsemble은 논문 수치와 거의 일치한다.
- XGBoost는 recall은 높지만 precision이 낮아 F1 기준으로 EasyEnsemble 재현값을 넘지 못했다.
- `EasyEnsemble_n50_original`은 n=10 대비 recall/PR-AUC가 일부 개선되지만, F1은 크게 개선되지 않았다.

## 4. Recall 제약 Threshold 결과

선택 기준:

```python
MIN_RECALL_CONSTRAINT = 0.30
best = max(validation_f1) among rows where validation_recall >= 0.30
```

상위 test F1 결과:

| Variant | Model | Threshold | Test F1 | Test Recall | Test Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| with ZIP | `BalancedBagging_original` | 0.50 | 0.1526 | 0.5872 | 0.0877 |
| with ZIP | `LogisticRegression_SMOTE` | 0.46 | 0.1507 | 0.3028 | 0.1003 |
| without ZIP | `BalancedBagging_original` | 0.44 | 0.1353 | 0.7064 | 0.0748 |
| without ZIP | `LogisticRegression_SMOTE` | 0.37 | 0.1330 | 0.4862 | 0.0770 |

핵심 해석:

- default threshold에서 `LogisticRegression_SMOTE`는 최고 F1이지만 recall `0.2661`로 30% 제약에 미달한다.
- recall 제약을 명시하면 `BalancedBagging_original`이 실무형 best가 된다.
- 이 결과는 논문의 business-driven threshold 관점과 잘 맞는다.

## 5. 재현 명세

- Dataset: `Baza customer Telecom v2.csv`
- Raw rows: 8,453
- PID 중복 제거 후 rows: 8,436
- Target after dedup: `No=7,891`, `Yes=545`
- Train/Test: 80:20 stratified split, `random_state=42`
- Resampling: `SVMSMOTE(random_state=42)`, train only
- Threshold validation: train split 내부 25%, stratified, `random_state=42`
- Test set: 최종 평가에만 사용
- Environment checked: `scikit-learn 1.8.0`, `imbalanced-learn 0.14.1`, `xgboost 3.2.0`

## 6. 실행 명령

전체 비교와 threshold 결과를 빠르게 갱신:

```powershell
.\.venv\Scripts\python.exe preprocess_churn.py --skip-error-analysis
```

전체 error/permutation analysis까지 포함:

```powershell
.\.venv\Scripts\python.exe preprocess_churn.py
```

주의: 전체 error/permutation analysis는 실행 시간이 길다.

