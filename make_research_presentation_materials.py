import os
from pathlib import Path
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
from imblearn.over_sampling import SVMSMOTE
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from preprocess_churn import (
    RANDOM_STATE,
    THRESHOLD_GRID,
    build_comparison_models,
    fit_model,
    load_native_catboost_split,
    load_processed_split,
    positive_scores,
)


OUTPUT_ROOT = Path("processed") / "research_presentation"
COLUMN_SPLIT_ROOT = Path("processed") / "column_split_datasets"

MODEL_VARIANT_DIRS = {
    "with_billing_zip": Path("processed") / "model_a_with_billing_zip",
    "without_billing_zip": Path("processed") / "model_b_without_billing_zip",
}

PAPER_REFERENCE = {
    "model": "EasyEnsembleClassifier + SVMSMOTE",
    "f1": 0.1290,
    "recall": 0.3820,
    "precision": 0.0770,
    "accuracy": 0.6640,
    "balanced_accuracy": 0.5330,
    "roc_auc": 0.5510,
    "pr_auc": 0.0790,
    "mcc": 0.0340,
}

COLUMN_KO = {
    "CRM_PID_Value_Segment": "CRM 고객가치 등급",
    "EffectiveSegment": "실질 비즈니스 세그먼트",
    "Billing_ZIP": "청구 우편번호",
    "Active_subscribers": "활성 가입자 수",
    "Not_Active_subscribers": "비활성 가입자 수",
    "Suspended_subscribers": "정지 가입자 수",
    "Total_SUBs": "전체 가입자 수",
    "AvgMobileRevenue": "평균 모바일 매출",
    "AvgFIXRevenue": "평균 유선 매출",
    "TotalRevenue": "총 매출",
    "ARPU": "가입자당 평균 매출",
}

COLUMN_ROLE = {
    "CRM_PID_Value_Segment": "CRM이 판단한 고객가치 등급별 이탈 위험을 확인한다.",
    "EffectiveSegment": "실제 사업 규모/유형별 이탈 위험을 확인한다.",
    "Billing_ZIP": "지역별 이탈 위험과 논문 SHAP의 geographic billing zone 신호를 검증한다.",
    "Active_subscribers": "서비스를 실제 사용하는 가입자 규모를 확인한다.",
    "Not_Active_subscribers": "가입 후 미활성 상태인 고객의 이탈 위험을 확인한다.",
    "Suspended_subscribers": "정지 가입자 존재 여부와 결측 자체의 위험 신호를 확인한다.",
    "Total_SUBs": "고객의 전체 계약 규모와 대형 계정 이탈 위험을 확인한다.",
    "AvgMobileRevenue": "모바일 매출 규모가 이탈과 어떤 관계인지 확인한다.",
    "AvgFIXRevenue": "유선/번들 이용 여부가 이탈 방어 효과를 갖는지 확인한다.",
    "TotalRevenue": "고객 총 매출 규모와 이탈 위험을 확인한다.",
    "ARPU": "가입자당 수익성과 이탈 위험을 확인한다.",
}

PREPROCESS_REASON = {
    "CRM_PID_Value_Segment": "범주 수가 작아 one-hot 또는 CatBoost가 적합하며, 세그먼트와 교차하면 고위험 조합을 설명할 수 있다.",
    "EffectiveSegment": "SOHO/VSE/SME처럼 사업 유형 자체가 이탈률 차이를 만들기 때문에 범주형으로 보존한다.",
    "Billing_ZIP": "456개 고유값으로 one-hot만 쓰면 차원이 커지므로 CatBoost 또는 leakage-safe target encoding이 적합하다.",
    "Active_subscribers": "우측 왜곡과 이상치가 있어 log1p와 outlier flag가 필요하다.",
    "Not_Active_subscribers": "결측률이 높고 결측 자체가 의미를 가질 수 있어 missing flag를 추가한다.",
    "Suspended_subscribers": "95% 이상 결측이므로 값보다 존재 여부가 중요해 missing/exists flag를 분리한다.",
    "Total_SUBs": "대형 계정에서 이탈률이 높아 구간화, log1p, outlier flag가 필요하다.",
    "AvgMobileRevenue": "매출 규모가 클수록 이탈률이 상승하는 패턴이 있어 log/sqrt 변환과 선형 기준모델을 비교한다.",
    "AvgFIXRevenue": "0이 많은 변수라 유선 매출 존재 여부 flag가 번들 lock-in 신호가 된다.",
    "TotalRevenue": "AvgMobileRevenue와 거의 중복되므로 단독 사용보다 비율/상호작용 변수로 변환한다.",
    "ARPU": "왜도가 커서 log1p가 필요하고, 단순 평균보다 구간별 위험 확인이 중요하다.",
}

DISPLAY_NAMES = {
    "comparison_item": "비교항목",
    "paper_method": "논문 방식",
    "our_method": "우리 방식",
    "reason": "진행 이유",
    "source": "구분",
    "variant": "데이터/피처 조건",
    "model": "모델",
    "threshold": "임계값",
    "selected_threshold": "선택 임계값",
    "f1": "F1",
    "recall": "Recall",
    "precision": "Precision",
    "pr_auc": "PR-AUC",
    "mcc": "MCC",
    "column_ko": "한글 컬럼명",
    "csv_role": "CSV 역할",
    "highest_risk_group": "최고위험 값/구간",
    "highest_risk_churn_percent": "최고위험 이탈률(%)",
    "best_single_column_model": "단일 컬럼 best model",
    "best_single_column_f1": "단일 컬럼 F1",
}


def metrics_from_scores(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def fit_and_score_candidate(
    *,
    candidate: tuple[str, Any, str],
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
    X_fit_native: pd.DataFrame,
    X_eval_native: pd.DataFrame,
) -> np.ndarray:
    model_name, model, train_kind = candidate
    fitted = clone(model)

    if train_kind == "native_categorical":
        fit_model(fitted, X_fit_native, y_fit, train_kind)
        return positive_scores(fitted, X_eval_native)

    if train_kind == "resampled":
        smote = SVMSMOTE(random_state=RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X_fit, y_fit)
        fit_model(
            fitted,
            pd.DataFrame(X_resampled, columns=X_fit.columns),
            pd.Series(y_resampled).astype(int),
            train_kind,
        )
    else:
        fit_model(fitted, X_fit, y_fit, train_kind)

    return positive_scores(fitted, X_eval)


def run_ensemble_experiment() -> pd.DataFrame:
    output_path = OUTPUT_ROOT / "ensemble_model_comparison.csv"
    if output_path.exists():
        return pd.read_csv(output_path)

    available = {name: (name, model, kind) for name, model, kind in build_comparison_models()}
    ensemble_sets = {
        "soft_avg_lr_balancedbagging": [
            "LogisticRegression_SMOTE",
            "BalancedBagging_original",
        ],
        "soft_avg_lr_balancedbagging_catboost": [
            "LogisticRegression_SMOTE",
            "BalancedBagging_original",
            "CatBoost_original_balanced",
        ],
        "soft_avg_lr_bagging_catboost_native": [
            "LogisticRegression_SMOTE",
            "BalancedBagging_original",
            "CatBoost_original_balanced",
            "CatBoost_native_categorical",
        ],
    }

    rows = []
    for variant, variant_dir in MODEL_VARIANT_DIRS.items():
        (
            X_train,
            X_test,
            y_train,
            y_test,
            _X_train_resampled,
            _y_train_resampled,
        ) = load_processed_split(variant_dir)
        X_train_native, X_test_native = load_native_catboost_split(variant_dir)

        train_idx, valid_idx = train_test_split(
            np.arange(len(y_train)),
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

        X_fit = X_train.iloc[train_idx].reset_index(drop=True)
        X_valid = X_train.iloc[valid_idx].reset_index(drop=True)
        y_fit = y_train.iloc[train_idx].reset_index(drop=True)
        y_valid = y_train.iloc[valid_idx].reset_index(drop=True)
        X_fit_native = X_train_native.iloc[train_idx].reset_index(drop=True)
        X_valid_native = X_train_native.iloc[valid_idx].reset_index(drop=True)

        for ensemble_name, model_names in ensemble_sets.items():
            candidates = [available[name] for name in model_names]
            validation_scores = []
            test_scores = []
            for candidate in candidates:
                validation_scores.append(
                    fit_and_score_candidate(
                        candidate=candidate,
                        X_fit=X_fit,
                        y_fit=y_fit,
                        X_eval=X_valid,
                        X_fit_native=X_fit_native,
                        X_eval_native=X_valid_native,
                    )
                )
                test_scores.append(
                    fit_and_score_candidate(
                        candidate=candidate,
                        X_fit=X_train.reset_index(drop=True),
                        y_fit=y_train.reset_index(drop=True),
                        X_eval=X_test.reset_index(drop=True),
                        X_fit_native=X_train_native.reset_index(drop=True),
                        X_eval_native=X_test_native.reset_index(drop=True),
                    )
                )

            validation_mean = np.mean(validation_scores, axis=0)
            threshold_rows = [
                metrics_from_scores(y_valid, validation_mean, threshold)
                for threshold in THRESHOLD_GRID
            ]
            best_threshold = pd.DataFrame(threshold_rows).sort_values(
                ["f1", "recall", "precision"], ascending=[False, False, False]
            ).iloc[0]["threshold"]

            test_mean = np.mean(test_scores, axis=0)
            row = {
                "variant": variant,
                "model": ensemble_name,
                "train_data": "soft_average",
                "members": "+".join(model_names),
                "selected_threshold": float(best_threshold),
            }
            row.update(metrics_from_scores(y_test, test_mean, float(best_threshold)))
            rows.append(row)

    result = pd.DataFrame(rows).sort_values(["f1", "recall"], ascending=[False, False])
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def top_group_for_column(all_rates: pd.DataFrame, column: str, min_rows: int = 10) -> pd.Series:
    subset = all_rates[(all_rates["column"] == column) & (all_rates["rows"] >= min_rows)]
    if subset.empty:
        subset = all_rates[all_rates["column"] == column]
    return subset.sort_values(["yes_rate", "rows"], ascending=[False, False]).iloc[0]


def low_group_for_column(all_rates: pd.DataFrame, column: str, min_rows: int = 10) -> pd.Series:
    subset = all_rates[(all_rates["column"] == column) & (all_rates["rows"] >= min_rows)]
    if subset.empty:
        subset = all_rates[all_rates["column"] == column]
    return subset.sort_values(["yes_rate", "rows"], ascending=[True, False]).iloc[0]


def build_csv_role_summary() -> pd.DataFrame:
    profile = pd.read_csv(COLUMN_SPLIT_ROOT / "03_profiles" / "column_profile.csv")
    best_models = pd.read_csv(COLUMN_SPLIT_ROOT / "05_single_column_best_models.csv")
    all_rates = pd.read_csv(
        COLUMN_SPLIT_ROOT / "03_profiles" / "all_column_value_yes_no_rate_summary.csv"
    )

    rows = []
    for _, profile_row in profile.iterrows():
        col = profile_row["column"]
        best = best_models[best_models["column"] == col].iloc[0]
        high = top_group_for_column(all_rates, col)
        low = low_group_for_column(all_rates, col)
        rows.append(
            {
                "column": col,
                "column_ko": COLUMN_KO.get(col, col),
                "csv_role": COLUMN_ROLE.get(col, ""),
                "why_preprocess_this_way": PREPROCESS_REASON.get(col, ""),
                "missing_rate_percent": float(profile_row["missing_rate"] * 100),
                "unique_values": int(profile_row["unique_values"]),
                "highest_risk_group": str(high["group"]),
                "highest_risk_rows": int(high["rows"]),
                "highest_risk_yes": int(high["yes_count"]),
                "highest_risk_no": int(high["no_count"]),
                "highest_risk_churn_percent": float(high["yes_rate_percent"]),
                "lowest_risk_group": str(low["group"]),
                "lowest_risk_churn_percent": float(low["yes_rate_percent"]),
                "best_single_column_model": best["model"],
                "best_single_column_f1": float(best["f1"]),
                "best_single_column_recall": float(best["recall"]),
                "best_single_column_precision": float(best["precision"]),
                "best_single_column_pr_auc": float(best["pr_auc"]),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        "best_single_column_f1", ascending=False
    )
    summary.to_csv(OUTPUT_ROOT / "csv_role_churn_model_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def build_paper_vs_ours_tables(ensemble_result: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_summary = pd.read_csv("final_model_summary.csv")
    model_comparison = pd.read_csv("processed/model_comparison_billing_zip.csv")

    rows = [
        {
            "comparison_item": "데이터 규모",
            "paper_method": "8,454 unique business accounts",
            "our_method": "원본 8,453행을 기준으로 분석하고 PID 중복/충돌을 별도 데이터 품질 이슈로 제시",
            "reason": "동일 PID에 Yes/No가 섞인 사례가 있어 단순 unique 처리보다 중복 기준을 명확히 해야 함",
        },
        {
            "comparison_item": "PID",
            "paper_method": "중복 제거 키로 사용",
            "our_method": "원본은 보존하고 모델/분리 CSV에서는 제외",
            "reason": "식별자는 예측 신호가 아니라 leakage 위험이 있으므로 학습 변수에서 제외",
        },
        {
            "comparison_item": "KA_name",
            "paper_method": "범주형 feature로 포함 후 label encoding",
            "our_method": "요청 기준에 따라 모델 데이터에서 제외",
            "reason": "담당자명은 조직 변화에 취약하고 운영상 민감한 변수라 고객 행동 중심 모델을 우선 구성",
        },
        {
            "comparison_item": "결측 처리",
            "paper_method": "비활성/정지 가입자 결측은 0, CRM 결측은 Unknown, ZIP/ARPU는 median",
            "our_method": "값 대체와 별도로 missing flag/exists flag를 보존",
            "reason": "정지 가입자 수처럼 결측 자체가 이탈률 차이를 보이는 MNAR 신호일 수 있음",
        },
        {
            "comparison_item": "범주형 처리",
            "paper_method": "Billing_ZIP 등 고카디널리티도 label encoding 중심",
            "our_method": "값별 이탈률 CSV, CatBoost native categorical, leakage-safe target encoding 후보로 분리",
            "reason": "지역별 이탈률 차이가 크고 ordinal label 값 자체에는 순서 의미가 없기 때문",
        },
        {
            "comparison_item": "로그 변환",
            "paper_method": "매출 변수 중심 log transform",
            "our_method": "매출뿐 아니라 가입자 수 변수도 왜도/이상치를 보고 log1p 후보로 처리",
            "reason": "가입자 수 변수도 강한 우측 왜곡과 이상치 이탈률 상승이 관찰됨",
        },
        {
            "comparison_item": "피처 엔지니어링",
            "paper_method": "active subscriber rate, interaction terms 등 22개 feature",
            "our_method": "active_rate, missing flag, outlier flag, CRM x Segment, zip risk 등을 추가 후보로 제안",
            "reason": "단일 변수 상관은 약하지만 구간/교차/결측에서 이탈 신호가 강하게 나타남",
        },
        {
            "comparison_item": "모델 선택",
            "paper_method": PAPER_REFERENCE["model"],
            "our_method": "LogisticRegression_SMOTE, BalancedBagging, CatBoost, soft ensemble 비교",
            "reason": "불균형 데이터라 F1형/Recall형 운영 목적별 모델을 분리해 비교해야 함",
        },
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_ROOT / "paper_vs_ours_comparison.csv", index=False, encoding="utf-8-sig")

    best_current = model_comparison.sort_values("f1", ascending=False).iloc[0]
    best_recall = model_comparison.sort_values("recall", ascending=False).iloc[0]
    best_ensemble = ensemble_result.sort_values("f1", ascending=False).iloc[0]
    performance_rows = [
        {
            "source": "논문 기준 best",
            "variant": "external_paper",
            "model": PAPER_REFERENCE["model"],
            "threshold": "not_reported",
            **{k: PAPER_REFERENCE[k] for k in ["f1", "recall", "precision", "accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "mcc"]},
        },
        {
            "source": "우리 단일 목적 F1 best",
            "variant": best_current["variant"],
            "model": best_current["model"],
            "threshold": 0.5,
            **{k: float(best_current[k]) for k in ["f1", "recall", "precision", "accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "mcc"]},
        },
        {
            "source": "우리 Recall 캠페인형 best",
            "variant": best_recall["variant"],
            "model": best_recall["model"],
            "threshold": 0.5,
            **{k: float(best_recall[k]) for k in ["f1", "recall", "precision", "accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "mcc"]},
        },
        {
            "source": "우리 soft ensemble best",
            "variant": best_ensemble["variant"],
            "model": best_ensemble["model"],
            "threshold": float(best_ensemble["selected_threshold"]),
            **{k: float(best_ensemble[k]) for k in ["f1", "recall", "precision", "accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "mcc"]},
        },
    ]
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(OUTPUT_ROOT / "paper_vs_ours_performance.csv", index=False, encoding="utf-8-sig")

    final_summary.to_csv(OUTPUT_ROOT / "final_model_summary_snapshot.csv", index=False, encoding="utf-8-sig")
    return comparison, performance


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> list[str]:
    shown = df[columns].copy()
    if max_rows is not None:
        shown = shown.head(max_rows)
    headers = [DISPLAY_NAMES.get(col, col) for col in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in shown.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_presentation_markdown(
    csv_summary: pd.DataFrame,
    paper_comparison: pd.DataFrame,
    performance: pd.DataFrame,
    ensemble_result: pd.DataFrame,
) -> None:
    lines = [
        "# 연구 발표 자료: 논문 대비 이탈 예측 개선 분석 [김시환]",
        "",
        "## 발표 핵심 메시지",
        "",
        "논문은 B2B 통신 고객 이탈 예측을 위해 SVMSMOTE와 EasyEnsembleClassifier를 중심으로 통합 전처리 파이프라인을 구성했다. 우리는 같은 문제를 더 설명 가능한 연구 흐름으로 확장하기 위해 원본을 보존하고, `PID`와 `KA_name`을 제외한 뒤 컬럼별/값별 CSV를 별도 데이터로 분리하여 각 변수의 역할, 이탈률, 단일 변수 성능 한계를 먼저 확인했다.",
        "",
        "핵심 결론은 단일 컬럼만으로는 F1이 0.15 근처에서 한계가 있으며, 최종 성능은 결측 flag, 이상치 flag, 활성 비율, 지역/세그먼트 교차 신호, 임계값 튜닝을 결합해야 올라간다는 점이다.",
        "",
        "## 논문은 이렇게, 우리는 이렇게",
        "",
    ]
    lines += markdown_table(
        paper_comparison,
        ["comparison_item", "paper_method", "our_method", "reason"],
    )
    lines += [
        "",
        "## 성능 비교",
        "",
    ]
    lines += markdown_table(
        performance,
        ["source", "variant", "model", "threshold", "f1", "recall", "precision", "pr_auc", "mcc"],
    )
    lines += [
        "",
        "## 컬럼별 CSV 역할과 단일 컬럼 모델 한계",
        "",
    ]
    lines += markdown_table(
        csv_summary,
        [
            "column_ko",
            "csv_role",
            "highest_risk_group",
            "highest_risk_churn_percent",
            "best_single_column_model",
            "best_single_column_f1",
        ],
        max_rows=11,
    )
    lines += [
        "",
        "## 복수 모델 앙상블 실험",
        "",
    ]
    lines += markdown_table(
        ensemble_result,
        ["variant", "model", "selected_threshold", "f1", "recall", "precision", "pr_auc", "mcc"],
    )
    lines += [
        "",
        "앙상블은 여러 모델의 예측 확률을 평균냈다. 검증셋에서 threshold를 고른 뒤 테스트셋에서 평가했으므로, 테스트셋 threshold 직접 최적화보다 보수적인 비교다.",
        "",
        "## 왜 이런 방식으로 진행했는가",
        "",
        "- 원본 CSV를 유지했다: 재현성과 감사 가능성을 위해 원본 데이터는 수정하지 않았다.",
        "- 컬럼별 CSV를 만들었다: 통합 전처리를 바로 하면 어떤 컬럼이 어떤 신호를 주는지 설명하기 어렵다.",
        "- `PID`를 제외했다: 고객 식별자는 예측 가능한 행동 신호가 아니라 leakage 위험이 있다.",
        "- `KA_name`을 제외했다: 담당자명은 조직 개편에 취약하고 개인/운영 민감도가 높아 고객 행동 중심 모델을 우선했다.",
        "- 결측 flag를 보존했다: 정지/비활성 가입자 결측은 단순 결측이 아니라 운영 기록의 부재라는 신호일 수 있다.",
        "- 수치형을 구간화해 이탈률을 봤다: Pearson 상관은 낮지만 특정 구간과 이상치에서 이탈률이 상승한다.",
        "- 여러 모델을 비교했다: 불균형 데이터에서는 정확도보다 F1, Recall, Precision, PR-AUC를 함께 봐야 한다.",
        "",
        "## 성능을 더 올리는 방안",
        "",
        "1. `CRM_PID_Value_Segment x EffectiveSegment` 교차 target encoding을 train fold 내부에서만 계산한다.",
        "2. `Billing_ZIP`은 최소 표본수 기준 smoothing target encoding 또는 CatBoost ordered statistics로 처리한다.",
        "3. `Suspended_subscribers_exists`, `Not_Active_subscribers_missing`, `has_fix_revenue`, `mobile_only` 같은 flag를 최종 모델에 명시적으로 넣는다.",
        "4. `TotalRevenue`와 `AvgMobileRevenue`는 상관이 매우 높으므로 둘을 동시에 넣기보다 비율/차이/대표 변수로 정리한다.",
        "5. 가입자 수와 매출 극단값은 제거하지 말고 outlier flag로 보존한다.",
        "6. 단일 holdout보다 Stratified K-Fold 반복 검증으로 성능 신뢰구간을 제시한다.",
        "7. 최종 운영 목적을 F1형과 Recall형으로 분리한다. 캠페인 대상 누락이 치명적이면 Recall형 모델을 별도 채택한다.",
        "8. threshold를 고정 0.5로 두지 말고 retention 예산과 상담 가능 인원에 맞춰 조정한다.",
        "",
        "## 발표 흐름 제안",
        "",
        "1. 문제 정의: B2B 통신 고객 이탈은 적은 수의 이탈자라도 매출 영향이 크다.",
        "2. 논문 요약: SVMSMOTE + EasyEnsemble, F1 0.129, SHAP 주요 변수.",
        "3. 데이터 재점검: PID 중복, 결측, 지역/세그먼트 이탈률 차이.",
        "4. 우리 방식: 원본 보존, PID/KA 제외, 컬럼별 CSV 분리, 한글 사전화.",
        "5. 컬럼별 역할: 각 CSV가 무슨 질문에 답하는지 설명.",
        "6. 모델 비교: 단일 컬럼 한계와 통합 모델 개선.",
        "7. 성능 한계와 개선안: target encoding, 교차 피처, threshold, K-Fold.",
    ]

    Path("RESEARCH_PRESENTATION_MATERIAL_KIM_SIHWAN.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_slide_outline() -> None:
    lines = [
        "# 발표 슬라이드 구성안 [김시환]",
        "",
        "1. 연구 배경: B2B 통신 고객 이탈 예측의 필요성",
        "2. 기준 논문 요약: 데이터, 전처리, 모델, 성능",
        "3. 우리 연구 질문: 같은 데이터에서 더 설명 가능한 전처리와 피처 설계가 가능한가",
        "4. 데이터 품질 확인: PID 중복, 결측, 클래스 불균형",
        "5. 컬럼별 CSV 분리: 각 변수의 역할과 이탈률 확인",
        "6. 주요 EDA 결과: 고가치 등급, VSE/SME, ZIP, 가입자 수 구간",
        "7. 논문 대비 전처리 차이: 결측 flag, 로그 범위 확대, PID/KA 제외",
        "8. 단일 컬럼 모델 한계: best F1 약 0.15",
        "9. 통합 모델 결과: 논문 F1 0.129 대비 우리 F1 0.168",
        "10. Recall형 운영 모델: 더 많은 이탈자 탐지와 false positive trade-off",
        "11. 앙상블 실험 결과: 복수 모델 평균의 효과와 한계",
        "12. 성능 향상 방안: target encoding, interaction, K-Fold, threshold 최적화",
        "13. 결론: 고매출/고가치 고객의 미활용 신호를 조기에 탐지하는 방향",
    ]
    (OUTPUT_ROOT / "slide_outline.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensemble_result = run_ensemble_experiment()
    csv_summary = build_csv_role_summary()
    paper_comparison, performance = build_paper_vs_ours_tables(ensemble_result)
    write_presentation_markdown(csv_summary, paper_comparison, performance, ensemble_result)
    write_slide_outline()
    print(f"Research presentation materials created under: {OUTPUT_ROOT}")
    print("Main file: RESEARCH_PRESENTATION_MATERIAL_KIM_SIHWAN.md")
    print("Best ensemble:")
    best = ensemble_result.iloc[0]
    print(
        best["variant"],
        best["model"],
        f"f1={best['f1']:.4f}",
        f"recall={best['recall']:.4f}",
        f"precision={best['precision']:.4f}",
    )


if __name__ == "__main__":
    main()
