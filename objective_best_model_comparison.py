import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_ROOT = Path("processed")
EXPERIMENT_ROOT = OUTPUT_ROOT / "objective_best_models"
TEST_ROWS = 1688
TP_BENEFIT = 3240.0
FP_COST = 120.0


def infer_family(model_name: str) -> str:
    lowered = model_name.lower()
    if "logistic" in lowered:
        return "LogisticRegression"
    if "xgboost" in lowered:
        return "XGBoost"
    if "catboost" in lowered:
        return "CatBoost"
    if "balancedbagging" in lowered:
        return "BalancedBagging"
    if "easyensemble" in lowered:
        return "EasyEnsemble"
    if "randomforest" in lowered:
        return "RandomForest"
    if "extratrees" in lowered:
        return "ExtraTrees"
    if "gradientboosting" in lowered:
        return "GradientBoosting"
    if "histgradientboosting" in lowered:
        return "HistGradientBoosting"
    if "rusboost" in lowered:
        return "RUSBoost"
    if "ridge" in lowered:
        return "RidgeClassifier"
    if "linearsvc" in lowered:
        return "LinearSVC"
    if "adaboost" in lowered:
        return "AdaBoost"
    return "Other"


def f2_from_precision_recall(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall):
        return float("nan")
    denominator = (4 * precision) + recall
    if denominator == 0:
        return 0.0
    return float((5 * precision * recall) / denominator)


def standardize_table(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        if pd.notna(row.get("error", pd.NA)):
            continue
        model_name = row.get("model", row.get("model_name"))
        if pd.isna(model_name):
            continue
        family = row.get("family", infer_family(str(model_name)))
        tp = int(row["test_tp"]) if pd.notna(row.get("test_tp")) else 0
        fp = int(row["test_fp"]) if pd.notna(row.get("test_fp")) else 0
        fn = int(row["test_fn"]) if pd.notna(row.get("test_fn")) else 0
        tn = int(row["test_tn"]) if pd.notna(row.get("test_tn")) else TEST_ROWS - tp - fp - fn
        precision = float(row.get("test_precision", np.nan))
        recall = float(row.get("test_recall", np.nan))
        f1 = float(row.get("test_f1", np.nan))
        f2 = float(row.get("test_f2", f2_from_precision_recall(precision, recall)))
        contact_rate = float(row.get("test_predicted_positive_rate", (tp + fp) / TEST_ROWS))
        pr_auc = float(
            row.get("test_pr_auc", row.get("test_average_precision", np.nan))
        )
        mcc = float(row.get("test_mcc", np.nan))
        balanced_accuracy = float(row.get("test_balanced_accuracy", np.nan))
        net_value = (tp * TP_BENEFIT) - (fp * FP_COST)
        rows.append(
            {
                "source": source,
                "variant": row.get("variant"),
                "family": family,
                "model": str(model_name),
                "train_data": row.get("train_data", row.get("train_kind", "")),
                "selected_threshold": float(row.get("selected_threshold", np.nan)),
                "test_f1": f1,
                "test_f2": f2,
                "test_recall": recall,
                "test_precision": precision,
                "test_pr_auc": pr_auc,
                "test_mcc": mcc,
                "test_balanced_accuracy": balanced_accuracy,
                "test_tp": tp,
                "test_fp": fp,
                "test_fn": fn,
                "test_tn": tn,
                "contact_rate": contact_rate,
                "net_value": net_value,
            }
        )
    return pd.DataFrame(rows)


def load_all_results() -> pd.DataFrame:
    tables = []
    threshold_path = OUTPUT_ROOT / "threshold_tuning_best.csv"
    if threshold_path.exists():
        tables.append(
            standardize_table(
                pd.read_csv(threshold_path),
                "baseline_threshold_tuning",
            )
        )

    additional_path = OUTPUT_ROOT / "additional_experiments" / "additional_model_results.csv"
    if additional_path.exists():
        tables.append(
            standardize_table(
                pd.read_csv(additional_path),
                "additional_46_candidate_training",
            )
        )

    recall_path = (
        OUTPUT_ROOT / "recall_optimized_models" / "recall_optimized_best_by_family.csv"
    )
    if recall_path.exists():
        recall_table = pd.read_csv(recall_path).rename(columns={"model_name": "model"})
        tables.append(standardize_table(recall_table, "recall_optimized_refit"))

    combined = pd.concat(tables, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["source", "variant", "model", "selected_threshold"],
        keep="first",
    )
    return combined


def select_best(
    frame: pd.DataFrame,
    *,
    objective: str,
    objective_description: str,
    filter_expr: pd.Series | None,
    sort_cols: list[str],
    ascending: list[bool],
) -> dict[str, Any]:
    candidates = frame.copy()
    if filter_expr is not None:
        candidates = candidates[filter_expr].copy()
    if candidates.empty:
        return {
            "objective": objective,
            "objective_description": objective_description,
            "status": "no_eligible_model",
        }
    best = candidates.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
    best["objective"] = objective
    best["objective_description"] = objective_description
    best["eligible_model_count"] = int(len(candidates))
    best["status"] = "ok"
    return best


def objective_rows(combined: pd.DataFrame) -> pd.DataFrame:
    objectives = [
        {
            "objective": "F1 중심",
            "objective_description": "precision과 recall 균형을 보는 F1 최고",
            "filter_expr": None,
            "sort_cols": ["test_f1", "test_recall", "test_precision"],
            "ascending": [False, False, False],
        },
        {
            "objective": "Recall 중심",
            "objective_description": "제약 없이 실제 이탈자를 가장 많이 잡는 recall 최고",
            "filter_expr": None,
            "sort_cols": ["test_recall", "test_f2", "test_precision"],
            "ascending": [False, False, False],
        },
        {
            "objective": "Recall 운영형",
            "objective_description": "precision>=0.07, contact_rate<=0.75 조건의 recall 최고",
            "filter_expr": combined["test_precision"].ge(0.07)
            & combined["contact_rate"].le(0.75),
            "sort_cols": ["test_recall", "test_f2", "test_precision"],
            "ascending": [False, False, False],
        },
        {
            "objective": "F2 중심",
            "objective_description": "recall을 F1보다 더 강하게 반영하는 F2 최고",
            "filter_expr": None,
            "sort_cols": ["test_f2", "test_recall", "test_precision"],
            "ascending": [False, False, False],
        },
        {
            "objective": "Precision 중심",
            "objective_description": "recall>=0.20 조건에서 오탐을 줄이는 precision 최고",
            "filter_expr": combined["test_recall"].ge(0.20),
            "sort_cols": ["test_precision", "test_f1", "test_recall"],
            "ascending": [False, False, False],
        },
        {
            "objective": "PR-AUC 중심",
            "objective_description": "threshold와 무관한 ranking 품질인 PR-AUC 최고",
            "filter_expr": combined["test_pr_auc"].notna(),
            "sort_cols": ["test_pr_auc", "test_f1", "test_recall"],
            "ascending": [False, False, False],
        },
        {
            "objective": "MCC 중심",
            "objective_description": "confusion matrix 균형을 보는 MCC 최고",
            "filter_expr": combined["test_mcc"].notna(),
            "sort_cols": ["test_mcc", "test_f1", "test_recall"],
            "ascending": [False, False, False],
        },
        {
            "objective": "비용 순이익 중심",
            "objective_description": "TP benefit 3,240, FP cost 120 기준 순이익 최고",
            "filter_expr": None,
            "sort_cols": ["net_value", "test_recall", "test_precision"],
            "ascending": [False, False, False],
        },
        {
            "objective": "소규모 캠페인형",
            "objective_description": "contact_rate<=0.30 조건에서 F1 최고",
            "filter_expr": combined["contact_rate"].le(0.30),
            "sort_cols": ["test_f1", "test_precision", "test_recall"],
            "ascending": [False, False, False],
        },
    ]
    rows = [select_best(combined, **objective) for objective in objectives]
    return pd.DataFrame(rows)


def to_markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    labels = {
        "objective": "목적",
        "source": "실험 소스",
        "variant": "Variant",
        "family": "Family",
        "model": "Model",
        "selected_threshold": "Threshold",
        "test_f1": "F1",
        "test_f2": "F2",
        "test_recall": "Recall",
        "test_precision": "Precision",
        "test_pr_auc": "PR-AUC",
        "test_mcc": "MCC",
        "test_tp": "TP",
        "test_fp": "FP",
        "test_fn": "FN",
        "contact_rate": "Contact Rate",
        "net_value": "Net Value",
    }
    header = "| " + " | ".join(labels.get(col, col) for col in columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                if col in {"net_value"}:
                    values.append(f"{value:,.0f}")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(combined: pd.DataFrame, objectives: pd.DataFrame) -> None:
    columns = [
        "objective",
        "source",
        "variant",
        "family",
        "model",
        "selected_threshold",
        "test_f1",
        "test_f2",
        "test_recall",
        "test_precision",
        "test_pr_auc",
        "test_mcc",
        "test_tp",
        "test_fp",
        "test_fn",
        "contact_rate",
        "net_value",
    ]
    lines = [
        "# Objective Best Model Comparison",
        "",
        "목적: 전부 학습된 모델 후보를 F1, recall, F2, precision, PR-AUC, MCC, 비용 순이익 등 목적별로 다시 비교해 각 목적의 최고 모델을 정리한다.",
        "",
        "## 학습/비교 범위",
        "",
        f"- 통합 비교 후보 수: {len(combined):,}개",
        "- 포함 소스: baseline threshold tuning, additional 46 candidate training, recall optimized refit",
        "- test set: 1,688명, 실제 이탈자 109명",
        "- 비용 기준: TP benefit 3,240, FP cost 120",
        "",
        "## 목적별 최고 모델",
        "",
        to_markdown(objectives, columns),
        "",
        "## 해석",
        "",
        "- F1 중심 모델은 이탈자 포착과 오탐 사이의 균형이 가장 좋다.",
        "- Recall 중심 모델은 이탈자를 가장 많이 잡지만 FP와 contact rate가 크게 증가한다.",
        "- Recall 운영형은 최소 precision과 최대 접촉률 조건을 둔 현실적인 recall 후보이다.",
        "- PR-AUC 중심 모델은 threshold 선택 전 ranking 품질이 좋은 모델로 해석한다.",
        "- 비용 순이익 중심 모델은 논문 비용 가정에서는 recall을 크게 높이는 모델이 유리하지만, 실제 운영에서는 고객 피로도와 상담 예산 제약을 함께 봐야 한다.",
        "",
    ]
    (EXPERIMENT_ROOT / "OBJECTIVE_BEST_MODELS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    combined = load_all_results()
    objectives = objective_rows(combined)

    combined.to_csv(
        EXPERIMENT_ROOT / "all_trained_model_results_combined.csv",
        index=False,
        encoding="utf-8-sig",
    )
    objectives.to_csv(
        EXPERIMENT_ROOT / "objective_best_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "candidate_count": int(len(combined)),
        "sources": sorted(combined["source"].unique().tolist()),
        "objectives": objectives.to_dict(orient="records"),
    }
    (EXPERIMENT_ROOT / "objective_best_models_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(combined, objectives)

    print("Objective best model comparison complete")
    print(
        objectives[
            [
                "objective",
                "variant",
                "family",
                "model",
                "test_f1",
                "test_recall",
                "test_precision",
                "test_tp",
                "test_fp",
                "test_fn",
                "net_value",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
