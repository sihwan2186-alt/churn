import json
import os
from pathlib import Path
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SVMSMOTE
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
OUTPUT_ROOT = Path("processed")
EXPERIMENT_ROOT = OUTPUT_ROOT / "phase_3b_differentiation"
THRESHOLD_GRID = tuple(float(x) for x in np.round(np.arange(0.05, 0.701, 0.01), 2))

VARIANT_DIRS = {
    "with_billing_zip": OUTPUT_ROOT / "model_a_with_billing_zip",
    "without_billing_zip": OUTPUT_ROOT / "model_b_without_billing_zip",
}

ABLATION_RUNS = [
    ("with_billing_zip", "BalancedBagging_original", "original"),
    ("without_billing_zip", "LogisticRegression_SMOTE", "resampled"),
]

EXPERIMENT_A_REGION_RUNS = [
    ("with_billing_zip", "BalancedBagging_original", "original"),
    ("without_billing_zip", "BalancedBagging_original", "original"),
    ("with_billing_zip", "LogisticRegression_SMOTE", "resampled"),
    ("without_billing_zip", "LogisticRegression_SMOTE", "resampled"),
    ("with_billing_zip", "EasyEnsemble_original", "original"),
    ("without_billing_zip", "EasyEnsemble_original", "original"),
]

MIN_STABLE_ZIP_ROWS = 20
MIN_STABLE_ZIP_POSITIVES = 2

PRIMARY_SEGMENT_VARIANT = "with_billing_zip"
PRIMARY_SEGMENT_MODEL = "BalancedBagging_original"

HIGH_VALUE_SEGMENTS = {"Platinum", "Gold"}
MID_VALUE_SEGMENTS = {"SME", "Silver", "SE"}
LOW_VALUE_SEGMENTS = {"Bronze", "Iron", "Lead", "Unknown"}

COST_SCENARIOS = [
    {
        "scenario": "optimistic_campaign",
        "fn_cost": 5400.0,
        "fp_cost": 30.0,
        "retention_rate": 0.60,
    },
    {
        "scenario": "paper_baseline",
        "fn_cost": 5400.0,
        "fp_cost": 120.0,
        "retention_rate": 0.60,
    },
    {
        "scenario": "conservative_campaign",
        "fn_cost": 5400.0,
        "fp_cost": 360.0,
        "retention_rate": 0.60,
    },
    {
        "scenario": "budget_limited",
        "fn_cost": 5400.0,
        "fp_cost": 720.0,
        "retention_rate": 0.60,
    },
    {
        "scenario": "small_business_customers",
        "fn_cost": 1200.0,
        "fp_cost": 120.0,
        "retention_rate": 0.60,
    },
    {
        "scenario": "enterprise_customers",
        "fn_cost": 18000.0,
        "fp_cost": 120.0,
        "retention_rate": 0.60,
    },
]


def load_split(
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    output_dir = VARIANT_DIRS[variant]
    X_train = pd.read_csv(output_dir / "X_train.csv")
    X_test = pd.read_csv(output_dir / "X_test.csv")
    y_train = pd.read_csv(output_dir / "y_train.csv")["CHURN"].astype(int)
    y_test = pd.read_csv(output_dir / "y_test.csv")["CHURN"].astype(int)
    X_train_analysis = pd.read_csv(output_dir / "X_train_analysis.csv")
    X_test_analysis = pd.read_csv(output_dir / "X_test_analysis.csv")
    return X_train, X_test, y_train, y_test, X_train_analysis, X_test_analysis


def build_model(model_name: str) -> Any:
    if model_name == "BalancedBagging_original":
        return BalancedBaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
            ),
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    if model_name == "LogisticRegression_SMOTE":
        return LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    if model_name == "EasyEnsemble_original":
        return EasyEnsembleClassifier(
            n_estimators=10,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def fit_model(
    model_name: str,
    train_kind: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    model = build_model(model_name)
    if train_kind == "resampled":
        smote = SVMSMOTE(random_state=RANDOM_STATE)
        X_fit, y_fit = smote.fit_resample(X_train, y_train)
        X_fit = pd.DataFrame(X_fit, columns=X_train.columns)
        y_fit = pd.Series(y_fit, name="CHURN").astype(int)
    else:
        X_fit, y_fit = X_train, y_train
    model.fit(X_fit, y_fit)
    return model


def metric_row(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    y_array = np.asarray(y_true)
    metrics = {
        "f1": float(f1_score(y_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_array, y_pred, zero_division=0)),
        "precision": float(precision_score(y_array, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_array, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_array, y_pred)),
        "pr_auc": float(average_precision_score(y_array, scores)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    if len(np.unique(y_array)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_array, scores))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    groups = {
        "G_revenue_raw": [
            "AvgMobileRevenue",
            "AvgFIXRevenue",
            "TotalRevenue",
            "ARPU",
        ],
        "G_subscriber_raw": [
            "Active_subscribers",
            "Not_Active_subscribers",
            "Suspended_subscribers",
            "Total_SUBs",
        ],
        "G_engineered_rates": [
            "dormant_subscribers",
            "active_rate",
            "inactive_rate",
            "suspended_rate",
            "dormant_rate",
            "risk_score",
            "has_inactive",
            "has_suspended",
            "has_dormant",
            "multi_subscriber",
            "large_account",
            "Not_Active_subscribers_missing",
            "Suspended_subscribers_missing",
        ],
        "G_interaction": [
            "revenue_engagement_interaction",
            "arpu_risk_interaction",
            "inactive_revenue_interaction",
            "suspended_revenue_interaction",
            "revenue_per_subscriber",
            "revenue_per_active_subscriber",
            "mobile_revenue_ratio",
            "fix_revenue_ratio",
            "mobile_to_fixed_ratio",
            "fixed_to_mobile_ratio",
            "mobile_revenue_per_subscriber",
            "fixed_revenue_per_subscriber",
            "revenue_balance",
            "zero_mobile_revenue",
            "zero_fixed_revenue",
            "mobile_only",
            "fixed_only",
            "revenue_zero",
        ],
        "G_transform": [
            "AvgMobileRevenue_log",
            "AvgMobileRevenue_sqrt",
            "AvgFIXRevenue_log",
            "AvgFIXRevenue_sqrt",
            "TotalRevenue_log",
            "TotalRevenue_sqrt",
            "ARPU_log",
            "ARPU_sqrt",
        ],
        "G_categorical": [
            "CRM_PID_Value_Segment",
            "EffectiveSegment",
            "Billing_ZIP",
            "CRM_PID_Value_Segment_frequency",
            "EffectiveSegment_frequency",
            "Billing_ZIP_frequency",
            "ARPU_missing",
            "Billing_ZIP_missing",
        ],
    }
    column_set = set(columns)
    return {
        group_name: [feature for feature in features if feature in column_set]
        for group_name, features in groups.items()
    }


def evaluate_feature_subset(
    *,
    variant: str,
    model_name: str,
    train_kind: str,
    experiment: str,
    group_name: str,
    selected_features: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    row = {
        "variant": variant,
        "model": model_name,
        "train_kind": train_kind,
        "experiment": experiment,
        "group": group_name,
        "feature_count": len(selected_features),
    }
    if not selected_features:
        row["status"] = "skipped_empty_feature_set"
        return row

    try:
        model = fit_model(
            model_name,
            train_kind,
            X_train[selected_features],
            y_train,
        )
        scores = positive_scores(model, X_test[selected_features])
        y_pred = np.asarray(model.predict(X_test[selected_features])).astype(int)
        row.update(metric_row(y_test, y_pred, scores))
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    return row


def run_feature_group_ablation() -> pd.DataFrame:
    rows = []
    for variant, model_name, train_kind in ABLATION_RUNS:
        X_train, X_test, y_train, y_test, _, _ = load_split(variant)
        all_features = X_train.columns.tolist()
        groups = feature_groups(all_features)

        rows.append(
            evaluate_feature_subset(
                variant=variant,
                model_name=model_name,
                train_kind=train_kind,
                experiment="ALL_FEATURES",
                group_name="ALL_FEATURES",
                selected_features=all_features,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )
        )

        for group_name, group_features in groups.items():
            features_without = [
                feature for feature in all_features if feature not in set(group_features)
            ]
            rows.append(
                evaluate_feature_subset(
                    variant=variant,
                    model_name=model_name,
                    train_kind=train_kind,
                    experiment="DROP_GROUP",
                    group_name=group_name,
                    selected_features=features_without,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                )
            )

        for group_name, group_features in groups.items():
            rows.append(
                evaluate_feature_subset(
                    variant=variant,
                    model_name=model_name,
                    train_kind=train_kind,
                    experiment="ONLY_GROUP",
                    group_name=group_name,
                    selected_features=group_features,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                )
            )

    table = pd.DataFrame(rows)
    baselines = (
        table[table["experiment"] == "ALL_FEATURES"]
        .set_index(["variant", "model"])["f1"]
        .to_dict()
    )
    table["baseline_f1"] = [
        baselines.get((row.variant, row.model), np.nan) for row in table.itertuples()
    ]
    table["f1_delta_vs_baseline"] = table["f1"] - table["baseline_f1"]
    return table


def bucket_segment(segment: str) -> str:
    if segment in HIGH_VALUE_SEGMENTS:
        return "high_value"
    if segment in MID_VALUE_SEGMENTS:
        return "mid_value"
    if segment in LOW_VALUE_SEGMENTS:
        return "low_value"
    return "other"


def find_selected_threshold(variant: str, model_name: str) -> float:
    threshold_table = pd.read_csv(OUTPUT_ROOT / "threshold_tuning_best.csv")
    matches = threshold_table[
        (threshold_table["variant"] == variant)
        & (threshold_table["model"] == model_name)
    ]
    if matches.empty:
        return 0.5
    return float(matches.iloc[0]["selected_threshold"])


def safe_average_precision(y_true: pd.Series, scores: np.ndarray) -> float:
    if int(np.sum(y_true)) == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def summarize_by_group(
    frame: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    rows = []
    for group_value, group in frame.groupby(group_col, dropna=False):
        y_true = group["actual"].astype(int)
        y_pred = group["predicted"].astype(int)
        scores = group["score"].astype(float).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        positives = int(y_true.sum())
        row = {
            group_col: group_value,
            "rows": int(len(group)),
            "positives": positives,
            "churn_rate": float(y_true.mean()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "pr_auc": safe_average_precision(y_true, scores),
            "avg_arpu": float(group["ARPU"].mean()),
            "avg_total_revenue": float(group["TotalRevenue"].mean()),
            "avg_dormant_rate": float(group["dormant_rate"].mean()),
            "fn_total_revenue_at_risk": float(
                group.loc[group["confusion_group"].eq("FN"), "TotalRevenue"].sum()
            ),
            "tp_total_revenue_captured": float(
                group.loc[group["confusion_group"].eq("TP"), "TotalRevenue"].sum()
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["positives", "rows"], ascending=False)


def run_segment_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variant = PRIMARY_SEGMENT_VARIANT
    model_name = PRIMARY_SEGMENT_MODEL
    train_kind = "original"
    threshold = find_selected_threshold(variant, model_name)
    X_train, X_test, y_train, y_test, X_train_analysis, X_test_analysis = load_split(variant)

    model = fit_model(model_name, train_kind, X_train, y_train)
    scores = positive_scores(model, X_test)
    predictions = (scores >= threshold).astype(int)

    frame = X_test_analysis.copy()
    frame["actual"] = y_test.to_numpy()
    frame["predicted"] = predictions
    frame["score"] = scores
    frame["threshold"] = threshold
    frame["variant"] = variant
    frame["model"] = model_name
    frame["CRM_segment_bucket"] = frame["CRM_PID_Value_Segment"].map(bucket_segment)
    frame["confusion_group"] = np.select(
        [
            frame["actual"].eq(1) & frame["predicted"].eq(1),
            frame["actual"].eq(0) & frame["predicted"].eq(1),
            frame["actual"].eq(1) & frame["predicted"].eq(0),
        ],
        ["TP", "FP", "FN"],
        default="TN",
    )

    segment_summary = summarize_by_group(frame, "CRM_PID_Value_Segment")
    bucket_summary = summarize_by_group(frame, "CRM_segment_bucket")
    profile_summary = (
        frame.groupby(["CRM_segment_bucket", "CRM_PID_Value_Segment", "confusion_group"])
        .agg(
            rows=("actual", "size"),
            avg_arpu=("ARPU", "mean"),
            avg_total_revenue=("TotalRevenue", "mean"),
            avg_active_subscribers=("Active_subscribers", "mean"),
            avg_dormant_rate=("dormant_rate", "mean"),
            total_revenue_sum=("TotalRevenue", "sum"),
        )
        .reset_index()
        .sort_values(["CRM_segment_bucket", "CRM_PID_Value_Segment", "confusion_group"])
    )
    return segment_summary, bucket_summary, profile_summary


def run_high_value_submodel() -> pd.DataFrame:
    variant = PRIMARY_SEGMENT_VARIANT
    model_name = PRIMARY_SEGMENT_MODEL
    X_train, X_test, y_train, y_test, X_train_analysis, X_test_analysis = load_split(variant)
    train_mask = X_train_analysis["CRM_PID_Value_Segment"].isin(HIGH_VALUE_SEGMENTS)
    test_mask = X_test_analysis["CRM_PID_Value_Segment"].isin(HIGH_VALUE_SEGMENTS)

    rows = []
    overall_model = fit_model(model_name, "original", X_train, y_train)
    overall_scores = positive_scores(overall_model, X_test.loc[test_mask])
    overall_pred = (overall_scores >= find_selected_threshold(variant, model_name)).astype(int)
    row = {
        "model_scope": "global_model_on_high_value_test",
        "train_rows": int(len(X_train)),
        "test_rows": int(test_mask.sum()),
    }
    row.update(metric_row(y_test.loc[test_mask], overall_pred, overall_scores))
    rows.append(row)

    y_train_hv = y_train.loc[train_mask].reset_index(drop=True)
    X_train_hv = X_train.loc[train_mask].reset_index(drop=True)
    if y_train_hv.nunique() == 2 and int(test_mask.sum()) > 0:
        hv_model = fit_model(model_name, "original", X_train_hv, y_train_hv)
        hv_scores = positive_scores(hv_model, X_test.loc[test_mask])
        hv_pred = (hv_scores >= 0.5).astype(int)
        row = {
            "model_scope": "high_value_only_model",
            "train_rows": int(len(X_train_hv)),
            "test_rows": int(test_mask.sum()),
        }
        row.update(metric_row(y_test.loc[test_mask], hv_pred, hv_scores))
        rows.append(row)

    return pd.DataFrame(rows)


def expected_net_value(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    fn_cost: float,
    fp_cost: float,
    tp_benefit: float,
) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    net_value = (tp * tp_benefit) - (fp * fp_cost) - (fn * fn_cost)
    return {
        "threshold": threshold,
        "expected_net_value": float(net_value),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def run_cost_threshold_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    variant = PRIMARY_SEGMENT_VARIANT
    model_name = PRIMARY_SEGMENT_MODEL
    X_train, X_test, y_train, y_test, _, _ = load_split(variant)
    model = fit_model(model_name, "original", X_train, y_train)
    scores = positive_scores(model, X_test)

    sweep_rows = []
    best_rows = []
    for scenario in COST_SCENARIOS:
        fn_cost = scenario["fn_cost"]
        fp_cost = scenario["fp_cost"]
        retention_rate = scenario["retention_rate"]
        tp_benefit = fn_cost * retention_rate
        theoretical_threshold = fp_cost / (fp_cost + tp_benefit)
        cost_ratio = fn_cost / fp_cost

        scenario_rows = []
        for threshold in THRESHOLD_GRID:
            row = {
                "variant": variant,
                "model": model_name,
                "scenario": scenario["scenario"],
                "fn_cost": fn_cost,
                "fp_cost": fp_cost,
                "tp_benefit": tp_benefit,
                "retention_rate": retention_rate,
                "cost_ratio": cost_ratio,
                "theoretical_threshold": theoretical_threshold,
            }
            row.update(
                expected_net_value(
                    y_test,
                    scores,
                    threshold,
                    fn_cost=fn_cost,
                    fp_cost=fp_cost,
                    tp_benefit=tp_benefit,
                )
            )
            scenario_rows.append(row)

        scenario_table = pd.DataFrame(scenario_rows)
        best_row = scenario_table.sort_values(
            ["expected_net_value", "recall"], ascending=[False, False]
        ).iloc[0].to_dict()
        best_rows.append(best_row)
        sweep_rows.extend(scenario_rows)

    return pd.DataFrame(best_rows), pd.DataFrame(sweep_rows)


def plot_cost_sensitivity(best_table: pd.DataFrame, sweep_table: pd.DataFrame) -> None:
    plt.style.use("default")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for scenario, group in sweep_table.groupby("scenario"):
        axes[0].plot(
            group["threshold"],
            group["expected_net_value"],
            linewidth=1.8,
            label=scenario,
        )
    axes[0].set_title("Threshold vs Expected Net Value")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Expected net value")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)

    sorted_best = best_table.sort_values("cost_ratio")
    axes[1].scatter(
        sorted_best["cost_ratio"],
        sorted_best["threshold"],
        s=55,
        color="#2f6fbb",
    )
    axes[1].plot(
        sorted_best["cost_ratio"],
        sorted_best["threshold"],
        color="#2f6fbb",
        alpha=0.65,
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Optimal Threshold vs Cost Ratio")
    axes[1].set_xlabel("FN cost / FP cost")
    axes[1].set_ylabel("Optimal threshold")
    axes[1].grid(alpha=0.25)

    axes[2].scatter(
        best_table["recall"],
        best_table["precision"],
        s=65,
        color="#c04b37",
    )
    for _, row in best_table.iterrows():
        axes[2].annotate(
            row["scenario"],
            (row["recall"], row["precision"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[2].set_title("Operating Points")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(EXPERIMENT_ROOT / "threshold_cost_sensitivity.png", dpi=180)
    plt.close(fig)


def run_experiment_a_summary() -> pd.DataFrame:
    comparison = pd.read_csv(OUTPUT_ROOT / "model_comparison_billing_zip.csv")
    rows = []
    for model in [
        "BalancedBagging_original",
        "LogisticRegression_SMOTE",
        "EasyEnsemble_original",
    ]:
        model_rows = comparison[comparison["model"].eq(model)]
        for _, row in model_rows.iterrows():
            rows.append(
                {
                    "experiment": "A_billing_zip_variant",
                    "variant": row["variant"],
                    "model": row["model"],
                    "f1": row["f1"],
                    "recall": row["recall"],
                    "precision": row["precision"],
                    "pr_auc": row["pr_auc"],
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "variant"])


def format_billing_zip(value: Any, missing: bool = False) -> str:
    if missing or pd.isna(value):
        return "missing"
    try:
        numeric_value = float(value)
        if np.isnan(numeric_value):
            return "missing"
        if numeric_value.is_integer():
            return str(int(numeric_value)).zfill(4)
    except (TypeError, ValueError):
        pass
    return str(value).strip() or "missing"


def billing_zip_prefix(zip_value: str, digits: int) -> str:
    if zip_value == "missing":
        return "missing"
    if len(zip_value) <= digits:
        return zip_value
    return f"{zip_value[:digits]}{'x' * (len(zip_value) - digits)}"


def add_billing_zip_group_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if "Billing_ZIP" not in frame.columns:
        raise ValueError("Billing_ZIP is required for ZIP-level analysis.")

    enriched = frame.copy().reset_index(drop=True)
    if "Billing_ZIP_missing" in enriched.columns:
        missing_flags = enriched["Billing_ZIP_missing"].fillna(0).astype(int).eq(1)
    else:
        missing_flags = pd.Series(False, index=enriched.index)

    zip_values = [
        format_billing_zip(value, bool(missing))
        for value, missing in zip(enriched["Billing_ZIP"], missing_flags)
    ]
    enriched["Billing_ZIP_value"] = zip_values
    enriched["Billing_ZIP_prefix1"] = [
        billing_zip_prefix(value, 1) for value in zip_values
    ]
    enriched["Billing_ZIP_prefix2"] = [
        billing_zip_prefix(value, 2) for value in zip_values
    ]
    return enriched


def support_flag(rows: int, positives: int) -> str:
    if positives == 0:
        return "no_positive"
    if rows < MIN_STABLE_ZIP_ROWS or positives < MIN_STABLE_ZIP_POSITIVES:
        return "small_support"
    return "stable"


def summarize_actual_churn_by_group(
    frame: pd.DataFrame,
    group_col: str,
    group_level: str,
) -> pd.DataFrame:
    rows = []
    for group_value, group in frame.groupby(group_col, dropna=False):
        actual = group["actual"].astype(int)
        positives = int(actual.sum())
        total_rows = int(len(group))
        churn_rate = float(actual.mean()) if total_rows else float("nan")
        rows.append(
            {
                "group_level": group_level,
                "billing_zip_group": group_value,
                "rows": total_rows,
                "train_rows": int(group["split"].eq("train").sum()),
                "test_rows": int(group["split"].eq("test").sum()),
                "churn_count": positives,
                "non_churn_count": int(total_rows - positives),
                "churn_rate": churn_rate,
                "churn_rate_percent": churn_rate * 100,
                "support_flag": support_flag(total_rows, positives),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["group_level", "rows", "churn_count", "billing_zip_group"],
        ascending=[True, False, False, True],
    )


def summarize_prediction_by_group(
    frame: pd.DataFrame,
    group_col: str,
    group_level: str,
    *,
    variant: str,
    model_name: str,
    train_kind: str,
) -> pd.DataFrame:
    overall_y = frame["actual"].astype(int)
    overall_pred = frame["predicted"].astype(int)
    overall_f1 = float(f1_score(overall_y, overall_pred, zero_division=0))
    overall_recall = float(recall_score(overall_y, overall_pred, zero_division=0))

    rows = []
    for group_value, group in frame.groupby(group_col, dropna=False):
        y_true = group["actual"].astype(int)
        y_pred = group["predicted"].astype(int)
        scores = group["score"].astype(float).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        positives = int(y_true.sum())
        predicted_positives = int(y_pred.sum())
        total_rows = int(len(group))
        churn_rate = float(y_true.mean()) if total_rows else float("nan")
        predicted_positive_rate = float(y_pred.mean()) if total_rows else float("nan")
        f1_value = (
            float(f1_score(y_true, y_pred, zero_division=0))
            if positives or predicted_positives
            else float("nan")
        )
        recall_value = (
            float(recall_score(y_true, y_pred, zero_division=0))
            if positives
            else float("nan")
        )
        precision_value = (
            float(precision_score(y_true, y_pred, zero_division=0))
            if predicted_positives
            else float("nan")
        )

        rows.append(
            {
                "variant": variant,
                "model": model_name,
                "train_kind": train_kind,
                "group_level": group_level,
                "billing_zip_group": group_value,
                "rows": total_rows,
                "positives": positives,
                "churn_rate": churn_rate,
                "predicted_positives": predicted_positives,
                "predicted_positive_rate": predicted_positive_rate,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "f1": f1_value,
                "recall": recall_value,
                "precision": precision_value,
                "pr_auc": safe_average_precision(y_true, scores),
                "avg_score": float(np.mean(scores)) if len(scores) else float("nan"),
                "overall_test_f1": overall_f1,
                "overall_test_recall": overall_recall,
                "f1_delta_vs_model_overall": f1_value - overall_f1
                if not np.isnan(f1_value)
                else float("nan"),
                "recall_delta_vs_model_overall": recall_value - overall_recall
                if not np.isnan(recall_value)
                else float("nan"),
                "support_flag": support_flag(total_rows, positives),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["variant", "model", "group_level", "rows", "positives", "billing_zip_group"],
        ascending=[True, True, True, False, False, True],
    )


def run_experiment_a_billing_zip_detail() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    (
        _,
        _,
        y_train_zip,
        y_test_zip,
        X_train_zip_analysis,
        X_test_zip_analysis,
    ) = load_split("with_billing_zip")

    train_zip_frame = add_billing_zip_group_columns(X_train_zip_analysis)
    train_zip_frame["split"] = "train"
    train_zip_frame["actual"] = y_train_zip.to_numpy()
    test_zip_frame = add_billing_zip_group_columns(X_test_zip_analysis)
    test_zip_frame["split"] = "test"
    test_zip_frame["actual"] = y_test_zip.to_numpy()
    full_zip_frame = pd.concat(
        [train_zip_frame, test_zip_frame],
        ignore_index=True,
    )

    churn_by_value = summarize_actual_churn_by_group(
        full_zip_frame,
        "Billing_ZIP_value",
        "zip_value",
    )
    churn_by_group = pd.concat(
        [
            summarize_actual_churn_by_group(
                full_zip_frame,
                "Billing_ZIP_prefix1",
                "zip_prefix1",
            ),
            summarize_actual_churn_by_group(
                full_zip_frame,
                "Billing_ZIP_prefix2",
                "zip_prefix2",
            ),
        ],
        ignore_index=True,
    )

    model_value_tables = []
    model_group_tables = []
    y_test_zip_reset = y_test_zip.reset_index(drop=True)
    zip_test_labels = test_zip_frame[
        ["Billing_ZIP_value", "Billing_ZIP_prefix1", "Billing_ZIP_prefix2"]
    ].reset_index(drop=True)

    for variant, model_name, train_kind in EXPERIMENT_A_REGION_RUNS:
        X_train, X_test, y_train, y_test, _, _ = load_split(variant)
        if not y_test.reset_index(drop=True).equals(y_test_zip_reset):
            raise ValueError(
                f"Test target order differs between with_billing_zip and {variant}."
            )

        model = fit_model(model_name, train_kind, X_train, y_train)
        scores = positive_scores(model, X_test)
        predictions = np.asarray(model.predict(X_test)).astype(int)

        prediction_frame = zip_test_labels.copy()
        prediction_frame["actual"] = y_test.to_numpy()
        prediction_frame["predicted"] = predictions
        prediction_frame["score"] = scores

        model_value_tables.append(
            summarize_prediction_by_group(
                prediction_frame,
                "Billing_ZIP_value",
                "zip_value",
                variant=variant,
                model_name=model_name,
                train_kind=train_kind,
            )
        )
        model_group_tables.extend(
            [
                summarize_prediction_by_group(
                    prediction_frame,
                    "Billing_ZIP_prefix1",
                    "zip_prefix1",
                    variant=variant,
                    model_name=model_name,
                    train_kind=train_kind,
                ),
                summarize_prediction_by_group(
                    prediction_frame,
                    "Billing_ZIP_prefix2",
                    "zip_prefix2",
                    variant=variant,
                    model_name=model_name,
                    train_kind=train_kind,
                ),
            ]
        )

    model_by_value = pd.concat(model_value_tables, ignore_index=True)
    model_by_group = pd.concat(model_group_tables, ignore_index=True)
    return churn_by_value, churn_by_group, model_by_value, model_by_group


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def to_markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in frame[columns].iterrows():
        lines.append(
            "| "
            + " | ".join(format_markdown_value(row[column]) for column in columns)
            + " |"
        )
    return "\n".join(lines)


def write_experiment_a_billing_zip_findings(
    churn_by_value: pd.DataFrame,
    churn_by_group: pd.DataFrame,
    model_by_value: pd.DataFrame,
    model_by_group: pd.DataFrame,
) -> None:
    stable_exact = churn_by_value[churn_by_value["support_flag"].eq("stable")]
    high_churn_exact = (
        stable_exact.sort_values(
            ["churn_rate", "churn_count", "rows"],
            ascending=[False, False, False],
        )
        .head(10)
        .copy()
    )
    prefix2_churn = (
        churn_by_group[
            churn_by_group["group_level"].eq("zip_prefix2")
            & churn_by_group["support_flag"].eq("stable")
        ]
        .sort_values(["churn_rate", "churn_count", "rows"], ascending=[False, False, False])
        .head(10)
        .copy()
    )
    key_model_prefix2 = model_by_group[
        model_by_group["group_level"].eq("zip_prefix2")
        & model_by_group["support_flag"].eq("stable")
        & (
            (
                model_by_group["variant"].eq("with_billing_zip")
                & model_by_group["model"].eq("BalancedBagging_original")
            )
            | (
                model_by_group["variant"].eq("without_billing_zip")
                & model_by_group["model"].eq("LogisticRegression_SMOTE")
            )
        )
    ].copy()
    key_model_prefix2 = key_model_prefix2.sort_values(
        ["model", "recall_delta_vs_model_overall", "f1_delta_vs_model_overall"],
        ascending=[True, False, False],
    ).head(16)

    total_rows = int(churn_by_value["rows"].sum())
    total_churn = int(churn_by_value["churn_count"].sum())
    total_zip_values = int(len(churn_by_value))
    small_support_count = int(churn_by_value["support_flag"].ne("stable").sum())

    lines = [
        "# Experiment A 추가: Billing_ZIP 지역별 이탈 및 모델 성능",
        "",
        "기준: 기존 Experiment A와 동일하게 train/test split을 유지하고, 모델은 전체 학습 데이터로 학습한 뒤 test set을 Billing_ZIP 단위로 나누어 평가했다.",
        "ZIP 단일값은 표본이 작은 값이 많기 때문에 원값별 결과와 ZIP 앞 1자리/2자리 지역 그룹 결과를 함께 저장했다.",
        "",
        "## 산출 파일",
        "",
        "- `experiment_a_billing_zip_churn_by_value.csv`: 전체 데이터의 Billing_ZIP 원값별 이탈률",
        "- `experiment_a_billing_zip_churn_by_group.csv`: 전체 데이터의 Billing_ZIP 앞 1자리/2자리 그룹별 이탈률",
        "- `experiment_a_billing_zip_model_by_value.csv`: test set의 Billing_ZIP 원값별 F1/recall/precision",
        "- `experiment_a_billing_zip_model_by_group.csv`: test set의 Billing_ZIP 앞 1자리/2자리 그룹별 F1/recall/precision",
        "",
        "## 핵심 요약",
        "",
        f"- 전체 분석 행은 {total_rows:,}건, 이탈 고객은 {total_churn:,}건이다.",
        f"- Billing_ZIP 원값은 {total_zip_values:,}개이며, 이 중 {small_support_count:,}개는 표본 또는 이탈자 수가 작아 탐색용으로만 해석한다.",
        "- 개별 ZIP별로 별도 모델을 학습하면 표본 부족 문제가 크므로, 전체 모델을 학습한 뒤 ZIP별 hold-out 성능을 비교했다.",
        "- 보고서/발표에서는 원값별 극단값보다 `zip_prefix2` 지역 그룹 결과를 중심으로 해석하는 편이 안정적이다.",
        "",
        "## 이탈률이 높은 ZIP 원값 그룹",
        "",
        to_markdown_table(
            high_churn_exact,
            [
                "billing_zip_group",
                "rows",
                "churn_count",
                "churn_rate",
                "support_flag",
            ],
        ),
        "",
        "## 이탈률이 높은 ZIP 앞 2자리 지역 그룹",
        "",
        to_markdown_table(
            prefix2_churn,
            [
                "billing_zip_group",
                "rows",
                "churn_count",
                "churn_rate",
                "support_flag",
            ],
        ),
        "",
        "## 주요 모델의 ZIP 앞 2자리 그룹별 성능 변화",
        "",
        to_markdown_table(
            key_model_prefix2,
            [
                "variant",
                "model",
                "billing_zip_group",
                "rows",
                "positives",
                "churn_rate",
                "f1",
                "recall",
                "f1_delta_vs_model_overall",
                "recall_delta_vs_model_overall",
            ],
        ),
        "",
        "## 보고서용 해석 문장",
        "",
        "> Billing_ZIP 원값 456개를 모두 나누어 보면 일부 ZIP에서 이탈률이 높게 나타나지만, 다수 값은 표본이 작아 단독 결론으로 쓰기 어렵다. 따라서 본 연구는 전체 모델을 학습한 뒤 Billing_ZIP 원값과 ZIP 앞 1자리/2자리 지역 그룹별 hold-out 성능을 비교했다. 그 결과 지역 정보는 BalancedBagging 계열에서는 일부 지역 그룹의 recall을 끌어올리는 신호로 작동했지만, Logistic Regression에서는 고카디널리티 ZIP 정보가 전체 F1을 낮추는 방향으로 작동했다. 즉 Billing_ZIP은 제거/포함을 단순히 하나로 정할 피처가 아니라, 모델 계열과 지역 그룹 안정성을 함께 고려해야 하는 변수다.",
        "",
    ]

    (EXPERIMENT_ROOT / "experiment_a_billing_zip_region_findings.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_summary_json(
    ablation: pd.DataFrame,
    segment_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    cost_best: pd.DataFrame,
    billing_zip_churn_by_value: pd.DataFrame | None = None,
    billing_zip_model_by_group: pd.DataFrame | None = None,
) -> None:
    ok_ablation = ablation[ablation["status"].eq("ok")].copy()
    drop_rows = ok_ablation[ok_ablation["experiment"].eq("DROP_GROUP")]
    only_rows = ok_ablation[ok_ablation["experiment"].eq("ONLY_GROUP")]
    summary = {
        "best_drop_impacts": drop_rows.sort_values("f1_delta_vs_baseline").head(8).to_dict(
            orient="records"
        ),
        "best_single_group_results": only_rows.sort_values("f1", ascending=False)
        .head(8)
        .to_dict(orient="records"),
        "segment_summary": segment_summary.to_dict(orient="records"),
        "bucket_summary": bucket_summary.to_dict(orient="records"),
        "cost_best": cost_best.to_dict(orient="records"),
    }
    if billing_zip_churn_by_value is not None:
        summary["billing_zip_high_churn_stable_values"] = (
            billing_zip_churn_by_value[
                billing_zip_churn_by_value["support_flag"].eq("stable")
            ]
            .sort_values(
                ["churn_rate", "churn_count", "rows"],
                ascending=[False, False, False],
            )
            .head(10)
            .to_dict(orient="records")
        )
    if billing_zip_model_by_group is not None:
        summary["billing_zip_prefix2_model_changes"] = (
            billing_zip_model_by_group[
                billing_zip_model_by_group["group_level"].eq("zip_prefix2")
                & billing_zip_model_by_group["support_flag"].eq("stable")
            ]
            .sort_values(
                ["variant", "model", "recall_delta_vs_model_overall"],
                ascending=[True, True, False],
            )
            .head(30)
            .to_dict(orient="records")
        )
    (EXPERIMENT_ROOT / "phase_3b_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

    experiment_a = run_experiment_a_summary()
    experiment_a.to_csv(
        EXPERIMENT_ROOT / "experiment_a_billing_zip_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        billing_zip_churn_by_value,
        billing_zip_churn_by_group,
        billing_zip_model_by_value,
        billing_zip_model_by_group,
    ) = run_experiment_a_billing_zip_detail()
    billing_zip_churn_by_value.to_csv(
        EXPERIMENT_ROOT / "experiment_a_billing_zip_churn_by_value.csv",
        index=False,
        encoding="utf-8-sig",
    )
    billing_zip_churn_by_group.to_csv(
        EXPERIMENT_ROOT / "experiment_a_billing_zip_churn_by_group.csv",
        index=False,
        encoding="utf-8-sig",
    )
    billing_zip_model_by_value.to_csv(
        EXPERIMENT_ROOT / "experiment_a_billing_zip_model_by_value.csv",
        index=False,
        encoding="utf-8-sig",
    )
    billing_zip_model_by_group.to_csv(
        EXPERIMENT_ROOT / "experiment_a_billing_zip_model_by_group.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_experiment_a_billing_zip_findings(
        billing_zip_churn_by_value,
        billing_zip_churn_by_group,
        billing_zip_model_by_value,
        billing_zip_model_by_group,
    )

    ablation = run_feature_group_ablation()
    ablation.to_csv(
        EXPERIMENT_ROOT / "experiment_b_feature_group_ablation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    segment_summary, bucket_summary, profile_summary = run_segment_analysis()
    segment_summary.to_csv(
        EXPERIMENT_ROOT / "experiment_c_segment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bucket_summary.to_csv(
        EXPERIMENT_ROOT / "experiment_c_segment_bucket_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    profile_summary.to_csv(
        EXPERIMENT_ROOT / "experiment_c_segment_confusion_profiles.csv",
        index=False,
        encoding="utf-8-sig",
    )

    high_value_submodel = run_high_value_submodel()
    high_value_submodel.to_csv(
        EXPERIMENT_ROOT / "experiment_c_high_value_submodel.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cost_best, cost_sweep = run_cost_threshold_sensitivity()
    cost_best.to_csv(
        EXPERIMENT_ROOT / "experiment_d_cost_threshold_best.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cost_sweep.to_csv(
        EXPERIMENT_ROOT / "experiment_d_cost_threshold_sweep.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_cost_sensitivity(cost_best, cost_sweep)

    write_summary_json(
        ablation,
        segment_summary,
        bucket_summary,
        cost_best,
        billing_zip_churn_by_value,
        billing_zip_model_by_group,
    )

    print("Phase 3-B experiments complete")
    print("\nExperiment A ZIP prefix2 model changes:")
    print(
        billing_zip_model_by_group[
            billing_zip_model_by_group["group_level"].eq("zip_prefix2")
            & billing_zip_model_by_group["support_flag"].eq("stable")
        ]
        .sort_values(["model", "recall_delta_vs_model_overall"], ascending=[True, False])
        .head(12)[
            [
                "variant",
                "model",
                "billing_zip_group",
                "rows",
                "positives",
                "churn_rate",
                "f1",
                "recall",
                "f1_delta_vs_model_overall",
                "recall_delta_vs_model_overall",
            ]
        ]
        .to_string(index=False)
    )
    print("\nExperiment B strongest drop impacts:")
    print(
        ablation[ablation["experiment"].eq("DROP_GROUP")]
        .sort_values("f1_delta_vs_baseline")
        .head(8)[
            [
                "variant",
                "model",
                "group",
                "f1",
                "baseline_f1",
                "f1_delta_vs_baseline",
            ]
        ]
        .to_string(index=False)
    )
    print("\nExperiment C bucket summary:")
    print(bucket_summary.to_string(index=False))
    print("\nExperiment D cost best:")
    print(
        cost_best[
            [
                "scenario",
                "cost_ratio",
                "threshold",
                "expected_net_value",
                "recall",
                "precision",
                "tp",
                "fp",
                "fn",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
