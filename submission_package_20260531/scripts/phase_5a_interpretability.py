import json
import os
from pathlib import Path
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


RANDOM_STATE = 42
OUTPUT_ROOT = Path("processed")
EXPERIMENT_ROOT = OUTPUT_ROOT / "phase_5a_interpretability"
VARIANT_DIR = OUTPUT_ROOT / "model_b_without_billing_zip"
PERMUTATION_IMPORTANCE_PATH = OUTPUT_ROOT / "feature_importance_top.csv"
TOP_N = 20
LOCAL_EXPLANATION_CUSTOMERS = 5
LOCAL_EXPLANATION_FEATURES = 10


def load_lr_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(VARIANT_DIR / "X_train_resampled.csv")
    y_train = pd.read_csv(VARIANT_DIR / "y_train_resampled.csv")["CHURN"].astype(int)
    X_test = pd.read_csv(VARIANT_DIR / "X_test.csv")
    y_test = pd.read_csv(VARIANT_DIR / "y_test.csv")["CHURN"].astype(int)
    return X_train, X_test, y_train, y_test


def fit_lr_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def coefficient_table(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    coefficients = model.coef_[0]
    contributions = X_test.to_numpy() * coefficients
    mean_abs_contribution = np.mean(np.abs(contributions), axis=0)
    mean_contribution = np.mean(contributions, axis=0)

    correlations = []
    for feature in X_test.columns:
        if X_test[feature].std(ddof=0) == 0:
            corr = 0.0
        else:
            corr = X_test[feature].corr(y_test)
        correlations.append(0.0 if pd.isna(corr) else float(corr))

    table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
            "odds_ratio_per_1sd": np.exp(coefficients),
            "direction": np.where(coefficients >= 0, "Churn_up", "Churn_down"),
            "correlation_direction": np.where(
                np.asarray(correlations) >= 0, "Churn_up", "Churn_down"
            ),
            "mean_abs_logit_contribution": mean_abs_contribution,
            "mean_logit_contribution": mean_contribution,
            "correlation_with_churn": correlations,
            "abs_correlation_with_churn": np.abs(correlations),
        }
    )
    table = table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    table["coef_rank"] = np.arange(1, len(table) + 1)
    table["mean_abs_contribution_rank"] = (
        table["mean_abs_logit_contribution"].rank(ascending=False, method="min").astype(int)
    )
    table["correlation_rank"] = (
        table["abs_correlation_with_churn"].rank(ascending=False, method="min").astype(int)
    )
    return table


def permutation_comparison(coef_table: pd.DataFrame) -> pd.DataFrame:
    permutation = pd.read_csv(PERMUTATION_IMPORTANCE_PATH)
    permutation = permutation[
        (permutation["operating_point"] == "main_f1_baseline")
        & (permutation["variant"] == "without_billing_zip")
        & (permutation["model"] == "LogisticRegression_SMOTE")
    ].copy()
    permutation = permutation.sort_values("f1_importance", ascending=False).reset_index(
        drop=True
    )
    permutation["permutation_rank"] = np.arange(1, len(permutation) + 1)

    comparison = coef_table.merge(
        permutation[
            [
                "feature",
                "f1_importance",
                "baseline_f1",
                "permuted_f1_mean",
                "permutation_rank",
            ]
        ],
        on="feature",
        how="left",
    )
    comparison["rank_gap_coef_minus_permutation"] = (
        comparison["coef_rank"] - comparison["permutation_rank"]
    )
    return comparison.sort_values(
        ["permutation_rank", "coef_rank"], na_position="last"
    ).reset_index(drop=True)


def local_linear_explanations(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficients = model.coef_[0]
    intercept = float(model.intercept_[0])
    logits = intercept + X_test.to_numpy().dot(coefficients)
    probabilities = sigmoid(logits)
    order = np.argsort(probabilities)[::-1][:LOCAL_EXPLANATION_CUSTOMERS]

    customer_rows = []
    explanation_rows = []
    for rank, row_idx in enumerate(order, start=1):
        row = X_test.iloc[row_idx]
        contributions = row.to_numpy() * coefficients
        contribution_order = np.argsort(np.abs(contributions))[::-1][
            :LOCAL_EXPLANATION_FEATURES
        ]
        customer_rows.append(
            {
                "risk_rank": rank,
                "test_row_id": int(row_idx),
                "actual": int(y_test.iloc[row_idx]),
                "predicted_probability": float(probabilities[row_idx]),
                "logit": float(logits[row_idx]),
                "intercept": intercept,
            }
        )
        for feature_rank, feature_idx in enumerate(contribution_order, start=1):
            contribution = float(contributions[feature_idx])
            explanation_rows.append(
                {
                    "risk_rank": rank,
                    "test_row_id": int(row_idx),
                    "feature_rank": feature_rank,
                    "feature": X_test.columns[feature_idx],
                    "standardized_value": float(row.iloc[feature_idx]),
                    "coefficient": float(coefficients[feature_idx]),
                    "logit_contribution": contribution,
                    "direction": "Churn_up" if contribution >= 0 else "Churn_down",
                }
            )
    return pd.DataFrame(customer_rows), pd.DataFrame(explanation_rows)


def plot_coefficient_importance(table: pd.DataFrame) -> None:
    top = table.sort_values("abs_coefficient", ascending=True).tail(TOP_N)
    colors = ["#c83e3a" if value > 0 else "#2f6fbb" for value in top["coefficient"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    axes[0].barh(top["feature"], top["coefficient"], color=colors)
    axes[0].axvline(0, color="black", linewidth=0.9)
    axes[0].set_title("LR Coefficients with Direction")
    axes[0].set_xlabel("Standardized coefficient")
    axes[0].grid(axis="x", alpha=0.25)

    odds_colors = [
        "#c83e3a" if value > 1.0 else "#2f6fbb" for value in top["odds_ratio_per_1sd"]
    ]
    axes[1].barh(top["feature"], top["odds_ratio_per_1sd"], color=odds_colors)
    axes[1].axvline(1, color="black", linewidth=0.9)
    axes[1].set_title("Odds Ratio per 1 SD")
    axes[1].set_xlabel("Odds ratio")
    axes[1].grid(axis="x", alpha=0.25)

    red = mpatches.Patch(color="#c83e3a", label="Increases churn risk")
    blue = mpatches.Patch(color="#2f6fbb", label="Decreases churn risk")
    axes[0].legend(handles=[red, blue], loc="lower right")
    fig.tight_layout()
    fig.savefig(EXPERIMENT_ROOT / "lr_coefficient_importance.png", dpi=180)
    plt.close(fig)


def plot_linear_contribution_importance(table: pd.DataFrame) -> None:
    top = table.sort_values("mean_abs_logit_contribution", ascending=True).tail(TOP_N)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["feature"], top["mean_abs_logit_contribution"], color="#4b7f52")
    ax.set_title("Mean Absolute Linear Logit Contribution")
    ax.set_xlabel("mean(|coefficient x standardized value|)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXPERIMENT_ROOT / "lr_linear_contribution_importance.png", dpi=180)
    plt.close(fig)


def plot_pdp(model: LogisticRegression, X_test: pd.DataFrame, top_features: list[str]) -> None:
    n_cols = 3
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows))
    axes_array = np.asarray(axes).ravel()
    PartialDependenceDisplay.from_estimator(
        model,
        X_test,
        top_features,
        kind="average",
        response_method="predict_proba",
        grid_resolution=50,
        ax=axes_array[: len(top_features)],
    )
    for ax in axes_array[len(top_features) :]:
        ax.axis("off")
    fig.suptitle("Partial Dependence: LR Feature Effects on Churn Probability", y=1.01)
    fig.tight_layout()
    fig.savefig(EXPERIMENT_ROOT / "lr_pdp_top_features.png", dpi=180)
    plt.close(fig)


def metric_summary(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    return {
        "variant": "without_billing_zip",
        "model": "LogisticRegression_SMOTE",
        "threshold": 0.5,
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
    }


def main() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_lr_data()
    model = fit_lr_model(X_train, y_train)

    coef = coefficient_table(model, X_test, y_test)
    comparison = permutation_comparison(coef)
    high_risk, local_explanations = local_linear_explanations(model, X_test, y_test)

    coef.to_csv(
        EXPERIMENT_ROOT / "lr_coefficient_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coef.sort_values("mean_abs_logit_contribution", ascending=False).to_csv(
        EXPERIMENT_ROOT / "lr_linear_contribution_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coef.sort_values("abs_correlation_with_churn", ascending=False).to_csv(
        EXPERIMENT_ROOT / "lr_feature_churn_correlation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        EXPERIMENT_ROOT / "lr_permutation_vs_coefficient_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    high_risk.to_csv(
        EXPERIMENT_ROOT / "lr_high_risk_customers.csv",
        index=False,
        encoding="utf-8-sig",
    )
    local_explanations.to_csv(
        EXPERIMENT_ROOT / "lr_local_linear_explanations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_coef_features = coef.head(6)["feature"].tolist()
    plot_coefficient_importance(coef)
    plot_linear_contribution_importance(coef)
    plot_pdp(model, X_test, top_coef_features)

    summary = {
        "metric_summary": metric_summary(model, X_test, y_test),
        "interpretability_position": [
            "For a standardized LogisticRegression model, coefficients provide global direction and magnitude.",
            "Local additive logit contributions are exactly coefficient * standardized feature value.",
            "This is a linear-model alternative to SHAP in logit space, not a replacement for nonlinear Tree-SHAP.",
        ],
        "top_by_abs_coefficient": coef.head(15).to_dict(orient="records"),
        "top_by_mean_abs_logit_contribution": coef.sort_values(
            "mean_abs_logit_contribution", ascending=False
        )
        .head(15)
        .to_dict(orient="records"),
        "top_by_permutation_importance": comparison.dropna(
            subset=["permutation_rank"]
        )
        .sort_values("permutation_rank")
        .head(15)
        .to_dict(orient="records"),
        "top_high_risk_customers": high_risk.to_dict(orient="records"),
    }
    (EXPERIMENT_ROOT / "phase_5a_interpretability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Phase 5-A interpretability complete")
    print("\nTop coefficient features:")
    print(
        coef.head(12)[
            [
                "feature",
                "coefficient",
                "odds_ratio_per_1sd",
                "direction",
                "mean_abs_logit_contribution",
            ]
        ].to_string(index=False)
    )
    print("\nTop permutation vs coefficient comparison:")
    print(
        comparison.dropna(subset=["permutation_rank"])
        .sort_values("permutation_rank")
        .head(12)[
            [
                "feature",
                "permutation_rank",
                "f1_importance",
                "coef_rank",
                "coefficient",
                "direction",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
