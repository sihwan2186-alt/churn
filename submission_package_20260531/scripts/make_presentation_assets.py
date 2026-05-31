from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
ASSET_DIR = ROOT / "presentation_assets"


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def short_model_name(value: str) -> str:
    return (
        value.replace("LogisticRegression_SMOTE", "LogReg SMOTE")
        .replace("BalancedBagging_original", "BalancedBagging")
        .replace("CatBoost_native_categorical", "CatBoost Native")
        .replace("CatBoost_original_balanced", "CatBoost Encoded")
        .replace("EasyEnsemble", "Paper EasyEns.")
    )


def plot_model_metric_comparison(summary: pd.DataFrame) -> None:
    rows = summary[summary["selection_purpose"] != "paper_reference_best"].copy()
    rows["label"] = rows["model"].map(short_model_name)
    x = np.arange(len(rows))
    width = 0.24

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width, rows["f1"], width, label="F1", color="#2563eb")
    plt.bar(x, rows["recall"], width, label="Recall", color="#059669")
    plt.bar(x + width, rows["precision"], width, label="Precision", color="#dc2626")
    plt.xticks(x, rows["label"], rotation=0)
    plt.ylabel("Score")
    plt.ylim(0, max(rows["recall"].max(), rows["f1"].max()) * 1.18)
    plt.title("Final Model Metric Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(ASSET_DIR / "01_model_metric_comparison.png")


def plot_confusion_counts(summary: pd.DataFrame) -> None:
    rows = summary[summary["selection_purpose"] != "paper_reference_best"].copy()
    rows["label"] = rows["model"].map(short_model_name)
    x = np.arange(len(rows))

    plt.figure(figsize=(10, 5.5))
    bottom = np.zeros(len(rows))
    for metric, color in [
        ("tp", "#16a34a"),
        ("fp", "#f97316"),
        ("fn", "#ef4444"),
        ("tn", "#64748b"),
    ]:
        values = rows[metric].astype(float).to_numpy()
        plt.bar(x, values, bottom=bottom, label=metric.upper(), color=color)
        bottom += values

    plt.xticks(x, rows["label"])
    plt.ylabel("Test Rows")
    plt.title("Confusion Matrix Counts by Operating Point")
    plt.legend(ncol=4)
    plt.grid(axis="y", alpha=0.25)
    savefig(ASSET_DIR / "02_confusion_counts.png")


def plot_precision_recall_tradeoff(threshold_sweep: pd.DataFrame) -> None:
    targets = [
        ("with_billing_zip", "BalancedBagging_original"),
        ("without_billing_zip", "LogisticRegression_SMOTE"),
        ("with_billing_zip", "CatBoost_native_categorical"),
    ]
    plt.figure(figsize=(9, 6))
    for variant, model in targets:
        subset = threshold_sweep[
            (threshold_sweep["variant"] == variant)
            & (threshold_sweep["model"] == model)
        ].sort_values("threshold")
        if subset.empty:
            continue
        label = short_model_name(model)
        plt.plot(subset["recall"], subset["precision"], marker="o", ms=3, label=label)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Trade-off Across Thresholds")
    plt.grid(alpha=0.25)
    plt.legend()
    savefig(ASSET_DIR / "03_precision_recall_tradeoff.png")


def plot_feature_importance(feature_importance: pd.DataFrame) -> None:
    subset = feature_importance[
        feature_importance["operating_point"] == "main_f1_baseline"
    ].head(10)
    subset = subset.sort_values("f1_importance")

    plt.figure(figsize=(10, 6))
    plt.barh(subset["feature"], subset["f1_importance"], color="#7c3aed")
    plt.xlabel("F1 drop after permutation")
    plt.title("Top Feature Importance for Main F1 Model")
    plt.grid(axis="x", alpha=0.25)
    savefig(ASSET_DIR / "04_feature_importance_main.png")


def plot_paper_comparison(summary: pd.DataFrame) -> None:
    selected = summary[
        summary["selection_purpose"].isin(["main_f1_model", "paper_reference_best"])
    ].copy()
    selected["label"] = selected["selection_purpose"].map(
        {
            "main_f1_model": "Our LogReg SMOTE",
            "paper_reference_best": "Paper EasyEns.",
        }
    )
    x = np.arange(len(selected))
    width = 0.28

    plt.figure(figsize=(8, 5.5))
    plt.bar(x - width / 2, selected["f1"], width, label="F1", color="#2563eb")
    plt.bar(x + width / 2, selected["recall"], width, label="Recall", color="#059669")
    plt.xticks(x, selected["label"])
    plt.ylabel("Score")
    plt.title("Project Result vs Reference Paper")
    plt.ylim(0, max(selected["recall"].max(), selected["f1"].max()) * 1.2)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(ASSET_DIR / "05_paper_comparison.png")


def main() -> None:
    ASSET_DIR.mkdir(exist_ok=True)
    summary = pd.read_csv(ROOT / "final_model_summary.csv")
    threshold_sweep = pd.read_csv(PROCESSED / "threshold_tuning_sweep.csv")
    feature_importance = pd.read_csv(PROCESSED / "feature_importance_top.csv")

    plot_model_metric_comparison(summary)
    plot_confusion_counts(summary)
    plot_precision_recall_tradeoff(threshold_sweep)
    plot_feature_importance(feature_importance)
    plot_paper_comparison(summary)

    print(f"Presentation assets written to {ASSET_DIR}")


if __name__ == "__main__":
    main()
