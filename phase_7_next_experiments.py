import json
import os
from pathlib import Path
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from phase_4_cross_validation import prepare_fold
from preprocess_churn import (
    INPUT_FILE,
    RANDOM_STATE,
    build_comparison_models,
    load_base_dataframe,
)


OUTPUT_ROOT = Path("processed")
PAPER_ABLATION_ROOT = OUTPUT_ROOT / "paper_ablation_variants"
PHASE_OUTPUT = OUTPUT_ROOT / "phase_7_next_experiments"
CV_FOLDS = 5
VALIDATION_SIZE = 0.25
THRESHOLD_GRID = tuple(float(x) for x in np.round(np.arange(0.05, 0.701, 0.01), 2))
MIN_RECALL_CONSTRAINT = 0.30


def ensure_output_dir() -> None:
    PHASE_OUTPUT.mkdir(parents=True, exist_ok=True)


def read_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one-column series file: {path}")
    return frame.iloc[:, 0].astype(int)


def positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def safe_roc_auc(y_true: pd.Series, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def metric_row(y_true: pd.Series, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": safe_roc_auc(y_true, scores),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def model_lookup() -> dict[str, tuple[Any, str]]:
    lookup = {
        model_name: (model, train_kind)
        for model_name, model, train_kind in build_comparison_models()
    }
    lookup["BalancedBagging_tree_depthnone_leaf25"] = (
        BalancedBaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=None,
                min_samples_leaf=25,
                random_state=RANDOM_STATE,
            ),
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "original",
    )
    return lookup


def phase7_model_specs() -> list[dict[str, Any]]:
    lookup = model_lookup()
    names = [
        "LogisticRegression_SMOTE",
        "BalancedBagging_original",
        "BalancedBagging_tree_depthnone_leaf25",
        "EasyEnsemble_original",
    ]
    return [
        {
            "model": name,
            "estimator": lookup[name][0],
            "train_kind": lookup[name][1],
        }
        for name in names
    ]


def fit_for_train_kind(
    model: Any,
    train_kind: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_train_resampled: pd.DataFrame | None = None,
    y_train_resampled: pd.Series | None = None,
) -> Any:
    fitted = clone(model)
    if train_kind == "resampled":
        if X_train_resampled is None or y_train_resampled is None:
            smote = SVMSMOTE(random_state=RANDOM_STATE)
            X_fit, y_fit = smote.fit_resample(X_train, y_train)
            X_fit = pd.DataFrame(X_fit, columns=X_train.columns)
            y_fit = pd.Series(y_fit, name="CHURN").astype(int)
        else:
            X_fit, y_fit = X_train_resampled, y_train_resampled
    else:
        X_fit, y_fit = X_train, y_train
    fitted.fit(X_fit, y_fit)
    return fitted


def select_validation_threshold(
    model: Any,
    train_kind: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Any]:
    X_fit_raw, X_valid, y_fit_raw, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    fitted = fit_for_train_kind(model, train_kind, X_fit_raw, y_fit_raw)
    scores = positive_scores(fitted, X_valid)
    rows = []
    for threshold in THRESHOLD_GRID:
        pred = (scores >= threshold).astype(int)
        row = {
            "threshold": threshold,
            **metric_row(y_valid, pred, scores),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    constrained = table[table["recall"].ge(MIN_RECALL_CONSTRAINT)]
    if constrained.empty:
        candidates = table
        strategy = "max_f1_fallback_no_threshold_met_min_recall"
    else:
        candidates = constrained
        strategy = "max_f1_with_min_recall_constraint"
    best = candidates.sort_values(
        ["f1", "precision", "recall"],
        ascending=[False, False, False],
    ).iloc[0]
    result = {f"validation_{key}": value for key, value in best.to_dict().items()}
    result["threshold_selection_strategy"] = strategy
    result["min_recall_constraint"] = MIN_RECALL_CONSTRAINT
    return result


def load_paper_variant(variant: str) -> dict[str, pd.DataFrame | pd.Series]:
    variant_dir = PAPER_ABLATION_ROOT / variant
    return {
        "X_train": pd.read_csv(variant_dir / "X_train.csv"),
        "X_test": pd.read_csv(variant_dir / "X_test.csv"),
        "X_train_resampled": pd.read_csv(variant_dir / "X_train_resampled.csv"),
        "y_train": read_series(variant_dir / "y_train.csv"),
        "y_test": read_series(variant_dir / "y_test.csv"),
        "y_train_resampled": read_series(variant_dir / "y_train_resampled.csv"),
    }


def run_paper_ablation_benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    variant_summary = pd.read_csv(PAPER_ABLATION_ROOT / "variant_summary.csv")
    rows = []
    for _, variant_info in variant_summary.iterrows():
        variant = str(variant_info["variant"])
        data = load_paper_variant(variant)
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        X_train_resampled = data["X_train_resampled"]
        y_train_resampled = data["y_train_resampled"]

        for spec in phase7_model_specs():
            model_name = spec["model"]
            train_kind = spec["train_kind"]
            estimator = spec["estimator"]
            validation = select_validation_threshold(estimator, train_kind, X_train, y_train)
            fitted = fit_for_train_kind(
                estimator,
                train_kind,
                X_train,
                y_train,
                X_train_resampled,
                y_train_resampled,
            )
            scores = positive_scores(fitted, X_test)

            for mode, threshold in [
                ("fixed_0.50", 0.50),
                ("validation_threshold", float(validation["validation_threshold"])),
            ]:
                pred = (scores >= threshold).astype(int)
                row = {
                    "experiment": "paper_ablation_benchmark",
                    "variant": variant,
                    "zip_mode": variant_info["zip_mode"],
                    "ka_mode": variant_info["ka_mode"],
                    "transform_mode": variant_info["transform_mode"],
                    "include_extended_interactions": bool(
                        variant_info["include_extended_interactions"]
                    ),
                    "feature_count": int(variant_info["feature_count"]),
                    "model": model_name,
                    "train_kind": train_kind,
                    "evaluation": mode,
                    "threshold": threshold,
                    "test_rows": int(len(y_test)),
                    "test_positives": int(y_test.sum()),
                }
                row.update(metric_row(y_test, pred, scores))
                row.update(validation)
                rows.append(row)

    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(
        PHASE_OUTPUT / "paper_ablation_benchmark.csv",
        index=False,
        encoding="utf-8-sig",
    )

    threshold_rows = benchmark[benchmark["evaluation"].eq("validation_threshold")]
    best_by_variant = (
        threshold_rows.sort_values(["f1", "pr_auc"], ascending=[False, False])
        .groupby("variant", as_index=False)
        .head(1)
        .sort_values("f1", ascending=False)
    )
    best_by_variant.to_csv(
        PHASE_OUTPUT / "paper_ablation_best_by_variant.csv",
        index=False,
        encoding="utf-8-sig",
    )
    threshold_rows.sort_values(["f1", "pr_auc"], ascending=[False, False]).head(30).to_csv(
        PHASE_OUTPUT / "paper_ablation_top30.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return benchmark, best_by_variant


def summarize_grouped_metrics(table: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        "f1",
        "recall",
        "precision",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "mcc",
    ]
    rows = []
    for group_values, group in table.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values))
        row["folds"] = int(len(group))
        for metric in metric_cols:
            mean_value = float(group[metric].mean())
            sd_value = float(group[metric].std(ddof=1))
            se_value = sd_value / np.sqrt(len(group))
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_sd"] = sd_value
            row[f"{metric}_ci95_low"] = mean_value - 1.96 * se_value
            row[f"{metric}_ci95_high"] = mean_value + 1.96 * se_value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False)


def run_tuned_candidate_cv() -> tuple[pd.DataFrame, pd.DataFrame]:
    df, _ = load_base_dataframe(INPUT_FILE)
    X = df.drop(columns=["CHURN"])
    y = df["CHURN"]
    specs = {
        spec["model"]: spec
        for spec in phase7_model_specs()
        if spec["model"]
        in {
            "LogisticRegression_SMOTE",
            "BalancedBagging_original",
            "BalancedBagging_tree_depthnone_leaf25",
            "EasyEnsemble_original",
        }
    }
    cv_runs = [
        {
            "variant": "with_billing_zip",
            "model": "BalancedBagging_original",
            "use_billing_zip": True,
        },
        {
            "variant": "with_billing_zip",
            "model": "BalancedBagging_tree_depthnone_leaf25",
            "use_billing_zip": True,
        },
        {
            "variant": "without_billing_zip",
            "model": "LogisticRegression_SMOTE",
            "use_billing_zip": False,
        },
        {
            "variant": "with_billing_zip",
            "model": "EasyEnsemble_original",
            "use_billing_zip": True,
        },
    ]

    fold_rows = []
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train_raw = X.iloc[train_idx].reset_index(drop=True)
        X_valid_raw = X.iloc[valid_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)
        prepared_cache: dict[bool, tuple[pd.DataFrame, pd.DataFrame]] = {}

        for run in cv_runs:
            use_billing_zip = bool(run["use_billing_zip"])
            if use_billing_zip not in prepared_cache:
                prepared_cache[use_billing_zip] = prepare_fold(
                    X_train_raw,
                    X_valid_raw,
                    use_billing_zip,
                )
            X_train_fold, X_valid_fold = prepared_cache[use_billing_zip]
            spec = specs[run["model"]]
            fitted = fit_for_train_kind(
                spec["estimator"],
                spec["train_kind"],
                X_train_fold,
                y_train,
            )
            scores = positive_scores(fitted, X_valid_fold)
            pred = (scores >= 0.50).astype(int)
            row = {
                "experiment": "tuned_candidate_cv_default_threshold",
                "fold": fold_idx,
                "variant": run["variant"],
                "model": run["model"],
                "train_kind": spec["train_kind"],
                "use_billing_zip": use_billing_zip,
                "threshold": 0.50,
                "train_rows": int(len(y_train)),
                "valid_rows": int(len(y_valid)),
                "valid_positives": int(y_valid.sum()),
            }
            row.update(metric_row(y_valid, pred, scores))
            fold_rows.append(row)

    fold_table = pd.DataFrame(fold_rows)
    summary = summarize_grouped_metrics(
        fold_table,
        ["variant", "model", "train_kind"],
    )
    fold_table.to_csv(
        PHASE_OUTPUT / "tuned_candidate_cv_fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        PHASE_OUTPUT / "tuned_candidate_cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return fold_table, summary


def write_summary_json(
    paper_best: pd.DataFrame,
    cv_summary: pd.DataFrame,
) -> None:
    summary = {
        "purpose": [
            "Validate whether the additional tuned BalancedBagging candidate is stable under 5-fold CV.",
            "Benchmark already-generated paper ablation variants that were previously preprocessed but not model-scored.",
        ],
        "guardrails": [
            "Paper ablation threshold is selected on a validation split from training data, then applied once to test.",
            "CV preprocessing fits imputation, encoding, scaling, and SVMSMOTE inside each training fold.",
            "CV results use the default 0.50 operating threshold to match the existing Phase 4 convention.",
        ],
        "paper_ablation_best_overall": paper_best.head(10).to_dict(orient="records"),
        "cv_summary": cv_summary.to_dict(orient="records"),
    }
    (PHASE_OUTPUT / "phase_7_next_experiments_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ensure_output_dir()
    _, paper_best = run_paper_ablation_benchmark()
    _, cv_summary = run_tuned_candidate_cv()
    write_summary_json(paper_best, cv_summary)

    print("Phase 7 next experiments complete")
    print("\nPaper ablation best by variant:")
    print(
        paper_best[
            ["variant", "model", "evaluation", "threshold", "f1", "recall", "precision"]
        ].head(12).to_string(index=False)
    )
    print("\nTuned candidate CV summary:")
    print(
        cv_summary[
            [
                "variant",
                "model",
                "f1_mean",
                "f1_sd",
                "recall_mean",
                "precision_mean",
                "pr_auc_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
