import os
from pathlib import Path
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.ensemble import (
    BalancedBaggingClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from imblearn.over_sampling import SVMSMOTE
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
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
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
OUTPUT_ROOT = Path("processed")
EXPERIMENT_ROOT = OUTPUT_ROOT / "additional_experiments"
THRESHOLD_GRID = tuple(float(x) for x in np.round(np.arange(0.05, 0.611, 0.02), 2))
VALIDATION_SIZE = 0.25
TOP_K_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def load_processed_split(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(output_dir / "X_train.csv")
    X_test = pd.read_csv(output_dir / "X_test.csv")
    y_train = pd.read_csv(output_dir / "y_train.csv")["CHURN"].astype(int)
    y_test = pd.read_csv(output_dir / "y_test.csv")["CHURN"].astype(int)
    X_train_resampled = pd.read_csv(output_dir / "X_train_resampled.csv")
    y_train_resampled = pd.read_csv(output_dir / "y_train_resampled.csv")[
        "CHURN"
    ].astype(int)
    return X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled


def load_native_catboost_split(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train_native = pd.read_csv(output_dir / "X_train_analysis.csv")
    X_test_native = pd.read_csv(output_dir / "X_test_analysis.csv")
    for frame in [X_train_native, X_test_native]:
        for col in native_cat_features(frame):
            frame[col] = frame[col].fillna("Unknown").astype(str)
    return X_train_native, X_test_native


def native_cat_features(X: pd.DataFrame) -> list[str]:
    return [
        col
        for col in ["CRM_PID_Value_Segment", "EffectiveSegment", "Billing_ZIP"]
        if col in X.columns
    ]


def positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def fit_model(model: Any, X: pd.DataFrame, y: pd.Series, train_kind: str) -> Any:
    if train_kind == "native_categorical":
        model.fit(X, y, cat_features=native_cat_features(X))
    else:
        model.fit(X, y)
    return model


def classification_metrics(
    y_true: pd.Series | np.ndarray, y_pred: np.ndarray, scores: np.ndarray, prefix: str = ""
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pr_auc = float(average_precision_score(y_true, scores))
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = float("nan")
    return {
        prefix + "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        prefix + "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        prefix + "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        prefix + "accuracy": float(accuracy_score(y_true, y_pred)),
        prefix + "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        prefix + "roc_auc": roc_auc,
        prefix + "pr_auc": pr_auc,
        prefix + "average_precision": pr_auc,
        prefix + "mcc": float(matthews_corrcoef(y_true, y_pred)),
        prefix + "tp": int(tp),
        prefix + "fp": int(fp),
        prefix + "fn": int(fn),
        prefix + "tn": int(tn),
    }


def fit_threshold_training_data(
    X_train: pd.DataFrame, y_train: pd.Series, train_kind: str
) -> tuple[pd.DataFrame, pd.Series]:
    if train_kind != "resampled":
        return X_train, y_train

    smote = SVMSMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return (
        pd.DataFrame(X_resampled, columns=X_train.columns),
        pd.Series(y_resampled, name="CHURN").astype(int),
    )


def base_experiment_models() -> list[tuple[str, Any, str]]:
    models: list[tuple[str, Any, str]] = [
        (
            "LogisticRegression_SMOTE_C0.1",
            LogisticRegression(max_iter=5000, C=0.1, random_state=RANDOM_STATE),
            "resampled",
        ),
        (
            "LogisticRegression_SMOTE_C1.0",
            LogisticRegression(max_iter=5000, C=1.0, random_state=RANDOM_STATE),
            "resampled",
        ),
        (
            "LogisticRegression_balanced_C0.1",
            LogisticRegression(
                max_iter=5000,
                C=0.1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "original",
        ),
        (
            "LogisticRegression_balanced_C1.0",
            LogisticRegression(
                max_iter=5000,
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "original",
        ),
        (
            "RidgeClassifier_balanced_alpha0.3",
            RidgeClassifier(alpha=0.3, class_weight="balanced", random_state=RANDOM_STATE),
            "original",
        ),
        (
            "LinearSVC_balanced_C0.1",
            LinearSVC(
                C=0.1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=8000,
            ),
            "original",
        ),
        (
            "RandomForest_balanced_depth5",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "original",
        ),
        (
            "ExtraTrees_balanced_depth5",
            ExtraTreesClassifier(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "original",
        ),
        (
            "GradientBoosting_SMOTE_depth2_lr0.05",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                random_state=RANDOM_STATE,
            ),
            "resampled",
        ),
        (
            "HistGradientBoosting_balanced_lr0.03",
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_iter=300,
                max_leaf_nodes=15,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "original",
        ),
        (
            "AdaBoost_balanced_stump",
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                n_estimators=200,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            ),
            "original",
        ),
        (
            "EasyEnsemble_n10",
            EasyEnsembleClassifier(n_estimators=10, random_state=RANDOM_STATE, n_jobs=1),
            "original",
        ),
        (
            "RUSBoost_n100_lr0.05",
            RUSBoostClassifier(
                n_estimators=100,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            ),
            "original",
        ),
    ]

    for depth in [3, 5, None]:
        depth_name = "none" if depth is None else str(depth)
        for leaf in [10, 25]:
            models.append(
                (
                    f"BalancedBagging_tree_depth{depth_name}_leaf{leaf}",
                    BalancedBaggingClassifier(
                        estimator=DecisionTreeClassifier(
                            max_depth=depth,
                            min_samples_leaf=leaf,
                            random_state=RANDOM_STATE,
                        ),
                        n_estimators=100,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                    "original",
                )
            )

    for depth in [3, 4]:
        for lr in [0.03]:
            models.append(
                (
                    f"CatBoost_encoded_depth{depth}_lr{lr}",
                    CatBoostClassifier(
                        iterations=300,
                        learning_rate=lr,
                        depth=depth,
                        loss_function="Logloss",
                        eval_metric="F1",
                        auto_class_weights="Balanced",
                        random_seed=RANDOM_STATE,
                        thread_count=1,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    "original",
                )
            )
            models.append(
                (
                    f"CatBoost_native_depth{depth}_lr{lr}",
                    CatBoostClassifier(
                        iterations=300,
                        learning_rate=lr,
                        depth=depth,
                        loss_function="Logloss",
                        eval_metric="F1",
                        auto_class_weights="Balanced",
                        random_seed=RANDOM_STATE,
                        thread_count=1,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    "native_categorical",
                )
            )

    return models


def evaluate_candidate(
    *,
    variant: str,
    model_name: str,
    model: Any,
    train_kind: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_resampled: pd.DataFrame,
    y_train_resampled: pd.Series,
    X_train_native: pd.DataFrame,
    X_test_native: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    if train_kind == "native_categorical":
        threshold_X_train = X_train_native
        threshold_X_test = X_test_native
    else:
        threshold_X_train = X_train
        threshold_X_test = X_test

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        threshold_X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    X_fit = X_fit.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    y_fit = y_fit.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)

    threshold_model = clone(model)
    fit_X, fit_y = fit_threshold_training_data(X_fit, y_fit, train_kind)
    fit_model(threshold_model, fit_X, fit_y, train_kind)
    validation_scores = positive_scores(threshold_model, X_valid)

    sweep_rows = []
    for threshold in THRESHOLD_GRID:
        y_valid_pred = (validation_scores >= threshold).astype(int)
        row = {
            "variant": variant,
            "model": model_name,
            "train_data": train_kind,
            "threshold": threshold,
        }
        row.update(classification_metrics(y_valid, y_valid_pred, validation_scores))
        sweep_rows.append(row)

    validation_table = pd.DataFrame(sweep_rows).sort_values(
        ["f1", "recall", "precision"], ascending=[False, False, False]
    )
    best_validation = validation_table.iloc[0].to_dict()
    selected_threshold = float(best_validation["threshold"])

    final_model = clone(model)
    if train_kind == "native_categorical":
        final_fit_X, final_fit_y = X_train_native, y_train
        final_test_X = X_test_native
    elif train_kind == "resampled":
        final_fit_X, final_fit_y = X_train_resampled, y_train_resampled
        final_test_X = X_test
    else:
        final_fit_X, final_fit_y = X_train, y_train
        final_test_X = X_test
    fit_model(final_model, final_fit_X, final_fit_y, train_kind)
    test_scores = positive_scores(final_model, final_test_X)
    test_pred = (test_scores >= selected_threshold).astype(int)

    test_oracle_rows = []
    for threshold in THRESHOLD_GRID:
        y_test_pred = (test_scores >= threshold).astype(int)
        oracle_row = {"threshold": threshold}
        oracle_row.update(classification_metrics(y_test, y_test_pred, test_scores))
        test_oracle_rows.append(oracle_row)
    test_oracle_best = pd.DataFrame(test_oracle_rows).sort_values(
        ["f1", "recall", "precision"], ascending=[False, False, False]
    ).iloc[0]

    result = {
        "variant": variant,
        "model": model_name,
        "train_data": train_kind,
        "selected_threshold": selected_threshold,
        "validation_f1": float(best_validation["f1"]),
        "validation_recall": float(best_validation["recall"]),
        "validation_precision": float(best_validation["precision"]),
        "validation_pr_auc": float(best_validation["pr_auc"]),
        "validation_mcc": float(best_validation["mcc"]),
        "test_oracle_threshold": float(test_oracle_best["threshold"]),
        "test_oracle_f1": float(test_oracle_best["f1"]),
        "test_oracle_recall": float(test_oracle_best["recall"]),
        "test_oracle_precision": float(test_oracle_best["precision"]),
    }
    result.update(classification_metrics(y_test, test_pred, test_scores, prefix="test_"))
    return result, sweep_rows, test_scores


def top_k_budget_rows(
    *,
    variant: str,
    model: str,
    threshold: float,
    y_test: pd.Series,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    y_values = y_test.to_numpy()
    order = np.argsort(scores)[::-1]
    total_churn = int(y_values.sum())
    for fraction in TOP_K_FRACTIONS:
        k = max(1, int(round(len(y_values) * fraction)))
        selected = order[:k]
        captured = int(y_values[selected].sum())
        rows.append(
            {
                "variant": variant,
                "model": model,
                "selected_threshold": threshold,
                "top_fraction": fraction,
                "selected_count": k,
                "captured_churn": captured,
                "total_churn": total_churn,
                "precision_at_k": captured / k,
                "recall_at_k": captured / total_churn if total_churn else 0.0,
            }
        )
    return rows


def main() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    variant_dirs = [
        ("with_billing_zip", OUTPUT_ROOT / "model_a_with_billing_zip"),
        ("without_billing_zip", OUTPUT_ROOT / "model_b_without_billing_zip"),
    ]

    results = []
    sweep_rows = []
    score_cache: dict[tuple[str, str], tuple[float, np.ndarray, pd.Series]] = {}

    for variant, output_dir in variant_dirs:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_resampled,
            y_train_resampled,
        ) = load_processed_split(output_dir)
        X_train_native, X_test_native = load_native_catboost_split(output_dir)
        for model_name, model, train_kind in base_experiment_models():
            if train_kind == "native_categorical" and not native_cat_features(X_train_native):
                continue
            try:
                result, model_sweep_rows, test_scores = evaluate_candidate(
                    variant=variant,
                    model_name=model_name,
                    model=model,
                    train_kind=train_kind,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    X_train_resampled=X_train_resampled,
                    y_train_resampled=y_train_resampled,
                    X_train_native=X_train_native,
                    X_test_native=X_test_native,
                )
            except Exception as exc:
                results.append(
                    {
                        "variant": variant,
                        "model": model_name,
                        "train_data": train_kind,
                        "error": repr(exc),
                    }
                )
                continue
            results.append(result)
            sweep_rows.extend(model_sweep_rows)
            score_cache[(variant, model_name)] = (
                float(result["selected_threshold"]),
                test_scores,
                y_test,
            )
            print(
                f"{variant} {model_name} "
                f"test_f1={result['test_f1']:.4f} "
                f"recall={result['test_recall']:.4f} "
                f"precision={result['test_precision']:.4f}",
                flush=True,
            )
            pd.DataFrame(results).to_csv(
                EXPERIMENT_ROOT / "additional_model_results_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

    result_table = pd.DataFrame(results)
    if "error" not in result_table.columns:
        result_table["error"] = pd.NA
    ok_table = result_table[result_table["error"].isna()].copy()
    ok_table = ok_table.sort_values(
        ["test_f1", "test_recall", "test_precision"], ascending=[False, False, False]
    )
    result_table = pd.concat(
        [ok_table, result_table[result_table["error"].notna()]], ignore_index=True
    )

    result_table.to_csv(
        EXPERIMENT_ROOT / "additional_model_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(sweep_rows).to_csv(
        EXPERIMENT_ROOT / "additional_threshold_sweep.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_cols = [
        "variant",
        "model",
        "train_data",
        "selected_threshold",
        "validation_f1",
        "test_f1",
        "test_recall",
        "test_precision",
        "test_pr_auc",
        "test_mcc",
        "test_tp",
        "test_fp",
        "test_fn",
        "test_tn",
        "test_oracle_threshold",
        "test_oracle_f1",
        "test_oracle_recall",
        "test_oracle_precision",
    ]
    ok_table[summary_cols].head(25).to_csv(
        EXPERIMENT_ROOT / "additional_top25_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    budget_rows = []
    for _, row in ok_table.head(10).iterrows():
        cache_key = (row["variant"], row["model"])
        if cache_key not in score_cache:
            continue
        threshold, scores, y_test = score_cache[cache_key]
        budget_rows.extend(
            top_k_budget_rows(
                variant=row["variant"],
                model=row["model"],
                threshold=threshold,
                y_test=y_test,
                scores=scores,
            )
        )
    pd.DataFrame(budget_rows).to_csv(
        EXPERIMENT_ROOT / "operating_budget_topk.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nBest validation-selected test result:")
    best = ok_table.iloc[0]
    print(
        f"{best['variant']} {best['model']} "
        f"threshold={best['selected_threshold']:.2f} "
        f"f1={best['test_f1']:.4f} "
        f"recall={best['test_recall']:.4f} "
        f"precision={best['test_precision']:.4f}"
    )
    print(f"Outputs written to: {EXPERIMENT_ROOT}")


if __name__ == "__main__":
    main()
