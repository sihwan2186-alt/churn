import json
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
from sklearn.model_selection import StratifiedKFold

from preprocess_churn import (
    INPUT_FILE,
    RANDOM_STATE,
    add_engineered_features,
    build_comparison_models,
    encode_categoricals,
    impute_train_test,
    load_base_dataframe,
    scale_numeric_features,
)


OUTPUT_ROOT = Path("processed") / "phase_4_paper_comparison"
CV_FOLDS = 5

CV_RUNS = [
    {
        "variant": "with_billing_zip",
        "model": "EasyEnsemble_original",
        "train_kind": "original",
        "use_billing_zip": True,
    },
    {
        "variant": "without_billing_zip",
        "model": "EasyEnsemble_original",
        "train_kind": "original",
        "use_billing_zip": False,
    },
    {
        "variant": "with_billing_zip",
        "model": "BalancedBagging_original",
        "train_kind": "original",
        "use_billing_zip": True,
    },
    {
        "variant": "without_billing_zip",
        "model": "LogisticRegression_SMOTE",
        "train_kind": "resampled",
        "use_billing_zip": False,
    },
]


def prepare_fold(
    X_train_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    use_billing_zip: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train_raw.copy()
    X_valid = X_valid_raw.copy()

    if use_billing_zip:
        cat_cols = ["CRM_PID_Value_Segment", "EffectiveSegment", "Billing_ZIP"]
    else:
        X_train = X_train.drop(columns=["Billing_ZIP"])
        X_valid = X_valid.drop(columns=["Billing_ZIP"])
        cat_cols = ["CRM_PID_Value_Segment", "EffectiveSegment"]

    X_train, X_valid, _ = impute_train_test(X_train, X_valid, use_billing_zip)

    if use_billing_zip:
        X_train["Billing_ZIP"] = X_train["Billing_ZIP"].astype(str)
        X_valid["Billing_ZIP"] = X_valid["Billing_ZIP"].astype(str)

    X_train = add_engineered_features(X_train)
    X_valid = add_engineered_features(X_valid)
    X_train, X_valid, _ = encode_categoricals(X_train, X_valid, cat_cols)
    X_train, X_valid, _, _ = scale_numeric_features(X_train, X_valid)
    return X_train, X_valid


def metric_row(y_true: pd.Series, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def fit_for_run(
    model: Any,
    train_kind: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    fitted = clone(model)
    if train_kind == "resampled":
        smote = SVMSMOTE(random_state=RANDOM_STATE)
        X_fit, y_fit = smote.fit_resample(X_train, y_train)
        X_fit = pd.DataFrame(X_fit, columns=X_train.columns)
        y_fit = pd.Series(y_fit, name="CHURN").astype(int)
    else:
        X_fit, y_fit = X_train, y_train
    fitted.fit(X_fit, y_fit)
    return fitted


def summarize_cv(fold_table: pd.DataFrame) -> pd.DataFrame:
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
    for (variant, model, train_kind), group in fold_table.groupby(
        ["variant", "model", "train_kind"]
    ):
        row = {
            "variant": variant,
            "model": model,
            "train_kind": train_kind,
            "folds": int(len(group)),
        }
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


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df, base_summary = load_base_dataframe(INPUT_FILE)
    X = df.drop(columns=["CHURN"])
    y = df["CHURN"]

    model_lookup = {
        model_name: (model, train_kind)
        for model_name, model, train_kind in build_comparison_models()
    }

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train_raw = X.iloc[train_idx].reset_index(drop=True)
        X_valid_raw = X.iloc[valid_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)

        prepared_cache: dict[bool, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for run in CV_RUNS:
            use_billing_zip = bool(run["use_billing_zip"])
            if use_billing_zip not in prepared_cache:
                prepared_cache[use_billing_zip] = prepare_fold(
                    X_train_raw, X_valid_raw, use_billing_zip
                )
            X_train_fold, X_valid_fold = prepared_cache[use_billing_zip]

            model, expected_train_kind = model_lookup[run["model"]]
            if expected_train_kind != run["train_kind"]:
                raise ValueError(f"Unexpected train kind for {run['model']}")

            fitted = fit_for_run(model, run["train_kind"], X_train_fold, y_train)
            scores = positive_scores(fitted, X_valid_fold)
            predictions = np.asarray(fitted.predict(X_valid_fold)).astype(int)

            row = {
                "fold": fold_idx,
                "variant": run["variant"],
                "model": run["model"],
                "train_kind": run["train_kind"],
                "use_billing_zip": use_billing_zip,
                "train_rows": int(len(y_train)),
                "valid_rows": int(len(y_valid)),
                "valid_positives": int(y_valid.sum()),
            }
            row.update(metric_row(y_valid, predictions, scores))
            fold_rows.append(row)

    fold_table = pd.DataFrame(fold_rows)
    summary_table = summarize_cv(fold_table)

    fold_table.to_csv(
        OUTPUT_ROOT / "phase_4_cv_fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_table.to_csv(
        OUTPUT_ROOT / "phase_4_cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "base_summary": base_summary,
        "cv_folds": CV_FOLDS,
        "cv_strategy": "StratifiedKFold(shuffle=True, random_state=42)",
        "leakage_guardrails": [
            "Each fold fits imputation, encoding, scaling, and SVMSMOTE on the training fold only.",
            "Validation folds are transformed with training-fold artifacts only.",
            "No threshold tuning is applied inside this CV; predictions use each model's default threshold.",
        ],
        "results": summary_table.to_dict(orient="records"),
    }
    (OUTPUT_ROOT / "phase_4_cv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Phase 4 CV complete")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
