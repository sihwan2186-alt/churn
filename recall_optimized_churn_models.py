import json
import os
from itertools import product
from pathlib import Path
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SVMSMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


RANDOM_STATE = 42
OUTPUT_ROOT = Path("processed")
EXPERIMENT_ROOT = OUTPUT_ROOT / "recall_optimized_models"
VARIANT_DIRS = {
    "with_billing_zip": OUTPUT_ROOT / "model_a_with_billing_zip",
    "without_billing_zip": OUTPUT_ROOT / "model_b_without_billing_zip",
}

VALIDATION_SIZE = 0.25
THRESHOLD_GRID = tuple(float(x) for x in np.round(np.arange(0.01, 0.801, 0.01), 2))
MIN_VALIDATION_PRECISION = 0.07
MAX_VALIDATION_CONTACT_RATE = 0.75


def load_processed_split(
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    output_dir = VARIANT_DIRS[variant]
    X_train = pd.read_csv(output_dir / "X_train.csv")
    X_test = pd.read_csv(output_dir / "X_test.csv")
    y_train = pd.read_csv(output_dir / "y_train.csv")["CHURN"].astype(int)
    y_test = pd.read_csv(output_dir / "y_test.csv")["CHURN"].astype(int)
    return X_train, X_test, y_train, y_test


def positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    prefix: str = "",
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = float("nan")
    return {
        prefix + "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        prefix + "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        prefix + "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        prefix + "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        prefix + "accuracy": float(accuracy_score(y_true, y_pred)),
        prefix + "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        prefix + "roc_auc": roc_auc,
        prefix + "pr_auc": float(average_precision_score(y_true, scores)),
        prefix + "mcc": float(matthews_corrcoef(y_true, y_pred)),
        prefix + "tp": int(tp),
        prefix + "fp": int(fp),
        prefix + "fn": int(fn),
        prefix + "tn": int(tn),
        prefix + "predicted_positive_rate": float(np.mean(y_pred)),
    }


def threshold_metrics(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    row = {
        "threshold": threshold,
    }
    row.update(classification_metrics(y_true, predictions, scores, prefix=""))
    return row


def select_recall_threshold(
    y_true: pd.Series,
    scores: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    rows = [threshold_metrics(y_true, scores, threshold) for threshold in THRESHOLD_GRID]
    sweep = pd.DataFrame(rows)
    eligible = sweep[
        sweep["precision"].ge(MIN_VALIDATION_PRECISION)
        & sweep["predicted_positive_rate"].le(MAX_VALIDATION_CONTACT_RATE)
    ].copy()
    strategy = (
        f"max_recall_precision>={MIN_VALIDATION_PRECISION}_"
        f"contact<={MAX_VALIDATION_CONTACT_RATE}"
    )
    if eligible.empty:
        eligible = sweep.copy()
        strategy = "fallback_max_f2_no_guardrail"
    best = eligible.sort_values(
        ["recall", "f2", "precision", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return best.to_dict(), sweep, strategy


def resample_if_needed(
    X: pd.DataFrame,
    y: pd.Series,
    train_kind: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if train_kind != "svmsmote":
        return X, y
    smote = SVMSMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return (
        pd.DataFrame(X_resampled, columns=X.columns),
        pd.Series(y_resampled, name="CHURN").astype(int),
    )


def candidate_logistic_models() -> list[dict[str, Any]]:
    candidates = []
    for train_kind, class_weight, c_value in product(
        ["original", "svmsmote"],
        ["balanced", None],
        [0.03, 0.1, 0.3, 1.0],
    ):
        if train_kind == "original" and class_weight is None:
            continue
        if train_kind == "svmsmote" and class_weight is not None:
            continue
        candidates.append(
            {
                "family": "LogisticRegression",
                "model_name": (
                    f"LogisticRegression_{train_kind}_"
                    f"{'balanced' if class_weight else 'plain'}_C{c_value}"
                ),
                "train_kind": train_kind,
                "params": {
                    "C": c_value,
                    "class_weight": class_weight,
                    "max_iter": 6000,
                    "solver": "lbfgs",
                    "random_state": RANDOM_STATE,
                },
            }
        )
    return candidates


def candidate_xgboost_models() -> list[dict[str, Any]]:
    candidates = []
    for max_depth, learning_rate, n_estimators, scale_pos_weight in product(
        [2, 3],
        [0.03, 0.06],
        [200],
        [14.5, 20.0],
    ):
        candidates.append(
            {
                "family": "XGBoost",
                "model_name": (
                    "XGBoost_weighted_"
                    f"depth{max_depth}_lr{learning_rate}_"
                    f"n{n_estimators}_spw{scale_pos_weight}"
                ),
                "train_kind": "original",
                "params": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "scale_pos_weight": scale_pos_weight,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "min_child_weight": 3,
                    "reg_lambda": 5.0,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "tree_method": "hist",
                    "random_state": RANDOM_STATE,
                    "n_jobs": 1,
                },
            }
        )
    for max_depth, learning_rate, n_estimators in product([2], [0.03, 0.06], [200]):
        candidates.append(
            {
                "family": "XGBoost",
                "model_name": (
                    "XGBoost_svmsmote_"
                    f"depth{max_depth}_lr{learning_rate}_n{n_estimators}"
                ),
                "train_kind": "svmsmote",
                "params": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "scale_pos_weight": 1.0,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "min_child_weight": 3,
                    "reg_lambda": 5.0,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "tree_method": "hist",
                    "random_state": RANDOM_STATE,
                    "n_jobs": 1,
                },
            }
        )
    return candidates


def candidate_catboost_models() -> list[dict[str, Any]]:
    candidates = []
    for depth, learning_rate, iterations, scale_pos_weight in product(
        [3, 4],
        [0.03, 0.06],
        [150],
        [14.5, 20.0],
    ):
        candidates.append(
            {
                "family": "CatBoost",
                "model_name": (
                    "CatBoost_weighted_"
                    f"depth{depth}_lr{learning_rate}_"
                    f"iter{iterations}_spw{scale_pos_weight}"
                ),
                "train_kind": "original",
                "params": {
                    "iterations": iterations,
                    "depth": depth,
                    "learning_rate": learning_rate,
                    "scale_pos_weight": scale_pos_weight,
                    "l2_leaf_reg": 6.0,
                    "loss_function": "Logloss",
                    "eval_metric": "Recall",
                    "random_seed": RANDOM_STATE,
                    "thread_count": 1,
                    "verbose": False,
                    "allow_writing_files": False,
                },
            }
        )
    return candidates


def build_model(candidate: dict[str, Any]) -> Any:
    family = candidate["family"]
    params = candidate["params"]
    if family == "LogisticRegression":
        return LogisticRegression(**params)
    if family == "XGBoost":
        return XGBClassifier(**params)
    if family == "CatBoost":
        return CatBoostClassifier(**params)
    raise ValueError(f"Unsupported family: {family}")


def evaluate_candidate(
    candidate: dict[str, Any],
    X_subtrain: pd.DataFrame,
    y_subtrain: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame]:
    X_fit, y_fit = resample_if_needed(X_subtrain, y_subtrain, candidate["train_kind"])
    model = build_model(candidate)
    model.fit(X_fit, y_fit)
    scores = positive_scores(model, X_val)
    best_threshold, sweep, strategy = select_recall_threshold(y_val, scores)
    row = {
        "family": candidate["family"],
        "model_name": candidate["model_name"],
        "train_kind": candidate["train_kind"],
        "params_json": json.dumps(candidate["params"], sort_keys=True),
        "selection_strategy": strategy,
    }
    row.update({f"validation_{key}": value for key, value in best_threshold.items()})
    sweep.insert(0, "family", candidate["family"])
    sweep.insert(1, "model_name", candidate["model_name"])
    sweep.insert(2, "train_kind", candidate["train_kind"])
    return row, sweep


def select_best_candidate(validation_results: pd.DataFrame) -> pd.Series:
    eligible = validation_results[
        validation_results["validation_precision"].ge(MIN_VALIDATION_PRECISION)
        & validation_results["validation_predicted_positive_rate"].le(
            MAX_VALIDATION_CONTACT_RATE
        )
    ].copy()
    if eligible.empty:
        eligible = validation_results.copy()
    return eligible.sort_values(
        [
            "validation_recall",
            "validation_f2",
            "validation_precision",
            "validation_threshold",
        ],
        ascending=[False, False, False, False],
    ).iloc[0]


def refit_and_test(
    candidate_row: pd.Series,
    candidate_lookup: dict[str, dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    candidate = candidate_lookup[candidate_row["model_name"]]
    X_fit, y_fit = resample_if_needed(X_train, y_train, candidate["train_kind"])
    model = build_model(candidate)
    model.fit(X_fit, y_fit)
    scores = positive_scores(model, X_test)
    threshold = float(candidate_row["validation_threshold"])
    predictions = (scores >= threshold).astype(int)

    result = {
        "family": candidate["family"],
        "model_name": candidate["model_name"],
        "train_kind": candidate["train_kind"],
        "selected_threshold": threshold,
        "selection_strategy": candidate_row["selection_strategy"],
        "params_json": candidate_row["params_json"],
    }
    for key in [
        "validation_f1",
        "validation_f2",
        "validation_recall",
        "validation_precision",
        "validation_predicted_positive_rate",
        "validation_tp",
        "validation_fp",
        "validation_fn",
        "validation_tn",
    ]:
        result[key] = candidate_row[key]
    result.update(classification_metrics(y_test, predictions, scores, prefix="test_"))
    return result


def run_family_for_variant(
    variant: str,
    family: str,
    candidates: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train, X_test, y_train, y_test = load_processed_split(variant)
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    X_subtrain = X_subtrain.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_subtrain = y_subtrain.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    validation_rows = []
    sweep_rows = []
    for candidate in candidates:
        row, sweep = evaluate_candidate(candidate, X_subtrain, y_subtrain, X_val, y_val)
        row["variant"] = variant
        validation_rows.append(row)
        sweep.insert(0, "variant", variant)
        sweep_rows.append(sweep)

    validation_results = pd.DataFrame(validation_rows)
    best_candidate = select_best_candidate(validation_results)
    candidate_lookup = {candidate["model_name"]: candidate for candidate in candidates}
    test_result = refit_and_test(
        best_candidate,
        candidate_lookup,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    test_result["variant"] = variant
    return (
        validation_results,
        pd.concat(sweep_rows, ignore_index=True),
        pd.DataFrame([test_result]),
    )


def main() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    family_candidates = {
        "LogisticRegression": candidate_logistic_models(),
        "XGBoost": candidate_xgboost_models(),
        "CatBoost": candidate_catboost_models(),
    }

    all_validation = []
    all_sweeps = []
    all_best = []
    for variant in VARIANT_DIRS:
        for family, candidates in family_candidates.items():
            print(
                f"Running {variant} / {family} ({len(candidates)} candidates)",
                flush=True,
            )
            validation, sweep, best = run_family_for_variant(variant, family, candidates)
            all_validation.append(validation)
            all_sweeps.append(sweep)
            all_best.append(best)

    validation_table = pd.concat(all_validation, ignore_index=True)
    sweep_table = pd.concat(all_sweeps, ignore_index=True)
    best_table = pd.concat(all_best, ignore_index=True)
    best_table = best_table.sort_values(
        ["test_recall", "test_f2", "test_precision"],
        ascending=[False, False, False],
    )

    validation_table.to_csv(
        EXPERIMENT_ROOT / "recall_optimized_validation_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sweep_table.to_csv(
        EXPERIMENT_ROOT / "recall_optimized_threshold_sweep.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_table.to_csv(
        EXPERIMENT_ROOT / "recall_optimized_best_by_family.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "selection_rule": (
            "Validation에서 precision >= 0.07, predicted_positive_rate <= 0.75를 "
            "만족하는 후보 중 recall을 최우선으로 선택하고, 동률이면 F2와 precision을 "
            "사용했다. Test set은 최종 1회 평가에만 사용했다."
        ),
        "validation_size": VALIDATION_SIZE,
        "min_validation_precision": MIN_VALIDATION_PRECISION,
        "max_validation_contact_rate": MAX_VALIDATION_CONTACT_RATE,
        "best_by_family": best_table.to_dict(orient="records"),
    }
    (EXPERIMENT_ROOT / "recall_optimized_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Recall-Optimized Churn Models",
        "",
        "목적: CatBoost, XGBoost, Logistic Regression을 ChurnRadar의 이탈 고객 포착 목적에 맞게 recall 중심으로 재최적화했다.",
        "",
        "선택 규칙: train 내부 validation set에서 `precision >= 0.07`, `predicted_positive_rate <= 0.75`를 만족하는 후보 중 recall을 최우선으로 선택했다. 동률이면 F2와 precision을 사용했다. Test set은 최종 평가에만 사용했다.",
        "",
        "## Best Test Results",
        "",
        "| Variant | Family | Model | Threshold | Recall | F2 | F1 | Precision | TP | FP | FN | Contact Rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in best_table.iterrows():
        report_lines.append(
            "| "
            f"{row['variant']} | {row['family']} | {row['model_name']} | "
            f"{row['selected_threshold']:.2f} | {row['test_recall']:.4f} | "
            f"{row['test_f2']:.4f} | {row['test_f1']:.4f} | "
            f"{row['test_precision']:.4f} | {int(row['test_tp'])} | "
            f"{int(row['test_fp'])} | {int(row['test_fn'])} | "
            f"{row['test_predicted_positive_rate']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- recall 최적화는 이탈 고객을 더 많이 잡는 대신 FP를 크게 늘린다.",
            "- 따라서 이 결과는 F1 대표 모델을 대체하기보다, 이탈 포착 우선 캠페인 후보로 해석한다.",
            "- 실무에서는 contact rate와 상담/혜택 예산을 함께 제한해야 한다.",
            "",
        ]
    )
    (EXPERIMENT_ROOT / "RECALL_OPTIMIZED_MODELS.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nBest recall-optimized results:")
    print(
        best_table[
            [
                "variant",
                "family",
                "model_name",
                "selected_threshold",
                "test_recall",
                "test_f2",
                "test_f1",
                "test_precision",
                "test_tp",
                "test_fp",
                "test_fn",
                "test_predicted_positive_rate",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
