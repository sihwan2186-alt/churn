import json
from math import erfc, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


RANDOM_STATE = 42
BOOTSTRAP_REPEATS = 5000
OUTPUT_ROOT = Path("processed") / "phase_8_statistical_validation"
PREDICTIONS_FILE = (
    Path("processed")
    / "phase_6_extended_case_studies"
    / "phase6_model_predictions.csv"
)

MODEL_CASES = [
    "LR_no_zip_f1",
    "BalancedBagging_with_zip",
    "EasyEnsemble_with_zip",
    "CatBoost_native_with_zip",
    "XGBoost_with_zip",
]

PAIRWISE_COMPARISONS = [
    ("LR_no_zip_f1", "BalancedBagging_with_zip"),
    ("LR_no_zip_f1", "EasyEnsemble_with_zip"),
    ("BalancedBagging_with_zip", "EasyEnsemble_with_zip"),
    ("CatBoost_native_with_zip", "XGBoost_with_zip"),
    ("BalancedBagging_with_zip", "CatBoost_native_with_zip"),
]


def ensure_output_dir() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    }


def bootstrap_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    y_true = predictions["actual"].to_numpy(dtype=int)
    n = len(y_true)
    summary_rows = []
    sample_rows = []

    for case in MODEL_CASES:
        y_pred = predictions[f"{case}_pred"].to_numpy(dtype=int)
        point = metric_bundle(y_true, y_pred)
        samples = {
            "f1": [],
            "recall": [],
            "precision": [],
        }
        for _ in range(BOOTSTRAP_REPEATS):
            idx = rng.integers(0, n, size=n)
            metrics = metric_bundle(y_true[idx], y_pred[idx])
            for metric, value in metrics.items():
                samples[metric].append(value)

        row: dict[str, Any] = {
            "case_label": case,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "rows": n,
            "positives": int(y_true.sum()),
        }
        for metric, values in samples.items():
            values_array = np.asarray(values, dtype=float)
            row[f"{metric}_point"] = point[metric]
            row[f"{metric}_ci95_low"] = float(np.percentile(values_array, 2.5))
            row[f"{metric}_ci95_high"] = float(np.percentile(values_array, 97.5))
            row[f"{metric}_bootstrap_mean"] = float(values_array.mean())
        summary_rows.append(row)

        for metric, values in samples.items():
            for value in values:
                sample_rows.append(
                    {
                        "case_label": case,
                        "metric": metric,
                        "value": float(value),
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def bootstrap_pairwise_differences(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE + 1)
    y_true = predictions["actual"].to_numpy(dtype=int)
    n = len(y_true)
    rows = []
    for left, right in PAIRWISE_COMPARISONS:
        left_pred = predictions[f"{left}_pred"].to_numpy(dtype=int)
        right_pred = predictions[f"{right}_pred"].to_numpy(dtype=int)
        left_point = metric_bundle(y_true, left_pred)
        right_point = metric_bundle(y_true, right_pred)
        samples = {"f1": [], "recall": [], "precision": []}
        for _ in range(BOOTSTRAP_REPEATS):
            idx = rng.integers(0, n, size=n)
            left_metrics = metric_bundle(y_true[idx], left_pred[idx])
            right_metrics = metric_bundle(y_true[idx], right_pred[idx])
            for metric in samples:
                samples[metric].append(left_metrics[metric] - right_metrics[metric])

        for metric, values in samples.items():
            values_array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "left_case": left,
                    "right_case": right,
                    "metric": metric,
                    "left_minus_right_point": left_point[metric] - right_point[metric],
                    "ci95_low": float(np.percentile(values_array, 2.5)),
                    "ci95_high": float(np.percentile(values_array, 97.5)),
                    "bootstrap_mean": float(values_array.mean()),
                    "share_left_greater": float((values_array > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def mcnemar_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    y_true = predictions["actual"].to_numpy(dtype=int)
    rows = []
    for left, right in PAIRWISE_COMPARISONS:
        left_pred = predictions[f"{left}_pred"].to_numpy(dtype=int)
        right_pred = predictions[f"{right}_pred"].to_numpy(dtype=int)
        left_correct = left_pred == y_true
        right_correct = right_pred == y_true
        left_only_correct = int((left_correct & ~right_correct).sum())
        right_only_correct = int((~left_correct & right_correct).sum())
        discordant = left_only_correct + right_only_correct
        if discordant == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = ((abs(left_only_correct - right_only_correct) - 1) ** 2) / discordant
            p_value = erfc(sqrt(statistic / 2.0))
        rows.append(
            {
                "left_case": left,
                "right_case": right,
                "left_only_correct_b": left_only_correct,
                "right_only_correct_c": right_only_correct,
                "discordant_pairs": discordant,
                "mcnemar_chi2_continuity": float(statistic),
                "approx_p_value": float(p_value),
                "interpretation": (
                    "different_error_pattern"
                    if p_value < 0.05
                    else "no_clear_accuracy_difference"
                ),
            }
        )
    return pd.DataFrame(rows)


def write_summary_json(
    metric_ci: pd.DataFrame,
    pairwise: pd.DataFrame,
    mcnemar: pd.DataFrame,
) -> None:
    summary = {
        "purpose": "Bootstrap confidence intervals and paired tests for selected operating points.",
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "note": [
            "Bootstrap CIs are computed on the fixed hold-out test set by resampling test rows.",
            "McNemar uses paired correctness and a continuity-corrected chi-square approximation.",
            "For imbalanced churn, McNemar tests overall correctness, not recall or F1 directly.",
        ],
        "metric_ci": metric_ci.to_dict(orient="records"),
        "pairwise_differences": pairwise.to_dict(orient="records"),
        "mcnemar": mcnemar.to_dict(orient="records"),
    }
    (OUTPUT_ROOT / "phase_8_statistical_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ensure_output_dir()
    predictions = pd.read_csv(PREDICTIONS_FILE)

    metric_ci, bootstrap_samples = bootstrap_metrics(predictions)
    pairwise = bootstrap_pairwise_differences(predictions)
    mcnemar = mcnemar_rows(predictions)

    metric_ci.to_csv(OUTPUT_ROOT / "bootstrap_metric_ci.csv", index=False, encoding="utf-8-sig")
    bootstrap_samples.to_csv(
        OUTPUT_ROOT / "bootstrap_metric_samples_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pairwise.to_csv(
        OUTPUT_ROOT / "bootstrap_pairwise_metric_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mcnemar.to_csv(OUTPUT_ROOT / "mcnemar_paired_tests.csv", index=False, encoding="utf-8-sig")
    write_summary_json(metric_ci, pairwise, mcnemar)

    print("Phase 8 statistical validation complete")
    print(metric_ci.to_string(index=False))
    print("\nMcNemar paired tests:")
    print(mcnemar.to_string(index=False))


if __name__ == "__main__":
    main()
