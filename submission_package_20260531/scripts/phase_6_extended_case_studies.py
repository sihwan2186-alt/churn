import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SVMSMOTE
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from preprocess_churn import (
    RANDOM_STATE,
    build_comparison_models,
    fit_model,
    native_cat_features,
    positive_scores,
)


OUTPUT_ROOT = Path("processed")
PHASE_OUTPUT = OUTPUT_ROOT / "phase_6_extended_case_studies"

VARIANT_DIRS = {
    "with_billing_zip": OUTPUT_ROOT / "model_a_with_billing_zip",
    "without_billing_zip": OUTPUT_ROOT / "model_b_without_billing_zip",
}

PAPER_ANNUAL_ARPU = 5400.0
RETENTION_RATE = 0.60
TP_BENEFIT = PAPER_ANNUAL_ARPU * RETENTION_RATE
DEFAULT_FP_COST = 120.0
PAPER_REPORTED_NET = 74200.0

THRESHOLD_GRID = np.round(np.arange(0.01, 0.991, 0.01), 2)
TOPK_PCTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]

HIGH_VALUE_SEGMENTS = {"Platinum", "Gold"}
MID_VALUE_SEGMENTS = {"SME", "Silver", "SE"}
LOW_VALUE_SEGMENTS = {"Bronze", "Iron", "Lead", "Unknown"}

COST_SCENARIOS = [
    {
        "scenario": "optimistic_low_campaign_cost",
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
        "scenario": "conservative_campaign_cost",
        "fn_cost": 5400.0,
        "fp_cost": 360.0,
        "retention_rate": 0.60,
    },
    {
        "scenario": "budget_limited_high_campaign_cost",
        "fn_cost": 5400.0,
        "fp_cost": 720.0,
        "retention_rate": 0.60,
    },
    {
        "scenario": "small_business_value",
        "fn_cost": 1200.0,
        "fp_cost": 120.0,
        "retention_rate": 0.60,
    },
    {
        "scenario": "enterprise_value",
        "fn_cost": 18000.0,
        "fp_cost": 120.0,
        "retention_rate": 0.60,
    },
]

MODEL_SPECS = [
    {
        "case_label": "LR_no_zip_f1",
        "variant": "without_billing_zip",
        "model": "LogisticRegression_SMOTE",
        "train_kind": "resampled",
        "operating_threshold": 0.50,
    },
    {
        "case_label": "LR_with_zip_recall_constraint",
        "variant": "with_billing_zip",
        "model": "LogisticRegression_SMOTE",
        "train_kind": "resampled",
        "operating_threshold": 0.46,
    },
    {
        "case_label": "BalancedBagging_with_zip",
        "variant": "with_billing_zip",
        "model": "BalancedBagging_original",
        "train_kind": "original",
        "operating_threshold": 0.50,
    },
    {
        "case_label": "BalancedBagging_no_zip",
        "variant": "without_billing_zip",
        "model": "BalancedBagging_original",
        "train_kind": "original",
        "operating_threshold": 0.44,
    },
    {
        "case_label": "EasyEnsemble_with_zip",
        "variant": "with_billing_zip",
        "model": "EasyEnsemble_original",
        "train_kind": "original",
        "operating_threshold": 0.50,
    },
    {
        "case_label": "CatBoost_native_with_zip",
        "variant": "with_billing_zip",
        "model": "CatBoost_native_categorical",
        "train_kind": "native_categorical",
        "operating_threshold": 0.35,
    },
    {
        "case_label": "CatBoost_balanced_with_zip",
        "variant": "with_billing_zip",
        "model": "CatBoost_original_balanced",
        "train_kind": "original",
        "operating_threshold": 0.34,
    },
    {
        "case_label": "XGBoost_with_zip",
        "variant": "with_billing_zip",
        "model": "XGBoost_SMOTE",
        "train_kind": "resampled",
        "operating_threshold": 0.16,
    },
]


def ensure_output_dir() -> None:
    PHASE_OUTPUT.mkdir(parents=True, exist_ok=True)


def load_variant(variant: str) -> dict[str, pd.DataFrame | pd.Series]:
    root = VARIANT_DIRS[variant]
    data: dict[str, pd.DataFrame | pd.Series] = {
        "X_train": pd.read_csv(root / "X_train.csv"),
        "X_test": pd.read_csv(root / "X_test.csv"),
        "X_train_resampled": pd.read_csv(root / "X_train_resampled.csv"),
        "y_train": pd.read_csv(root / "y_train.csv")["CHURN"].astype(int),
        "y_test": pd.read_csv(root / "y_test.csv")["CHURN"].astype(int),
        "y_train_resampled": pd.read_csv(root / "y_train_resampled.csv")[
            "CHURN"
        ].astype(int),
        "X_train_native": pd.read_csv(root / "X_train_analysis.csv"),
        "X_test_native": pd.read_csv(root / "X_test_analysis.csv"),
    }
    for native_key in ["X_train_native", "X_test_native"]:
        frame = data[native_key]
        assert isinstance(frame, pd.DataFrame)
        for col in native_cat_features(frame):
            frame[col] = frame[col].fillna("Unknown").astype(str)
    return data


def model_factory(model_name: str, train_kind: str) -> Any:
    for name, model, kind in build_comparison_models():
        if name == model_name and kind == train_kind:
            return clone(model)
    raise ValueError(f"Unknown model spec: {model_name}, {train_kind}")


def training_split(data: dict[str, pd.DataFrame | pd.Series], train_kind: str):
    if train_kind == "resampled":
        return data["X_train_resampled"], data["y_train_resampled"]
    if train_kind == "native_categorical":
        return data["X_train_native"], data["y_train"]
    return data["X_train"], data["y_train"]


def test_split(data: dict[str, pd.DataFrame | pd.Series], train_kind: str):
    if train_kind == "native_categorical":
        return data["X_test_native"], data["y_test"]
    return data["X_test"], data["y_test"]


def classification_row(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    prefix: str = "",
) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    net = tp * TP_BENEFIT - fp * DEFAULT_FP_COST
    return {
        prefix + "threshold": threshold,
        prefix + "f1": float(f1_score(y_true, pred, zero_division=0)),
        prefix + "recall": float(recall_score(y_true, pred, zero_division=0)),
        prefix + "precision": float(precision_score(y_true, pred, zero_division=0)),
        prefix + "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        prefix + "roc_auc": safe_roc_auc(y_true, scores),
        prefix + "pr_auc": safe_pr_auc(y_true, scores),
        prefix + "mcc": float(matthews_corrcoef(y_true, pred)),
        prefix + "tp": int(tp),
        prefix + "fp": int(fp),
        prefix + "fn": int(fn),
        prefix + "tn": int(tn),
        prefix + "contacts": int(tp + fp),
        prefix + "net_benefit": float(net),
        prefix + "paper_multiple": float(net / PAPER_REPORTED_NET),
    }


def safe_roc_auc(y_true: pd.Series, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_pr_auc(y_true: pd.Series, scores: np.ndarray) -> float:
    if int(np.sum(y_true)) == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def fit_selected_models() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows = []
    prediction_frame = None
    y_test_reference = None
    segment_reference = None

    loaded_variants = {variant: load_variant(variant) for variant in VARIANT_DIRS}

    for spec in MODEL_SPECS:
        data = loaded_variants[spec["variant"]]
        X_train, y_train = training_split(data, spec["train_kind"])
        X_test, y_test = test_split(data, spec["train_kind"])
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

        model = model_factory(spec["model"], spec["train_kind"])
        fit_model(model, X_train, y_train, spec["train_kind"])
        scores = positive_scores(model, X_test)
        metrics = classification_row(y_test, scores, spec["operating_threshold"])
        all_rows.append({**spec, **metrics})

        if prediction_frame is None:
            y_test_reference = y_test.reset_index(drop=True)
            native = loaded_variants["with_billing_zip"]["X_test_native"]
            assert isinstance(native, pd.DataFrame)
            segment_reference = native["CRM_PID_Value_Segment"].reset_index(drop=True)
            prediction_frame = pd.DataFrame(
                {
                    "row_id": np.arange(len(y_test_reference)),
                    "actual": y_test_reference,
                    "CRM_PID_Value_Segment": segment_reference,
                    "CRM_segment_bucket": segment_reference.map(bucket_segment),
                }
            )

        assert prediction_frame is not None
        prediction_frame[f"{spec['case_label']}_score"] = scores
        prediction_frame[f"{spec['case_label']}_pred"] = (
            scores >= spec["operating_threshold"]
        ).astype(int)

    assert prediction_frame is not None
    operating_metrics = pd.DataFrame(all_rows)
    operating_metrics.to_csv(
        PHASE_OUTPUT / "phase6_model_operating_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_frame.to_csv(
        PHASE_OUTPUT / "phase6_model_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return operating_metrics, prediction_frame, pd.DataFrame(MODEL_SPECS)


def bucket_segment(segment: str) -> str:
    if segment in HIGH_VALUE_SEGMENTS:
        return "high_value"
    if segment in MID_VALUE_SEGMENTS:
        return "mid_value"
    if segment in LOW_VALUE_SEGMENTS:
        return "low_value"
    return "other"


def net_value(tp: int, fp: int, fn_cost: float, fp_cost: float, retention_rate: float):
    return tp * (fn_cost * retention_rate) - fp * fp_cost


def run_topk_budget(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_true = predictions["actual"].to_numpy()
    n = len(predictions)
    positives = int(np.sum(y_true))
    for spec in MODEL_SPECS:
        scores = predictions[f"{spec['case_label']}_score"].to_numpy()
        order = np.argsort(scores)[::-1]
        for pct in TOPK_PCTS:
            k = max(1, int(round(n * pct)))
            selected = np.zeros(n, dtype=int)
            selected[order[:k]] = 1
            tp = int(np.sum((selected == 1) & (y_true == 1)))
            fp = int(np.sum((selected == 1) & (y_true == 0)))
            fn = int(np.sum((selected == 0) & (y_true == 1)))
            tn = int(np.sum((selected == 0) & (y_true == 0)))
            rows.append(
                {
                    "case_label": spec["case_label"],
                    "variant": spec["variant"],
                    "model": spec["model"],
                    "topk_pct": pct,
                    "contacts": k,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "recall_at_k": tp / positives if positives else np.nan,
                    "precision_at_k": tp / k if k else np.nan,
                    "lift_vs_random": (tp / k) / (positives / n),
                    "net_benefit": tp * TP_BENEFIT - fp * DEFAULT_FP_COST,
                    "paper_multiple": (tp * TP_BENEFIT - fp * DEFAULT_FP_COST)
                    / PAPER_REPORTED_NET,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(
        PHASE_OUTPUT / "phase6_topk_budget_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table


def run_threshold_cost_sweep(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    y_true = predictions["actual"].to_numpy()
    for spec in MODEL_SPECS:
        scores = predictions[f"{spec['case_label']}_score"].to_numpy()
        for scenario in COST_SCENARIOS:
            for threshold in THRESHOLD_GRID:
                pred = (scores >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
                expected = net_value(
                    int(tp),
                    int(fp),
                    scenario["fn_cost"],
                    scenario["fp_cost"],
                    scenario["retention_rate"],
                )
                rows.append(
                    {
                        "case_label": spec["case_label"],
                        "variant": spec["variant"],
                        "model": spec["model"],
                        "scenario": scenario["scenario"],
                        "fn_cost": scenario["fn_cost"],
                        "fp_cost": scenario["fp_cost"],
                        "retention_rate": scenario["retention_rate"],
                        "threshold": threshold,
                        "expected_net_value": expected,
                        "tp": int(tp),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tn": int(tn),
                        "recall": float(recall_score(y_true, pred, zero_division=0)),
                        "precision": float(
                            precision_score(y_true, pred, zero_division=0)
                        ),
                        "f1": float(f1_score(y_true, pred, zero_division=0)),
                    }
                )
    sweep = pd.DataFrame(rows)
    best = (
        sweep.sort_values("expected_net_value", ascending=False)
        .groupby(["case_label", "variant", "model", "scenario"], as_index=False)
        .first()
        .sort_values(["scenario", "expected_net_value"], ascending=[True, False])
    )
    sweep.to_csv(
        PHASE_OUTPUT / "phase6_cost_threshold_sweep.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best.to_csv(
        PHASE_OUTPUT / "phase6_cost_threshold_best_by_model.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return sweep, best


def expected_calibration_error(
    y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10
) -> tuple[float, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(scores, bins[1:-1], right=True)
    rows = []
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            rows.append(
                {
                    "bin": bin_id,
                    "bin_low": bins[bin_id],
                    "bin_high": bins[bin_id + 1],
                    "count": 0,
                    "mean_score": np.nan,
                    "observed_rate": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        mean_score = float(np.mean(scores[mask]))
        observed = float(np.mean(y_true[mask]))
        gap = abs(mean_score - observed)
        ece += (np.sum(mask) / len(y_true)) * gap
        rows.append(
            {
                "bin": bin_id,
                "bin_low": bins[bin_id],
                "bin_high": bins[bin_id + 1],
                "count": int(np.sum(mask)),
                "mean_score": mean_score,
                "observed_rate": observed,
                "abs_gap": gap,
            }
        )
    return float(ece), pd.DataFrame(rows)


def calibrate_scores(
    spec: dict, loaded_variants: dict[str, dict[str, pd.DataFrame | pd.Series]]
) -> dict:
    data = loaded_variants[spec["variant"]]

    if spec["train_kind"] == "native_categorical":
        X_full = data["X_train_native"]
        X_test = data["X_test_native"]
    else:
        X_full = data["X_train"]
        X_test = data["X_test"]
    y_full = data["y_train"]
    y_test = data["y_test"]
    assert isinstance(X_full, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_full, pd.Series)
    assert isinstance(y_test, pd.Series)

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_full,
        y_full,
        test_size=0.25,
        stratify=y_full,
        random_state=RANDOM_STATE,
    )

    if spec["train_kind"] == "resampled":
        smote = SVMSMOTE(random_state=RANDOM_STATE)
        X_fit_model, y_fit_model = smote.fit_resample(X_fit, y_fit)
    else:
        X_fit_model, y_fit_model = X_fit, y_fit

    base = model_factory(spec["model"], spec["train_kind"])
    fit_model(base, X_fit_model, y_fit_model, spec["train_kind"])
    cal_scores = positive_scores(base, X_cal)
    test_scores = positive_scores(base, X_test)

    y_cal_np = y_cal.to_numpy()
    y_test_np = y_test.to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_scores, y_cal_np)
    iso_test = np.clip(iso.transform(test_scores), 0.0, 1.0)

    platt = SkLogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    platt.fit(cal_scores.reshape(-1, 1), y_cal_np)
    platt_test = platt.predict_proba(test_scores.reshape(-1, 1))[:, 1]

    rows = []
    bin_rows = []
    for method, scores in [
        ("raw", np.clip(test_scores, 0.0, 1.0)),
        ("isotonic", iso_test),
        ("platt", platt_test),
    ]:
        ece, bins = expected_calibration_error(y_test_np, scores)
        bins["case_label"] = spec["case_label"]
        bins["model"] = spec["model"]
        bins["calibration_method"] = method
        bin_rows.append(bins)
        rows.append(
            {
                "case_label": spec["case_label"],
                "variant": spec["variant"],
                "model": spec["model"],
                "calibration_method": method,
                "brier_score": float(brier_score_loss(y_test_np, scores)),
                "ece_10bin": ece,
                "log_loss": float(log_loss(y_test_np, np.clip(scores, 1e-6, 1 - 1e-6))),
                "roc_auc": safe_roc_auc(y_test, scores),
                "pr_auc": safe_pr_auc(y_test, scores),
                "mean_score": float(np.mean(scores)),
                "observed_churn_rate": float(np.mean(y_test_np)),
            }
        )
    return {
        "metrics": rows,
        "bins": pd.concat(bin_rows, ignore_index=True),
    }


def run_calibration_case_study() -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_specs = [
        spec
        for spec in MODEL_SPECS
        if spec["case_label"]
        in {
            "LR_no_zip_f1",
            "BalancedBagging_with_zip",
            "CatBoost_native_with_zip",
            "XGBoost_with_zip",
        }
    ]
    loaded_variants = {variant: load_variant(variant) for variant in VARIANT_DIRS}
    metric_rows = []
    bin_tables = []
    for spec in calibration_specs:
        result = calibrate_scores(spec, loaded_variants)
        metric_rows.extend(result["metrics"])
        bin_tables.append(result["bins"])
    metrics = pd.DataFrame(metric_rows)
    bins = pd.concat(bin_tables, ignore_index=True)
    metrics.to_csv(
        PHASE_OUTPUT / "phase6_calibration_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bins.to_csv(
        PHASE_OUTPUT / "phase6_calibration_bins.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return metrics, bins


def run_segment_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in MODEL_SPECS:
        pred_col = f"{spec['case_label']}_pred"
        score_col = f"{spec['case_label']}_score"
        for bucket, group in predictions.groupby("CRM_segment_bucket"):
            y_true = group["actual"]
            y_pred = group[pred_col].to_numpy()
            scores = group[score_col].to_numpy()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            rows.append(
                {
                    "case_label": spec["case_label"],
                    "variant": spec["variant"],
                    "model": spec["model"],
                    "CRM_segment_bucket": bucket,
                    "rows": int(len(group)),
                    "positives": int(y_true.sum()),
                    "churn_rate": float(y_true.mean()),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tn": int(tn),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "precision": float(
                        precision_score(y_true, y_pred, zero_division=0)
                    ),
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                    "pr_auc": safe_pr_auc(y_true, scores),
                    "net_benefit": float(tp * TP_BENEFIT - fp * DEFAULT_FP_COST),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(
        PHASE_OUTPUT / "phase6_segment_operating_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table


def run_model_agreement(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_cols = [f"{spec['case_label']}_pred" for spec in MODEL_SPECS]
    frame = predictions.copy()
    frame["model_vote_count"] = frame[pred_cols].sum(axis=1)
    rows = []
    for votes, group in frame.groupby("model_vote_count"):
        rows.append(
            {
                "model_vote_count": int(votes),
                "rows": int(len(group)),
                "churners": int(group["actual"].sum()),
                "churn_rate": float(group["actual"].mean()),
                "share_of_test": float(len(group) / len(frame)),
            }
        )
    agreement = pd.DataFrame(rows).sort_values("model_vote_count")

    overlap_rows = []
    churners = frame[frame["actual"].eq(1)].copy()
    for spec in MODEL_SPECS:
        caught = churners[churners[f"{spec['case_label']}_pred"].eq(1)]
        only_mask = churners[f"{spec['case_label']}_pred"].eq(1)
        for other in MODEL_SPECS:
            if other["case_label"] != spec["case_label"]:
                only_mask &= churners[f"{other['case_label']}_pred"].eq(0)
        overlap_rows.append(
            {
                "case_label": spec["case_label"],
                "model": spec["model"],
                "caught_churners": int(len(caught)),
                "unique_churners_caught_only_by_this_model": int(only_mask.sum()),
            }
        )

    # Pairwise overlap among actual churners.
    for i, left in enumerate(MODEL_SPECS):
        for right in MODEL_SPECS[i + 1 :]:
            overlap = int(
                (
                    churners[f"{left['case_label']}_pred"].eq(1)
                    & churners[f"{right['case_label']}_pred"].eq(1)
                ).sum()
            )
            overlap_rows.append(
                {
                    "case_label": f"{left['case_label']} & {right['case_label']}",
                    "model": "pairwise_overlap",
                    "caught_churners": overlap,
                    "unique_churners_caught_only_by_this_model": np.nan,
                }
            )

    overlap = pd.DataFrame(overlap_rows)
    agreement.to_csv(
        PHASE_OUTPUT / "phase6_model_agreement_vote_groups.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overlap.to_csv(
        PHASE_OUTPUT / "phase6_churn_capture_overlap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return agreement, overlap


def plot_outputs(
    topk: pd.DataFrame,
    cost_best: pd.DataFrame,
    calibration: pd.DataFrame,
    segment: pd.DataFrame,
    agreement: pd.DataFrame,
) -> None:
    plot_topk(topk)
    plot_cost_best(cost_best)
    plot_calibration(calibration)
    plot_segment_heatmap(segment)
    plot_agreement(agreement)


def plot_topk(topk: pd.DataFrame) -> None:
    chosen = topk[topk["topk_pct"].isin([0.05, 0.10, 0.20, 0.30, 0.50])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, group in chosen.groupby("case_label"):
        ordered = group.sort_values("topk_pct")
        axes[0].plot(
            ordered["topk_pct"] * 100,
            ordered["recall_at_k"],
            marker="o",
            linewidth=1.5,
            label=label,
        )
        axes[1].plot(
            ordered["topk_pct"] * 100,
            ordered["net_benefit"],
            marker="o",
            linewidth=1.5,
            label=label,
        )
    axes[0].set_title("Recall by Campaign Budget")
    axes[0].set_xlabel("Top-k campaign share (%)")
    axes[0].set_ylabel("Recall@k")
    axes[1].set_title("Net Benefit by Campaign Budget")
    axes[1].set_xlabel("Top-k campaign share (%)")
    axes[1].set_ylabel("Net benefit")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    plt.tight_layout()
    fig.savefig(PHASE_OUTPUT / "phase6_topk_budget_curves.png", dpi=160)
    plt.close(fig)


def plot_cost_best(cost_best: pd.DataFrame) -> None:
    paper = cost_best[cost_best["scenario"].eq("paper_baseline")].copy()
    paper = paper.sort_values("expected_net_value", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(paper["case_label"], paper["expected_net_value"], color="#2c7fb8")
    ax.axvline(PAPER_REPORTED_NET, color="#777777", linestyle="--", linewidth=1.5)
    ax.set_title("Best Threshold Net Value under Paper Cost Scenario")
    ax.set_xlabel("Expected net value")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(PHASE_OUTPUT / "phase6_cost_best_paper_baseline.png", dpi=160)
    plt.close(fig)


def plot_calibration(calibration: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pivot = calibration.pivot(
        index="case_label", columns="calibration_method", values="brier_score"
    )
    pivot.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Brier Score by Calibration Method")
    axes[0].set_ylabel("Brier score, lower is better")
    axes[0].tick_params(axis="x", labelrotation=30)

    pivot_ece = calibration.pivot(
        index="case_label", columns="calibration_method", values="ece_10bin"
    )
    pivot_ece.plot(kind="bar", ax=axes[1])
    axes[1].set_title("10-bin Expected Calibration Error")
    axes[1].set_ylabel("ECE, lower is better")
    axes[1].tick_params(axis="x", labelrotation=30)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(PHASE_OUTPUT / "phase6_calibration_comparison.png", dpi=160)
    plt.close(fig)


def plot_segment_heatmap(segment: pd.DataFrame) -> None:
    pivot = segment.pivot_table(
        index="case_label",
        columns="CRM_segment_bucket",
        values="recall",
        aggfunc="mean",
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Recall by CRM Segment Bucket and Model")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(PHASE_OUTPUT / "phase6_segment_recall_heatmap.png", dpi=160)
    plt.close(fig)


def plot_agreement(agreement: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(
        agreement["model_vote_count"],
        agreement["rows"],
        color="#bdbdbd",
        label="Rows",
    )
    ax1.set_xlabel("Number of models flagging churn")
    ax1.set_ylabel("Customer count")
    ax2 = ax1.twinx()
    ax2.plot(
        agreement["model_vote_count"],
        agreement["churn_rate"],
        color="#d7301f",
        marker="o",
        linewidth=2,
        label="Observed churn rate",
    )
    ax2.set_ylabel("Observed churn rate")
    ax1.set_title("Model Agreement as Risk Stratification")
    ax1.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(PHASE_OUTPUT / "phase6_model_agreement.png", dpi=160)
    plt.close(fig)


def save_summary(
    operating: pd.DataFrame,
    topk: pd.DataFrame,
    cost_best: pd.DataFrame,
    calibration: pd.DataFrame,
    segment: pd.DataFrame,
    agreement: pd.DataFrame,
    overlap: pd.DataFrame,
) -> None:
    topk_best = (
        topk.sort_values("net_benefit", ascending=False)
        .groupby("topk_pct", as_index=False)
        .first()
        .sort_values("topk_pct")
    )
    paper_best = cost_best[cost_best["scenario"].eq("paper_baseline")].sort_values(
        "expected_net_value", ascending=False
    )
    calibration_best = calibration.sort_values("brier_score").head(10)
    segment_best = segment.sort_values("net_benefit", ascending=False).head(12)

    summary = {
        "operating_metrics_top": operating.sort_values(
            "net_benefit", ascending=False
        ).to_dict(orient="records"),
        "topk_best_by_budget": topk_best.to_dict(orient="records"),
        "paper_cost_best_by_model": paper_best.to_dict(orient="records"),
        "calibration_best_rows": calibration_best.to_dict(orient="records"),
        "segment_best_rows": segment_best.to_dict(orient="records"),
        "agreement": agreement.to_dict(orient="records"),
        "overlap_top_rows": overlap.head(20).to_dict(orient="records"),
    }
    with (PHASE_OUTPUT / "phase6_extended_case_studies_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    ensure_output_dir()
    operating, predictions, _ = fit_selected_models()
    topk = run_topk_budget(predictions)
    _, cost_best = run_threshold_cost_sweep(predictions)
    calibration, _ = run_calibration_case_study()
    segment = run_segment_metrics(predictions)
    agreement, overlap = run_model_agreement(predictions)
    plot_outputs(topk, cost_best, calibration, segment, agreement)
    save_summary(operating, topk, cost_best, calibration, segment, agreement, overlap)

    print("=== Phase 6 Extended Case Studies ===")
    print(f"Outputs: {PHASE_OUTPUT}")
    print("\nOperating metrics:")
    print(
        operating[
            [
                "case_label",
                "model",
                "threshold",
                "f1",
                "recall",
                "precision",
                "tp",
                "fp",
                "fn",
                "net_benefit",
            ]
        ].to_string(index=False)
    )
    print("\nBest paper-baseline cost threshold by model:")
    print(
        cost_best[cost_best["scenario"].eq("paper_baseline")][
            [
                "case_label",
                "model",
                "threshold",
                "expected_net_value",
                "tp",
                "fp",
                "fn",
                "recall",
                "precision",
            ]
        ].to_string(index=False)
    )
    print("\nTop-k best by budget:")
    print(
        topk.sort_values("net_benefit", ascending=False)
        .groupby("topk_pct", as_index=False)
        .first()[
            [
                "topk_pct",
                "case_label",
                "contacts",
                "tp",
                "fp",
                "recall_at_k",
                "precision_at_k",
                "net_benefit",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
