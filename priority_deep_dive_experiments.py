import json
import os
from pathlib import Path
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from phase_4_cross_validation import prepare_fold
from preprocess_churn import (
    INPUT_FILE,
    RANDOM_STATE,
    add_engineered_features,
    impute_train_test,
    load_base_dataframe,
    native_cat_features,
)


OUTPUT_ROOT = Path("processed")
EXPERIMENT_ROOT = OUTPUT_ROOT / "priority_deep_dive"
VARIANT_DIRS = {
    "with_billing_zip": OUTPUT_ROOT / "model_a_with_billing_zip",
    "without_billing_zip": OUTPUT_ROOT / "model_b_without_billing_zip",
}

CV_FOLDS = 5
TOPK_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.75)
REGION_TOPK_FRACTIONS = (0.30, 0.40, 0.50)
THRESHOLD_GRID = tuple(float(x) for x in np.round(np.arange(0.01, 0.991, 0.01), 2))
TP_BENEFIT = 3240.0
FP_COST = 120.0
RETENTION_RATE = 0.60


def ensure_output_dir() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)


def champion_specs() -> list[dict[str, Any]]:
    return [
        {
            "objective": "F1_MCC",
            "case_label": "F1_BalancedBagging_tuned",
            "variant": "with_billing_zip",
            "model": "BalancedBagging_tree_depthnone_leaf25",
            "train_kind": "original",
            "preprocess": "encoded",
            "threshold": 0.51,
        },
        {
            "objective": "Recall",
            "case_label": "Recall_HistGradientBoosting",
            "variant": "with_billing_zip",
            "model": "HistGradientBoosting_balanced_lr0.03",
            "train_kind": "original",
            "preprocess": "encoded",
            "threshold": 0.19,
        },
        {
            "objective": "Recall_XGBoost",
            "case_label": "Recall_XGBoost_weighted_no_zip",
            "variant": "without_billing_zip",
            "model": "XGBoost_weighted_depth3_lr0.06_n200_spw14.5",
            "train_kind": "original",
            "preprocess": "encoded",
            "threshold": 0.30,
        },
        {
            "objective": "Recall_operating",
            "case_label": "Recall_CatBoost_balanced",
            "variant": "with_billing_zip",
            "model": "CatBoost_original_balanced",
            "train_kind": "original",
            "preprocess": "encoded",
            "threshold": 0.34,
        },
        {
            "objective": "Cost",
            "case_label": "Cost_CatBoost_native",
            "variant": "with_billing_zip",
            "model": "CatBoost_native_categorical",
            "train_kind": "native_categorical",
            "preprocess": "native",
            "threshold": 0.35,
        },
        {
            "objective": "F2",
            "case_label": "F2_BalancedBagging_original",
            "variant": "with_billing_zip",
            "model": "BalancedBagging_original",
            "train_kind": "original",
            "preprocess": "encoded",
            "threshold": 0.50,
        },
        {
            "objective": "Precision",
            "case_label": "Precision_LR_SMOTE_C0.1",
            "variant": "with_billing_zip",
            "model": "LogisticRegression_SMOTE_C0.1",
            "train_kind": "resampled",
            "preprocess": "encoded",
            "threshold": 0.51,
        },
        {
            "objective": "Small_campaign",
            "case_label": "SmallCampaign_LR_SMOTE",
            "variant": "with_billing_zip",
            "model": "LogisticRegression_SMOTE",
            "train_kind": "resampled",
            "preprocess": "encoded",
            "threshold": 0.46,
        },
    ]


def build_model(model_name: str) -> Any:
    if model_name == "BalancedBagging_tree_depthnone_leaf25":
        return BalancedBaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=None,
                min_samples_leaf=25,
                random_state=RANDOM_STATE,
            ),
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
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
    if model_name == "HistGradientBoosting_balanced_lr0.03":
        return HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=300,
            max_leaf_nodes=15,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if model_name == "XGBoost_weighted_depth3_lr0.06_n200_spw14.5":
        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.06,
            scale_pos_weight=14.5,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=5.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    if model_name == "CatBoost_original_balanced":
        return CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=4,
            loss_function="Logloss",
            eval_metric="F1",
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
    if model_name == "CatBoost_native_categorical":
        return CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=4,
            loss_function="Logloss",
            eval_metric="F1",
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
    if model_name == "LogisticRegression_SMOTE_C0.1":
        return LogisticRegression(max_iter=5000, C=0.1, random_state=RANDOM_STATE)
    if model_name == "LogisticRegression_SMOTE":
        return LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    raise ValueError(f"Unsupported model: {model_name}")


def prepare_fold_native(
    X_train_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    use_billing_zip: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train_raw.copy()
    X_valid = X_valid_raw.copy()
    if not use_billing_zip:
        X_train = X_train.drop(columns=["Billing_ZIP"])
        X_valid = X_valid.drop(columns=["Billing_ZIP"])
    X_train, X_valid, _ = impute_train_test(X_train, X_valid, use_billing_zip)
    if use_billing_zip:
        X_train["Billing_ZIP"] = X_train["Billing_ZIP"].astype(str)
        X_valid["Billing_ZIP"] = X_valid["Billing_ZIP"].astype(str)
    X_train = add_engineered_features(X_train)
    X_valid = add_engineered_features(X_valid)
    for frame in [X_train, X_valid]:
        for col in native_cat_features(frame):
            frame[col] = frame[col].fillna("Unknown").astype(str)
    return X_train, X_valid


def positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def metric_row(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = float("nan")
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": roc_auc,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "contact_rate": float(np.mean(pred)),
        "net_value": float(tp * TP_BENEFIT - fp * FP_COST),
    }


def fit_model_for_spec(
    spec: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    model = build_model(spec["model"])
    if spec["train_kind"] == "resampled":
        smote = SVMSMOTE(random_state=RANDOM_STATE)
        X_fit, y_fit = smote.fit_resample(X_train, y_train)
        X_fit = pd.DataFrame(X_fit, columns=X_train.columns)
        y_fit = pd.Series(y_fit, name="CHURN").astype(int)
    else:
        X_fit, y_fit = X_train, y_train
    if spec["train_kind"] == "native_categorical":
        model.fit(X_fit, y_fit, cat_features=native_cat_features(X_fit))
    else:
        model.fit(X_fit, y_fit)
    return model


def summarize_cv(fold_table: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "f1",
        "f2",
        "recall",
        "precision",
        "pr_auc",
        "mcc",
        "contact_rate",
        "net_value",
    ]
    rows = []
    for (objective, case_label, variant, model), group in fold_table.groupby(
        ["objective", "case_label", "variant", "model"]
    ):
        row = {
            "objective": objective,
            "case_label": case_label,
            "variant": variant,
            "model": model,
            "folds": int(len(group)),
        }
        for metric in metrics:
            mean = float(group[metric].mean())
            sd = float(group[metric].std(ddof=1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["f1_mean", "recall_mean"], ascending=False)


def run_champion_cv() -> tuple[pd.DataFrame, pd.DataFrame]:
    df, _ = load_base_dataframe(INPUT_FILE)
    X = df.drop(columns=["CHURN"])
    y = df["CHURN"]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    specs = champion_specs()
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train_raw = X.iloc[train_idx].reset_index(drop=True)
        X_valid_raw = X.iloc[valid_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)
        encoded_cache: dict[bool, tuple[pd.DataFrame, pd.DataFrame]] = {}
        native_cache: dict[bool, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for spec in specs:
            use_zip = spec["variant"] == "with_billing_zip"
            if spec["preprocess"] == "native":
                if use_zip not in native_cache:
                    native_cache[use_zip] = prepare_fold_native(
                        X_train_raw, X_valid_raw, use_zip
                    )
                X_train_fold, X_valid_fold = native_cache[use_zip]
            else:
                if use_zip not in encoded_cache:
                    encoded_cache[use_zip] = prepare_fold(
                        X_train_raw, X_valid_raw, use_zip
                    )
                X_train_fold, X_valid_fold = encoded_cache[use_zip]
            fitted = fit_model_for_spec(spec, X_train_fold, y_train)
            scores = positive_scores(fitted, X_valid_fold)
            row = {
                "fold": fold_idx,
                "objective": spec["objective"],
                "case_label": spec["case_label"],
                "variant": spec["variant"],
                "model": spec["model"],
                "train_kind": spec["train_kind"],
                "preprocess": spec["preprocess"],
                "valid_rows": int(len(y_valid)),
                "valid_positives": int(y_valid.sum()),
            }
            row.update(metric_row(y_valid, scores, spec["threshold"]))
            rows.append(row)
            print(
                f"CV fold {fold_idx} {spec['case_label']} "
                f"f1={row['f1']:.4f} recall={row['recall']:.4f}",
                flush=True,
            )
    fold_table = pd.DataFrame(rows)
    summary_table = summarize_cv(fold_table)
    return fold_table, summary_table


def load_variant_data(variant: str) -> dict[str, pd.DataFrame | pd.Series]:
    root = VARIANT_DIRS[variant]
    data: dict[str, pd.DataFrame | pd.Series] = {
        "X_train": pd.read_csv(root / "X_train.csv"),
        "X_test": pd.read_csv(root / "X_test.csv"),
        "y_train": pd.read_csv(root / "y_train.csv")["CHURN"].astype(int),
        "y_test": pd.read_csv(root / "y_test.csv")["CHURN"].astype(int),
        "X_train_native": pd.read_csv(root / "X_train_analysis.csv"),
        "X_test_native": pd.read_csv(root / "X_test_analysis.csv"),
    }
    for key in ["X_train_native", "X_test_native"]:
        frame = data[key]
        assert isinstance(frame, pd.DataFrame)
        for col in native_cat_features(frame):
            frame[col] = frame[col].fillna("Unknown").astype(str)
    return data


def format_zip(value: Any, missing: bool = False) -> str:
    if missing or pd.isna(value):
        return "missing"
    numeric = float(value)
    return str(int(numeric)).zfill(4) if numeric.is_integer() else str(value)


def add_zip_prefixes(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy().reset_index(drop=True)
    missing = (
        enriched["Billing_ZIP_missing"].fillna(0).astype(int).eq(1)
        if "Billing_ZIP_missing" in enriched.columns
        else pd.Series(False, index=enriched.index)
    )
    zips = [
        format_zip(value, bool(flag))
        for value, flag in zip(enriched["Billing_ZIP"], missing)
    ]
    enriched["Billing_ZIP_value"] = zips
    enriched["Billing_ZIP_prefix2"] = [
        "missing" if value == "missing" else f"{value[:2]}xx" for value in zips
    ]
    return enriched


def holdout_scores_for_champions() -> pd.DataFrame:
    rows = []
    specs = champion_specs()
    data_cache: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}
    zip_frame = add_zip_prefixes(
        pd.read_csv(VARIANT_DIRS["with_billing_zip"] / "X_test_analysis.csv")
    )
    y_reference = pd.read_csv(VARIANT_DIRS["with_billing_zip"] / "y_test.csv")[
        "CHURN"
    ].astype(int)
    for spec in specs:
        variant = spec["variant"]
        if variant not in data_cache:
            data_cache[variant] = load_variant_data(variant)
        data = data_cache[variant]
        y_test = data["y_test"]
        assert isinstance(y_test, pd.Series)
        if not y_test.reset_index(drop=True).equals(y_reference.reset_index(drop=True)):
            raise ValueError(f"Unexpected y_test order for {variant}")

        if spec["preprocess"] == "native":
            X_train = data["X_train_native"]
            X_test = data["X_test_native"]
        else:
            X_train = data["X_train"]
            X_test = data["X_test"]
        y_train = data["y_train"]
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        fitted = fit_model_for_spec(spec, X_train, y_train)
        scores = positive_scores(fitted, X_test)
        metrics = metric_row(y_test, scores, spec["threshold"])
        for idx, score in enumerate(scores):
            rows.append(
                {
                    "row_id": idx,
                    "actual": int(y_test.iloc[idx]),
                    "case_label": spec["case_label"],
                    "objective": spec["objective"],
                    "variant": variant,
                    "model": spec["model"],
                    "score": float(score),
                    "threshold": spec["threshold"],
                    "predicted": int(score >= spec["threshold"]),
                    "Billing_ZIP_value": zip_frame.loc[idx, "Billing_ZIP_value"],
                    "Billing_ZIP_prefix2": zip_frame.loc[idx, "Billing_ZIP_prefix2"],
                    "CRM_PID_Value_Segment": zip_frame.loc[
                        idx, "CRM_PID_Value_Segment"
                    ],
                    "TotalRevenue": float(zip_frame.loc[idx, "TotalRevenue"]),
                    "ARPU": float(zip_frame.loc[idx, "ARPU"]),
                    "model_test_f1": metrics["f1"],
                    "model_test_recall": metrics["recall"],
                    "model_test_precision": metrics["precision"],
                    "model_test_net_value": metrics["net_value"],
                }
            )
    return pd.DataFrame(rows)


def topk_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_label, group in predictions.groupby("case_label"):
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        total_churn = int(ordered["actual"].sum())
        for fraction in TOPK_FRACTIONS:
            k = max(1, int(round(len(ordered) * fraction)))
            selected = ordered.head(k)
            tp = int(selected["actual"].sum())
            fp = int(k - tp)
            captured_revenue = float(
                selected.loc[selected["actual"].eq(1), "TotalRevenue"].sum()
            )
            rows.append(
                {
                    "case_label": case_label,
                    "top_fraction": fraction,
                    "selected_count": k,
                    "tp": tp,
                    "fp": fp,
                    "fn_remaining": int(total_churn - tp),
                    "precision_at_k": tp / k,
                    "recall_at_k": tp / total_churn if total_churn else 0.0,
                    "net_value": float(tp * TP_BENEFIT - fp * FP_COST),
                    "captured_total_revenue": captured_revenue,
                    "individualized_net_proxy": float(
                        captured_revenue * RETENTION_RATE - fp * FP_COST
                    ),
                }
            )
    return pd.DataFrame(rows)


def region_topk_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case_label, prefix), group in predictions.groupby(
        ["case_label", "Billing_ZIP_prefix2"]
    ):
        rows_total = int(len(group))
        positives = int(group["actual"].sum())
        if rows_total < 20 or positives < 2:
            continue
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        for fraction in REGION_TOPK_FRACTIONS:
            k = max(1, int(round(rows_total * fraction)))
            selected = ordered.head(k)
            tp = int(selected["actual"].sum())
            fp = int(k - tp)
            rows.append(
                {
                    "case_label": case_label,
                    "Billing_ZIP_prefix2": prefix,
                    "group_rows": rows_total,
                    "group_positives": positives,
                    "group_churn_rate": positives / rows_total,
                    "top_fraction": fraction,
                    "selected_count": k,
                    "tp": tp,
                    "fp": fp,
                    "recall_at_k": tp / positives,
                    "precision_at_k": tp / k,
                    "net_value": float(tp * TP_BENEFIT - fp * FP_COST),
                }
            )
    return pd.DataFrame(rows)


def region_threshold_oracle(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case_label, prefix), group in predictions.groupby(
        ["case_label", "Billing_ZIP_prefix2"]
    ):
        rows_total = int(len(group))
        positives = int(group["actual"].sum())
        if rows_total < 20 or positives < 2:
            continue
        best = None
        y = group["actual"].astype(int)
        scores = group["score"].to_numpy()
        for threshold in THRESHOLD_GRID:
            metrics = metric_row(y, scores, threshold)
            candidate = {
                "case_label": case_label,
                "Billing_ZIP_prefix2": prefix,
                "group_rows": rows_total,
                "group_positives": positives,
                "group_churn_rate": positives / rows_total,
                **metrics,
            }
            if best is None or (
                candidate["net_value"],
                candidate["recall"],
                candidate["precision"],
            ) > (best["net_value"], best["recall"], best["precision"]):
                best = candidate
        rows.append(best)
    return pd.DataFrame(rows)


def to_markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                if "value" in col or "revenue" in col:
                    values.append(f"{value:,.0f}")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    cv_summary: pd.DataFrame,
    global_topk: pd.DataFrame,
    region_topk: pd.DataFrame,
    region_oracle: pd.DataFrame,
) -> None:
    cv_view = cv_summary[
        [
            "case_label",
            "variant",
            "model",
            "f1_mean",
            "f1_sd",
            "recall_mean",
            "recall_sd",
            "precision_mean",
            "contact_rate_mean",
            "net_value_mean",
        ]
    ].copy()
    topk_best = (
        global_topk.sort_values(
            ["top_fraction", "net_value", "recall_at_k"],
            ascending=[True, False, False],
        )
        .groupby("top_fraction")
        .head(1)
    )
    region_best = (
        region_topk[region_topk["top_fraction"].eq(0.30)]
        .sort_values(["net_value", "recall_at_k"], ascending=[False, False])
        .head(12)
    )
    oracle_best = region_oracle.sort_values(
        ["net_value", "recall"], ascending=[False, False]
    ).head(12)
    lines = [
        "# Priority Deep Dive Experiments",
        "",
        "목적: 보고서에서 집중할 3대 실험을 더 깊게 검증했다. 1) 목적별 챔피언 5-fold CV, 2) ZIP 지역별 top-k 캠페인, 3) contact-rate/비용/ARPU proxy 운영 분석.",
        "",
        "## 1. 목적별 챔피언 5-fold CV",
        "",
        to_markdown(
            cv_view,
            [
                "case_label",
                "variant",
                "model",
                "f1_mean",
                "f1_sd",
                "recall_mean",
                "recall_sd",
                "precision_mean",
                "contact_rate_mean",
                "net_value_mean",
            ],
        ),
        "",
        "## 2. 전역 Top-k/Contact Rate별 최고 운영점",
        "",
        to_markdown(
            topk_best,
            [
                "top_fraction",
                "case_label",
                "selected_count",
                "tp",
                "fp",
                "precision_at_k",
                "recall_at_k",
                "net_value",
                "individualized_net_proxy",
            ],
        ),
        "",
        "## 3. ZIP 앞 2자리 지역별 Top 30% 캠페인 우수 조합",
        "",
        to_markdown(
            region_best,
            [
                "Billing_ZIP_prefix2",
                "case_label",
                "group_rows",
                "group_positives",
                "group_churn_rate",
                "tp",
                "fp",
                "precision_at_k",
                "recall_at_k",
                "net_value",
            ],
        ),
        "",
        "## 4. ZIP 앞 2자리 지역별 threshold oracle 진단",
        "",
        "이 표는 test label을 사용해 지역별 최적 threshold를 찾은 진단용 결과다. 운영 적용 전에는 별도 validation으로 다시 선택해야 한다.",
        "",
        to_markdown(
            oracle_best,
            [
                "Billing_ZIP_prefix2",
                "case_label",
                "threshold",
                "group_rows",
                "group_positives",
                "f1",
                "recall",
                "precision",
                "tp",
                "fp",
                "net_value",
            ],
        ),
        "",
    ]
    (EXPERIMENT_ROOT / "PRIORITY_DEEP_DIVE_EXPERIMENTS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    ensure_output_dir()
    cv_fold, cv_summary = run_champion_cv()
    predictions = holdout_scores_for_champions()
    global_topk = topk_rows(predictions)
    region_topk = region_topk_rows(predictions)
    region_oracle = region_threshold_oracle(predictions)

    cv_fold.to_csv(
        EXPERIMENT_ROOT / "priority_champion_cv_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_summary.to_csv(
        EXPERIMENT_ROOT / "priority_champion_cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        EXPERIMENT_ROOT / "priority_champion_holdout_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    global_topk.to_csv(
        EXPERIMENT_ROOT / "priority_global_topk_budget.csv",
        index=False,
        encoding="utf-8-sig",
    )
    region_topk.to_csv(
        EXPERIMENT_ROOT / "priority_zip_prefix2_topk.csv",
        index=False,
        encoding="utf-8-sig",
    )
    region_oracle.to_csv(
        EXPERIMENT_ROOT / "priority_zip_prefix2_threshold_oracle.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(cv_summary, global_topk, region_topk, region_oracle)
    summary = {
        "cv_rows": int(len(cv_fold)),
        "holdout_prediction_rows": int(len(predictions)),
        "global_topk_rows": int(len(global_topk)),
        "region_topk_rows": int(len(region_topk)),
        "region_threshold_oracle_rows": int(len(region_oracle)),
        "outputs": [
            "priority_champion_cv_summary.csv",
            "priority_global_topk_budget.csv",
            "priority_zip_prefix2_topk.csv",
            "priority_zip_prefix2_threshold_oracle.csv",
            "PRIORITY_DEEP_DIVE_EXPERIMENTS.md",
        ],
    }
    (EXPERIMENT_ROOT / "priority_deep_dive_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Priority deep dive complete")
    print(cv_summary.to_string(index=False))


if __name__ == "__main__":
    main()
