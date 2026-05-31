import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import joblib
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
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


INPUT_FILE = Path("Baza customer Telecom v2.csv")
OUTPUT_ROOT = Path("processed")
COMPARISON_OUTPUT = OUTPUT_ROOT / "model_comparison_billing_zip.csv"
THRESHOLD_SWEEP_OUTPUT = OUTPUT_ROOT / "threshold_tuning_sweep.csv"
THRESHOLD_BEST_OUTPUT = OUTPUT_ROOT / "threshold_tuning_best.csv"
ERROR_ANALYSIS_OUTPUT = OUTPUT_ROOT / "error_analysis_predictions.csv"
ERROR_GROUP_SUMMARY_OUTPUT = OUTPUT_ROOT / "error_analysis_group_summary.csv"
FEATURE_IMPORTANCE_OUTPUT = OUTPUT_ROOT / "feature_importance.csv"
FEATURE_IMPORTANCE_TOP_OUTPUT = OUTPUT_ROOT / "feature_importance_top.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD_VALIDATION_SIZE = 0.25
THRESHOLD_GRID = tuple(float(x) for x in np.round(np.arange(0.05, 0.501, 0.01), 2))
MIN_RECALL_CONSTRAINT = 0.30
PERMUTATION_REPEATS = 3
ENABLE_FN_TARGET_FEATURES_DEFAULT = False
THRESHOLD_TUNING_MODELS = {
    "BalancedBagging_original",
    "EasyEnsemble_original",
    "EasyEnsemble_n50_original",
    "CatBoost_native_categorical",
    "CatBoost_original_balanced",
    "LogisticRegression_SMOTE",
    "XGBoost_SMOTE",
}
ANALYSIS_OPERATING_POINTS = [
    {
        "operating_point": "main_f1_baseline",
        "variant": "without_billing_zip",
        "model": "LogisticRegression_SMOTE",
        "threshold": 0.5,
    },
    {
        "operating_point": "tuned_best_f1",
        "variant": "with_billing_zip",
        "model": "BalancedBagging_original",
        "threshold": None,
    },
    {
        "operating_point": "recall_heavy",
        "variant": "with_billing_zip",
        "model": "CatBoost_original_balanced",
        "threshold": None,
    },
    {
        "operating_point": "native_catboost",
        "variant": "with_billing_zip",
        "model": "CatBoost_native_categorical",
        "threshold": None,
    },
]
ERROR_GROUP_FEATURES = [
    "Active_subscribers",
    "Not_Active_subscribers",
    "Suspended_subscribers",
    "Total_SUBs",
    "AvgMobileRevenue",
    "AvgFIXRevenue",
    "TotalRevenue",
    "ARPU",
    "active_rate",
    "inactive_rate",
    "suspended_rate",
    "dormant_rate",
    "revenue_per_subscriber",
    "revenue_per_active_subscriber",
    "risk_score",
    "low_revenue_high_dormant",
    "small_account_dormant_risk",
    "inactive_to_revenue_ratio",
    "dormant_revenue_pressure",
    "single_or_small_account_risk",
    "large_account",
]

REVENUE_COLS = [
    "AvgMobileRevenue",
    "AvgFIXRevenue",
    "TotalRevenue",
    "ARPU",
]

NUMERIC_INPUT_COLS = [
    "Billing_ZIP",
    "Active_subscribers",
    "Not_Active_subscribers",
    "Suspended_subscribers",
    "Total_SUBs",
    "AvgMobileRevenue",
    "AvgFIXRevenue",
    "TotalRevenue",
    "ARPU",
]

MISSING_FLAG_COLS = [
    "Not_Active_subscribers",
    "Suspended_subscribers",
    "ARPU",
    "Billing_ZIP",
]


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def safe_divide(
    numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray
) -> pd.Series:
    numerator = pd.Series(numerator)
    denominator = pd.Series(denominator).replace(0, np.nan)
    return (
        numerator.divide(denominator)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def summarize_raw_quality(df: pd.DataFrame) -> dict:
    numeric_ranges = {}
    for col in [item for item in NUMERIC_INPUT_COLS if item in df.columns]:
        numeric = pd.to_numeric(df[col], errors="coerce")
        numeric_ranges[col] = {
            "min": None if numeric.dropna().empty else float(numeric.min()),
            "max": None if numeric.dropna().empty else float(numeric.max()),
            "missing_rate": float(numeric.isna().mean()),
        }

    target_distribution = {}
    if "CHURN" in df.columns:
        target_distribution = {
            str(k): int(v) for k, v in df["CHURN"].value_counts(dropna=False).items()
        }

    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_rate": {
            col: float(rate) for col, rate in df.isna().mean().sort_values(ascending=False).items()
        },
        "duplicate_pid_count": int(df.duplicated(subset=["PID"]).sum())
        if "PID" in df.columns
        else 0,
        "target_distribution_raw": target_distribution,
        "numeric_ranges": numeric_ranges,
    }


def load_base_dataframe(input_file: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(input_file)
    raw_shape = df.shape

    df.columns = df.columns.str.strip()

    required_columns = [
        "PID",
        "CRM_PID_Value_Segment",
        "EffectiveSegment",
        "Billing_ZIP",
        "KA_name",
        "Active_subscribers",
        "Not_Active_subscribers",
        "Suspended_subscribers",
        "Total_SUBs",
        "AvgMobileRevenue",
        "AvgFIXRevenue",
        "TotalRevenue",
        "ARPU",
        "CHURN",
    ]
    require_columns(df, required_columns)

    data_quality = summarize_raw_quality(df)

    for col in NUMERIC_INPUT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    duplicate_pid_count = int(df.duplicated(subset=["PID"]).sum())
    df = df.drop_duplicates(subset=["PID"]).copy()

    df["CHURN"] = (
        df["CHURN"].astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})
    )
    if df["CHURN"].isna().any():
        bad_values = df.loc[df["CHURN"].isna(), "CHURN"].unique().tolist()
        raise ValueError(f"Unexpected CHURN values after mapping: {bad_values}")
    df["CHURN"] = df["CHURN"].astype(int)

    df["CRM_PID_Value_Segment"] = df["CRM_PID_Value_Segment"].replace(
        {"Sliver": "Silver"}
    )

    df["CRM_PID_Value_Segment"] = df["CRM_PID_Value_Segment"].fillna("Unknown")
    df["EffectiveSegment"] = df["EffectiveSegment"].fillna("Unknown")

    df = df.drop(columns=["PID", "KA_name"])

    summary = {
        "input_file": str(input_file),
        "raw_shape": list(raw_shape),
        "data_quality": data_quality,
        "duplicate_pid_removed": duplicate_pid_count,
        "shape_after_pid_dedup": [int(df.shape[0]), int(df.shape[1])],
        "leakage_guardrails": [
            "CHURN is removed from feature matrices before preprocessing.",
            "Imputation and scaling values are fit on train split only.",
            "No feature is derived from target labels.",
        ],
    }
    return df, summary


def add_engineered_features(
    df: pd.DataFrame, enable_fn_target_features: bool = False
) -> pd.DataFrame:
    active = df["Active_subscribers"]
    inactive = df["Not_Active_subscribers"]
    suspended = df["Suspended_subscribers"]
    total_subs = df["Total_SUBs"]
    mobile = df["AvgMobileRevenue"]
    fixed = df["AvgFIXRevenue"]
    revenue = df["TotalRevenue"]
    arpu = df["ARPU"]
    dormant = inactive + suspended

    df["dormant_subscribers"] = dormant
    df["active_rate"] = safe_divide(active, total_subs).clip(0.0, 1.0)
    df["inactive_rate"] = safe_divide(inactive, total_subs).clip(0.0, 1.0)
    df["suspended_rate"] = safe_divide(suspended, total_subs).clip(0.0, 1.0)
    df["dormant_rate"] = safe_divide(dormant, total_subs).clip(0.0, 1.0)

    df["mobile_revenue_ratio"] = safe_divide(mobile, revenue).clip(0.0, 1.0)
    df["fix_revenue_ratio"] = safe_divide(fixed, revenue).clip(0.0, 1.0)
    df["mobile_to_fixed_ratio"] = safe_divide(mobile, fixed)
    df["fixed_to_mobile_ratio"] = safe_divide(fixed, mobile)
    df["revenue_per_subscriber"] = safe_divide(revenue, total_subs)
    df["revenue_per_active_subscriber"] = safe_divide(revenue, active)
    df["mobile_revenue_per_subscriber"] = safe_divide(mobile, total_subs)
    df["fixed_revenue_per_subscriber"] = safe_divide(fixed, total_subs)

    df["risk_score"] = df["dormant_rate"]
    if enable_fn_target_features:
        low_revenue_pressure = safe_divide(
            pd.Series(1.0, index=df.index), np.log1p(revenue.clip(lower=0)) + 1.0
        )
        small_account_pressure = safe_divide(
            pd.Series(1.0, index=df.index), np.sqrt(total_subs.clip(lower=1))
        )
        df["inactive_to_revenue_ratio"] = safe_divide(inactive, revenue + 1.0)
        df["dormant_to_revenue_ratio"] = safe_divide(dormant, revenue + 1.0)
        df["low_revenue_high_dormant"] = df["dormant_rate"] * low_revenue_pressure
        df["small_account_dormant_risk"] = (
            df["dormant_rate"] * small_account_pressure
        )
        df["dormant_revenue_pressure"] = (
            df["dormant_rate"] * df["dormant_to_revenue_ratio"]
        )
        df["inactive_low_revenue_pressure"] = (
            df["inactive_rate"] * low_revenue_pressure
        )
        df["suspended_low_revenue_pressure"] = (
            df["suspended_rate"] * low_revenue_pressure
        )
        df["single_or_small_account_risk"] = (
            total_subs.le(3).astype(np.int64) * df["dormant_rate"]
        )
    df["revenue_engagement_interaction"] = revenue * df["active_rate"]
    df["arpu_risk_interaction"] = arpu * df["risk_score"]
    df["inactive_revenue_interaction"] = df["inactive_rate"] * revenue
    df["suspended_revenue_interaction"] = df["suspended_rate"] * revenue
    df["revenue_balance"] = safe_divide(
        pd.concat([mobile, fixed], axis=1).min(axis=1),
        pd.concat([mobile, fixed], axis=1).max(axis=1),
    ).clip(0.0, 1.0)

    df["has_inactive"] = inactive.gt(0).astype(np.int64)
    df["has_suspended"] = suspended.gt(0).astype(np.int64)
    df["has_dormant"] = dormant.gt(0).astype(np.int64)
    df["multi_subscriber"] = total_subs.gt(1).astype(np.int64)
    df["large_account"] = total_subs.ge(10).astype(np.int64)
    df["zero_mobile_revenue"] = mobile.eq(0).astype(np.int64)
    df["zero_fixed_revenue"] = fixed.eq(0).astype(np.int64)
    df["mobile_only"] = (mobile.gt(0) & fixed.eq(0)).astype(np.int64)
    df["fixed_only"] = (fixed.gt(0) & mobile.eq(0)).astype(np.int64)
    df["revenue_zero"] = revenue.eq(0).astype(np.int64)

    for col in REVENUE_COLS:
        df[col + "_log"] = np.log1p(df[col].clip(lower=0))
        df[col + "_sqrt"] = np.sqrt(df[col].clip(lower=0))

    return df


def impute_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame, use_billing_zip: bool
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    imputation_values = {}

    active_missing_flags = []
    for col in MISSING_FLAG_COLS:
        if col == "Billing_ZIP" and not use_billing_zip:
            continue
        if col in X_train.columns:
            flag_col = col + "_missing"
            X_train[flag_col] = X_train[col].isna().astype(np.int64)
            X_test[flag_col] = X_test[col].isna().astype(np.int64)
            active_missing_flags.append(flag_col)

    X_train["Not_Active_subscribers"] = X_train["Not_Active_subscribers"].fillna(0)
    X_test["Not_Active_subscribers"] = X_test["Not_Active_subscribers"].fillna(0)
    X_train["Suspended_subscribers"] = X_train["Suspended_subscribers"].fillna(0)
    X_test["Suspended_subscribers"] = X_test["Suspended_subscribers"].fillna(0)

    train_arpu_from_revenue = safe_divide(X_train["TotalRevenue"], X_train["Total_SUBs"])
    test_arpu_from_revenue = safe_divide(X_test["TotalRevenue"], X_test["Total_SUBs"])
    X_train["ARPU"] = X_train["ARPU"].fillna(train_arpu_from_revenue)
    X_test["ARPU"] = X_test["ARPU"].fillna(test_arpu_from_revenue)

    arpu_median = float(X_train["ARPU"].median())
    X_train["ARPU"] = X_train["ARPU"].fillna(arpu_median)
    X_test["ARPU"] = X_test["ARPU"].fillna(arpu_median)
    imputation_values["ARPU_median"] = arpu_median
    imputation_values["Not_Active_subscribers_fill"] = 0.0
    imputation_values["Suspended_subscribers_fill"] = 0.0

    if use_billing_zip:
        billing_zip_median = float(X_train["Billing_ZIP"].median())
        X_train["Billing_ZIP"] = X_train["Billing_ZIP"].fillna(billing_zip_median)
        X_test["Billing_ZIP"] = X_test["Billing_ZIP"].fillna(billing_zip_median)
        imputation_values["Billing_ZIP_median"] = billing_zip_median

    imputation_values["missing_flag_cols"] = active_missing_flags

    return X_train, X_test, imputation_values


def encode_categoricals(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()

        train_values = X_train[col].fillna("Unknown").astype(str)
        test_values = X_test[col].fillna("Unknown").astype(str)
        frequency_map = train_values.value_counts(normalize=True).to_dict()
        X_train[col + "_frequency"] = train_values.map(frequency_map).fillna(0.0)
        X_test[col + "_frequency"] = test_values.map(frequency_map).fillna(0.0)

        le.fit(np.append(train_values.to_numpy(), "Unknown"))
        test_values = test_values.where(test_values.isin(le.classes_), "Unknown")

        X_train[col] = np.asarray(le.transform(train_values), dtype=np.int64)
        X_test[col] = np.asarray(le.transform(test_values), dtype=np.int64)

        encoders[col] = {
            "label_encoder": le,
            "frequency_map": {str(k): float(v) for k, v in frequency_map.items()},
        }

    return X_train, X_test, encoders


def scale_numeric_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, list[str]]:
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    scaler = StandardScaler()
    scaled_train = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index
    )
    scaled_test = pd.DataFrame(
        scaler.transform(X_test[num_cols]), columns=num_cols, index=X_test.index
    )

    for col in num_cols:
        X_train[col] = scaled_train[col]
        X_test[col] = scaled_test[col]

    return X_train, X_test, scaler, num_cols


def save_series(series: pd.Series, path: Path, name: str = "CHURN") -> None:
    pd.DataFrame({name: series}).to_csv(path, index=False, encoding="utf-8-sig")


def load_processed_split(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(output_dir / "X_train.csv")
    X_test = pd.read_csv(output_dir / "X_test.csv")
    y_train = pd.read_csv(output_dir / "y_train.csv")["CHURN"].astype(int)
    y_test = pd.read_csv(output_dir / "y_test.csv")["CHURN"].astype(int)
    X_train_resampled = pd.read_csv(output_dir / "X_train_resampled.csv")
    y_train_resampled = pd.read_csv(output_dir / "y_train_resampled.csv")[
        "CHURN"
    ].astype(int)
    return X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled


def load_analysis_test_features(output_dir: Path) -> pd.DataFrame:
    return pd.read_csv(output_dir / "X_test_analysis.csv")


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


def build_comparison_models() -> list[tuple[str, Any, str]]:
    return [
        (
            "CatBoost_native_categorical",
            CatBoostClassifier(
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
            ),
            "native_categorical",
        ),
        (
            "CatBoost_original_balanced",
            CatBoostClassifier(
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
            ),
            "original",
        ),
        (
            "EasyEnsemble_original",
            EasyEnsembleClassifier(
                n_estimators=10,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "original",
        ),
        (
            "EasyEnsemble_n50_original",
            EasyEnsembleClassifier(
                n_estimators=50,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "original",
        ),
        (
            "RUSBoost_original",
            RUSBoostClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            ),
            "original",
        ),
        (
            "BalancedBagging_original",
            BalancedBaggingClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=10,
                    random_state=RANDOM_STATE,
                ),
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "original",
        ),
        (
            "GradientBoosting_SMOTE",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            "resampled",
        ),
        (
            "HistGradientBoosting_SMOTE",
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "resampled",
        ),
        (
            "LogisticRegression_SMOTE",
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            "resampled",
        ),
        (
            "RandomForest_SMOTE",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "resampled",
        ),
        (
            "XGBoost_SMOTE",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=14,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
            "resampled",
        ),
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
    y_true: pd.Series, y_pred: np.ndarray, scores: np.ndarray, prefix: str = ""
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pr_auc = float(average_precision_score(y_true, scores))
    return {
        prefix + "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        prefix + "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        prefix + "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        prefix + "accuracy": float(accuracy_score(y_true, y_pred)),
        prefix + "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        prefix + "roc_auc": float(roc_auc_score(y_true, scores)),
        prefix + "pr_auc": pr_auc,
        prefix + "average_precision": pr_auc,
        prefix + "mcc": float(matthews_corrcoef(y_true, y_pred)),
        prefix + "tp": int(tp),
        prefix + "fp": int(fp),
        prefix + "fn": int(fn),
        prefix + "tn": int(tn),
    }


def evaluate_model(
    *,
    variant: str,
    model_name: str,
    model: Any,
    train_kind: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_train_resampled: pd.DataFrame,
    y_train_resampled: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train_native: pd.DataFrame | None = None,
    X_test_native: pd.DataFrame | None = None,
) -> dict:
    if train_kind == "native_categorical":
        if X_train_native is None or X_test_native is None:
            raise ValueError("Native CatBoost splits are required for native training.")
        fit_X, fit_y = X_train_native, y_train
        eval_X = X_test_native
    elif train_kind == "resampled":
        fit_X, fit_y = X_train_resampled, y_train_resampled
        eval_X = X_test
    else:
        fit_X, fit_y = X_train, y_train
        eval_X = X_test

    fit_model(model, fit_X, fit_y, train_kind)
    y_pred = np.asarray(model.predict(eval_X)).astype(int)
    scores = positive_scores(model, eval_X)

    row = {
        "variant": variant,
        "model": model_name,
        "train_data": train_kind,
    }
    row.update(classification_metrics(y_test, y_pred, scores))
    return row


def run_model_comparison(output_root: Path) -> pd.DataFrame:
    variant_dirs = [
        ("with_billing_zip", output_root / "model_a_with_billing_zip"),
        ("without_billing_zip", output_root / "model_b_without_billing_zip"),
    ]
    rows = []
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
        for model_name, model, train_kind in build_comparison_models():
            rows.append(
                evaluate_model(
                    variant=variant,
                    model_name=model_name,
                    model=model,
                    train_kind=train_kind,
                    X_train=X_train,
                    y_train=y_train,
                    X_train_resampled=X_train_resampled,
                    y_train_resampled=y_train_resampled,
                    X_test=X_test,
                    y_test=y_test,
                    X_train_native=X_train_native,
                    X_test_native=X_test_native,
                )
            )

    table = pd.DataFrame(rows).sort_values(
        ["f1", "roc_auc"], ascending=[False, False]
    )
    comparison_output = output_root / COMPARISON_OUTPUT.name
    table.to_csv(comparison_output, index=False, encoding="utf-8-sig")
    return table


def fit_threshold_training_data(
    X_train: pd.DataFrame, y_train: pd.Series, train_kind: str
) -> tuple[pd.DataFrame, pd.Series]:
    if train_kind != "resampled":
        return X_train, y_train

    smote = SVMSMOTE(random_state=RANDOM_STATE)
    resampled: Any = smote.fit_resample(X_train, y_train)
    return (
        pd.DataFrame(resampled[0], columns=X_train.columns),
        pd.Series(resampled[1], name="CHURN").astype(int),
    )


def threshold_metrics_row(
    *,
    variant: str,
    model_name: str,
    train_kind: str,
    threshold: float,
    y_true: pd.Series,
    scores: np.ndarray,
) -> dict:
    y_pred = (scores >= threshold).astype(int)
    row = {
        "variant": variant,
        "model": model_name,
        "train_data": train_kind,
        "threshold": threshold,
    }
    row.update(classification_metrics(y_true, y_pred, scores))
    return row


def tune_threshold_for_model(
    *,
    variant: str,
    model_name: str,
    model: Any,
    train_kind: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_train_resampled: pd.DataFrame,
    y_train_resampled: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train_native: pd.DataFrame | None = None,
    X_test_native: pd.DataFrame | None = None,
) -> tuple[dict, list[dict]]:
    if train_kind == "native_categorical":
        if X_train_native is None or X_test_native is None:
            raise ValueError("Native CatBoost splits are required for threshold tuning.")
        threshold_X_train = X_train_native
        threshold_X_test = X_test_native
    else:
        threshold_X_train = X_train
        threshold_X_test = X_test

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        threshold_X_train,
        y_train,
        test_size=THRESHOLD_VALIDATION_SIZE,
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

    sweep_rows = [
        threshold_metrics_row(
            variant=variant,
            model_name=model_name,
            train_kind=train_kind,
            threshold=threshold,
            y_true=y_valid,
            scores=validation_scores,
        )
        for threshold in THRESHOLD_GRID
    ]
    sweep_table = pd.DataFrame(sweep_rows)
    f1_sorted_sweep = sweep_table.sort_values(
        ["f1", "recall", "precision"], ascending=[False, False, False]
    )
    unconstrained_best = f1_sorted_sweep.iloc[0].to_dict()
    feasible_sweep = f1_sorted_sweep[
        f1_sorted_sweep["recall"] >= MIN_RECALL_CONSTRAINT
    ]
    if feasible_sweep.empty:
        best_validation = unconstrained_best
        selection_strategy = "max_f1_fallback_no_threshold_met_min_recall"
    else:
        best_validation = feasible_sweep.iloc[0].to_dict()
        selection_strategy = "max_f1_with_min_recall_constraint"
    selected_threshold = float(best_validation["threshold"])

    final_model = clone(model)
    if train_kind == "native_categorical":
        final_fit_X, final_fit_y = threshold_X_train, y_train
    elif train_kind == "resampled":
        final_fit_X, final_fit_y = X_train_resampled, y_train_resampled
    else:
        final_fit_X, final_fit_y = X_train, y_train
    fit_model(final_model, final_fit_X, final_fit_y, train_kind)
    test_scores = positive_scores(final_model, threshold_X_test)
    test_pred = (test_scores >= selected_threshold).astype(int)

    best_row = {
        "variant": variant,
        "model": model_name,
        "train_data": train_kind,
        "selected_threshold": selected_threshold,
        "threshold_selection_strategy": selection_strategy,
        "min_recall_constraint": MIN_RECALL_CONSTRAINT,
        "validation_recall_constraint_met": bool(
            best_validation["recall"] >= MIN_RECALL_CONSTRAINT
        ),
        "unconstrained_selected_threshold": float(unconstrained_best["threshold"]),
        "unconstrained_validation_f1": float(unconstrained_best["f1"]),
        "unconstrained_validation_recall": float(unconstrained_best["recall"]),
        "unconstrained_validation_precision": float(unconstrained_best["precision"]),
    }
    metric_names = [
        "f1",
        "recall",
        "precision",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "average_precision",
        "mcc",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    for metric_name in metric_names:
        best_row["validation_" + metric_name] = best_validation[metric_name]
    best_row.update(classification_metrics(y_test, test_pred, test_scores, prefix="test_"))

    return best_row, sweep_rows


def run_threshold_tuning(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    variant_dirs = [
        ("with_billing_zip", output_root / "model_a_with_billing_zip"),
        ("without_billing_zip", output_root / "model_b_without_billing_zip"),
    ]
    best_rows = []
    sweep_rows = []

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
        for model_name, model, train_kind in build_comparison_models():
            if model_name not in THRESHOLD_TUNING_MODELS:
                continue
            best_row, model_sweep_rows = tune_threshold_for_model(
                variant=variant,
                model_name=model_name,
                model=model,
                train_kind=train_kind,
                X_train=X_train,
                y_train=y_train,
                X_train_resampled=X_train_resampled,
                y_train_resampled=y_train_resampled,
                X_test=X_test,
                y_test=y_test,
                X_train_native=X_train_native,
                X_test_native=X_test_native,
            )
            best_rows.append(best_row)
            sweep_rows.extend(model_sweep_rows)

    best_table = pd.DataFrame(best_rows).sort_values(
        ["test_f1", "test_recall", "test_precision"], ascending=[False, False, False]
    )
    sweep_table = pd.DataFrame(sweep_rows).sort_values(
        ["variant", "model", "f1"], ascending=[True, True, False]
    )
    best_table.to_csv(output_root / THRESHOLD_BEST_OUTPUT.name, index=False, encoding="utf-8-sig")
    sweep_table.to_csv(output_root / THRESHOLD_SWEEP_OUTPUT.name, index=False, encoding="utf-8-sig")
    return best_table, sweep_table


def comparison_model_lookup() -> dict[str, tuple[Any, str]]:
    return {model_name: (model, train_kind) for model_name, model, train_kind in build_comparison_models()}


def variant_output_dirs(output_root: Path) -> dict[str, Path]:
    return {
        "with_billing_zip": output_root / "model_a_with_billing_zip",
        "without_billing_zip": output_root / "model_b_without_billing_zip",
    }


def resolve_operating_point_threshold(
    operating_point: dict, threshold_best_table: pd.DataFrame
) -> float:
    if operating_point["threshold"] is not None:
        return float(operating_point["threshold"])

    matches = threshold_best_table[
        (threshold_best_table["variant"] == operating_point["variant"])
        & (threshold_best_table["model"] == operating_point["model"])
    ]
    if matches.empty:
        raise ValueError(f"No tuned threshold found for {operating_point}")
    return float(matches.iloc[0]["selected_threshold"])


def fit_final_model_for_analysis(
    model: Any,
    train_kind: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_train_resampled: pd.DataFrame,
    y_train_resampled: pd.Series,
    X_train_native: pd.DataFrame | None = None,
) -> Any:
    fitted_model = clone(model)
    if train_kind == "native_categorical":
        if X_train_native is None:
            raise ValueError("Native CatBoost training split is required for analysis.")
        fit_X, fit_y = X_train_native, y_train
    elif train_kind == "resampled":
        fit_X, fit_y = X_train_resampled, y_train_resampled
    else:
        fit_X, fit_y = X_train, y_train
    fit_model(fitted_model, fit_X, fit_y, train_kind)
    return fitted_model


def prediction_group(actual: int, predicted: int) -> str:
    if actual == 1 and predicted == 1:
        return "TP"
    if actual == 0 and predicted == 1:
        return "FP"
    if actual == 1 and predicted == 0:
        return "FN"
    return "TN"


def build_error_prediction_rows(
    *,
    operating_point_name: str,
    variant: str,
    model_name: str,
    train_kind: str,
    threshold: float,
    y_test: pd.Series,
    scores: np.ndarray,
    X_test_analysis: pd.DataFrame,
) -> pd.DataFrame:
    predictions = (scores >= threshold).astype(int)
    base = pd.DataFrame(
        {
            "operating_point": operating_point_name,
            "variant": variant,
            "model": model_name,
            "train_data": train_kind,
            "threshold": threshold,
            "test_row_id": np.arange(len(y_test)),
            "actual": y_test.to_numpy(),
            "predicted": predictions,
            "score": scores,
        }
    )
    base["confusion_group"] = [
        prediction_group(int(actual), int(predicted))
        for actual, predicted in zip(base["actual"], base["predicted"])
    ]
    return pd.concat([base, X_test_analysis.reset_index(drop=True)], axis=1)


def build_error_group_summary(error_predictions: pd.DataFrame) -> pd.DataFrame:
    summary_features = [
        col for col in ERROR_GROUP_FEATURES if col in error_predictions.columns
    ]
    group_cols = ["operating_point", "variant", "model", "confusion_group"]
    rows = []
    for group_values, group in error_predictions.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, group_values))
        row["count"] = int(len(group))
        row["score_mean"] = float(group["score"].mean())
        row["score_median"] = float(group["score"].median())
        for col in summary_features:
            numeric = pd.to_numeric(group[col], errors="coerce")
            row[col + "_mean"] = None if numeric.dropna().empty else float(numeric.mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def permutation_importance_for_threshold(
    *,
    operating_point_name: str,
    variant: str,
    model_name: str,
    train_kind: str,
    threshold: float,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    baseline_metrics: dict,
) -> list[dict]:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for feature in X_test.columns:
        f1_scores = []
        recall_scores = []
        precision_scores = []
        for _ in range(PERMUTATION_REPEATS):
            X_permuted = X_test.copy()
            X_permuted[feature] = rng.permutation(X_permuted[feature].to_numpy())
            permuted_scores = positive_scores(model, X_permuted)
            permuted_pred = (permuted_scores >= threshold).astype(int)
            permuted_metrics = classification_metrics(
                y_test, permuted_pred, permuted_scores
            )
            f1_scores.append(permuted_metrics["f1"])
            recall_scores.append(permuted_metrics["recall"])
            precision_scores.append(permuted_metrics["precision"])

        rows.append(
            {
                "operating_point": operating_point_name,
                "variant": variant,
                "model": model_name,
                "train_data": train_kind,
                "threshold": threshold,
                "feature": feature,
                "baseline_f1": baseline_metrics["f1"],
                "baseline_recall": baseline_metrics["recall"],
                "baseline_precision": baseline_metrics["precision"],
                "permuted_f1_mean": float(np.mean(f1_scores)),
                "permuted_f1_std": float(np.std(f1_scores)),
                "f1_importance": float(baseline_metrics["f1"] - np.mean(f1_scores)),
                "recall_importance": float(
                    baseline_metrics["recall"] - np.mean(recall_scores)
                ),
                "precision_importance": float(
                    baseline_metrics["precision"] - np.mean(precision_scores)
                ),
            }
        )

    return rows


def run_error_and_feature_analysis(
    output_root: Path, threshold_best_table: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_lookup = comparison_model_lookup()
    output_dirs = variant_output_dirs(output_root)
    error_prediction_tables = []
    importance_rows = []

    for operating_point in ANALYSIS_OPERATING_POINTS:
        variant = operating_point["variant"]
        model_name = operating_point["model"]
        operating_point_name = operating_point["operating_point"]
        if model_name not in model_lookup:
            raise ValueError(f"Unknown analysis model: {model_name}")

        model, train_kind = model_lookup[model_name]
        output_dir = output_dirs[variant]
        (
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_resampled,
            y_train_resampled,
        ) = load_processed_split(output_dir)
        X_train_native, X_test_native = load_native_catboost_split(output_dir)
        X_test_analysis = load_analysis_test_features(output_dir)
        eval_X_test = X_test_native if train_kind == "native_categorical" else X_test
        threshold = resolve_operating_point_threshold(
            operating_point, threshold_best_table
        )
        fitted_model = fit_final_model_for_analysis(
            model,
            train_kind,
            X_train,
            y_train,
            X_train_resampled,
            y_train_resampled,
            X_train_native,
        )
        scores = positive_scores(fitted_model, eval_X_test)
        predictions = (scores >= threshold).astype(int)
        baseline_metrics = classification_metrics(y_test, predictions, scores)

        error_prediction_tables.append(
            build_error_prediction_rows(
                operating_point_name=operating_point_name,
                variant=variant,
                model_name=model_name,
                train_kind=train_kind,
                threshold=threshold,
                y_test=y_test,
                scores=scores,
                X_test_analysis=X_test_analysis,
            )
        )
        importance_rows.extend(
            permutation_importance_for_threshold(
                operating_point_name=operating_point_name,
                variant=variant,
                model_name=model_name,
                train_kind=train_kind,
                threshold=threshold,
                model=fitted_model,
                X_test=eval_X_test,
                y_test=y_test,
                baseline_metrics=baseline_metrics,
            )
        )

    error_predictions = pd.concat(error_prediction_tables, ignore_index=True)
    error_summary = build_error_group_summary(error_predictions)
    importance_table = pd.DataFrame(importance_rows).sort_values(
        ["operating_point", "f1_importance"], ascending=[True, False]
    )
    importance_top = (
        importance_table.sort_values(
            ["operating_point", "f1_importance"], ascending=[True, False]
        )
        .groupby("operating_point", group_keys=False)
        .head(15)
        .reset_index(drop=True)
    )

    error_predictions.to_csv(
        output_root / ERROR_ANALYSIS_OUTPUT.name, index=False, encoding="utf-8-sig"
    )
    error_summary.to_csv(
        output_root / ERROR_GROUP_SUMMARY_OUTPUT.name, index=False, encoding="utf-8-sig"
    )
    importance_table.to_csv(
        output_root / FEATURE_IMPORTANCE_OUTPUT.name, index=False, encoding="utf-8-sig"
    )
    importance_top.to_csv(
        output_root / FEATURE_IMPORTANCE_TOP_OUTPUT.name, index=False, encoding="utf-8-sig"
    )

    return error_predictions, error_summary, importance_table, importance_top


def preprocess_variant(
    input_file: Path,
    output_dir: Path,
    use_billing_zip: bool,
    enable_fn_target_features: bool = ENABLE_FN_TARGET_FEATURES_DEFAULT,
) -> dict:
    df, summary = load_base_dataframe(input_file)

    if use_billing_zip:
        cat_cols = ["CRM_PID_Value_Segment", "EffectiveSegment", "Billing_ZIP"]
        variant_name = "model_a_with_billing_zip"
    else:
        df = df.drop(columns=["Billing_ZIP"])
        cat_cols = ["CRM_PID_Value_Segment", "EffectiveSegment"]
        variant_name = "model_b_without_billing_zip"

    X = df.drop(columns=["CHURN"])
    y = df["CHURN"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train, X_test, imputation_values = impute_train_test(
        X_train, X_test, use_billing_zip
    )

    if use_billing_zip:
        X_train["Billing_ZIP"] = X_train["Billing_ZIP"].astype(str)
        X_test["Billing_ZIP"] = X_test["Billing_ZIP"].astype(str)

    X_train = add_engineered_features(X_train, enable_fn_target_features)
    X_test = add_engineered_features(X_test, enable_fn_target_features)
    X_train_analysis = X_train.copy()
    X_test_analysis = X_test.copy()

    X_train, X_test, encoders = encode_categoricals(X_train, X_test, cat_cols)
    X_train, X_test, scaler, num_cols = scale_numeric_features(X_train, X_test)

    smote = SVMSMOTE(random_state=RANDOM_STATE)
    resampled: Any = smote.fit_resample(X_train, y_train)
    X_train_resampled = pd.DataFrame(resampled[0], columns=X_train.columns)
    y_train_resampled = pd.Series(resampled[1], name="CHURN").astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False, encoding="utf-8-sig")
    X_test.to_csv(output_dir / "X_test.csv", index=False, encoding="utf-8-sig")
    X_train_analysis.to_csv(
        output_dir / "X_train_analysis.csv", index=False, encoding="utf-8-sig"
    )
    X_test_analysis.to_csv(
        output_dir / "X_test_analysis.csv", index=False, encoding="utf-8-sig"
    )
    save_series(y_train, output_dir / "y_train.csv")
    save_series(y_test, output_dir / "y_test.csv")

    X_train_resampled.to_csv(
        output_dir / "X_train_resampled.csv", index=False, encoding="utf-8-sig"
    )
    save_series(y_train_resampled, output_dir / "y_train_resampled.csv")

    artifacts = {
        "encoders": encoders,
        "scaler": scaler,
        "imputation_values": imputation_values,
        "cat_cols": cat_cols,
        "missing_flag_cols": imputation_values["missing_flag_cols"],
        "num_cols": num_cols,
        "feature_columns": X_train.columns.tolist(),
        "use_billing_zip": use_billing_zip,
        "enable_fn_target_features": enable_fn_target_features,
    }
    joblib.dump(artifacts, output_dir / "preprocessing_artifacts.joblib")

    feature_engineering_notes = [
        "Missingness is preserved through *_missing flags before imputation.",
        "Subscriber status ratios and dormant customer signals are derived from pre-target customer state.",
        "Revenue ratios, per-subscriber revenue, and revenue balance features expose domain structure explicitly.",
        "Categorical features keep ordinal labels for tree compatibility and add train-only frequency encoding.",
    ]
    if enable_fn_target_features:
        feature_engineering_notes.insert(
            3,
            "FN-targeted dormant and low-revenue pressure features highlight low-revenue high-dormancy churn risk.",
        )

    summary.update(
        {
            "variant": variant_name,
            "use_billing_zip": use_billing_zip,
            "enable_fn_target_features": enable_fn_target_features,
            "cat_cols": cat_cols,
            "imputation_values": imputation_values,
            "feature_engineering_notes": feature_engineering_notes,
            "num_cols": num_cols,
            "feature_count": int(X_train.shape[1]),
            "X_train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
            "X_test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
            "X_train_analysis_shape": [
                int(X_train_analysis.shape[0]),
                int(X_train_analysis.shape[1]),
            ],
            "X_test_analysis_shape": [
                int(X_test_analysis.shape[0]),
                int(X_test_analysis.shape[1]),
            ],
            "X_train_resampled_shape": [
                int(X_train_resampled.shape[0]),
                int(X_train_resampled.shape[1]),
            ],
            "y_train_distribution": {
                str(k): int(v)
                for k, v in y_train.value_counts().sort_index().to_dict().items()
            },
            "y_test_distribution": {
                str(k): int(v)
                for k, v in y_test.value_counts().sort_index().to_dict().items()
            },
            "y_train_resampled_distribution": {
                str(k): int(v)
                for k, v in y_train_resampled.value_counts()
                .sort_index()
                .to_dict()
                .items()
            },
        }
    )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess telecom churn data with optional Billing_ZIP variants."
    )
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--variant",
        choices=["both", "with-zip", "without-zip"],
        default="both",
        help="Which preprocessing variant to generate.",
    )
    parser.add_argument(
        "--skip-model-comparison",
        action="store_true",
        help="Only write preprocessing outputs and skip model metric comparison.",
    )
    parser.add_argument(
        "--skip-error-analysis",
        action="store_true",
        help="Run preprocessing, comparison, and threshold tuning without permutation/error analysis.",
    )
    parser.add_argument(
        "--enable-fn-target-features",
        action="store_true",
        help="Enable experimental low-revenue high-dormancy features from FN analysis.",
    )
    args = parser.parse_args()

    variants = []
    if args.variant in {"both", "with-zip"}:
        variants.append((True, args.output_root / "model_a_with_billing_zip"))
    if args.variant in {"both", "without-zip"}:
        variants.append((False, args.output_root / "model_b_without_billing_zip"))

    summaries = [
        preprocess_variant(
            args.input,
            output_dir,
            use_billing_zip,
            args.enable_fn_target_features,
        )
        for use_billing_zip, output_dir in variants
    ]

    comparison_table = None
    threshold_best_table = None
    error_summary_table = None
    feature_importance_top_table = None
    if not args.skip_model_comparison:
        comparison_table = run_model_comparison(args.output_root)
        threshold_best_table, _ = run_threshold_tuning(args.output_root)
        if not args.skip_error_analysis:
            _, error_summary_table, _, feature_importance_top_table = (
                run_error_and_feature_analysis(args.output_root, threshold_best_table)
            )
        for summary in summaries:
            comparison_variant = (
                "with_billing_zip"
                if summary["use_billing_zip"]
                else "without_billing_zip"
            )
            rows = comparison_table[
                comparison_table["variant"] == comparison_variant
            ].to_dict(orient="records")
            summary["model_comparison"] = rows
            threshold_rows = threshold_best_table[
                threshold_best_table["variant"] == comparison_variant
            ].to_dict(orient="records")
            summary["threshold_tuning"] = threshold_rows
            if error_summary_table is not None:
                error_rows = error_summary_table[
                    error_summary_table["variant"] == comparison_variant
                ].to_dict(orient="records")
                summary["error_analysis_group_summary"] = error_rows
            if feature_importance_top_table is not None:
                importance_rows = feature_importance_top_table[
                    feature_importance_top_table["variant"] == comparison_variant
                ].to_dict(orient="records")
                summary["feature_importance_top"] = importance_rows

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary_all.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Preprocessing complete")
    for summary in summaries:
        print(f"\n[{summary['variant']}]")
        print("X_train:", tuple(summary["X_train_shape"]))
        print("X_test:", tuple(summary["X_test_shape"]))
        print("X_train_resampled:", tuple(summary["X_train_resampled_shape"]))
        print("y_train_resampled distribution:")
        print(summary["y_train_resampled_distribution"])

    if comparison_table is not None and not comparison_table.empty:
        best = comparison_table.iloc[0]
        print("\nModel comparison complete")
        print(
            "Best:",
            best["variant"],
            best["model"],
            f"f1={best['f1']:.4f}",
            f"recall={best['recall']:.4f}",
            f"precision={best['precision']:.4f}",
        )

    if threshold_best_table is not None and not threshold_best_table.empty:
        best_threshold = threshold_best_table.iloc[0]
        print("\nThreshold tuning complete")
        print(
            "Best tuned:",
            best_threshold["variant"],
            best_threshold["model"],
            f"threshold={best_threshold['selected_threshold']:.2f}",
            f"test_f1={best_threshold['test_f1']:.4f}",
            f"test_recall={best_threshold['test_recall']:.4f}",
            f"test_precision={best_threshold['test_precision']:.4f}",
        )

    if (
        error_summary_table is not None
        and not error_summary_table.empty
        and feature_importance_top_table is not None
        and not feature_importance_top_table.empty
    ):
        print("\nError analysis and feature importance complete")
        print(
            "Generated:",
            ERROR_ANALYSIS_OUTPUT.name,
            ERROR_GROUP_SUMMARY_OUTPUT.name,
            FEATURE_IMPORTANCE_OUTPUT.name,
            FEATURE_IMPORTANCE_TOP_OUTPUT.name,
        )


if __name__ == "__main__":
    main()
