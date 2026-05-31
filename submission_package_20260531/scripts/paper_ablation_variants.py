import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


INPUT_FILE = Path("Baza customer Telecom v2.csv")
OUTPUT_ROOT = Path("processed") / "paper_ablation_variants"
RANDOM_STATE = 42
TEST_SIZE = 0.2
EPSILON = 1e-9

REVENUE_COLS = [
    "TotalRevenue",
    "ARPU",
    "AvgMobileRevenue",
    "AvgFIXRevenue",
]

SUBSCRIBER_SQRT_COLS = [
    "Not_Active_subscribers",
    "Suspended_subscribers",
    "Total_SUBs",
]

ZERO_FILL_COLS = [
    "Not_Active_subscribers",
    "Suspended_subscribers",
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

BASE_CATEGORICAL_COLS = [
    "CRM_PID_Value_Segment",
    "EffectiveSegment",
    "KA_name",
]

CODE_TYPE_KAS = {
    "VM",
    "RJ",
    "MT",
    "AD?",
    "VU",
    "DI",
    "AD",
    "VT",
}

PREMIUM_SEGMENTS = {
    "Gold",
    "Platinum",
    "SME",
}

BASE_NUMERIC_COLS = [
    "Active_subscribers",
    "Not_Active_subscribers",
    "Suspended_subscribers",
    "Total_SUBs",
    "AvgMobileRevenue",
    "AvgFIXRevenue",
    "TotalRevenue",
    "ARPU",
]

CORE_ENGINEERED_COLS = [
    "active_rate",
    "inactive_rate",
    "suspended_rate",
    "risk_score",
    "mobile_revenue_ratio",
    "fixed_revenue_ratio",
    "revenue_per_subscriber",
    "revenue_x_active_rate",
]

EXTENDED_ENGINEERED_COLS = [
    "revenue_x_risk",
    "inactive_x_fixed_ratio",
    "suspended_x_mobile_ratio",
    "arpu_per_active",
    "total_rev_rank_by_segment",
]

KA_ABSTRACT_COLS = [
    "KA_is_code_type",
    "KA_type_x_premium",
]

KA_RESEARCH_COLS = [
    "KA_is_code_type",
    "KA_churn_rate_encoded",
    "KA_customer_count",
    "KA_avg_portfolio_revenue",
    "KA_type_x_premium",
]


@dataclass(frozen=True)
class VariantConfig:
    name: str
    zip_mode: str
    transform_mode: str
    include_extended_interactions: bool
    ka_mode: str = "label"
    zip_top_n: int = 50


VARIANTS = [
    VariantConfig(
        name="paper_core_zip_log",
        zip_mode="encoded",
        transform_mode="log_revenue",
        include_extended_interactions=False,
    ),
    VariantConfig(
        name="paper_core_no_zip_log",
        zip_mode="drop",
        transform_mode="log_revenue",
        include_extended_interactions=False,
    ),
    VariantConfig(
        name="paper_core_zip_top50_log",
        zip_mode="top_n",
        transform_mode="log_revenue",
        include_extended_interactions=False,
    ),
    VariantConfig(
        name="extended_zip_log_sqrt_interactions",
        zip_mode="encoded",
        transform_mode="log_revenue_sqrt_subscribers",
        include_extended_interactions=True,
    ),
    VariantConfig(
        name="extended_no_zip_log_sqrt_interactions",
        zip_mode="drop",
        transform_mode="log_revenue_sqrt_subscribers",
        include_extended_interactions=True,
    ),
    VariantConfig(
        name="extended_zip_top50_log_sqrt_interactions",
        zip_mode="top_n",
        transform_mode="log_revenue_sqrt_subscribers",
        include_extended_interactions=True,
    ),
    VariantConfig(
        name="paper_core_zip_log_ka_abstract",
        zip_mode="encoded",
        transform_mode="log_revenue",
        include_extended_interactions=False,
        ka_mode="abstract",
    ),
    VariantConfig(
        name="extended_zip_top50_log_sqrt_ka_abstract",
        zip_mode="top_n",
        transform_mode="log_revenue_sqrt_subscribers",
        include_extended_interactions=True,
        ka_mode="abstract",
    ),
    VariantConfig(
        name="extended_zip_top50_log_sqrt_ka_research",
        zip_mode="top_n",
        transform_mode="log_revenue_sqrt_subscribers",
        include_extended_interactions=True,
        ka_mode="research_full",
    ),
]


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (
        numerator.divide(denominator)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def load_base_dataframe(input_file: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
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

    summary = {
        "input_file": str(input_file),
        "raw_shape": [int(raw_shape[0]), int(raw_shape[1])],
        "duplicate_pid_removed": duplicate_pid_count,
        "shape_after_pid_dedup": [int(df.shape[0]), int(df.shape[1])],
        "actual_columns": df.columns.tolist(),
        "actual_input_feature_count_excluding_target": int(df.shape[1] - 1),
        "actual_model_feature_count_excluding_pid_and_target": int(df.shape[1] - 2),
        "target_distribution_after_dedup": {
            str(k): int(v) for k, v in df["CHURN"].value_counts().sort_index().items()
        },
    }
    return df, summary


def fill_missing_values(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    use_zip: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    imputation_values: dict[str, Any] = {}

    for col in BASE_CATEGORICAL_COLS:
        X_train[col] = X_train[col].fillna("Unknown").astype(str)
        X_test[col] = X_test[col].fillna("Unknown").astype(str)

    for col in ZERO_FILL_COLS:
        X_train[col] = X_train[col].fillna(0.0)
        X_test[col] = X_test[col].fillna(0.0)
        imputation_values[f"{col}_fill"] = 0.0

    arpu_median = float(X_train["ARPU"].median())
    X_train["ARPU"] = X_train["ARPU"].fillna(arpu_median)
    X_test["ARPU"] = X_test["ARPU"].fillna(arpu_median)
    imputation_values["ARPU_median"] = arpu_median

    if use_zip:
        zip_median = float(X_train["Billing_ZIP"].median())
        X_train["Billing_ZIP"] = X_train["Billing_ZIP"].fillna(zip_median)
        X_test["Billing_ZIP"] = X_test["Billing_ZIP"].fillna(zip_median)
        imputation_values["Billing_ZIP_median"] = zip_median

    return X_train, X_test, imputation_values


def format_zip_value(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return str(value)


def apply_zip_mode(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: VariantConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    cat_cols = BASE_CATEGORICAL_COLS.copy()
    if config.ka_mode != "label":
        cat_cols = [col for col in cat_cols if col != "KA_name"]

    zip_metadata: dict[str, Any] = {"zip_mode": config.zip_mode}

    if config.zip_mode == "drop":
        X_train = X_train.drop(columns=["Billing_ZIP"])
        X_test = X_test.drop(columns=["Billing_ZIP"])
        return X_train, X_test, cat_cols, zip_metadata

    X_train["Billing_ZIP"] = X_train["Billing_ZIP"].map(format_zip_value)
    X_test["Billing_ZIP"] = X_test["Billing_ZIP"].map(format_zip_value)
    cat_cols.append("Billing_ZIP")

    if config.zip_mode == "top_n":
        top_values = (
            X_train["Billing_ZIP"]
            .value_counts(dropna=False)
            .head(config.zip_top_n)
            .index.astype(str)
            .tolist()
        )
        top_set = set(top_values)
        X_train["Billing_ZIP"] = X_train["Billing_ZIP"].where(
            X_train["Billing_ZIP"].isin(top_set), "Other"
        )
        X_test["Billing_ZIP"] = X_test["Billing_ZIP"].where(
            X_test["Billing_ZIP"].isin(top_set), "Other"
        )
        zip_metadata["zip_top_n"] = config.zip_top_n
        zip_metadata["zip_top_values"] = top_values
    elif config.zip_mode != "encoded":
        raise ValueError(f"Unsupported zip mode: {config.zip_mode}")

    return X_train, X_test, cat_cols, zip_metadata


def fit_ka_reference(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    reference_frame = X_train[["KA_name", "TotalRevenue"]].copy()
    reference_frame["CHURN"] = y_train.astype(int)

    churn_rate = reference_frame.groupby("KA_name")["CHURN"].mean()
    churn_sum = reference_frame.groupby("KA_name")["CHURN"].sum()
    customer_count = reference_frame.groupby("KA_name")["CHURN"].size()
    avg_revenue = reference_frame.groupby("KA_name")["TotalRevenue"].mean()

    return {
        "overall_churn_rate": float(y_train.mean()),
        "fallback_customer_count": float(customer_count.median()),
        "fallback_avg_portfolio_revenue": float(reference_frame["TotalRevenue"].mean()),
        "ka_churn_rate": churn_rate.to_dict(),
        "ka_churn_sum": churn_sum.to_dict(),
        "ka_customer_count": customer_count.to_dict(),
        "ka_avg_portfolio_revenue": avg_revenue.to_dict(),
    }


def apply_ka_features(
    X: pd.DataFrame,
    config: VariantConfig,
    ka_reference: dict[str, Any] | None,
    y: pd.Series | None = None,
) -> pd.DataFrame:
    if config.ka_mode == "label":
        return X
    if config.ka_mode not in {"abstract", "research_full"}:
        raise ValueError(f"Unsupported KA mode: {config.ka_mode}")

    X["KA_is_code_type"] = X["KA_name"].isin(CODE_TYPE_KAS).astype(np.int64)
    X["KA_type_x_premium"] = (
        X["KA_is_code_type"].eq(0)
        & X["CRM_PID_Value_Segment"].isin(PREMIUM_SEGMENTS)
    ).astype(np.int64)

    if config.ka_mode == "research_full":
        if ka_reference is None:
            raise ValueError("ka_reference is required for research_full KA mode.")
        if y is None:
            X["KA_churn_rate_encoded"] = (
                X["KA_name"]
                .map(ka_reference["ka_churn_rate"])
                .fillna(ka_reference["overall_churn_rate"])
                .astype(float)
            )
        else:
            group_sum = X["KA_name"].map(ka_reference["ka_churn_sum"]).astype(float)
            group_count = X["KA_name"].map(ka_reference["ka_customer_count"]).astype(float)
            y_aligned = y.reindex(X.index).astype(float)
            loo_rate = (group_sum - y_aligned).divide(group_count - 1.0)
            X["KA_churn_rate_encoded"] = (
                loo_rate.where(group_count > 1, ka_reference["overall_churn_rate"])
                .fillna(ka_reference["overall_churn_rate"])
                .astype(float)
            )
        X["KA_customer_count"] = (
            X["KA_name"]
            .map(ka_reference["ka_customer_count"])
            .fillna(ka_reference["fallback_customer_count"])
            .astype(float)
        )
        X["KA_avg_portfolio_revenue"] = (
            X["KA_name"]
            .map(ka_reference["ka_avg_portfolio_revenue"])
            .fillna(ka_reference["fallback_avg_portfolio_revenue"])
            .astype(float)
        )

    return X


def summarize_ka_reference(
    config: VariantConfig,
    ka_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "ka_mode": config.ka_mode,
        "code_type_kas": sorted(CODE_TYPE_KAS),
        "premium_segments": sorted(PREMIUM_SEGMENTS),
        "raw_ka_name_in_final_features": config.ka_mode == "label",
    }
    if config.ka_mode == "abstract":
        metadata["features"] = KA_ABSTRACT_COLS
        metadata["target_dependent"] = False
    elif config.ka_mode == "research_full":
        if ka_reference is None:
            raise ValueError("ka_reference is required for research_full KA mode.")
        churn_values = pd.Series(ka_reference["ka_churn_rate"], dtype=float)
        count_values = pd.Series(ka_reference["ka_customer_count"], dtype=float)
        revenue_values = pd.Series(
            ka_reference["ka_avg_portfolio_revenue"], dtype=float
        )
        metadata.update(
            {
                "features": KA_RESEARCH_COLS,
                "target_dependent": True,
                "fit_scope": "train_only",
                "unseen_ka_churn_rate_fill": ka_reference["overall_churn_rate"],
                "unseen_ka_customer_count_fill": ka_reference[
                    "fallback_customer_count"
                ],
                "unseen_ka_avg_portfolio_revenue_fill": ka_reference[
                    "fallback_avg_portfolio_revenue"
                ],
                "train_ka_churn_rate_min": float(churn_values.min()),
                "train_ka_churn_rate_max": float(churn_values.max()),
                "train_ka_customer_count_min": float(count_values.min()),
                "train_ka_customer_count_max": float(count_values.max()),
                "train_ka_avg_portfolio_revenue_min": float(revenue_values.min()),
                "train_ka_avg_portfolio_revenue_max": float(revenue_values.max()),
            }
        )
    else:
        metadata["features"] = ["KA_name"]
        metadata["target_dependent"] = False
    return metadata


def fit_segment_rank_reference(X_train: pd.DataFrame) -> dict[str, Any]:
    global_values = np.sort(X_train["TotalRevenue"].astype(float).to_numpy())
    segment_values = {}
    for segment, group in X_train.groupby("CRM_PID_Value_Segment"):
        segment_values[str(segment)] = np.sort(group["TotalRevenue"].astype(float).to_numpy())
    return {
        "global": global_values,
        "segments": segment_values,
    }


def empirical_percentile(values: pd.Series, reference: np.ndarray) -> pd.Series:
    if reference.size == 0:
        return pd.Series(0.0, index=values.index)
    ranks = np.searchsorted(reference, values.astype(float).to_numpy(), side="right")
    return pd.Series(ranks / reference.size, index=values.index)


def apply_segment_rank(
    X: pd.DataFrame,
    rank_reference: dict[str, Any],
) -> pd.Series:
    result = pd.Series(0.0, index=X.index)
    global_reference = rank_reference["global"]
    segment_references = rank_reference["segments"]

    for segment, index in X.groupby("CRM_PID_Value_Segment").groups.items():
        reference = segment_references.get(str(segment), global_reference)
        result.loc[index] = empirical_percentile(X.loc[index, "TotalRevenue"], reference)

    return result


def add_engineered_features(
    X: pd.DataFrame,
    include_extended_interactions: bool,
    rank_reference: dict[str, Any] | None,
) -> pd.DataFrame:
    active = X["Active_subscribers"]
    inactive = X["Not_Active_subscribers"]
    suspended = X["Suspended_subscribers"]
    total = X["Total_SUBs"]
    mobile = X["AvgMobileRevenue"]
    fixed = X["AvgFIXRevenue"]
    revenue = X["TotalRevenue"]
    arpu = X["ARPU"]

    X["active_rate"] = safe_divide(active, total).clip(0.0, 1.0)
    X["inactive_rate"] = safe_divide(inactive, total).clip(0.0, 1.0)
    X["suspended_rate"] = safe_divide(suspended, total).clip(0.0, 1.0)
    X["risk_score"] = safe_divide(inactive + suspended, total).clip(0.0, 1.0)
    X["mobile_revenue_ratio"] = safe_divide(mobile, revenue).clip(0.0, 1.0)
    X["fixed_revenue_ratio"] = safe_divide(fixed, revenue).clip(0.0, 1.0)
    X["revenue_per_subscriber"] = safe_divide(revenue, total)
    X["revenue_x_active_rate"] = np.log1p(revenue.clip(lower=0)) * X["active_rate"]

    if include_extended_interactions:
        if rank_reference is None:
            raise ValueError("rank_reference is required for extended interactions.")
        X["revenue_x_risk"] = np.log1p(arpu.clip(lower=0)) * X["risk_score"]
        X["inactive_x_fixed_ratio"] = X["inactive_rate"] * X["fixed_revenue_ratio"]
        X["suspended_x_mobile_ratio"] = X["suspended_rate"] * X["mobile_revenue_ratio"]
        X["arpu_per_active"] = arpu / (X["active_rate"] + EPSILON)
        X["total_rev_rank_by_segment"] = apply_segment_rank(X, rank_reference)

    return X


def apply_transform_mode(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    transform_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    transformed_cols: list[str] = []

    if transform_mode in {"log_revenue", "log_revenue_sqrt_subscribers"}:
        for col in REVENUE_COLS:
            X_train[col] = np.log1p(X_train[col].clip(lower=0))
            X_test[col] = np.log1p(X_test[col].clip(lower=0))
            transformed_cols.append(f"log1p:{col}")
    elif transform_mode != "none":
        raise ValueError(f"Unsupported transform mode: {transform_mode}")

    if transform_mode == "log_revenue_sqrt_subscribers":
        for col in SUBSCRIBER_SQRT_COLS:
            X_train[col] = np.sqrt(X_train[col].clip(lower=0))
            X_test[col] = np.sqrt(X_test[col].clip(lower=0))
            transformed_cols.append(f"sqrt:{col}")

    return X_train, X_test, transformed_cols


def encode_categoricals(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    encoders: dict[str, Any] = {}

    for col in cat_cols:
        train_values = X_train[col].fillna("Unknown").astype(str)
        test_values = X_test[col].fillna("Unknown").astype(str)

        class_values = pd.Index(
            train_values.dropna().unique().tolist() + ["Unknown", "Other"]
        ).drop_duplicates()
        encoder = LabelEncoder()
        encoder.fit(class_values.astype(str).to_numpy())

        test_values = test_values.where(test_values.isin(encoder.classes_), "Unknown")
        X_train[col] = encoder.transform(train_values)
        X_test[col] = encoder.transform(test_values)

        encoders[col] = {
            "classes": encoder.classes_.tolist(),
            "unseen_test_values_mapped_to": "Unknown",
        }

    return X_train, X_test, encoders


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scaler = StandardScaler()
    columns = X_train.columns.tolist()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=columns,
        index=X_test.index,
    )
    scaler_summary = {
        "scaled_columns": columns,
        "fit_scope": "train_only",
    }
    return X_train_scaled, X_test_scaled, scaler_summary


def target_distribution(y: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in y.value_counts().sort_index().items()}


def save_series(series: pd.Series, path: Path) -> None:
    pd.DataFrame({"CHURN": series.astype(int)}).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def preprocess_variant(
    df: pd.DataFrame,
    base_summary: dict[str, Any],
    config: VariantConfig,
    output_dir: Path,
) -> dict[str, Any]:
    feature_df = df.drop(columns=["CHURN", "PID"]).copy()
    y = df["CHURN"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    use_zip = config.zip_mode != "drop"
    X_train, X_test, imputation_values = fill_missing_values(X_train, X_test, use_zip)
    X_train, X_test, cat_cols, zip_metadata = apply_zip_mode(X_train, X_test, config)

    ka_reference = None
    if config.ka_mode == "research_full":
        ka_reference = fit_ka_reference(X_train, y_train)
    X_train = apply_ka_features(X_train, config, ka_reference, y_train)
    X_test = apply_ka_features(X_test, config, ka_reference)

    rank_reference = None
    if config.include_extended_interactions:
        rank_reference = fit_segment_rank_reference(X_train)

    X_train = add_engineered_features(
        X_train, config.include_extended_interactions, rank_reference
    )
    X_test = add_engineered_features(
        X_test, config.include_extended_interactions, rank_reference
    )

    X_train, X_test, transformed_cols = apply_transform_mode(
        X_train, X_test, config.transform_mode
    )

    feature_cols = cat_cols + BASE_NUMERIC_COLS + CORE_ENGINEERED_COLS
    if config.include_extended_interactions:
        feature_cols += EXTENDED_ENGINEERED_COLS
    if config.ka_mode == "abstract":
        feature_cols += KA_ABSTRACT_COLS
    elif config.ka_mode == "research_full":
        feature_cols += KA_RESEARCH_COLS
    if config.zip_mode == "drop":
        feature_cols = [col for col in feature_cols if col != "Billing_ZIP"]
    elif "Billing_ZIP" not in feature_cols:
        feature_cols.insert(len(BASE_CATEGORICAL_COLS), "Billing_ZIP")

    X_train = X_train[feature_cols].copy()
    X_test = X_test[feature_cols].copy()

    X_train, X_test, encoder_summary = encode_categoricals(X_train, X_test, cat_cols)
    X_train, X_test, scaler_summary = scale_features(X_train, X_test)

    smote = SVMSMOTE(random_state=RANDOM_STATE)
    resampled: Any = smote.fit_resample(X_train, y_train)
    X_train_resampled = pd.DataFrame(resampled[0], columns=X_train.columns)
    y_train_resampled = pd.Series(resampled[1], name="CHURN").astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False, encoding="utf-8-sig")
    X_test.to_csv(output_dir / "X_test.csv", index=False, encoding="utf-8-sig")
    save_series(y_train, output_dir / "y_train.csv")
    save_series(y_test, output_dir / "y_test.csv")
    X_train_resampled.to_csv(
        output_dir / "X_train_resampled.csv", index=False, encoding="utf-8-sig"
    )
    save_series(y_train_resampled, output_dir / "y_train_resampled.csv")

    feature_count_note = (
        "The paper describes 22 final features, but this CSV has 12 usable "
        "input columns after excluding PID and CHURN. With the 8 traced "
        "engineered features, the reproducible paper-aligned core is 20 "
        "features with ZIP, or 19 without ZIP."
    )

    summary = {
        **base_summary,
        "variant": config.name,
        "config": asdict(config),
        "feature_count": int(X_train.shape[1]),
        "feature_count_note": feature_count_note,
        "feature_columns": feature_cols,
        "categorical_columns_label_encoded": cat_cols,
        "imputation_values": imputation_values,
        "zip_metadata": zip_metadata,
        "ka_metadata": summarize_ka_reference(config, ka_reference),
        "transformed_columns": transformed_cols,
        "encoder_summary": encoder_summary,
        "scaler_summary": scaler_summary,
        "leakage_guardrails": [
            "PID duplicates are removed before the stratified split.",
            "CHURN and PID are excluded from feature matrices.",
            "Median imputation values are fit on X_train only.",
            "Label encoders are fit on X_train only; unseen test categories map to Unknown.",
            "Segment revenue ranks use empirical distributions fit on X_train only.",
            "KA target encodings, when enabled, are fit from X_train and y_train only.",
            "StandardScaler is fit on X_train only.",
            "SVMSMOTE is fit and applied to X_train only.",
        ],
        "X_train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "X_test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "X_train_resampled_shape": [
            int(X_train_resampled.shape[0]),
            int(X_train_resampled.shape[1]),
        ],
        "y_train_distribution": target_distribution(y_train),
        "y_test_distribution": target_distribution(y_test),
        "y_train_resampled_distribution": target_distribution(y_train_resampled),
    }

    (output_dir / "feature_columns.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    df, base_summary = load_base_dataframe(INPUT_FILE)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = []
    for config in VARIANTS:
        summaries.append(
            preprocess_variant(
                df=df,
                base_summary=base_summary,
                config=config,
                output_dir=OUTPUT_ROOT / config.name,
            )
        )

    summary_rows = []
    for item in summaries:
        summary_rows.append(
            {
                "variant": item["variant"],
                "zip_mode": item["config"]["zip_mode"],
                "ka_mode": item["config"]["ka_mode"],
                "transform_mode": item["config"]["transform_mode"],
                "include_extended_interactions": item["config"][
                    "include_extended_interactions"
                ],
                "feature_count": item["feature_count"],
                "X_train_rows": item["X_train_shape"][0],
                "X_test_rows": item["X_test_shape"][0],
                "X_train_resampled_rows": item["X_train_resampled_shape"][0],
                "y_train_positive": item["y_train_distribution"].get("1", 0),
                "y_test_positive": item["y_test_distribution"].get("1", 0),
                "y_train_resampled_positive": item[
                    "y_train_resampled_distribution"
                ].get("1", 0),
            }
        )

    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(
        OUTPUT_ROOT / "variant_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_ROOT / "summary_all.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
