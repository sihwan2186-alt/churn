import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


INPUT_FILE = Path("Baza customer Telecom v2.csv")
OUTPUT_ROOT = Path("processed") / "column_split_datasets"
RANDOM_STATE = 42
TEST_SIZE = 0.2

EXCLUDED_MODEL_COLUMNS = ["PID", "KA_name"]
TARGET_COL = "CHURN"

CATEGORICAL_COLUMNS = [
    "CRM_PID_Value_Segment",
    "EffectiveSegment",
    "Billing_ZIP",
]

NUMERIC_COLUMNS = [
    "Active_subscribers",
    "Not_Active_subscribers",
    "Suspended_subscribers",
    "Total_SUBs",
    "AvgMobileRevenue",
    "AvgFIXRevenue",
    "TotalRevenue",
    "ARPU",
]

REQUIRED_COLUMNS = (
    EXCLUDED_MODEL_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET_COL]
)

COLUMN_KOREAN_NAMES = {
    "PID": "고객ID",
    "CRM_PID_Value_Segment": "CRM 고객가치 등급",
    "EffectiveSegment": "실질 비즈니스 세그먼트",
    "Billing_ZIP": "청구 우편번호",
    "KA_name": "담당 키 어카운트 매니저",
    "Active_subscribers": "활성 가입자 수",
    "Not_Active_subscribers": "비활성 가입자 수",
    "Suspended_subscribers": "정지 가입자 수",
    "Total_SUBs": "전체 가입자 수",
    "AvgMobileRevenue": "평균 모바일 매출",
    "AvgFIXRevenue": "평균 유선 매출",
    "TotalRevenue": "총 매출",
    "ARPU": "가입자당 평균 매출",
    "CHURN": "이탈 여부",
    "churn_binary": "이탈 여부 숫자값",
}

COLUMN_DESCRIPTIONS = {
    "PID": "고객을 식별하는 ID이며 모델 학습 데이터에서는 제외한다.",
    "CRM_PID_Value_Segment": "CRM 기준 고객가치 등급이다. Bronze, Silver, Gold, Platinum, SME 등으로 구분된다.",
    "EffectiveSegment": "실제 사업 규모/유형 세그먼트다. SOHO, VSE, SME 등이 포함된다.",
    "Billing_ZIP": "고객 청구지 우편번호다. 지역별 이탈 신호를 확인하는 데 사용한다.",
    "KA_name": "고객 담당 키 어카운트 매니저 이름이며 이번 작업의 모델 데이터에서는 제외한다.",
    "Active_subscribers": "현재 사용 중인 활성 가입자 수다.",
    "Not_Active_subscribers": "가입은 되어 있지만 활성 상태가 아닌 가입자 수다. 결측 자체도 의미가 있을 수 있다.",
    "Suspended_subscribers": "정지 상태 가입자 수다. 결측률이 매우 높아 결측 여부를 별도 신호로 본다.",
    "Total_SUBs": "고객이 보유한 전체 가입자 수다.",
    "AvgMobileRevenue": "평균 모바일 서비스 매출이다.",
    "AvgFIXRevenue": "평균 유선 서비스 매출이다. 번들/고착 효과 확인에 사용한다.",
    "TotalRevenue": "고객 단위 총 매출이다.",
    "ARPU": "가입자 1명당 평균 매출이다.",
    "CHURN": "고객 이탈 여부다. Yes는 이탈, No는 잔류를 뜻한다.",
    "churn_binary": "모델용 이탈 여부 숫자값이다. Yes는 1, No는 0이다.",
}

SUMMARY_KOREAN_HEADERS = {
    "column": "원본컬럼명",
    "column_ko": "한글컬럼명",
    "column_description": "간단설명",
    "semantic_type": "데이터유형",
    "group_type": "집계유형",
    "group": "값_또는_구간",
    "value": "값",
    "bin": "구간",
    "rows": "전체건수",
    "yes_count": "이탈Yes수",
    "no_count": "잔류No수",
    "churners": "이탈자수",
    "churn_rate": "이탈률",
    "yes_rate": "이탈률",
    "no_rate": "잔류율",
    "yes_rate_percent": "이탈률_퍼센트",
    "no_rate_percent": "잔류율_퍼센트",
    "missing_count": "결측수",
    "missing_rate": "결측률",
    "missing_rate_percent": "결측률_퍼센트",
    "non_missing": "비결측수",
    "unique_values": "고유값수",
    "overall_churn_rate": "전체이탈률",
    "recommended_preprocessing": "추천전처리",
    "recommended_model_family": "추천모델계열",
    "top_value": "최빈값",
    "top_value_count": "최빈값건수",
    "min": "최솟값",
    "max": "최댓값",
    "mean": "평균",
    "median": "중앙값",
    "skew": "왜도",
    "kurtosis": "첨도",
    "total_revenue_mean": "평균총매출",
    "arpu_mean": "평균ARPU",
    "active_rate_mean": "평균활성비율",
    "path": "파일경로",
}


def safe_filename(value: Any, prefix: str = "") -> str:
    text = "missing" if pd.isna(value) else str(value).strip()
    text = text or "blank"
    text = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", text)
    text = text.strip("._-") or "value"
    if prefix:
        text = f"{prefix}_{text}"
    return text[:120]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator.divide(denominator).replace([np.inf, -np.inf], np.nan)


def column_ko(col: str) -> str:
    return COLUMN_KOREAN_NAMES.get(col, col)


def column_description(col: str) -> str:
    return COLUMN_DESCRIPTIONS.get(col, "")


def write_korean_header_copy(df: pd.DataFrame, path: Path) -> None:
    readable = df.rename(columns={k: v for k, v in SUMMARY_KOREAN_HEADERS.items() if k in df.columns})
    readable.to_csv(path, index=False, encoding="utf-8-sig")


def require_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_raw_data(input_file: Path) -> pd.DataFrame:
    df = pd.read_csv(
        input_file,
        dtype={
            "PID": "string",
            "Billing_ZIP": "string",
            "CRM_PID_Value_Segment": "string",
            "EffectiveSegment": "string",
            "KA_name": "string",
            TARGET_COL: "string",
        },
    )
    df.columns = df.columns.str.strip()
    require_columns(df)

    for col in CATEGORICAL_COLUMNS + ["KA_name", TARGET_COL]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA})

    df["CRM_PID_Value_Segment"] = df["CRM_PID_Value_Segment"].replace(
        {"Sliver": "Silver"}
    )

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    target = df[TARGET_COL].astype("string").str.lower().map({"no": 0, "yes": 1})
    if target.isna().any():
        bad_values = df.loc[target.isna(), TARGET_COL].drop_duplicates().tolist()
        raise ValueError(f"Unexpected CHURN values: {bad_values}")

    df["churn_binary"] = target.astype(int)
    return df


def modeling_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in EXCLUDED_MODEL_COLUMNS and col != "churn_binary"
    ]


def write_base_dataset(df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    base_cols = modeling_columns(df)
    base_df = df[base_cols + ["churn_binary"]].copy()
    base_df.to_csv(output_root / "00_base_without_pid_ka.csv", index=False, encoding="utf-8-sig")
    readable_root = output_root / "07_korean_readable_summaries"
    readable_root.mkdir(parents=True, exist_ok=True)
    base_df.rename(columns=COLUMN_KOREAN_NAMES).to_csv(
        readable_root / "00_base_without_pid_ka_korean_columns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return base_df


def write_data_dictionary(df: pd.DataFrame, output_root: Path) -> None:
    readable_root = output_root / "07_korean_readable_summaries"
    readable_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for col in REQUIRED_COLUMNS + ["churn_binary"]:
        rows.append(
            {
                "원본컬럼명": col,
                "한글컬럼명": column_ko(col),
                "간단설명": column_description(col),
                "모델사용여부": "제외" if col in EXCLUDED_MODEL_COLUMNS else "사용",
                "데이터유형": (
                    "범주형"
                    if col in CATEGORICAL_COLUMNS or col in EXCLUDED_MODEL_COLUMNS or col == TARGET_COL
                    else "수치형"
                ),
                "결측수": int(df[col].isna().sum()) if col in df.columns else 0,
                "결측률_퍼센트": float(df[col].isna().mean() * 100) if col in df.columns else 0.0,
                "고유값수": int(df[col].nunique(dropna=True)) if col in df.columns else 2,
            }
        )

    dictionary = pd.DataFrame(rows)
    dictionary.to_csv(output_root / "00_data_dictionary_korean.csv", index=False, encoding="utf-8-sig")
    dictionary.to_csv(
        readable_root / "00_data_dictionary_korean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 데이터 컬럼 사전",
        "",
        "원본 CSV는 수정하지 않고, 새 결과물에만 한글명과 설명을 추가했다.",
        "",
        "| 원본컬럼명 | 한글컬럼명 | 간단설명 | 모델사용여부 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['원본컬럼명']} | {row['한글컬럼명']} | {row['간단설명']} | {row['모델사용여부']} |"
        )
    (output_root / "00_data_dictionary_korean.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (readable_root / "00_data_dictionary_korean.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_column_churn_pairs(df: pd.DataFrame, output_root: Path) -> list[dict[str, Any]]:
    pair_dir = output_root / "01_column_churn_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for col in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS:
        path = pair_dir / f"{safe_filename(col)}_churn.csv"
        pair = df[[col, TARGET_COL, "churn_binary"]].copy()
        pair.to_csv(path, index=False, encoding="utf-8-sig")
        records.append({"column": col, "rows": int(pair.shape[0]), "path": str(path)})
    return records


def write_category_value_subsets(df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    subset_root = output_root / "02_category_value_subsets"
    subset_root.mkdir(parents=True, exist_ok=True)
    model_cols = modeling_columns(df)
    summary_rows = []

    for col in CATEGORICAL_COLUMNS:
        col_dir = subset_root / safe_filename(col)
        col_dir.mkdir(parents=True, exist_ok=True)
        values = df[col].fillna("missing").astype(str).drop_duplicates().sort_values()

        for value in values:
            mask = df[col].fillna("missing").astype(str).eq(value)
            subset = df.loc[mask, model_cols + ["churn_binary"]].copy()
            if col == "Billing_ZIP":
                filename = safe_filename(value, prefix="zip") + "_churn.csv"
            else:
                filename = safe_filename(value) + "_churn.csv"
            path = col_dir / filename
            subset.to_csv(path, index=False, encoding="utf-8-sig")

            y = subset["churn_binary"]
            yes_count = int(y.sum())
            no_count = int(len(y) - yes_count)
            yes_rate = float(y.mean()) if len(y) else np.nan
            summary_rows.append(
                {
                    "column": col,
                    "column_ko": column_ko(col),
                    "column_description": column_description(col),
                    "value": value,
                    "rows": int(subset.shape[0]),
                    "yes_count": yes_count,
                    "no_count": no_count,
                    "churners": yes_count,
                    "churn_rate": yes_rate,
                    "yes_rate": yes_rate,
                    "no_rate": float(no_count / len(y)) if len(y) else np.nan,
                    "yes_rate_percent": yes_rate * 100 if len(y) else np.nan,
                    "no_rate_percent": (no_count / len(y)) * 100 if len(y) else np.nan,
                    "total_revenue_mean": float(subset["TotalRevenue"].mean()),
                    "arpu_mean": float(subset["ARPU"].mean()),
                    "active_rate_mean": float(
                        safe_divide(subset["Active_subscribers"], subset["Total_SUBs"]).mean()
                    ),
                    "path": str(path),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["column", "rows", "churn_rate"], ascending=[True, False, False]
    )
    summary.to_csv(
        output_root / "03_profiles" / "category_value_churn_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_root / "03_profiles" / "category_value_yes_no_rate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    readable_root = output_root / "07_korean_readable_summaries"
    readable_root.mkdir(parents=True, exist_ok=True)
    write_korean_header_copy(
        summary,
        readable_root / "category_value_yes_no_rate_summary_ko.csv",
    )
    return summary


def numeric_profile(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = numeric.dropna()
    if non_missing.empty:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
        }
    return {
        "min": float(non_missing.min()),
        "max": float(non_missing.max()),
        "mean": float(non_missing.mean()),
        "median": float(non_missing.median()),
        "skew": float(non_missing.skew()),
        "kurtosis": float(non_missing.kurtosis()),
    }


def preprocessing_note(col: str, df: pd.DataFrame) -> str:
    if col in CATEGORICAL_COLUMNS:
        unique_count = int(df[col].nunique(dropna=True))
        if col == "Billing_ZIP":
            return "high-cardinality categorical; use smoothed cross-fold target encoding or CatBoost native categorical"
        if unique_count <= 20:
            return "low-cardinality categorical; one-hot/CatBoost native categorical, plus interaction with segment where meaningful"
        return "categorical; prefer CatBoost native categorical or frequency/target encoding"

    if col in NUMERIC_COLUMNS:
        skew = abs(pd.to_numeric(df[col], errors="coerce").skew())
        if skew >= 2:
            return "numeric and strongly right-skewed; add missing flag, median/zero impute by meaning, log1p, and outlier flag"
        return "numeric; add missing flag if needed, median impute, scale for linear models"

    return "inspect manually"


def model_note(col: str) -> str:
    if col == "Billing_ZIP":
        return "CatBoost or tree ensemble with leakage-safe target encoding"
    if col in CATEGORICAL_COLUMNS:
        return "CatBoost, BalancedBagging, or LogisticRegression with one-hot as a baseline"
    if col in {"AvgMobileRevenue", "TotalRevenue", "Active_subscribers", "Total_SUBs"}:
        return "tree ensemble after log1p/outlier flags; linear model only as weak baseline"
    if col in {"Not_Active_subscribers", "Suspended_subscribers", "ARPU", "AvgFIXRevenue"}:
        return "tree ensemble with missing indicator and log1p; keep zero/missing meaning separate"
    return "compare simple linear baseline against tree ensemble"


def write_column_profiles(df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    profile_rows = []
    y = df["churn_binary"]
    for col in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS:
        row: dict[str, Any] = {
            "column": col,
            "column_ko": column_ko(col),
            "column_description": column_description(col),
            "semantic_type": "categorical" if col in CATEGORICAL_COLUMNS else "numeric",
            "rows": int(df.shape[0]),
            "non_missing": int(df[col].notna().sum()),
            "missing": int(df[col].isna().sum()),
            "missing_rate": float(df[col].isna().mean()),
            "unique_values": int(df[col].nunique(dropna=True)),
            "overall_churn_rate": float(y.mean()),
            "recommended_preprocessing": preprocessing_note(col, df),
            "recommended_model_family": model_note(col),
        }

        if col in NUMERIC_COLUMNS:
            row.update(numeric_profile(df[col]))
        else:
            top_value = df[col].value_counts(dropna=False).index[0]
            row.update(
                {
                    "top_value": "missing" if pd.isna(top_value) else str(top_value),
                    "top_value_count": int(df[col].value_counts(dropna=False).iloc[0]),
                    "min": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "median": np.nan,
                    "skew": np.nan,
                    "kurtosis": np.nan,
                }
            )

        profile_rows.append(row)

    profile = pd.DataFrame(profile_rows)
    profile.to_csv(output_root / "03_profiles" / "column_profile.csv", index=False, encoding="utf-8-sig")
    readable_root = output_root / "07_korean_readable_summaries"
    readable_root.mkdir(parents=True, exist_ok=True)
    write_korean_header_copy(profile, readable_root / "column_profile_ko.csv")
    return profile


def write_numeric_bins(df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        work = pd.DataFrame({col: numeric, "churn_binary": df["churn_binary"]})
        work["missing"] = work[col].isna()
        missing = work[work["missing"]]
        if not missing.empty:
            yes_count = int(missing["churn_binary"].sum())
            no_count = int(missing.shape[0] - yes_count)
            yes_rate = float(missing["churn_binary"].mean())
            rows.append(
                {
                    "column": col,
                    "column_ko": column_ko(col),
                    "column_description": column_description(col),
                    "bin": "missing",
                    "rows": int(missing.shape[0]),
                    "yes_count": yes_count,
                    "no_count": no_count,
                    "churners": yes_count,
                    "churn_rate": yes_rate,
                    "yes_rate": yes_rate,
                    "no_rate": float(no_count / missing.shape[0]),
                    "yes_rate_percent": yes_rate * 100,
                    "no_rate_percent": (no_count / missing.shape[0]) * 100,
                    "min": np.nan,
                    "max": np.nan,
                }
            )

        non_missing = work.loc[~work["missing"], [col, "churn_binary"]].copy()
        if non_missing[col].nunique() <= 1:
            continue
        bins = pd.qcut(non_missing[col], q=5, duplicates="drop")
        for bin_label, group in non_missing.groupby(bins, observed=False):
            yes_count = int(group["churn_binary"].sum())
            no_count = int(group.shape[0] - yes_count)
            yes_rate = float(group["churn_binary"].mean())
            rows.append(
                {
                    "column": col,
                    "column_ko": column_ko(col),
                    "column_description": column_description(col),
                    "bin": str(bin_label),
                    "rows": int(group.shape[0]),
                    "yes_count": yes_count,
                    "no_count": no_count,
                    "churners": yes_count,
                    "churn_rate": yes_rate,
                    "yes_rate": yes_rate,
                    "no_rate": float(no_count / group.shape[0]),
                    "yes_rate_percent": yes_rate * 100,
                    "no_rate_percent": (no_count / group.shape[0]) * 100,
                    "min": float(group[col].min()),
                    "max": float(group[col].max()),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "03_profiles" / "numeric_bins_churn_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_root / "03_profiles" / "numeric_bins_yes_no_rate_summary.csv", index=False, encoding="utf-8-sig")
    readable_root = output_root / "07_korean_readable_summaries"
    readable_root.mkdir(parents=True, exist_ok=True)
    write_korean_header_copy(
        summary,
        readable_root / "numeric_bins_yes_no_rate_summary_ko.csv",
    )
    return summary


def write_yes_no_rate_summaries(
    df: pd.DataFrame,
    category_summary: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    output_root: Path,
) -> None:
    rate_root = output_root / "06_yes_no_rate_by_column"
    rate_root.mkdir(parents=True, exist_ok=True)
    readable_root = output_root / "07_korean_readable_summaries"
    readable_root.mkdir(parents=True, exist_ok=True)
    readable_rate_root = readable_root / "yes_no_rate_by_column_ko"
    readable_rate_root.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    for col in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS:
        y = df["churn_binary"]
        yes_count = int(y.sum())
        no_count = int(len(y) - yes_count)
        yes_rate = float(y.mean())
        overall_rows.append(
            {
                "column": col,
                "column_ko": column_ko(col),
                "column_description": column_description(col),
                "semantic_type": "categorical" if col in CATEGORICAL_COLUMNS else "numeric",
                "rows": int(len(y)),
                "yes_count": yes_count,
                "no_count": no_count,
                "yes_rate": yes_rate,
                "no_rate": float(no_count / len(y)),
                "yes_rate_percent": yes_rate * 100,
                "no_rate_percent": (no_count / len(y)) * 100,
                "missing_count": int(df[col].isna().sum()),
                "missing_rate_percent": float(df[col].isna().mean() * 100),
            }
        )

    overall = pd.DataFrame(overall_rows)
    overall.to_csv(
        output_root / "03_profiles" / "column_yes_no_rate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_korean_header_copy(
        overall,
        readable_root / "column_yes_no_rate_summary_ko.csv",
    )

    combined_frames = []
    if not category_summary.empty:
        category_values = category_summary.copy()
        category_values["group_type"] = "category_value"
        category_values["group"] = category_values["value"]
        combined_frames.append(
            category_values[
                [
                    "column",
                    "column_ko",
                    "column_description",
                    "group_type",
                    "group",
                    "rows",
                    "yes_count",
                    "no_count",
                    "yes_rate",
                    "no_rate",
                    "yes_rate_percent",
                    "no_rate_percent",
                    "path",
                ]
            ]
        )

    if not numeric_summary.empty:
        numeric_values = numeric_summary.copy()
        numeric_values["group_type"] = "numeric_bin"
        numeric_values["group"] = numeric_values["bin"]
        numeric_values["path"] = ""
        combined_frames.append(
            numeric_values[
                [
                    "column",
                    "column_ko",
                    "column_description",
                    "group_type",
                    "group",
                    "rows",
                    "yes_count",
                    "no_count",
                    "yes_rate",
                    "no_rate",
                    "yes_rate_percent",
                    "no_rate_percent",
                    "path",
                ]
            ]
        )

    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        combined = combined.sort_values(
            ["column", "yes_rate", "rows"], ascending=[True, False, False]
        )
        combined.to_csv(
            output_root / "03_profiles" / "all_column_value_yes_no_rate_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        write_korean_header_copy(
            combined,
            readable_root / "all_column_value_yes_no_rate_summary_ko.csv",
        )

    export_cols = [
        "column_ko",
        "column_description",
        "group",
        "rows",
        "yes_count",
        "no_count",
        "yes_rate",
        "no_rate",
        "yes_rate_percent",
        "no_rate_percent",
    ]
    if not category_summary.empty:
        for col in CATEGORICAL_COLUMNS:
            per_col = category_summary.loc[category_summary["column"].eq(col)].copy()
            if per_col.empty:
                continue
            per_col["group"] = per_col["value"]
            per_col = per_col.sort_values(["yes_rate", "rows"], ascending=[False, False])
            per_col_output = per_col[export_cols + ["path"]]
            per_col_output.to_csv(
                rate_root / f"{safe_filename(col)}_yes_no_rate.csv",
                index=False,
                encoding="utf-8-sig",
            )
            write_korean_header_copy(
                per_col_output,
                readable_rate_root / f"{safe_filename(col)}_yes_no_rate_ko.csv",
            )

    if not numeric_summary.empty:
        for col in NUMERIC_COLUMNS:
            per_col = numeric_summary.loc[numeric_summary["column"].eq(col)].copy()
            if per_col.empty:
                continue
            per_col["group"] = per_col["bin"]
            per_col = per_col.sort_values(["yes_rate", "rows"], ascending=[False, False])
            per_col_output = per_col[export_cols + ["min", "max"]]
            per_col_output.to_csv(
                rate_root / f"{safe_filename(col)}_yes_no_rate.csv",
                index=False,
                encoding="utf-8-sig",
            )
            write_korean_header_copy(
                per_col_output,
                readable_rate_root / f"{safe_filename(col)}_yes_no_rate_ko.csv",
            )


def write_column_preprocessed(df: pd.DataFrame, output_root: Path) -> list[dict[str, Any]]:
    preprocessed_root = output_root / "04_column_preprocessed"
    preprocessed_root.mkdir(parents=True, exist_ok=True)
    records = []

    for col in CATEGORICAL_COLUMNS:
        values = df[col].fillna("missing").astype(str)
        frequency = values.value_counts(normalize=True).to_dict()
        categories = {value: idx for idx, value in enumerate(sorted(values.unique()))}
        out = pd.DataFrame(
            {
                col: df[col],
                f"{col}_missing": df[col].isna().astype(int),
                f"{col}_frequency": values.map(frequency).astype(float),
                f"{col}_label": values.map(categories).astype(int),
                TARGET_COL: df[TARGET_COL],
                "churn_binary": df["churn_binary"],
            }
        )
        path = preprocessed_root / f"{safe_filename(col)}_preprocessed.csv"
        out.to_csv(path, index=False, encoding="utf-8-sig")
        records.append({"column": col, "path": str(path)})

    for col in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        median = float(numeric.median()) if numeric.notna().any() else 0.0
        imputed = numeric.fillna(median)
        zscore_std = float(imputed.std(ddof=0))
        if zscore_std == 0:
            zscore = pd.Series(0.0, index=df.index)
        else:
            zscore = (imputed - float(imputed.mean())) / zscore_std

        q1 = imputed.quantile(0.25)
        q3 = imputed.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        out = pd.DataFrame(
            {
                col: df[col],
                f"{col}_missing": numeric.isna().astype(int),
                f"{col}_imputed_median": imputed,
                f"{col}_log1p": np.log1p(imputed.clip(lower=0)),
                f"{col}_zscore": zscore,
                f"{col}_iqr_outlier": ((imputed < lower) | (imputed > upper)).astype(int),
                TARGET_COL: df[TARGET_COL],
                "churn_binary": df["churn_binary"],
            }
        )
        path = preprocessed_root / f"{safe_filename(col)}_preprocessed.csv"
        out.to_csv(path, index=False, encoding="utf-8-sig")
        records.append({"column": col, "path": str(path)})

    return records


def build_preprocessor(col: str) -> ColumnTransformer:
    if col in CATEGORICAL_COLUMNS:
        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer([("cat", cat_pipe, [col])])

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer([("num", numeric_pipe, [col])])


def build_ordinal_preprocessor(col: str) -> ColumnTransformer:
    if col in CATEGORICAL_COLUMNS:
        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "ordinal",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
        return ColumnTransformer([("cat", cat_pipe, [col])])
    return build_preprocessor(col)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float]:
    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = np.nan
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            metrics["pr_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
    return metrics


def screen_single_column_models(df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows = []
    y = df["churn_binary"].astype(int)

    for col in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS:
        X = df[[col]].copy()
        if col in CATEGORICAL_COLUMNS:
            X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        candidates = [
            (
                "Dummy_stratified",
                Pipeline(
                    steps=[
                        ("preprocess", build_preprocessor(col)),
                        (
                            "model",
                            DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
                        ),
                    ]
                ),
            ),
            (
                "LogisticRegression_balanced",
                Pipeline(
                    steps=[
                        ("preprocess", build_preprocessor(col)),
                        (
                            "model",
                            LogisticRegression(
                                class_weight="balanced",
                                max_iter=1000,
                                solver="liblinear",
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "DecisionTree_balanced",
                Pipeline(
                    steps=[
                        ("preprocess", build_ordinal_preprocessor(col)),
                        (
                            "model",
                            DecisionTreeClassifier(
                                max_depth=4,
                                min_samples_leaf=25,
                                class_weight="balanced",
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "BalancedBagging_tree",
                Pipeline(
                    steps=[
                        ("preprocess", build_ordinal_preprocessor(col)),
                        (
                            "model",
                            BalancedBaggingClassifier(
                                estimator=DecisionTreeClassifier(
                                    max_depth=3,
                                    min_samples_leaf=20,
                                    random_state=RANDOM_STATE,
                                ),
                                n_estimators=15,
                                random_state=RANDOM_STATE,
                                n_jobs=1,
                            ),
                        ),
                    ]
                ),
            ),
        ]

        for model_name, pipeline in candidates:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            if hasattr(pipeline, "predict_proba"):
                y_score = pipeline.predict_proba(X_test)[:, 1]
            else:
                y_score = None
            metrics = evaluate_predictions(y_test, y_pred, y_score)
            rows.append(
                {
                    "column": col,
                    "semantic_type": "categorical" if col in CATEGORICAL_COLUMNS else "numeric",
                    "model": model_name,
                    **metrics,
                }
            )

    result = pd.DataFrame(rows).sort_values(
        ["column", "f1", "balanced_accuracy"], ascending=[True, False, False]
    )
    result.to_csv(output_root / "05_single_column_model_screening.csv", index=False, encoding="utf-8-sig")

    best = (
        result.sort_values(["column", "f1", "balanced_accuracy"], ascending=[True, False, False])
        .groupby("column", as_index=False)
        .head(1)
        .sort_values("f1", ascending=False)
    )
    best.to_csv(output_root / "05_single_column_best_models.csv", index=False, encoding="utf-8-sig")
    return result


def write_performance_limit_note(df: pd.DataFrame, output_root: Path) -> None:
    best_path = output_root / "05_single_column_best_models.csv"
    best = pd.read_csv(best_path) if best_path.exists() else pd.DataFrame()

    lines = [
        "# Column Split Data Understanding Notes",
        "",
        "## Scope",
        "",
        "- Original CSV is not modified.",
        "- Modeling/split datasets exclude `PID` and `KA_name`.",
        "- Each original feature is exported as a `feature + CHURN` CSV.",
        "- Category values such as `Bronze` and `SOHO` are exported as separate churn datasets.",
        "",
        "## Modeling Interpretation",
        "",
        "Single-column screening is a quick limit check, not the final model. If one feature alone cannot get high F1/PR-AUC, the final model needs interaction features, leakage-safe target encoding, and imbalance-aware threshold tuning.",
        "",
    ]

    if not best.empty:
        top = best.sort_values("f1", ascending=False).head(8)
        lines += [
            "## Best Single-Column Signals",
            "",
            "| column | best model | F1 | recall | precision | PR-AUC |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for _, row in top.iterrows():
            lines.append(
                f"| {row['column']} | {row['model']} | {row['f1']:.4f} | {row['recall']:.4f} | {row['precision']:.4f} | {row['pr_auc']:.4f} |"
            )
        lines.append("")

    lines += [
        "## Next Paper Comparison Checkpoints",
        "",
        "- Compare paper preprocessing against this split-first setup.",
        "- Check whether paper used `PID` or `KA_name`; this workflow excludes both from model datasets.",
        "- Verify how the paper handled `Billing_ZIP`, missing values, skewed subscriber counts, and class imbalance.",
        "- For target/risk encoding, compute rates inside train folds only to avoid target leakage.",
        "",
    ]
    (output_root / "COLUMN_SPLIT_README.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    df: pd.DataFrame,
    output_root: Path,
    pair_records: list[dict[str, Any]],
    preprocessed_records: list[dict[str, Any]],
    screening: pd.DataFrame | None,
) -> None:
    manifest = {
        "input_file": str(INPUT_FILE),
        "output_root": str(output_root),
        "original_shape": [int(df.shape[0]), int(df.shape[1])],
        "excluded_model_columns": EXCLUDED_MODEL_COLUMNS,
        "target_column": TARGET_COL,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "target_distribution": {
            str(k): int(v) for k, v in df[TARGET_COL].value_counts(dropna=False).items()
        },
        "column_pair_files": pair_records,
        "preprocessed_column_files": preprocessed_records,
        "single_column_model_screening": screening is not None,
        "korean_readable_outputs": {
            "data_dictionary": str(output_root / "00_data_dictionary_korean.csv"),
            "readable_root": str(output_root / "07_korean_readable_summaries"),
            "original_csv_modified": False,
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-column and per-category churn datasets without PID/KA_name."
    )
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--skip-model-screening",
        action="store_true",
        help="Only create CSV splits/profiles, without fitting single-column screening models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output
    (output_root / "03_profiles").mkdir(parents=True, exist_ok=True)

    df = load_raw_data(args.input)
    write_base_dataset(df, output_root)
    write_data_dictionary(df, output_root)
    pair_records = write_column_churn_pairs(df, output_root)
    write_column_profiles(df, output_root)
    category_summary = write_category_value_subsets(df, output_root)
    numeric_summary = write_numeric_bins(df, output_root)
    write_yes_no_rate_summaries(df, category_summary, numeric_summary, output_root)
    preprocessed_records = write_column_preprocessed(df, output_root)

    screening = None
    if not args.skip_model_screening:
        screening = screen_single_column_models(df, output_root)

    write_performance_limit_note(df, output_root)
    write_manifest(df, output_root, pair_records, preprocessed_records, screening)

    print(f"Created column split datasets under: {output_root}")
    print(f"Rows: {df.shape[0]}, columns after excluding PID/KA_name: {len(modeling_columns(df))}")
    print(f"Column pair files: {len(pair_records)}")
    print(f"Preprocessed column files: {len(preprocessed_records)}")
    if screening is not None:
        print("Single-column model screening: enabled")


if __name__ == "__main__":
    main()
