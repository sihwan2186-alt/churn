from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FEATURES = ["TotalRevenue", "Active_subscribers", "ARPU"]
DEFAULT_REFERENCE = Path("Baza customer Telecom v2.csv")


def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """PSI(Population Stability Index)를 계산한다."""
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()

    if expected.empty or actual.empty:
        raise ValueError("PSI 계산에 사용할 유효한 숫자 값이 없습니다.")

    if expected.nunique() == 1:
        lower = float(expected.iloc[0]) - 0.5
        upper = float(expected.iloc[0]) + 0.5
        bin_edges = np.linspace(lower, upper, buckets + 1)
    else:
        quantiles = np.linspace(0, 1, buckets + 1)
        bin_edges = np.unique(np.quantile(expected, quantiles))
        if len(bin_edges) < 2:
            bin_edges = np.linspace(float(expected.min()), float(expected.max()), buckets + 1)

    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual.clip(bin_edges[0], bin_edges[-1]), bins=bin_edges)[0]

    expected_percents = np.clip(expected_counts / len(expected), 0.0001, 1)
    actual_percents = np.clip(actual_counts / len(actual), 0.0001, 1)

    return float(np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents)))


def drift_status(psi: float) -> str:
    if psi < 0.1:
        return "OK"
    if psi < 0.2:
        return "Warning (Minor Drift)"
    return "Critical (Action Required)"


def run_drift_check(
    reference_path: Path,
    current_path: Path,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """주요 변수의 분포가 기준 데이터와 달라졌는지 점검한다."""
    features_to_check = features or DEFAULT_FEATURES

    if not reference_path.exists():
        return {"ok": False, "error": f"Reference file not found: {reference_path}"}
    if not current_path.exists():
        return {"ok": False, "error": f"Current file not found: {current_path}"}

    df_ref = pd.read_csv(reference_path)
    df_cur = pd.read_csv(current_path)

    results = []
    missing_features = []
    for col in features_to_check:
        if col not in df_ref.columns or col not in df_cur.columns:
            missing_features.append(col)
            continue

        try:
            psi = calculate_psi(df_ref[col], df_cur[col])
            results.append({"feature": col, "psi": psi, "status": drift_status(psi)})
        except ValueError as exc:
            results.append({"feature": col, "psi": None, "status": "Skipped", "error": str(exc)})

    return {
        "ok": bool(results),
        "reference": str(reference_path),
        "current": str(current_path),
        "checked_features": [row["feature"] for row in results],
        "missing_features": missing_features,
        "drift_summary": results,
        "threshold_info": "PSI < 0.1: Stable | 0.1-0.2: Monitor | > 0.2: Retrain",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ChurnRadar input data drift with PSI.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Baseline CSV path")
    parser.add_argument("--current", type=Path, default=DEFAULT_REFERENCE, help="New CSV path to check")
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Numeric feature names to compare",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_drift_check(args.reference, args.current, args.features)
    output = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")

    print(output)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
