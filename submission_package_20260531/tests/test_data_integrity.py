import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Baza customer Telecom v2.csv"
REQUIRED_COLUMNS = [
    "PID",
    "CHURN",
    "TotalRevenue",
    "Active_subscribers",
    "Total_SUBs",
    "ARPU",
    "Billing_ZIP",
]


def load_data():
    return pd.read_csv(DATA_PATH)


def test_file_exists():
    """데이터 파일이 존재하는지 확인"""
    assert DATA_PATH.exists(), f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}"


def test_data_schema():
    """기본 스키마 및 필수 컬럼 확인"""
    df = load_data()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"필수 컬럼 누락: {col}"


def test_target_values():
    """CHURN 컬럼의 값이 예상된 범위(Yes/No)인지 확인"""
    df = load_data()
    unique_values = df["CHURN"].unique()
    assert set(unique_values).issubset({"Yes", "No"}), (
        f"CHURN 컬럼에 예상치 못한 값이 포함됨: {unique_values}"
    )


def test_no_empty_dataset():
    """데이터셋이 비어있지 않은지 확인"""
    df = load_data()
    assert len(df) > 0, "데이터셋이 비어있습니다."


def run_all_checks():
    test_file_exists()
    test_data_schema()
    test_target_values()
    test_no_empty_dataset()


if __name__ == "__main__":
    print("데이터 무결성 검사를 시작합니다...")
    try:
        run_all_checks()
        print("모든 검사를 통과했습니다.")
    except Exception as e:
        print(f"검사 실패: {e}")
        sys.exit(1)
