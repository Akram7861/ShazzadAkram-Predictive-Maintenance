"""Validate the Predictive Maintenance dataset before processing."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/engine_data.csv")
EXPECTED_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
    "Engine Condition",
]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected dataset at {DATA_PATH}. Add engine_data.csv to data/.")
    df = pd.read_csv(DATA_PATH)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if df["Engine Condition"].isna().any():
        raise ValueError("Target column 'Engine Condition' contains missing values.")
    if df[EXPECTED_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains missing values; review before training.")
    if not set(df["Engine Condition"].unique()).issubset({0, 1}):
        raise ValueError("Target 'Engine Condition' must contain binary labels 0 and 1.")
    print(f"Validation passed: {len(df):,} rows, {len(df.columns)} columns.")
    print(f"Duplicate rows: {int(df.duplicated().sum()):,}")
    print(f"Target distribution: {df['Engine Condition'].value_counts().sort_index().to_dict()}")
    if extra_columns:
        print(f"Note: extra columns found: {extra_columns}")

if __name__ == "__main__":
    main()
