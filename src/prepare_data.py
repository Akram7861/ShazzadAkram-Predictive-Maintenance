"""Create a reproducible, stratified train/test split for the training workflow."""
from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Engine Condition"
RANDOM_STATE = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/engine_data.csv")
    parser.add_argument("--output-dir", default="artifacts/splits")
    args = parser.parse_args()

    source = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source).drop_duplicates().reset_index(drop=True)
    if TARGET not in df.columns:
        raise ValueError(f"Target column not found: {TARGET}")
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    train = X_train.copy()
    train[TARGET] = y_train.values
    test = X_test.copy()
    test[TARGET] = y_test.values
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(f"Saved {len(train):,} train rows and {len(test):,} test rows to {out_dir}")

if __name__ == "__main__":
    main()
