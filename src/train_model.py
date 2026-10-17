"""Tune candidate classifiers using training-only CV and evaluate only the selected model on test data."""
from pathlib import Path
import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
TARGET = "Engine Condition"
RANDOM_STATE = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="artifacts/splits")
    parser.add_argument("--output-dir", default="artifacts/model")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(split_dir / "train.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET].astype(int)
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET].astype(int)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grids = {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {"max_depth": [8, None], "min_samples_leaf": [1], "criterion": ["gini"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced"),
            {"n_estimators": [20, 40], "max_depth": [None], "min_samples_leaf": [1]},
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [20, 40], "learning_rate": [1.0]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [20, 40], "learning_rate": [0.1], "max_depth": [2]},
        ),
    }

    experiment_rows = []
    best_estimators = {}
    best_scores = {}
    for model_name, (estimator, param_grid) in grids.items():
        search = GridSearchCV(
            estimator=estimator, param_grid=param_grid, scoring="f1_weighted",
            cv=cv, n_jobs=1, refit=True, return_train_score=True
        )
        started = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - started
        best_estimators[model_name] = search.best_estimator_
        best_scores[model_name] = float(search.best_score_)
        results = pd.DataFrame(search.cv_results_)
        for _, row in results.iterrows():
            experiment_rows.append({
                "model": model_name,
                "params": json.dumps(row["params"], sort_keys=True),
                "mean_cv_f1_weighted": float(row["mean_test_score"]),
                "std_cv_f1_weighted": float(row["std_test_score"]),
                "mean_train_f1_weighted": float(row["mean_train_score"]),
                "rank": int(row["rank_test_score"]),
                "fit_time_seconds_mean": float(row["mean_fit_time"]),
                "search_elapsed_seconds": round(elapsed, 3),
                "is_best_for_model": bool(row["rank_test_score"] == 1),
            })

    # Select using CV on training data only. Do not use test scores for model selection.
    selected_name = sorted(best_scores, key=lambda name: (-best_scores[name], name))[0]
    selected_model = best_estimators[selected_name]
    selected_pred = selected_model.predict(X_test)
    selected_metrics = {
        "model": selected_name,
        "accuracy": float(accuracy_score(y_test, selected_pred)),
        "precision_weighted": float(precision_score(y_test, selected_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, selected_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, selected_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, selected_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, selected_model.predict_proba(X_test)[:, 1]))
                    if hasattr(selected_model, "predict_proba") else None,
    }
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = {
        "model": "Dummy baseline",
        "accuracy": float(accuracy_score(y_test, baseline_pred)),
        "precision_weighted": float(precision_score(y_test, baseline_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, baseline_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, baseline_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, baseline_pred, average="macro", zero_division=0)),
        "roc_auc": None,
    }

    pd.DataFrame(experiment_rows).sort_values(
        ["mean_cv_f1_weighted", "std_cv_f1_weighted"], ascending=[False, True]
    ).to_csv(output_dir / "experiment_tracking.csv", index=False)
    cv_summary = pd.DataFrame([
        {"model": name, "best_cv_weighted_f1": score,
         "best_parameters": json.dumps(best_estimators[name].get_params(), default=str, sort_keys=True)}
        for name, score in best_scores.items()
    ]).sort_values("best_cv_weighted_f1", ascending=False)
    cv_summary.to_csv(output_dir / "cv_model_comparison.csv", index=False)
    pd.DataFrame([selected_metrics, baseline_metrics]).to_csv(output_dir / "model_evaluation.csv", index=False)
    joblib.dump(selected_model, output_dir / "best_predictive_maintenance_model.joblib")
    metadata = {
        "project": "Predictive Maintenance",
        "target": TARGET,
        "features": list(X_train.columns),
        "selected_model": selected_name,
        "selection_metric": "mean stratified 3-fold cross-validation weighted F1 on training data",
        "mean_cv_weighted_f1": best_scores[selected_name],
        "best_parameters": selected_model.get_params(),
        "test_metrics_selected_model_only": selected_metrics,
        "test_metrics_dummy_baseline": baseline_metrics,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "important_caveat": "Confirm target label semantics before operational use. Test metrics are not used to select the model."
    }
    (output_dir / "best_model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_test, selected_pred, zero_division=0)
    )
    print(f"Selected model (training-only CV): {selected_name}")
    print(f"Mean CV weighted F1: {best_scores[selected_name]:.4f}")
    print("Final held-out test metrics for selected model:")
    print(json.dumps(selected_metrics, indent=2))
    print(f"Artifacts written to {output_dir}")

if __name__ == "__main__":
    main()
