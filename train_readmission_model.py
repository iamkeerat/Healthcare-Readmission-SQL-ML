#!/usr/bin/env python3
"""Train a Random Forest classifier for 30-day readmission.

Why this uses ``readmission_features`` instead of ``patient_features``
----------------------------------------------------------------------
``patient_features`` has one row per patient. Its "latest admission" is by
definition the last encounter in the dataset, so there is no subsequent
admission to observe; every ``readmitted_30d`` value would be 0. Training on
that table would produce a model that always predicts "no readmission."

``readmission_features`` has one row per admission and keeps the target on the
index admission. It uses the same patient history features (days since the last
admission and patient average blood pressure), plus current admission data. The
model is evaluated on patients that never appear in training.

Usage:
    python train_readmission_model.py
    python train_readmission_model.py --seed 7 --threshold 0.55
    python train_readmission_model.py --rebuild-features
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "readmitted_30d"

NUMERIC_FEATURES = [
    "age_at_admission",
    "comorbidity_count",
    "days_since_last_admission",
    "prior_admission_count",
    "prior_readmission_count",
    "length_of_stay_days",
    "systolic_bp",
    "diastolic_bp",
    "patient_avg_systolic_bp",
    "patient_avg_diastolic_bp",
    "lab_count",
    "abnormal_lab_rate",
    "creatinine",
    "bnp",
    "hba1c",
    "hemoglobin",
    "wbc",
    "lactate",
    "egfr",
    "is_first_admission",
]

CATEGORICAL_FEATURES = [
    "gender",
    "race",
    "ethnicity",
    "insurance_type",
    "admission_type",
    "department",
    "primary_diagnosis",
    "discharge_disposition",
]

ID_COLUMNS = ["admission_id", "patient_id"]
DATE_COLUMNS = ["admission_date", "discharge_date", "previous_admission_date"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest readmission model."
    )
    parser.add_argument("--db", type=Path, default=Path("healthcare.db"),
                        help="SQLite database path (default: healthcare.db)")
    parser.add_argument("--output-dir", type=Path, default=Path("model_artifacts"),
                        help="Directory for model, metrics, and predictions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--test-size", type=float, default=0.20,
                        help="Fraction of patients held out for testing (default: 0.20)")
    parser.add_argument("--trees", type=int, default=500,
                        help="Number of trees in the forest (default: 500)")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Maximum tree depth (default: 3)")
    parser.add_argument("--min-samples-leaf", type=int, default=5,
                        help="Minimum samples per leaf (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Probability threshold for a positive prediction (default: 0.50)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Patient-grouped CV folds on the training set (default: 5)")
    parser.add_argument("--rebuild-features", action="store_true",
                        help="Rebuild readmission_features from readmission_features.sql")
    args = parser.parse_args(argv)

    if not args.db.is_file():
        parser.error(f"database not found: {args.db}")
    if not 0.05 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.05 and 0.5")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def load_features(db_path: Path, rebuild: bool) -> pd.DataFrame:
    """Load the modeling table, building it from SQL when necessary."""
    conn = sqlite3.connect(db_path)
    try:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'readmission_features'"
        ).fetchone()

        if rebuild or not table_exists:
            sql_path = Path(__file__).with_name("readmission_features.sql")
            if not sql_path.is_file():
                raise FileNotFoundError(
                    f"readmission_features table is missing and {sql_path} was not found"
                )
            print(f"Building readmission_features from {sql_path}...")
            conn.executescript(sql_path.read_text(encoding="utf-8"))
            conn.commit()

        df = pd.read_sql_query(
            "SELECT * FROM readmission_features ORDER BY admission_date, admission_id",
            conn,
        )
    finally:
        conn.close()
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, leak-safe feature shaping and validate columns."""
    required = set(
        [TARGET, "patient_id", "admission_id"]
        + [feature for feature in NUMERIC_FEATURES if feature != "is_first_admission"]
        + CATEGORICAL_FEATURES
    )
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")

    if df[TARGET].isna().any():
        raise ValueError("Target contains null values")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary (0/1)")

    df = df.copy()
    # Missing gap means this is the patient's first admission, not an unknown
    # value. Preserve that information explicitly; RF cannot consume NaN.
    df["is_first_admission"] = df["days_since_last_admission"].isna().astype(int)
    df["days_since_last_admission"] = df["days_since_last_admission"].fillna(0)
    return df


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=1.0,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }
    prediction_frame = pd.DataFrame(
        {
            "admission_id": X_test["admission_id"].to_numpy(),
            "patient_id": X_test["patient_id"].to_numpy(),
            "y_true": y_test.to_numpy(),
            "y_probability": probabilities,
            "y_pred": predictions,
        }
    )
    return metrics, prediction_frame


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    df = load_features(args.db, rebuild=args.rebuild_features)
    df = prepare_features(df)

    print(f"Loaded {len(df):,} admissions from readmission_features")
    print(f"Target prevalence: {df[TARGET].mean():.1%} "
          f"({int(df[TARGET].sum()):,} positive admissions)")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ID_COLUMNS]
    y = df[TARGET].astype(int)
    groups = df["patient_id"]

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    train_patients = set(groups.iloc[train_idx])
    test_patients = set(groups.iloc[test_idx])
    if train_patients & test_patients:
        raise RuntimeError("Patient leakage detected between train and test")

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    print(
        f"Train: {len(X_train):,} admissions / {len(train_patients):,} patients, "
        f"{y_train.mean():.1%} positive"
    )
    print(
        f"Test:  {len(X_test):,} admissions / {len(test_patients):,} patients, "
        f"{y_test.mean():.1%} positive"
    )

    model = build_pipeline(args)

    # Patient-grouped CV gives a more stable estimate than one small holdout.
    cv = StratifiedGroupKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.seed
    )
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
            "f1": "f1",
        },
        n_jobs=1,
        error_score="raise",
    )
    cv_summary = {
        metric.replace("test_", ""): {
            "mean": float(cv_scores[metric].mean()),
            "std": float(cv_scores[metric].std()),
        }
        for metric in ("test_roc_auc", "test_pr_auc", "test_f1")
    }

    print("\nPatient-grouped cross-validation on the training set:")
    for metric, values in cv_summary.items():
        print(f"  {metric:8s} {values['mean']:.3f} +/- {values['std']:.3f}")

    # Baseline for context: always predicts the training prevalence.
    baseline = DummyClassifier(strategy="prior", random_state=args.seed)
    baseline.fit(X_train[["age_at_admission"]], y_train)
    baseline_prob = baseline.predict_proba(X_test[["age_at_admission"]])[:, 1]
    baseline_metrics = {
        "roc_auc": float(roc_auc_score(y_test, baseline_prob)),
        "pr_auc": float(average_precision_score(y_test, baseline_prob)),
    }

    model.fit(X_train, y_train)
    metrics, predictions = evaluate(model, X_test, y_test, args.threshold)
    metrics["cv"] = cv_summary
    metrics["baseline"] = baseline_metrics
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))
    metrics["n_train_patients"] = int(len(train_patients))
    metrics["n_test_patients"] = int(len(test_patients))

    print("\nHoldout metrics:")
    print(f"  ROC-AUC         {metrics['roc_auc']:.3f}")
    print(f"  PR-AUC          {metrics['pr_auc']:.3f}")
    print(f"  Accuracy        {metrics['accuracy']:.3f}")
    print(f"  Balanced acc.   {metrics['balanced_accuracy']:.3f}")
    print(f"  Precision       {metrics['precision']:.3f}")
    print(f"  Recall          {metrics['recall']:.3f}")
    print(f"  F1              {metrics['f1']:.3f}")
    print(f"  Brier score     {metrics['brier_score']:.3f}")
    print(f"  Baseline PR-AUC {baseline_metrics['pr_auc']:.3f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(f"  {metrics['confusion_matrix']}")

    # Feature importance at the transformed-feature level.
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("\nTop 15 features:")
    print(importance_df.head(15).to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "readmission_rf.joblib"
    metrics_path = args.output_dir / "metrics.json"
    importance_path = args.output_dir / "feature_importance.csv"
    predictions_path = args.output_dir / "test_predictions.csv"

    joblib.dump(
        {
            "model": model,
            "target": TARGET,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "threshold": args.threshold,
            "seed": args.seed,
        },
        model_path,
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    importance_df.to_csv(importance_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("\nSaved artifacts:")
    print(f"  {model_path}")
    print(f"  {metrics_path}")
    print(f"  {importance_path}")
    print(f"  {predictions_path}")
    print(
        "\nReminder: this is synthetic data and a demonstration model, not a "
        "clinical tool. The threshold is not calibrated for clinical use."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
