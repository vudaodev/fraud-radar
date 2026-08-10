"""Train the fraud-detection pipeline and serialise it to model/artifacts/.

Extracted from `model/notebooks/02_train_and_threshold.ipynb`. The notebook keeps
the exploration (precision-recall plots, the threshold sweep, the write-up); this
script keeps the reproducible part: load raw data, fit the pipeline, report the
metrics at the locked operating point, save the artifact.

The transform is imported from `common.transforms` — the same module the API
imports — so training and inference provably share one definition. The pickle
stores it by reference, which is why it must live in an importable module rather
than being defined here or as a lambda.

Preprocessing lives *inside* the pipeline: raw Time/Amount/V1-V28 go in, and the
ColumnTransformer rewrites the matrix before the forest sees it. Nothing is
transformed in pandas beforehand.

How to run
----------
From the repo root, with data/creditcard.csv present (gitignored, from Kaggle):

    uv run python -m model.train

Module form, not `python model/train.py`: the latter puts model/ on sys.path
instead of the repo root, so `import common.transforms` would fail.

Overwrites model/artifacts/isolation_forest.pkl. With random_state fixed
throughout, a rerun on the same data and library versions reproduces it.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from common.anomaly import FRAUD_THRESHOLD, anomaly_scores, flagged
from common.transforms import secs_to_hour_of_day


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "creditcard.csv"
ARTIFACT_PATH = REPO_ROOT / "model" / "artifacts" / "isolation_forest.pkl"

# Column order is baked into the fitted ColumnTransformer — changing it
# invalidates the artifact.
RAW_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
TARGET_COLUMN = "Class"

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 250
MAX_SAMPLES = 1024
CONTAMINATION = 0.0017

# The operating point itself lives in common/anomaly.py as FRAUD_THRESHOLD, next
# to the convention it depends on. Training reports metrics at that point; it
# does not re-select it — that choice was made by hand in the notebook.


def load_raw_frames() -> tuple[pd.DataFrame, pd.Series]:
    """Load the source CSV and return the raw feature matrix and labels."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Source data not found: {DATA_PATH}")

    source = pd.read_csv(DATA_PATH)
    missing_columns = set(RAW_COLUMNS + [TARGET_COLUMN]).difference(source.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Source data is missing required columns: {missing}")

    return source[RAW_COLUMNS], source[TARGET_COLUMN]


def build_pipeline() -> Pipeline:
    """Build the unfitted preprocessing + IsolationForest pipeline."""
    # Named module-level functions, not lambdas: the pickle stores them by
    # reference and a lambda has no importable path to resolve at load time.
    time_transformer = FunctionTransformer(
        secs_to_hour_of_day, feature_names_out="one-to-one"
    )
    amount_transformer = FunctionTransformer(np.log1p, feature_names_out="one-to-one")

    column_transformations = ColumnTransformer(
        transformers=(
            ("time_to_hour_of_day", time_transformer, ["Time"]),
            ("amount_to_log", amount_transformer, ["Amount"]),
        ),
        remainder="passthrough",
    )

    return Pipeline(
        [
            ("column_transformations", column_transformations),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=N_ESTIMATORS,
                    max_samples=MAX_SAMPLES,
                    contamination=CONTAMINATION,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Score the held-out set and return ranking metrics plus metrics at the threshold."""
    scores = anomaly_scores(pipeline, X_test)
    flags = flagged(scores)

    precision = precision_score(y_test, flags, zero_division=0)
    recall = recall_score(y_test, flags, zero_division=0)

    return {
        "pr_auc": average_precision_score(y_test, scores),
        "roc_auc": roc_auc_score(y_test, scores),
        "precision": precision,
        "recall": recall,
        "alert_to_fraud": 1 / precision if precision else float("inf"),
        "n_flagged": int(flags.sum()),
        "n_fraud": int(y_test.sum()),
        "n_caught": int(flags[y_test == 1].sum()),
    }


def report(metrics: dict[str, float]) -> None:
    """Print the metrics block for the README."""
    print("\nRanking metrics (threshold-independent)")
    print(f"  PR-AUC          {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC         {metrics['roc_auc']:.4f}")
    print(f"\nAt the locked threshold ({FRAUD_THRESHOLD})")
    print(f"  Recall          {metrics['recall']:.4f}")
    print(f"  Precision       {metrics['precision']:.4f}")
    print(f"  Alert-to-fraud  {metrics['alert_to_fraud']:.2f}:1")
    print(f"  Frauds caught   {metrics['n_caught']} of {metrics['n_fraud']}")
    print(f"  Flags raised    {metrics['n_flagged']}")


def main() -> None:
    """Train on a stratified split, report metrics, and save the fitted pipeline."""
    X, y = load_raw_frames()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Loaded {len(X)} rows from {DATA_PATH}; training on {len(X_train)}.")

    pipeline = build_pipeline()
    pipeline.fit(X_train)  # unsupervised — y is held back for evaluation only

    report(evaluate(pipeline, X_test, y_test))

    # Saved last: a failure above leaves the committed artifact untouched.
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_PATH)
    print(f"\nSaved fitted pipeline to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
