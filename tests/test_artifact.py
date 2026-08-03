"""Acceptance test for the serialised model artifact.

Runs as a fresh process (no training context), exactly like the API does. Proves
the fitted pipeline survives serialisation and scores a raw transaction:
  - the artifact loads standalone (secs_to_hour_of_day resolves by reference),
  - raw Time/Amount/V1-V28 go in and a finite anomaly score comes out,
  - the transforms travel with the artifact (raw input, not pre-transformed).

Uses no data files, so it runs in CI where creditcard.csv is absent.

How to run
----------
Run from the REPO ROOT (fraud-radar/). Prefer `uv run`, which sets up the project
environment and path correctly:

    # all tests in this file, one line of output per test
    uv run pytest tests/test_artifact.py -v

    # the whole test suite
    uv run pytest -v

    # a single test by name
    uv run pytest tests/test_artifact.py::test_scores_raw_row -v

For bare `pytest` to resolve `from common.transforms import ...`, the repo root
must be on the path. This is configured in pyproject.toml:

    [tool.pytest.ini_options]
    pythonpath = ["."]

Prerequisite: the artifact must exist at model/artifacts/isolation_forest.pkl.
If it is missing, run `python model/train.py` (or re-run notebook 02) first;
the load fixture will otherwise fail with a clear "run train.py" message.

Common failure: `ModuleNotFoundError: No module named 'common'` means the repo
root is not on the path — use `uv run`, or add the pythonpath config above.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

# Importing this here (even though the pipeline calls it internally) documents
# the dependency and fails loudly if the shared module goes missing.
from common.transforms import secs_to_hour_of_day  # noqa: F401

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "model" / "artifacts" / "isolation_forest.pkl"
THRESHOLD = -0.12175  # locked operating point; see 02_train_and_threshold.ipynb

RAW_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]


def _raw_row(**overrides):
    """One raw transaction, shaped exactly as Kafka delivers it."""
    row = {f"V{i}": 0.0 for i in range(1, 29)}
    row["Time"] = 406.0
    row["Amount"] = 59.99
    row.update(overrides)
    return pd.DataFrame([row])


@pytest.fixture(scope="module")
def model():
    """Load the committed artifact once for all tests in this module."""
    assert ARTIFACT_PATH.exists(), (
        f"Artifact not found at {ARTIFACT_PATH}. Run model/train.py to generate it."
    )
    return joblib.load(ARTIFACT_PATH)


def test_artifact_loads(model):
    """The pickle loads standalone — secs_to_hour_of_day resolved by reference."""
    assert model is not None


def test_scores_raw_row(model):
    """A raw row produces a single finite score — the transforms travelled."""
    score = model.decision_function(_raw_row())
    assert score.shape == (1,)
    assert np.isfinite(score[0])


def test_accepts_raw_columns_only(model):
    """The pipeline takes RAW columns (Time, Amount) and transforms internally.

    If preprocessing had leaked outside the artifact, scoring raw input would
    fail here (missing Hour/Amount_log). Passing proves it did not.
    """
    row = _raw_row()
    assert set(RAW_COLUMNS).issubset(row.columns)
    score = model.decision_function(row)
    assert np.isfinite(score[0])


def test_threshold_decision(model):
    """The threshold produces a boolean flag decision without error."""
    score = model.decision_function(_raw_row())[0]
    flagged = bool(score < THRESHOLD)
    assert isinstance(flagged, bool)