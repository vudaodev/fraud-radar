"""Acceptance test for the serialised model artifact.

Runs as a fresh process (no training context), exactly like the API does. Proves
the fitted pipeline survives serialisation and scores a raw transaction:
  - the artifact loads standalone (secs_to_hour_of_day resolves by reference),
  - raw Time/Amount/V1-V28 go in and a finite anomaly score comes out,
  - the transforms travel with the artifact (raw input, not pre-transformed),
  - the threshold is applied in the right direction (common/anomaly.py).

Uses only committed fixtures — synthetic rows plus data/test_sample.csv — so it
runs in CI, where the gitignored creditcard.csv is absent.

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
If it is missing, run `uv run python -m model.train` (or re-run notebook 02)
first; the load fixture will otherwise fail with a clear "run train.py" message.

Common failure: `ModuleNotFoundError: No module named 'common'` means the repo
root is not on the path — use `uv run`, or add the pythonpath config above.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

# The scoring helpers are imported rather than reimplemented, so these tests
# exercise the exact code the API will call. Importing the transform (even though
# the pipeline calls it internally) documents the dependency and fails loudly if
# the shared module goes missing.
from common.anomaly import FRAUD_THRESHOLD, anomaly_scores, flagged
from common.transforms import secs_to_hour_of_day  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = REPO_ROOT / "model" / "artifacts" / "isolation_forest.pkl"
SAMPLE_PATH = REPO_ROOT / "data" / "test_sample.csv"

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
    decision = flagged(anomaly_scores(model, _raw_row()))
    assert decision.shape == (1,)
    assert isinstance(bool(decision[0]), bool)


def test_threshold_flags_an_anomalous_row(model):
    """The comparison direction is right way round.

    A wildly out-of-distribution row must flag while an ordinary one must not.
    Inverting the comparison fails this, which the isinstance check above cannot
    catch on its own. The V values below exaggerate the directions real fraud
    tends to move in; an all-zero row sits at the centre of the PCA space and is
    by construction the least anomalous input available.
    """
    ordinary = anomaly_scores(model, _raw_row())[0]
    anomalous = anomaly_scores(
        model,
        _raw_row(
            V1=-30.0,
            V3=-30.0,
            V4=12.0,
            V10=-20.0,
            V12=-19.0,
            V14=-19.0,
            V17=-25.0,
            Amount=9999.99,
        ),
    )[0]

    assert anomalous > ordinary
    assert bool(anomalous >= FRAUD_THRESHOLD)
    assert not bool(ordinary >= FRAUD_THRESHOLD)


def test_flags_most_known_fraud_in_the_sample():
    """End-to-end sanity on real labelled rows, not synthetic ones.

    The committed fixture holds 20 known frauds. At the locked threshold the
    model should catch most of them, consistent with its ~74% recall. This is the
    check that would have caught the inverted comparison immediately: with the
    sign flipped it catches none.
    """
    sample = pd.read_csv(SAMPLE_PATH)
    fraud = sample.loc[sample["Class"] == 1, RAW_COLUMNS]
    assert len(fraud) == 20

    model = joblib.load(ARTIFACT_PATH)
    caught = int(flagged(anomaly_scores(model, fraud)).sum())
    assert caught >= 12, (
        f"only {caught} of 20 known frauds flagged — check the threshold direction"
    )
