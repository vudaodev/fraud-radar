from api.scoring import score, load_model
import pandas as pd
from common.transforms import secs_to_hour_of_day  # noqa: F401  # pickle resolves by reference
from common.paths import DATA, MODEL
from common.schemas import Transaction
import pytest


FLAGGED_ROWS = (477, 1602)
CLEAN_ROWS = (2627, 1000)
_features = pd.read_csv(DATA / "demo_sample.csv").drop(columns=["Class"])

def _transaction(row):
    """Input: a row label in demo_sample.csv
    Output: that row as a Transaction, minus the Class label the API never sees
    """
    return Transaction(**_features.loc[row])

flagged_1, flagged_2 = (_transaction(r) for r in FLAGGED_ROWS)
clean_1, clean_2 = (_transaction(r) for r in CLEAN_ROWS)

@pytest.fixture(scope="module")
def model():
    return load_model(MODEL)

def test_scoring_fraud_rows_flag(model):
    _1, is_flagged_1 = score(model, flagged_1)
    _2, is_flagged_2 = score(model, flagged_2)
    assert is_flagged_1 and is_flagged_2

def test_scoring_legit_rows_do_not_flag(model):
    _1, is_flagged_1 = score(model, clean_1)
    _2, is_flagged_2 = score(model, clean_2)
    assert not is_flagged_1 and not is_flagged_2
