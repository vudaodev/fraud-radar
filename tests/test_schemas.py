"""
Contract test for the pydantic schemas in common/schemas.py.

Transaction is the boundary every HTTP request crosses before it reaches the
model (api/main.py, post_score).

The dict from Transaction.model_dump() becomes the single-row DataFrame handed to sklearn in
api/scoring.py, so what this schema accepts is exactly what gets scored.
"""

from common.schemas import Transaction
from pydantic import ValidationError
import pytest


@pytest.fixture
def valid_payload():
    """
    A valid payload that we will use for the tests. Function scoped on purpose, so it can be adapted for different tests.
    """
    return {
        "Time": 93860.0,
        "Amount": 188.52,
        "V1": -10.6323749061596,
        "V2": 7.25193622855414,
        "V3": -17.6810718207918,
        "V4": 8.20414440620562,
        "V5": -10.1665907519072,
        "V6": -4.51034377036334,
        "V7": -12.9816061559658,
        "V8": 6.78358879797499,
        "V9": -4.65932958355558,
        "V10": -14.9246547735487,
        "V11": 8.38914233451929,
        "V12": -16.4655039422141,
        "V13": 0.33851695978655,
        "V14": -14.224403603167,
        "V15": 0.556584472572111,
        "V16": -11.683998043525,
        "V17": -15.8416159780561,
        "V18": -5.75319975278369,
        "V19": 3.81304079276336,
        "V20": -0.810146481561289,
        "V21": 2.71535704420309,
        "V22": 0.695602689761576,
        "V23": -1.13812206664164,
        "V24": 0.459442241911828,
        "V25": 0.386337323495895,
        "V26": 0.522438449202614,
        "V27": -1.41660373652915,
        "V28": -0.488307035713995
    }


def test_schemas_valid_payload_constructs(valid_payload):
    transaction = Transaction(**valid_payload)
    assert transaction.Amount == valid_payload["Amount"]
    assert transaction.Time == valid_payload["Time"]


def test_schemas_has_expected_fields():
    expected_set = {"Time", "Amount"} | {f"V{i}" for i in range(1, 29)}
    assert set(Transaction.model_fields) == expected_set


def test_schemas_missing_field_raises(valid_payload):
    del valid_payload["V28"]
    with pytest.raises(ValidationError):
        Transaction(**valid_payload)


def test_schemas_wrong_type_raises(valid_payload):
    valid_payload["V28"] = "Hello"
    with pytest.raises(ValidationError):
        Transaction(**valid_payload)
