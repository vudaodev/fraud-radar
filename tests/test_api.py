"""
TestClient hits /score with a valid body -> 200 + score
TestClient hits /score with a malformed body -> 422
No running server needed
"""

from fastapi.testclient import TestClient
import pytest
from api.main import app


@pytest.fixture
def valid_payload():
    """
    A valid payload that we will use for the tests.
    Function scoped on purpose, so it can be adapted for different tests.
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
        "V28": -0.488307035713995,
    }


def test_api_valid_body(valid_payload):
    with TestClient(app) as client:
        response = client.post("/score", json=valid_payload)
        assert response.status_code == 200
        assert response.json()["flagged"] is True
        assert isinstance(response.json()["score"], float)


def test_api_malformed_body(valid_payload):
    malformed_payload = valid_payload
    malformed_payload["Time"] = "Hello"
    with TestClient(app) as client:
        response = client.post("/score", json=malformed_payload)
        assert response.status_code == 422
