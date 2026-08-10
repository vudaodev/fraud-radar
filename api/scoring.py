import joblib
import pandas as pd

from common.anomaly import FRAUD_THRESHOLD, anomaly_scores, flagged
from common.transforms import secs_to_hour_of_day  # noqa: F401  # pickle resolves by reference


def load_model(path):
    model = joblib.load(path)
    return model


def score(model, txn, threshold=FRAUD_THRESHOLD):
    df = pd.DataFrame([txn.model_dump()])
    adjusted_scores = anomaly_scores(model, df)
    flags = flagged(adjusted_scores, threshold)
    return float(adjusted_scores[0]), bool(flags[0])
