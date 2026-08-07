"""
Shared definition of anomaly output. Used by training, tests, and the API.

Owns what a score MEANS — which direction it runs and where the cut sits — not
the act of scoring a request; that belongs to api/scoring.py, which imports this.

IsolationForest's decision_function runs backwards from intuition: LOWER means
more anomalous. precision_recall_curve — and therefore every threshold derived
from it — expects the opposite. Negating once, here, means no caller has to
remember which way round it goes.

FRAUD_THRESHOLD lives in ANOMALY-SCORE space (the negated one). Comparing it
against a raw decision_function value inverts the rule and flags almost nothing:
on the held-out split that mistake flags 2 transactions instead of 816 and
catches 0 of 98 frauds. Always go through anomaly_scores() first.

structure.md plans a FRAUD_THRESHOLD env var for the API; the constant below is
the default that overrides, not a competing source of truth.
"""

FRAUD_THRESHOLD = -0.1218

""" Operating point chosen in 02_train_and_threshold.ipynb: ~74.5% recall at
    ~8.95% precision, roughly 11 alerts per fraud caught.

    Derived as the midpoint between the chosen score and the next lower one, not
    by rounding. precision_recall_curve returns thresholds that EQUAL observed
    scores, so a threshold sitting on one is knife-edge; round-to-nearest can
    land just above a fraud's score and silently drop the catch. The nearest
    scores here are -0.12190943 and -0.12175136, so -0.1218 sits in the gap
    between them by construction.
"""


def anomaly_scores(model, X):
    """Input: a fitted pipeline and a raw feature frame
    Output: one score per row, where HIGHER means more anomalous
    """
    return -model.decision_function(X)


def flagged(scores, threshold=FRAUD_THRESHOLD):
    """Input: anomaly scores from anomaly_scores() — not raw decision_function
    Output: boolean flag decisions, one per row
    """
    return scores >= threshold
