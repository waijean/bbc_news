from typing import Dict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def custom_scorer(clf, X, y) -> Dict[str, float]:
    """
    Get the micro-averaged precision, recall and f1.

    Each score is a floating point number that quantifies the estimator prediction quality on X,
    with reference to y. By convention higher numbers are better.

    Notes
    https://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object
    https://scikit-learn.org/stable/modules/model_evaluation.html#using-multiple-metric-evaluation

    Args:
        clf: the model that should be evaluated
        X: validation data
        y: ground truth target

    Returns: A dictionary where the keys are the metric names and the values are the metric scores.

    """
    y_pred = clf.predict(X)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="micro"
    )
    return {"precision": precision, "recall": recall, "f1": f1}
