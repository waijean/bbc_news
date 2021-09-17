import logging
import pprint
from typing import Tuple, List

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV

from src.model_config import (
    tfidf_pipeline,
    doc2vec_pipeline,
    tfidf_distribution,
    config_params,
    doc2vec_distribution,
)
from src.scorer import custom_scorer

logger = logging.getLogger(__name__)


def create_pipeline_distribution() -> Tuple[Pipeline, dict]:
    """
    Load pipeline and randoms search distributions from config based on the features mode.

    Args:


    Returns:

    """
    pipeline = tfidf_pipeline
    distribution = tfidf_distribution
    logger.info(f"Pipeline constructed: \n{pipeline}")
    return pipeline, distribution


def random_search(
    pipeline: Pipeline, distribution: dict, X_train: pd.DataFrame, y_train: pd.Series
) -> RandomizedSearchCV:
    # todo solve the multiple log files issue to enable njobs=-1
    logger.info(
        f"Running random search on the following distribution: \n{pprint.pformat(distribution)}"
    )
    rscv = RandomizedSearchCV(
        pipeline,
        distribution,
        scoring=custom_scorer,
        refit="f1",
        random_state=0,
        n_iter=20,
        return_train_score=True,
        verbose=2,
        cv=3,
    )
    logger.info("Running random search...")
    rscv = rscv.fit(X_train, y_train)
    logger.info("Random search done!")
    logger.info(
        f"Best parameters based on accuracy: \n{pprint.pformat(rscv.best_params_)}"
    )

    return rscv


def get_cv_metrics(rscv: RandomizedSearchCV) -> pd.DataFrame:
    cv_metrics_df = pd.DataFrame(rscv.cv_results_)
    pd.options.display.float_format = "{:.3f}".format
    train_metrics = cv_metrics_df.loc[
        rscv.best_index_,
        [
            "mean_train_recall",
            "mean_train_precision",
            "mean_train_f1",
        ],
    ]
    test_metrics = cv_metrics_df.loc[
        rscv.best_index_,
        [
            "mean_test_recall",
            "mean_test_precision",
            "mean_test_f1",
        ],
    ]
    logger.info(f"Train metrics: \n{train_metrics}")
    logger.info(f"Validation metrics: \n{test_metrics}")
    return cv_metrics_df


def fit_pipeline(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """
    Load the params for the pipeline and fit the pipeline using training data.

    Args:
        pipeline:
        X_train:
        y_train:

    Returns: A fitted pipeline
    """
    pipeline.set_params(**config_params)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate the pipeline on both train and test set.

    Args:
        pipeline:
        X_train:
        X_test:
        y_train:
        y_test:

    Returns:

    """
    train_metrics = custom_scorer(pipeline, X_train, y_train)
    test_metrics = custom_scorer(pipeline, X_test, y_test)
    metrics_df = pd.DataFrame([train_metrics, test_metrics], index=["train", "test"])
    pd.options.display.float_format = "{:.3f}".format
    logger.info(f"Test metrics: \n{metrics_df}")
    return metrics_df


def inspect_pipeline(pipeline: Pipeline) -> List[str]:
    # inspect tfidf
    tfidf = pipeline["union"].named_transformers_["tfidf"]
    feature_names = tfidf.get_feature_names()
    logger.info(f"Number of unique words in tfidf matrix:{len(feature_names)}")

    # inspect classifier
    clf = pipeline["clf"]

    if hasattr(clf, "feature_log_prob_"):
        pos_class_prob_sorted = clf.feature_log_prob_[1, :].argsort()[::-1]
        neg_class_prob_sorted = clf.feature_log_prob_[0, :].argsort()[::-1]

        top_10_positive = np.take(feature_names, pos_class_prob_sorted[:10])
        top_10_negative = np.take(feature_names, neg_class_prob_sorted[:10])

        logger.info(f"Top 10 words in positive class {top_10_positive}")
        logger.info(f"Top 10 words in negative class {top_10_negative}")

    elif hasattr(clf, "coef_"):
        prob_sorted = clf.coef_[0].argsort()

        top_10_negative = np.take(feature_names, prob_sorted[:10])
        top_10_positive = np.take(feature_names, prob_sorted[-10:])

        logger.info(f"Top 10 words in positive class {top_10_positive}")
        logger.info(f"Top 10 words in negative class {top_10_negative}")

    elif hasattr(clf, "feature_importances_"):
        sorted_features = pd.Series(
            clf.feature_importances_, index=feature_names
        ).sort_values(ascending=False)
        logger.info(f"Top 10 most important words: \n{sorted_features[:10]}")

    return feature_names


def change_threshold(best_estimator, X, y):
    # get the probability of the positive label
    proba = best_estimator.predict_proba(X)
    pos_proba = proba[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, pos_proba)
    plot_precision_recall_curve(best_estimator, X, y)
    return pd.DataFrame(
        {
            "thresholds": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )


def get_prediction(
    pipeline: Pipeline, df, filter=None, cols: List[str] = None, order: bool = True
):
    """
    Get automatable probabilities for each control.

    Args:
        df: A DataFrame which contains "Control Text" column to be passed into pipeline.
        filter: A boolean expression to filter rows for predictions. None if no filter to perform.
                For example, df[CONTROL_EXECUTION] != "Automatic" to keep only non-automatic controls
        cols: A list of columns to keep in the prediction df. None to keep all the columns.
                For example, [ROW_ID, CONTROL_TITLE, CONTROL_DESCRIPTION, PROBABILITY]
        order: A boolean flag to determine whether to sort the predictions from highest to lowest probability.

    Returns:

    """
    if filter is not None:
        df = df.loc[filter]

    # get the probability of positive label
    X = df[[CONTROL_TEXT]]
    proba = pipeline.predict_proba(X)
    prediction = df.assign(**{PROBABILITY: proba[:, 1]})

    # only keep these columns in model output
    if cols is not None:
        prediction = prediction[cols]

    # order by the probability
    if order:
        prediction.sort_values(by=PROBABILITY, ascending=False, inplace=True)
    return prediction
