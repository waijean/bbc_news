import logging
from argparse import ArgumentParser

import pandas as pd

from src.logger import configure_parent_logger
from src.modeling import (
    create_pipeline_distribution,
    evaluate_pipeline,
    inspect_pipeline,
    random_search,
    fit_pipeline,
    get_cv_metrics,
)
from src.preprocessing import create_train_test_set


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "tune",
            "test",
            "full",
        ],
        required=True,
        help="To control whether to run hyperparameter tuning, evaluate on test set, or train on the full set",
    )
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    configure_parent_logger(f"{args.mode}")

    df = pd.read_csv("../data/preprocessed_text.csv")

    X = df[["combined_text"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = create_train_test_set(X, y)

    pipeline, distribution = create_pipeline_distribution()

    if args.mode == "tune":
        rscv = random_search(pipeline, distribution, X_train, y_train)
        cv_metrics_df = get_cv_metrics(rscv)
        feature_names = inspect_pipeline(rscv.best_estimator_)

    elif args.mode == "test":
        pipeline = fit_pipeline(pipeline, X_train, y_train)
        metrics_df = evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test)
        feature_names = inspect_pipeline(pipeline)

    elif args.mode == "full":
        pipeline = fit_pipeline(pipeline, X, y)

    # close the file handler
    logging.shutdown()
