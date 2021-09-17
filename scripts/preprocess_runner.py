from argparse import ArgumentParser

import pandas as pd
from src.logger import configure_parent_logger
from src.preprocessing import preprocess_text


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--feature",
        type=str,
        choices=["bow", "doc2vec",],
        help="To determine which vectorization technique to use",
    )
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    configure_parent_logger(args.feature)

    df = pd.read_csv("../data/text.csv")

    preprocessed_df = preprocess_text(df)

    preprocessed_df.to_csv("../data/preprocessed_text.csv", index=False)
