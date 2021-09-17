import logging

import spacy
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def preprocess_text(df, title_col="title", des_col="text"):
    """
    Create a `Combined Text` column by:
    1. Combining the title and description
    2. Perform tokenize, lemmatize and stopword removal for the text

    Notes:
    Store the output as tokens separated by whitespace to be passed into CustomVectorizer.

    Args:
        df: A raw DataFrame with `Title` and `Text` columns.
        title_col: A string for column name of Title
        des_col: A string for column name of Text

    Returns: The raw DataFrame with additional `Combined Text` column.

    """
    if not all(df.isnull().sum() == 0):
        raise ValueError("Raw table contains missing values.")
    texts = df[title_col] + " " + df[des_col]
    # rule-based lemmatizer requires tagger and attribute_ruler
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

    docs = []
    logger.info("Preprocessing text started...")
    for doc in nlp.pipe(texts):
        tokens = " ".join(
            (token.lemma_ for token in doc if not token.is_stop and token.is_alpha)
        )
        docs.append(tokens)
    logger.info("Preprocessing text done!")

    assert len(df) == len(docs)
    df["combined_text"] = docs
    return df


def create_train_test_set(X: pd.DataFrame, y: pd.Series):
    """
    Split into train and test set.

    X needs to be a DataFrame because imbalanced-learn's API expects a DataFrame
    https://imbalanced-learn.org/stable/introduction.html#api-s-of-imbalanced-learn-samplers

    Args:
        X: A DataFrame which has a single text column
        y: A Series which has binary target {0, 1}

    Returns: X_train, X_test, y_train, y_test

    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    logger.info(f"Target distribution for train set: \n{y_train.value_counts()}")
    logger.info(f"Target distribution for test set: \n{y_test.value_counts()}")
    return X_train, X_test, y_train, y_test
