from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform

from src.vectorizer import CustomVectorizer


PIPELINE_CONFIG = Pipeline(
    [
        ("undersampler", RandomUnderSampler()),
        ("union", ColumnTransformer([("tfidf", CustomVectorizer(), "combined_text",)]),),
        ("clf", RandomForestClassifier()),
    ]
)

DISTRIBUTION_CONFIG = {
    "undersampler__random_state": list(range(0, 10)),
    "union__tfidf__ngram_range": [(1, 1), (1, 2), (2, 2)],
    "union__tfidf__min_df": uniform(loc=0, scale=0.2),  # [0, 0.2]
    "union__tfidf__max_df": uniform(loc=0.8, scale=0.2),  # [0.8, 1]
    "union__tfidf__use_idf": [True, False],
    "clf__max_depth": list(range(5, 10)),
    "clf__random_state": list(range(0, 10)),
    "clf__max_features": ["sqrt", "log2"],
}
