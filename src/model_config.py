from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform

from src.vectorizer import CustomVectorizer

# doc2vec config
doc2vec_params = {
    "vector_size": 50,
    "dm": 1,  # {1:"distributed memory", 0:"distributed bag of words"}
    "window": 3,
    "seed": 0,
    "min_count": 3,
    "epochs": 40,  # increase training passes can help with small datasets
}


tfidf_pipeline = Pipeline(
    [
        ("undersampler", RandomUnderSampler()),
        ("union", ColumnTransformer([("tfidf", CustomVectorizer(), "combined_text",)]),),
        ("clf", RandomForestClassifier()),
    ]
)

doc2vec_pipeline = Pipeline(
    [("undersampler", RandomUnderSampler()), ("clf", RandomForestClassifier()),]
)

tfidf_distribution = {
    "undersampler__random_state": list(range(0, 10)),
    "union__tfidf__ngram_range": [(1, 1), (1, 2), (2, 2)],
    "union__tfidf__min_df": uniform(loc=0, scale=0.2),  # [0, 0.2]
    "union__tfidf__max_df": uniform(loc=0.8, scale=0.2),  # [0.8, 1]
    "union__tfidf__use_idf": [True, False],
    "clf__max_depth": list(range(5, 10)),
    "clf__random_state": list(range(0, 10)),
    "clf__max_features": ["sqrt", "log2"],
}

doc2vec_distribution = {
    "undersampler__random_state": list(range(0, 10)),
    "clf__n_estimators": list(range(100, 1000, 100)),
    "clf__max_depth": list(range(2, 5)),
    "clf__min_samples_leaf": list(range(2, 5)),
    "clf__random_state": list(range(0, 10)),
    "clf__max_features": uniform(loc=0, scale=0.2),  # [0, 0.2]
}

config_params = {
    "undersampler__random_state": 7,
    "union__tfidf__ngram_range": (1, 1),
    "union__tfidf__max_df": 0.843,
    "union__tfidf__min_df": 0.027,
    "union__tfidf__use_idf": False,
    "clf__random_state": 2,
    "clf__max_features": "sqrt",
    "clf__max_depth": 8,
}
