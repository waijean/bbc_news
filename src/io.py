import joblib
import json


def save_pipeline(pipeline):
    joblib.dump(pipeline, "pipeline.joblib")


def load_pipeline():
    pipeline = joblib.load("pipeline.joblib")
    return pipeline


def save_params(params):
    with open("../model/params.json", "w") as file:
        json.dump(params, file)


def load_params():
    with open("../model/params.json", "r") as fp:
        params = json.load(fp)
    return params
