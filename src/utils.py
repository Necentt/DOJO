import os
import joblib
import re
import string

german_pattern = re.compile(r"([äöüß]+| auf | geht | ich | ist | die )", re.UNICODE)


def save_model(model, model_name: str, compress=0):
    joblib.dump(model, os.path.join("results", model_name))

def save_model_compressed(model, model_name: str, compress=0):
    joblib.dump(model, os.path.join("results", model_name), compress=compress)

def load_model(model_name: str):
    return joblib.load(os.path.join("results", model_name))

def is_german(line):
    line = line.lower()
    res = german_pattern.search(line)
    return res is not None
    