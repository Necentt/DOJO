import os
import joblib
import re
import string
import numpy as np

german_pattern = re.compile(r"([äöüß]+| auf | geht | ich | ist | die )", re.UNICODE)
english_etalon = set(string.ascii_lowercase)


def in_english(line):
    return np.all(np.vectorize(lambda x: x in english_etalon)(list(line)))

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
