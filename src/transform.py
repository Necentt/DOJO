import re
from typing import Hashable, Iterable

import nltk
import numpy as np
from nltk.corpus import stopwords
from pandas import Series
from sklearn.base import BaseEstimator

nltk.download("stopwords")


class Normalizer(BaseEstimator):
    def __init__(self, lang: str = "english"):
        self.stop_words = set(stopwords.words(lang))
        self.lang = lang
        if lang == "english":
            self.pattern = r"[^a-z]"
        elif lang == "russian":
            self.pattern = r"[^а-я]"
        else:
            print("Unknown language")
            self.pattern = None

    def specify_stop_words(self, stop_words: Iterable[Hashable]) -> None:
        self.stop_words = stop_words

    def fit(self, x, y):
        return self

    def _transform(self, line: str) -> str:
        line = line.lower()
        if self.pattern is not None:
            line = re.sub(self.pattern, " ", line)
        word_list = [x for x in line.split() if x not in self.stop_words]

        return " ".join(word_list)

    def transform(self, lines: Series) -> str:
        if isinstance(lines, Series):
            return lines.apply(self._transform)
        else:
            return [self._transform(x) for x in lines]


class LengthScaler(BaseEstimator):
    def __init__(self, to_array=True):
        self.to_array = to_array

    def fit(self, x, y):
        return self

    def transform(self, X):
        if self.to_array:
            X = X.toarray()

        return (X.T / (np.sum(X, axis=1).T + 1e-20)).T


class ToArray(BaseEstimator):
    def fit(self, x, y):
        return self

    def transform(self, X):
        return X.toarray()
