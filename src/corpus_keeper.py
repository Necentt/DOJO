from Levenshtein import distance
from typing import Sequence, Self
import src.utils as utils
import numpy as np
import pandas as pd


class CorpusSearcher:
    def __init__(self):
        self.corpus = np.array(utils.load_model("corpus.pt"))
        self.vectorizer = utils.load_model("char_grams.pt")
        self.index = utils.load_model("index.pt")

    def _find_k_neib(self, data: Sequence[str], k: int = 25) -> Sequence[Sequence[int]]:
        data = self.vectorizer.transform(data)
        return self.index.search(data, k)[1]

    def _best_candidate(self, x, candidates: Sequence[str]):
        distances = np.vectorize(lambda y: distance(x, y))(candidates)
        idx = np.argmin(distances)
        return candidates[idx]

    def find(self, data: Sequence[str]):
        data = np.array(data)
        indexes = self._find_k_neib(data)
        result = []
        for x, idx in zip(data, indexes):
            candidates = self.corpus[idx]
            result.append(self._best_candidate(x, candidates))
        return result


class CorpusReplacer:
    def __init__(self):
        self.searcher = CorpusSearcher()
        self.corpus = set(utils.load_model("corpus.pt"))

    def _build_sequence(self, seq=Sequence[str]) -> Sequence[str]:
        result = seq.copy()
        missing = []
        missing_words = []

        for i, word in enumerate(seq):
            if word not in self.corpus:
                missing.append(i)
                missing_words.append(word)

        if len(missing) == 0:
            return result

        replacing_words = self.searcher.find(missing_words)

        for i, word in enumerate(replacing_words):
            result[missing[i]] = word

        return result

    def fit(self, x, y) -> Self:
        return self

    def transform(self, data: Sequence[Sequence[str]]) -> Sequence[Sequence[str]]:
        if isinstance(data, pd.Series):
            return data.apply(self._build_sequence)
        else:
            return [self._build_sequence(x) for x in data]
