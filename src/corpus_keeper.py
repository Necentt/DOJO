from Levenshtein import distance
from typing import Sequence, Self
from src.transform import LengthScaler
import src.utils as utils
import numpy as np
import pandas as pd
import faiss
from sklearn.pipeline import Pipeline

import src.config as config

def build_flat(dim, build_data):
    n = 1024
    index = faiss.IndexIVFFlat(
        faiss.IndexFlatL2(dim), dim, n, faiss.METRIC_INNER_PRODUCT
    )
    index.train(build_data)
    index.add(build_data)

    return index

class CorpusBuilder:
    def __init__(self):
        self.vectorizer = utils.load_model(config.CHAR_NGRAM_PATH)

        self.main_corpus = utils.load_model(config.MAIN_CORPUS_PATH)
        self.ngram_num = len(self.vectorizer["vectorizer"].vocabulary_)
        self.corpus = np.array(self.main_corpus)
        self.main_corpus = set(self.main_corpus)

    def create_corpus(self, inner_dataset: Sequence[str]|None):
        if inner_dataset is not None:
            self.priorities = np.zeros(len(self.corpus))
            counts = {key: val for key, val in zip(*np.unique(inner_dataset, return_counts=True))}
            for idx, word in enumerate(self.corpus):
                if word in counts:
                    self.priorities[idx] = counts[word]

            sorted_args = np.argsort(self.priorities)[::-1]
            self.corpus = self.corpus[sorted_args]
            self.priorities = self.priorities[sorted_args]

        X = self.vectorizer.transform(self.corpus)
        self.index = build_flat(self.ngram_num, X)

        utils.save_model_compressed(self.index, config.INDEX_PATH, 9)
        utils.save_model_compressed(self.corpus, config.CORPUS_PATH, 9)


class CorpusSearcher:
    def __init__(self):
        self.corpus = np.array(utils.load_model(config.CORPUS_PATH))
        self.vectorizer = utils.load_model(config.CHAR_NGRAM_PATH)
        self.index = utils.load_model(config.INDEX_PATH)
        self.index.nprobe=16

    def _find_k_neib(self, data: Sequence[str], k: int = 50) -> Sequence[Sequence[int]]:
        data = self.vectorizer.transform(data)
        return np.sort(self.index.search(data, k)[1])

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
        self.corpus = set(utils.load_model(config.CORPUS_PATH))

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
        

class FairSearch:
    def __init__(self):
        self.corpus = np.array(utils.load_model(config.CORPUS_PATH))

    def find(self, x):
        idx = np.argmin(np.vectorize(lambda y: distance(x, y))(self.corpus))
        return self.corpus[idx]

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

        result[missing] = np.vectorize(self.find)(missing_words)

        return result

    def transform(self, data: Sequence[Sequence[str]]) -> Sequence[Sequence[str]]:
        if isinstance(data, pd.Series):
            return data.apply(self._build_sequence)
        else:
            return [self._build_sequence(np.array(x)) for x in data]
