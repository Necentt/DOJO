from Levenshtein import distance
from typing import Sequence
import src.utils as utils
import numpy as np

class CorpusSearcher:
    def __init__(self):
        self.corpus = np.array(utils.load_model("corpus.pt"))
        self.vectorizer = utils.load_model("char_grams.pt")
        self.index = utils.load_model("index.pt")

    def _find_k_neib(self, data: Sequence[str], k: int=25)->Sequence[Sequence[int]]:
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