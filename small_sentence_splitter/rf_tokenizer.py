import importlib.resources
import json
from typing import Dict, List, Optional

from small_sentence_splitter.sentence_tokenizer import BaseSentenceTokenizer
from small_sentence_splitter.decision_tree import DecisionTree
from small_sentence_splitter.feature_extractors import to_features

_CONTEXT_SIZE = 4

class RandomForestSentenceTokenizer(BaseSentenceTokenizer):

    def __init__(self, dt_set=None):
        if dt_set is None:
            dt_set = list()
            for i in range(10):
                data = importlib.resources.open_text("small_sentence_splitter.trained_models", f"dt{i}.json")
                dt = DecisionTree.from_dict(json.load(data))
                dt_set.append(dt)
        self.dt_set = dt_set

    def is_sentence(self, left: str, right: str) -> bool:
        features = to_features(left, right, context_size=_CONTEXT_SIZE)

        votes = { 0: 0, 1: 0 }
        for dt in self.dt_set:
            pred = dt.infer(features)
            if 0 in pred:
                votes[0] += pred[0] # TODO: Binary snap?
            if 1 in pred:
                votes[1] += pred[1]
        return votes[1] > votes[0]

    # TODO: Implement split_all with an override because we can pop features faster than recomputing all.
