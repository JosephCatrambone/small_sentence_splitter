import importlib.resources
import json
from typing import Dict, List, Optional

from small_sentence_splitter.sentence_tokenizer import BaseSentenceTokenizer
from small_sentence_splitter.decision_tree import DecisionTree
from small_sentence_splitter.feature_extractors import to_features


_CONTEXT_SIZE = 4


class DecisionTreeSentenceTokenizer(BaseSentenceTokenizer):

    def __init__(self, dt=None):
        if dt is None:
            data = importlib.resources.open_text("small_sentence_splitter.trained_models", "dt.json")
            dt = DecisionTree.from_dict(json.load(data))
        self.dt = dt

    def is_sentence(self, left: str, right: str) -> bool:
        pred = self.dt.infer(to_features(left, right, context_size=_CONTEXT_SIZE))
        assert len(pred) > 0
        if len(pred) == 1:
            return pred[list(pred.keys())[0]] > 0.5 # Ick
        return pred[1] > pred[0]

    # TODO: Implement split_all with an override because we can pop features faster than recomputing all.


def _make_dataset(linereader, min_positive_examples: int):
    import random
    cs = _CONTEXT_SIZE
    examples = list()
    labels = list() # 0 -> No break, 1 -> sentence break
    next_sent = linereader.readline().decode('utf-8').strip()
    for _ in range(min_positive_examples):
        previous_sent = next_sent
        next_sent = linereader.readline().decode('utf-8').strip()
        # Make three true positive examples: foo.Bar, foo. Bar, foo.\nBar.
        # Make random negative examples by splitting the previous or next sentence at random points.
        positive_nospace = to_features(previous_sent[-cs:], next_sent[:cs])
        positive_space = to_features(previous_sent[-cs:], " " + next_sent[:cs])
        positive_newline = to_features(previous_sent[-cs:], "\n" + next_sent[:cs])
        positive_eos = to_features(previous_sent[-cs:], "")
        examples.append(positive_nospace)
        labels.append(1)
        examples.append(positive_space)
        labels.append(1)
        examples.append(positive_newline)
        labels.append(1)
        examples.append(positive_eos)
        labels.append(1)
        # Random negative examples by splitting next in the middle of words or spaces.
        for s_idx in range(cs, len(next_sent) - cs):
            examples.append(to_features(next_sent[s_idx-cs:s_idx], next_sent[s_idx:s_idx+cs]))
            labels.append(0)
            if random.random() > 0.9:
                examples.append(to_features(next_sent[s_idx-cs:s_idx], "\n" + next_sent[s_idx:s_idx+cs]))
                labels.append(0)
    return examples, labels

