from typing import Dict, List, Optional

from small_sentence_splitter import BaseSentenceTokenizer
from small_sentence_splitter.decision_tree import DecisionTree

_CONTEXT_SIZE = 4

class DecisionTreeSentenceTokenizer(BaseSentenceTokenizer):

    def __init__(self, dt=None):
        self.dt = dt

    def is_sentence(self, left: str, right: str) -> bool:
        pred = self.dt.infer(DecisionTreeSentenceTokenizer.to_features(left, right))
        if len(pred) == 1:
            return pred[list(pred.keys())[0]] > 0.5 # Ick
        return pred[1] > pred[0]

    @staticmethod
    def char_to_features(c: str) -> List[float]:
        basic_features = [1.0 * boolean_feature for boolean_feature in [
                c.isdecimal(),
                c.isalnum(),
                c.isdigit(),
                c.isnumeric(),
                c.isspace(),
                c.islower(),
                c.isupper(),
        ]]
        lower_c = c.lower()
        extended_features = [1.0 * (lower_c == character) for character in "abcdefghijklmnopqrstuvwxyz"]
        return basic_features + extended_features
        
    @staticmethod
    def to_features(left: str, right: str) -> List[float]:
        features = list()
        # If either left or right is too short, add spaces for padding.
        if len(left) < _CONTEXT_SIZE:
            left = left.rjust(_CONTEXT_SIZE)
        if len(right) < _CONTEXT_SIZE:
            right = right.ljust(_CONTEXT_SIZE)
        for c in left[-_CONTEXT_SIZE:]:
            features.extend(DecisionTreeSentenceTokenizer.char_to_features(c))
        for c in right[:_CONTEXT_SIZE]:
            features.extend(DecisionTreeSentenceTokenizer.char_to_features(c))
        return features


def _make_dataset(linereader, min_positive_examples: int):
    cs = _CONTEXT_SIZE
    examples = list()
    labels = list() # 0 -> No break, 1 -> sentence break
    next_sent = linereader.readline().decode('utf-8').strip()
    for _ in range(min_positive_examples):
        previous_sent = next_sent
        next_sent = linereader.readline().decode('utf-8').strip()
        # Make three true positive examples: foo.Bar, foo. Bar, foo.\nBar.
        # Make random negative examples by splitting the previous or next sentence at random points.
        positive_nospace = DecisionTreeSentenceTokenizer.to_features(previous_sent[-cs:], next_sent[:cs])
        positive_space = DecisionTreeSentenceTokenizer.to_features(previous_sent[-cs:], " " + next_sent[:cs])
        positive_newline = DecisionTreeSentenceTokenizer.to_features(previous_sent[-cs:], "\n" + next_sent[:cs])
        positive_eos = DecisionTreeSentenceTokenizer.to_features(previous_sent[-cs:], "")
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
            examples.append(DecisionTreeSentenceTokenizer.to_features(next_sent[s_idx-cs:s_idx], next_sent[s_idx:s_idx+cs]))
            labels.append(0)
    return examples, labels


