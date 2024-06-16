from typing import Dict, List, Optional


class DecisionTree:
    def __init__(self):
        self.left: Optional['DecisionTree'] = None
        self.right: Optional['DecisionTree'] = None
        self.feature_index: int = 0
        self.feature_threshold: float = 0.5
        self.gini_impurity: float = 1e5
        self.label_confidence: Dict[int, float] = dict()
        self.label: Optional[int] = None

    def save(self):
        return {
            "label": self.label,
            "confidence": self.label_confidence,
            "gini": self.gini_impurity,
            "threshold": self.feature_threshold,
            "feature_idx": self.feature_index,
            "left": self.left.save() if self.left is not None else None,
            "right": self.right.save() if self.right is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict):
        dt = cls()
        dt.label = int(data["label"]) if data["label"] is not None else None
        dt.label_confidence = {int(k): v for k, v in data["confidence"].items()}
        dt.gini_impurity = data["gini"]
        dt.feature_threshold = data["threshold"]
        dt.feature_index = data["feature_idx"]
        if data["left"]:
            dt.left = cls.from_dict(data["left"])
        if data["right"]:
            dt.right = cls.from_dict(data["right"])
        return dt

    @classmethod
    def new_trained(cls, examples: List[List[float]], labels: List[int], max_depth:int = -1):
        root = cls()
        # Find the base impurity of this state
        cat_to_p = _probability_by_category(labels)
        num_categories = len(cat_to_p)
        root.label_confidence = cat_to_p

        if num_categories == 0:
            return root

        if num_categories < 2:
            root.label = labels[0]
            return root

        best_split_column = 0
        best_split_value = 0.5
        lowest_impurity_candidate = 1.0
        for candidate_feature in range(len(examples[0])):
            for candidate_feature_threshold in (0.25, 0.5, 0.75):
                left_candidate_labels = list()
                right_candidate_labels = list()
                for idx in range(len(examples)):
                    if examples[idx][candidate_feature] < candidate_feature_threshold:
                        left_candidate_labels.append(labels[idx])
                    else:
                        right_candidate_labels.append(labels[idx])
                if len(left_candidate_labels) == 0 or len(right_candidate_labels) == 0:
                    continue  # Don't bother. This doesn't change our data.
                left_impurity = 1.0 - _probability_to_gini_impurity(_probability_by_category(left_candidate_labels))
                right_impurity = 1.0 - _probability_to_gini_impurity(_probability_by_category(right_candidate_labels))
                weighted_impurity = (float(len(left_candidate_labels))/float(len(labels))*left_impurity + float(len(right_candidate_labels))/float(len(labels))*right_impurity)
                if weighted_impurity < lowest_impurity_candidate:
                    best_split_column = candidate_feature
                    best_split_value = candidate_feature_threshold
                    lowest_impurity_candidate = weighted_impurity

        root.feature_index = best_split_column
        root.feature_threshold = best_split_value
        root.gini_impurity = lowest_impurity_candidate

        if max_depth == 0:
            return root

        left_examples = list()
        left_labels = list()
        right_examples = list()
        right_labels = list()
        for idx in range(0, len(examples)):
            if examples[idx][root.feature_index] < root.feature_threshold:
                left_examples.append(examples[idx])
                left_labels.append(labels[idx])
            else:
                right_examples.append(examples[idx])
                right_labels.append(labels[idx])
        root.left = cls.new_trained(left_examples, left_labels, max_depth=max_depth-1)
        root.right = cls.new_trained(right_examples, right_labels, max_depth=max_depth-1)
        return root

    def infer(self, example: List[float]) -> Dict[int, float]:
        """Returns a mapping from class -> probability."""
        if self.left is None or self.right is None:
            if self.label is not None:
                return {self.label: 1.0}
            else:
                return self.label_confidence
        elif example[self.feature_index] < self.feature_threshold:
            return self.left.infer(example)
        else:
            return self.right.infer(example)


def _count_by_category_and_total(labels: List[int]) -> Dict[int, int]:
    """Return the number of instances of each label."""
    counts = dict()
    total = 0
    for c in labels:
        if c not in counts:
            counts[c] = 0
        counts[c] += 1
        total += 1
    return counts, total

def _probability_by_category(labels: List[int]) -> Dict[int, float]:
    """Returns a mapping from each class to the probability."""
    class_count, total_count = _count_by_category_and_total(labels)
    assert total_count > 0
    for c, v in class_count.items():
        class_count[c] = float(v)/float(total_count)
    return class_count

def _probability_to_gini_impurity(category_to_probability: Dict[int, float]) -> float:
    gini_impurity = 0.0
    for v in category_to_probability.values():
        gini_impurity += v*v
    return gini_impurity

