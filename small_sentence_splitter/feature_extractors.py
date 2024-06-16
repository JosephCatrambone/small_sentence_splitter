from typing import Dict, List, Optional


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

    
def to_features(left: str, right: str, context_size: int) -> List[float]:
    features = list()
    # If either left or right is too short, add spaces for padding.
    if len(left) < context_size:
        left = left.rjust(context_size)
    if len(right) < context_size:
        right = right.ljust(context_size)
    for c in left[-context_size:]:
        features.extend(char_to_features(c))
    for c in right[:context_size]:
        features.extend(char_to_features(c))
    return features



