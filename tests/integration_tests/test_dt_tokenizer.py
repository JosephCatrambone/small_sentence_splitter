import random

from small_sentence_splitter import DecisionTreeSentenceTokenizer


FORMAL_ENGLISH_SENTENCES = [
    "Hello, world!",
    "My friends, how are you today?",
    "Dr. Davies will see you now.",
    "I was born in W. St. Paul, MN.",
    "Dr. M. D. Emdee grew up on W. Adams St. in Chicago.",
]


def test_dt_tokenizer_simple():
    splitter = DecisionTreeSentenceTokenizer()

    # Combine all the sentences into one blob, randomly adding and removing spaces and newlines between sentences.
    blob_of_text = ""
    for s in FORMAL_ENGLISH_SENTENCES:
        blob_of_text += s
        blob_of_text += random.choice(["", " ", "\n"])
    
    # Now split it up.
    estimated_split = splitter.split_all(blob_of_text)
    assert len(estimated_split) == len(FORMAL_ENGLISH_SENTENCES)
    for actual, predicted in zip(FORMAL_ENGLISH_SENTENCES, estimated_split):
        assert actual == predicted
