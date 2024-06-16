import random

from small_sentence_splitter import RandomForestSentenceTokenizer


SIMPLE_ENGLISH_SENTENCES = [
    "Hello, world!",
    "My friends, how are you today?",
]

DIFFICULT_ENGLISH_SENTENCES = [
    "Dr. Davies will see you now.",
    "I was born in W. St. Paul, MN.",
    "Dr. M. D. Emdee grew up on W. Adams St. in Chicago.",
]


def test_rf_tokenizer_simple():
    splitter = RandomForestSentenceTokenizer()

    # Combine all the sentences into one blob, randomly adding and removing spaces and newlines between sentences.
    blob_of_text = ""
    for s in SIMPLE_ENGLISH_SENTENCES:
        blob_of_text += s
        blob_of_text += random.choice([" ", "\n"])
    
    # Now split it up.
    estimated_split = splitter.split_all(blob_of_text)
    assert len(estimated_split) == len(SIMPLE_ENGLISH_SENTENCES)
    for actual, predicted in zip(SIMPLE_ENGLISH_SENTENCES, estimated_split):
        assert actual == predicted.strip()


def test_rf_tokenizer_difficult():
    splitter = RandomForestSentenceTokenizer()
    blob_of_text = ""
    for s in DIFFICULT_ENGLISH_SENTENCES:
        blob_of_text += s
        blob_of_text += random.choice(["  ", " ", "\n"])
    
    # Now split it up.
    estimated_split = splitter.split_all(blob_of_text)
    assert len(estimated_split) == len(SIMPLE_ENGLISH_SENTENCES)
    for actual, predicted in zip(SIMPLE_ENGLISH_SENTENCES, estimated_split):
        assert actual == predicted.strip()

