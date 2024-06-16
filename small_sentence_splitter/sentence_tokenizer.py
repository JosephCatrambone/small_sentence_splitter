from abc import ABC, abstractmethod

class BaseSentenceTokenizer(ABC):
    @abstractmethod
    def is_sentence(left: str, right: str) -> bool:
        """Returns 'true' if the last token of the left string delimits the end of a sentence."""
        ...

    def split_all(self, text: str, filter_empty: bool = True) -> list[str]:
        """Given a block of text, returns a list of all sentences."""
        last_sentence_break = 0
        sentences = list()
        for idx in range(0, len(text)):
            if self.is_sentence(text[last_sentence_break:idx], text[idx:]):
                sentences.append(text[last_sentence_break:idx].strip())
                last_sentence_break = idx
        # If we have anything left at the end, add it as the last sentence.
        if idx != last_sentence_break:
            sentences.append(text[last_sentence_break:])
        # Remove empties.
        if filter_empty:
            sentences = list(filter(None, sentences))
        return sentences

