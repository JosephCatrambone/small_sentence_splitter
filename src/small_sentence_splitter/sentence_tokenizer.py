from abc import ABC, abstractmethod

class BaseSentenceTokenizer(ABC):
    @abstractmethod
    def is_sentence(left: str, right: str) -> bool:
        """Returns 'true' if the last token of the left string delimits the end of a sentence."""
        ...

