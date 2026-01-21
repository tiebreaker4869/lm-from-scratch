from abc import ABC
import regex as re
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(string: str):
    for match in re.finditer(PAT, string):
        yield string[match.start():match.end()]

class Tokenizer(ABC):
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Construct and return a tokenizer from a serialized vocabulary, a list of merges and a list of special tokens.
        Args:
            vocab_filepath (str): _description_
            merges_filepath (str): _description_
            special_tokens (_type_, optional): _description_. Defaults to None.
        """
        pass
    def encode(self, string: str) -> list[int]:
        """
        Encode a string to a list of tokens

        Args:
            string (str): input string

        Returns:
            list[int]: tokenized input
        """
        pass
    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of tokens to string

        Args:
            tokens (list[int]): token indices

        Returns:
            str: decoded string
        """
        pass
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token ids. This can be used for memory-effcient tokenization of large files.

        Args:
            iterable (Iterable[str]): iterable of strings.

        Yields:
            Iterator[int]: iterator of encoded tokens. 
        """
        pass