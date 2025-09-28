import os
from secrets import token_bytes
from typing import Generator, Iterable, Iterator

import regex as re

from src.bpe import read_vocab_merges
from src.utils.constants import PAT
from src.utils.pretokenization import get_bytes_tuple


class Tokenizer:
    """Class representing a trained byte-level BPE tokenizer"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Initialization method of the Tokenizer class

        Args:
            vocab (dict[int, bytes]): _description_
            merges (list[tuple[bytes, bytes]]): _description_
            special_tokens (list[str] | None, optional): _description_. Defaults to None.
        """

        if special_tokens is not None and len(special_tokens):
            for special_token in special_tokens:
                encoded_special_token = special_token.encode(encoding="utf-8")
                if encoded_special_token not in vocab.values():
                    vocab[len(vocab)] = encoded_special_token

            self.special_pattern = (
                "("
                + "|".join(
                    re.escape(tok)
                    for tok in sorted(special_tokens, key=len, reverse=True)
                )
                + ")"
            )

        self.vocab = vocab
        self.inv_vocab = {
            pretoken: idx for idx, pretoken in self.vocab.items()
        }
        self.merges = merges
        self.special_tokens = special_tokens

        self.special_match = re.escape("<|") + ".*" + re.escape("|>")

    @classmethod
    def from_file(
        cls,
        vocab_filepath: str,
        merge_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """Class method to construct the Tokenizer from files paths

        Args:
            vocab_filepath (str): Vocabulary file path.
            merge_filepath (str): Merges file path.
            special_tokens (list[str] | None = None): Special tokens of the tokenizer.
        """

        vocab, merges = read_vocab_merges(vocab_filepath, merge_filepath)

        return Tokenizer(vocab, merges, special_tokens)

    def _encode_solo_text(self, text: str) -> list[int]:
        """Helper function to encode text without special tokens.
        If a special token is in the text, it is consider disallowed and raise an error.

        Args:
            text (str): Input text. It should not contains special tokens.

        Returns:
            list[int]: List of token ids.
        """
        disallowed = re.findall(self.special_match, text)

        if disallowed:
            raise ValueError(
                f"Disallowed special tokens encountered in text: {disallowed}"
            )
        pretokens = re.findall(PAT, text)

        text_ids = []
        for pretoken in pretokens:
            pretoken_bytes = get_bytes_tuple(pretoken)

            for _, merge in enumerate(self.merges):
                length_pretoken_bytes = len(pretoken_bytes)
                if length_pretoken_bytes == 1:
                    break
                # if len(max(merge, key=len)) > len(
                #     max(pretoken_bytes, key=len)
                # ):
                #     continue

                index_1, index_2 = 0, 1
                new_pretoken_bytes = []
                while index_2 < length_pretoken_bytes:
                    bytes_1, bytes_2 = (
                        pretoken_bytes[index_1],
                        pretoken_bytes[index_2],
                    )
                    if bytes_1 == merge[0] and bytes_2 == merge[1]:
                        new_pretoken_bytes.append(bytes_1 + bytes_2)
                        index_1 += 2
                        index_2 += 2
                    else:
                        new_pretoken_bytes.append(bytes_1)
                        index_1 += 1
                        index_2 += 1
                if index_1 < length_pretoken_bytes:
                    new_pretoken_bytes.append(pretoken_bytes[index_1])

                pretoken_bytes = tuple(new_pretoken_bytes)

            pretoken_ids = [
                self.inv_vocab[symbol] for symbol in pretoken_bytes
            ]

            text_ids.extend(pretoken_ids)

        return text_ids

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs.

        Args:
            text (str): Input text. It may contains special tokens.

        Returns:
            list[int]: List of token ids.
        """

        last_index = 0
        tokens_ids = []
        if self.special_tokens:
            for match in re.finditer(self.special_pattern, text):
                special_start, special_end, special_text = (
                    match.start(),
                    match.end(),
                    match.group(0),
                )
                clean_text = text[last_index:special_start]
                last_index = special_end
                clean_tokens_ids = self._encode_solo_text(clean_text)
                tokens_ids.extend(clean_tokens_ids)
                tokens_ids.append(
                    self.inv_vocab[special_text.encode(encoding="utf-8")]
                )

        if last_index < len(text):
            tokens_ids.extend(self._encode_solo_text(text[last_index:]))

        return tokens_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs.  This is required
        for memory-efficient tokenization of large files that we cannot directly load into memory.

        Args:
            iterable (Iterable[str]): _description_

        Yields:
            Iterator[int]: _description_
        """
        ids = []
        for text in iterable:
            ids = self.encode(text)
            yield from ids  # == for id in ids \n yield id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text.

        Args:
            ids (list[int]): List of tokens ids.

        Returns:
            str: Original decoded text
        """

        tokens_bytes = b""
        for token_id in ids:
            # tokens_bytes.append(self.vocab[token_id])
            tokens_bytes = tokens_bytes + self.vocab[token_id]

        return tokens_bytes.decode(encoding="utf-8")


if __name__ == "__main__":
    vocab_filepath = "/Users/jrodriguez/Documentos/personal_projects/cs336/assignment1-basics/bpe_tokenizer/TinyStoriesV2-GPT4-train-vocab.json"
    merge_filepath = "/Users/jrodriguez/Documentos/personal_projects/cs336/assignment1-basics/bpe_tokenizer/TinyStoriesV2-GPT4-train-merges.txt"
    special_tokens = ["<|endoftext|>", "<|im_start|>"]

    bpe_tokenizer = Tokenizer.from_file(
        vocab_filepath, merge_filepath, special_tokens
    )

    from tests.test_tokenizer import (
        test_roundtrip_unicode_string_with_special_tokens,
    )

    # test_roundtrip_single_unicode_character()

    test_roundtrip_unicode_string_with_special_tokens()
    # sample_text = (
    #     " <|endoftext|> <|im_start|>Once the you have a vocabulary, you could, in principle, count how often bytes occur next \
    #     to each other in your text and begin merging them starting with the most frequent pair of bytes. However, \
    #     this is quite computationally expensive, since we'd have to go take a full pass over the corpus each time \
    #     we merge.   <|endoftext|> In addition, directly merging bytes across the corpus may result in tokens that differ only in \
    #     punctuation <|endoftext|>"
    # )

    # f_2 = open(
    #     "/Users/jrodriguez/Documentos/personal_projects/cs336/assignment1-basics/tests/fixtures/tinystories_sample.txt",
    #     mode="r",
    # )

    # sample_text = f_2.read()

    # text_ids = bpe_tokenizer.encode(sample_text)

    # print(text_ids)
    # print(len(text_ids))

    # decoded_text = bpe_tokenizer.decode(text_ids)

    # assert sample_text == decoded_text, (
    #     "Original text and decoded text are not equal"
    # )

    # from io import StringIO

    # f = StringIO(sample_text)

    # f_2 = open(
    #     "/Users/jrodriguez/Documentos/personal_projects/cs336/assignment1-basics/tests/fixtures/tinystories_sample.txt",
    #     mode="r",
    # )

    # generator = bpe_tokenizer.encode_iterable(f_2)

    # gen_ids = []
    # for id in generator:
    #     gen_ids.append(id)

    # print(gen_ids)
    # print(len(gen_ids))

    # assert text_ids == gen_ids
