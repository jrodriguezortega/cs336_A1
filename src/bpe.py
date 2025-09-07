from collections import defaultdict

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def count_byte_pairs(pretoken: bytes, num_appaerances: int):
    """Compute the count of byte-pairs for a byte sequence of a pre-token

    Args:
        bytes_seq (list(byte)): Sequence of bytes of a pre-token
        num_appaerances (_type_): _description_

    Returns:
        _type_: _description_
    """
    counts = defaultdict(int)
    for bytes_1, bytes_2 in zip(pretoken[:-1], pretoken[1:]):
        counts[(bytes_1, bytes_2)] += num_appaerances

    return counts


def merge(pretoken_counts: dict[tuple[bytes], int], bytes_tuple: tuple[bytes]):
    merge_pretoken_counts = defaultdict(int)
    for pretoken, count in pretoken_counts.items():
        new_pretoken = []
        index_1, index_2 = 0, 1
        while index_2 < len(pretoken):
            bytes_1, bytes_2 = pretoken[index_1], pretoken[index_2]
            if bytes_1 == bytes_tuple[0] and bytes_2 == bytes_tuple[1]:
                new_pretoken.append(bytes_1 + bytes_2)
                index_1 += 2
                index_2 += 2
            else:
                new_pretoken.append(bytes_1)
                # new_pretoken.append(bytes_2)
                index_1 += 1
                index_2 += 1
        if index_1 < len(pretoken):
            new_pretoken.append(pretoken[index_1])
        merge_pretoken_counts[tuple(new_pretoken)] = count

    return merge_pretoken_counts


def get_top_max_values(counts):
    """Get top k maximum keys.

    Args:
        counts (dict[bytes, int]): Counts of the pretoken bytes.
    Return:
        list: List the maximum keys.
    """

    max_value = max(counts.values())

    max_keys = [key for key in counts.keys() if counts[key] == max_value]

    return max_keys


def get_max_pair(counts: dict[tuple[int, int], int], vocab: dict[int, bytes]):
    pairs = get_top_max_values(counts)  # max(counts, key=counts.get)

    if len(pairs) == 1:
        return pairs[0]
    elif len(pairs) > 1:  # Get the greater lexicographical pair
        # Get each pair in string
        string_pairs = []
        string_2_byte = {}
        for pair in pairs:
            string_1 = pair[0].decode()
            string_2 = pair[1].decode()
            string_2_byte[(string_1, string_2)] = pair

            string_pairs.append((string_1, string_2))

        max_string_pair = max(string_pairs)

        return string_2_byte[max_string_pair]

    else:
        raise ValueError()


def get_bytes_tuple(string):
    bytes_seq = list(string.encode("utf-8"))
    list_bytes = []
    for byte in bytes_seq:
        list_bytes.append(bytes([byte]))

    return tuple(list_bytes)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Function to train the byte-level BPE tokenizer

    Args:
        input_path (str): Path to file where the BPE should be trained.
        vocab_size (int): The maximum vocabulary size including the initial ASCII character bytes, the special tokens and the merge generated tokens.
        special_tokens (list[str]): A list with the special tokens of our tokenizer.

    Returns:
        vocab (dict[int, bytes]): The vocabulary of our tokenizer.
        merges (list[tuple[bytes, bytes]]): The merges resulted from the training.
    """
    with open(input_path, encoding="utf_8") as f:
        text = f.read()

    # Get and remove special tokens before pre-tokenization
    pattern = "|".join([re.escape(special_token) for special_token in special_tokens])
    split_text = re.split(pattern, text)

    # Init vocab with ASCII character bytes and special tokens
    vocab = defaultdict(bytes)
    for idx, special_token in enumerate(special_tokens):
        vocab[idx] = special_token.encode(encoding="utf-8")
    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])

    # Init merges
    merges: list[tuple[bytes]] = []

    pretoken_counts = defaultdict(int)  # dict[tuple(bytes), int]
    # Pre-tokenize and obtain the initial pretoken_counts
    for small_text in split_text:
        if len(small_text) == 0:
            continue

        for match in re.finditer(PAT, small_text):
            string = match.group(0)
            pretoken_counts[get_bytes_tuple(string)] += 1

    # Merge pairs until you reach vocab_size vocabulary size
    while len(vocab) < vocab_size:
        counts = defaultdict(int)

        # TODO: Substitute this naive implementation
        for pretoken, num_appaerances in pretoken_counts.items():
            byte_pair_count = count_byte_pairs(pretoken, num_appaerances)
            for byte_pair, count in byte_pair_count.items():
                counts[byte_pair] += count

        # Find the most common byte pair and merge
        bytes_tuple = get_max_pair(counts, vocab)
        merges.append(bytes_tuple)
        vocab[len(vocab)] = bytes_tuple[0] + bytes_tuple[1]
        pretoken_counts = merge(pretoken_counts, bytes_tuple)

    return vocab, merges


if __name__ == "__main__":
    import os
    import time

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # long_text = "<|endoftext|> Once you have a vocabulary, you could, in principle, count how often bytes occur next \
    # to each other in your text and begin merging them starting with the most frequent pair of bytes. However, \
    # this is quite computationally expensive, since we’d have to go take a full pass over the corpus each time \
    # we merge. In addition, directly merging bytes across the corpus may result in tokens that differ only in \
    # punctuation <|endoftext|>"

    # print(re.findall(PAT, "some text that i'll pre-tokenize"))

    # print(re.findall(PAT, long_text))
    FIXTURES = "/Users/jrodriguez/Documentos/personal_projects/cs336/assignment1-basics/tests/fixtures"

    input_path = os.path.join(FIXTURES, "corpus.en")
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()

    print(vocab)
    print("-" * 30)
    print(merges)
    print(f"Elapsed time: {end_time - start_time:.3f}s")
