import json
import multiprocessing as mp
import os
from collections import defaultdict

import regex as re

from src.utils.constants import MAX_THREADS, PAT
from src.utils.pretokenization import (
    count_byte_pairs,
    find_chunk_boundaries,
    get_bytes_tuple,
    get_max_pair,
    get_pretoken_count,
    merge,
    merge_efficient,
)


def pretokenize_text(text, special_tokens):
    """Pretokenize text

    Args:
        text (_type_): _description_
        special_tokens (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Get and remove special tokens before pre-tokenization
    pattern = "|".join(
        [re.escape(special_token) for special_token in special_tokens]
    )
    split_text = re.split(pattern, text)

    pretoken_counts = defaultdict(int)  # dict[tuple(bytes), int]
    # Pre-tokenize and obtain the initial pretoken_counts
    for small_text in split_text:
        if len(small_text) == 0:
            continue
        for match in re.finditer(PAT, small_text):
            string = match.group(0)
            pretoken_counts[get_bytes_tuple(string)] += 1

    return pretoken_counts


def pretokenize_text_2(text, special_tokens):
    """Pretokenize text

    Args:
        text (_type_): _description_
        special_tokens (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Get and remove special tokens before pre-tokenization
    pattern = "|".join(
        [re.escape(special_token) for special_token in special_tokens]
    )
    split_text = re.split(pattern, text)

    pretoken_counts = defaultdict(int)  # dict[tuple(bytes), int]
    # Pre-tokenize and obtain the initial pretoken_counts
    for small_text in split_text:
        if len(small_text) == 0:
            continue
        for match in re.finditer(PAT, small_text):
            string = match.group(0)
            pretoken_counts[get_bytes_tuple(string)] += 1

    return pretoken_counts


def pretokenize_text_binary(file_path, special_tokens):
    """_summary_

    Args:
        text (_type_): _description_
        split_special_token (_type_): _description_
    """

    num_processes = MAX_THREADS
    pretoken_counts = defaultdict(int)
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

            if len(chunks) == mp.cpu_count() or end == boundaries[-1]:
                with mp.Pool(mp.cpu_count()) as p:
                    small_pretoken_counts = p.starmap(
                        pretokenize_text_2,
                        [(chunk, special_tokens) for chunk in chunks],
                    )

                for small_pretoken_count in small_pretoken_counts:
                    for pretoken, count in small_pretoken_count.items():
                        pretoken_counts[pretoken] += count

                chunks = []

    idx_to_pretoken_counts = {
        idx: {"pretoken": pretoken, "counts": counts}
        for idx, (pretoken, counts) in enumerate(pretoken_counts.items())
    }
    return idx_to_pretoken_counts


def pretokenize_text_parallel(file_path, special_tokens):
    """Pretokenize text

    Args:
        text (_type_): _description_
        special_tokens (_type_): _description_

    Returns:
        _type_: _description_
    """

    with open(file_path, encoding="utf_8") as f:
        text = f.read()

    # Get and remove special tokens before pre-tokenization
    pattern = "|".join(
        [re.escape(special_token) for special_token in special_tokens]
    )
    split_text = re.split(pattern, text)

    # Pre-tokenize and obtain the initial pretoken_counts
    with mp.Pool(mp.cpu_count()) as p:
        small_pretoken_counts = p.map(get_pretoken_count, split_text)

    pretoken_counts = defaultdict(int)
    for small_pretoken_count in small_pretoken_counts:
        for pretoken, count in small_pretoken_count.items():
            pretoken_counts[pretoken] += count

    idx_to_pretoken_counts = {
        idx: {"pretoken": pretoken, "counts": counts}
        for idx, (pretoken, counts) in enumerate(pretoken_counts.items())
    }
    return idx_to_pretoken_counts


def train_bpe_slow(
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

    pretoken_counts = pretokenize_text(input_path, special_tokens)

    # Init vocab with ASCII character bytes and special tokens
    vocab = defaultdict(bytes)
    for idx, special_token in enumerate(special_tokens):
        vocab[idx] = special_token.encode(encoding="utf-8")
    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])

    # Init merges
    merges: list[tuple[bytes]] = []

    # Merge pairs until you reach vocab_size vocabulary size
    while len(vocab) < vocab_size:
        counts = defaultdict(int)

        # Naive implementation: Compute all the byte pair counts for each merge
        for pretoken, num_appaerances in pretoken_counts.items():
            byte_pair_count = count_byte_pairs(pretoken, num_appaerances)
            for byte_pair, count in byte_pair_count.items():
                counts[byte_pair] += count

        # Find the most common byte pair and merge
        bytes_tuple = get_max_pair(counts)
        merges.append(bytes_tuple)
        vocab[len(vocab)] = bytes_tuple[0] + bytes_tuple[1]
        pretoken_counts = merge(pretoken_counts, bytes_tuple)

    return vocab, merges


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

    print("Starting pre-tokenization")
    # idx_to_pretoken_counts = pretokenize_text_parallel(input_path, special_tokens)
    idx_to_pretoken_counts = pretokenize_text_binary(
        input_path, special_tokens
    )
    print("End pretokenization")
    # Init vocab with ASCII character bytes and special tokens
    vocab = {}
    for idx, special_token in enumerate(special_tokens):
        vocab[idx] = special_token.encode(encoding="utf-8")
    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])

    # Init merges
    merges: list[tuple[bytes]] = []

    # First count
    print("Starting first count...")
    pairs_counts = defaultdict(int)
    pairs_tokens: dict[tuple[bytes], dict[int, bool]]
    pairs_tokens = defaultdict(dict)
    for idx, dict_pretoken_count in idx_to_pretoken_counts.items():
        pretoken = dict_pretoken_count["pretoken"]
        counts = dict_pretoken_count["counts"]
        byte_pair_count = count_byte_pairs(pretoken, counts)
        for byte_pair, count in byte_pair_count.items():
            pairs_counts[byte_pair] += count
            pairs_tokens[byte_pair][idx] = True

    print("Starting merging process...")

    # Merge pairs until you reach vocab_size vocabulary size
    while len(vocab) < vocab_size:
        if len(vocab) % 100 == 0:
            print(f"Current vocabulary length: {len(vocab)}")
        if len(pairs_counts) == 0:
            print("WARNING: No more possible symbol pairs")
            print(f"Maximum vocab size reached: {len(vocab)}")
            break
        # Find the most common byte pair and merge
        bytes_tuple = get_max_pair(pairs_counts)
        merges.append(bytes_tuple)
        vocab[len(vocab)] = bytes_tuple[0] + bytes_tuple[1]

        # Merge and update data structures for efficient implementation
        pairs_counts, pairs_tokens, idx_to_pretoken_counts = merge_efficient(
            bytes_tuple, pairs_counts, pairs_tokens, idx_to_pretoken_counts
        )
        idx += 1

    return vocab, merges


def save_vocab_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    dir_path: str,
    name: str,
):
    """Function to save vocabulary and merges of a trained BPE tokenizer

    Args:
        vocab (dict[int, bytes]): Vocabulary.
        merges (list[tuple[bytes, bytes]]): Merges.
        dir_path (str): Directory to save the vocabulary and merges.
        name (str): Name to identify the trained BPE tokenizer.
    """

    os.makedirs(dir_path, exist_ok=True)

    vocab_json_path = os.path.join(dir_path, f"{name}-vocab.json")
    merges_txt_path = os.path.join(dir_path, f"{name}-merges.txt")

    with open(vocab_json_path, "w", encoding="utf-8") as f:
        vocab_str = {
            str(vocab_item): vocab_idx
            for vocab_idx, vocab_item in vocab.items()
        }
        json.dump(vocab_str, f, indent=2)

    with open(merges_txt_path, "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(f"{str(merge)}\n")


def read_vocab_merges(vocab_filepath: str, merges_filepath: str):
    with open(vocab_filepath, encoding="utf-8") as f:
        vocab_str_dict = json.load(f)

        vocab = {
            vocab_idx: eval(vocab_str)
            for vocab_str, vocab_idx in vocab_str_dict.items()
        }

    with open(merges_filepath, encoding="utf-8") as f:
        # merges_lines = f.readlines()
        merges = []
        for line in f:
            merges.append(eval(line))

    return vocab, merges


def find_repeated_elements(lst):
    from collections import Counter

    counts = Counter(lst)
    # Only keep elements with count > 1
    repeated = [item for item, count in counts.items() if count > 1]
    return repeated
