import os
from collections import defaultdict
from typing import BinaryIO

import regex as re

from src.utils.constants import PAT


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


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
                index_1 += 1
                index_2 += 1
        if index_1 < len(pretoken):
            new_pretoken.append(pretoken[index_1])
        merge_pretoken_counts[tuple(new_pretoken)] = count

    return merge_pretoken_counts


def merge_efficient(
    bytes_tuple: tuple[bytes],
    pairs_counts: dict[tuple[bytes], int],
    pairs_tokens: dict[tuple[bytes], dict[int, bool]],
    idx_to_pretoken_counts: dict[int, dict],
):
    # Find the associated
    associated_idxs = pairs_tokens[bytes_tuple].keys()
    joined_bytes = bytes_tuple[0] + bytes_tuple[1]

    sum_counts = 0
    for idx in associated_idxs:
        pretoken = idx_to_pretoken_counts[idx]["pretoken"]
        counts = idx_to_pretoken_counts[idx]["counts"]
        sum_counts += counts
        new_pretoken = []
        prev_pair = None
        current_pair = None
        next_pair = None
        index_1, index_2 = 0, 1
        while index_2 < len(pretoken):
            current_pair = pretoken[index_1], pretoken[index_2]
            if current_pair[0] == bytes_tuple[0] and current_pair[1] == bytes_tuple[1]:
                # Select and update previous pair if possible
                if index_1 > 0:
                    prev_pair = pretoken[index_1 - 1], pretoken[index_1]

                    # Update pair_counts
                    pairs_counts[prev_pair] -= counts
                    if pairs_counts[prev_pair] == 0:
                        del pairs_counts[prev_pair]

                # Select and update next pair if possible
                if index_2 < (len(pretoken) - 1):
                    next_pair = pretoken[index_2], pretoken[index_2 + 1]

                    # Update pair_counts
                    pairs_counts[next_pair] -= counts

                    if pairs_counts[next_pair] == 0:
                        del pairs_counts[next_pair]

                new_pretoken.append(joined_bytes)
                index_1 += 2
                index_2 += 2
            else:
                new_pretoken.append(current_pair[0])
                index_1 += 1
                index_2 += 1

        if index_1 < len(pretoken):
            new_pretoken.append(pretoken[index_1])

        new_pretoken = tuple(new_pretoken)
        # Update pretoken
        idx_to_pretoken_counts[idx]["pretoken"] = new_pretoken

        # Compute counts for the new pretoken
        for bytes_1, bytes_2 in zip(new_pretoken[:-1], new_pretoken[1:]):
            if bytes_1 == joined_bytes or bytes_2 == joined_bytes:
                pairs_counts[(bytes_1, bytes_2)] += counts
                pairs_tokens[(bytes_1, bytes_2)][idx] = True

    del pairs_counts[bytes_tuple]
    del pairs_tokens[bytes_tuple]
    return (pairs_counts, pairs_tokens, idx_to_pretoken_counts)


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


def get_max_pair(counts: dict[tuple[int, int], int]):
    pairs = get_top_max_values(counts)  # max(counts, key=counts.get)

    return max(pairs)


def transform_2_bytes(integer):
    return bytes([integer])


def get_bytes_tuple(string):
    bytes_seq = list(string.encode("utf-8"))
    list_bytes = []
    for byte in bytes_seq:
        list_bytes.append(bytes([byte]))

    return tuple(list_bytes)


def get_pretoken_count(text: str):
    pretoken_counts = defaultdict(int)
    if len(text) > 0:
        for match in re.finditer(PAT, text):
            string = match.group(0)
            pretoken_counts[get_bytes_tuple(string)] += 1

    return pretoken_counts


if __name__ == "__main__":
    ## Usage
    input_path = "/Users/jrodriguez/Documentos/personal_projects/cs336/assignment1-basics/data/TinyStories-train.txt"
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
