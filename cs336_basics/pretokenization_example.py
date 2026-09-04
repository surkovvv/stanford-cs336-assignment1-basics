import os
import regex as re

from typing import BinaryIO
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_DEFAULT_TOKENS = 256

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


def pretokenize_chunk(path: str, start: int, end: int, special_tokens: list[str]) -> dict[bytes, int]:
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    pretoken_counter: dict[str, int] = {}
    for doc in re.split("|".join([re.escape(special_tokens)]), chunk):
        for pretoken in re.finditer(PAT, doc):
            pretoken = pretoken.group(0)
            if pretoken not in pretoken_counter:
                pretoken_counter[pretoken] = 0

            pretoken_counter[pretoken] += 1

    return pretoken_counter


def main(path: str, special_tokens=["<|endoftext|>"], vocab_size: int = 666):
    with open(path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((path, start, end, special_tokens))
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(pretokenize_chunk, tasks)
    pretokens_counter = {}
    for pretoken_counts4chunk in results:
        pretokens_counter.update(pretoken_counts4chunk)

    # print(pretokens_counter)
    # print(special_tokens[0] in pretokens_counter)

    pair_counters: dict[tuple[bytes, bytes], int] = {}
    pair_to_pretokens: dict[tuple[bytes, bytes], set[str]] = {}  # defaultdict pls
    pretoken_to_current_view: dict[str, tuple[bytes, ...]] = {}
    for pretoken in pretokens_counter:
        encoded_pretoken = pretoken.encode('utf-8')  # -> bytes, but not 1 by 1
        tuple_of_bytes = tuple([encoded_pretoken[i: i + 1] for i in range(len(encoded_pretoken))])
        pretoken_to_current_view[pretoken] = tuple_of_bytes
        for idx in range(len(tuple_of_bytes) - 1):
            pair = tuple_of_bytes[idx: idx + 2]
            if pair not in pair_counters:
                pair_counters[pair] = 0
            pair_counters[pair] += 1

            if pair not in pair_to_pretokens:
                pair_to_pretokens[pair] = set()
            pair_to_pretokens[pair].add(pretoken)

    num_merges = vocab_size - len(special_tokens) - NUM_DEFAULT_TOKENS
    all_bytes = bytes(range(NUM_DEFAULT_TOKENS))
    vocab = [all_bytes[i : i + 1] for i in range(NUM_DEFAULT_TOKENS)] + [st.encode("utf-8") for st in special_tokens]
    merges: list[tuple[bytes, bytes]] = []

    for _ in range(num_merges):
        sorted_pairs = sorted(pair_counters, key=lambda x: (pair_counters[x], x))
        new_merge_pair = sorted_pairs[-1]
        print(sorted_pairs[-2:])
        new_merge_symbol = b''.join(new_merge_pair)

        vocab.append(new_merge_symbol)
        merges.append(new_merge_pair)

        for pretoken in pair_to_pretokens[new_merge_pair]:
            current_view = pretoken_to_current_view[pretoken]  # not splitted by pairs, just view
            pairs_split = [current_view[idx: idx + 2] for idx in range(len(current_view) - 1)]
            previous_pair_was_merged = False
            merged_pairs_ids = []
            merged_pair_counter = 0
            for pair_idx, current_pair in enumerate(pairs_split):
                if previous_pair_was_merged:
                    previous_pair_was_merged = False
                    continue
                if current_pair == new_merge_pair:
                    merged_pairs_ids.append(pair_idx)
                    previous_pair_was_merged = True
                    merged_pair_counter += 1
                else:
                    previous_pair_was_merged = False

            new_view = tuple()
            prev_idx = 0
            for idx in merged_pairs_ids:
                new_view += current_view[prev_idx: idx] + (new_merge_symbol, )
                prev_idx = idx + 2
            
            new_view += current_view[prev_idx:]
            
            pretoken_to_current_view[pretoken] = new_view
            new_pairs_split = [new_view[idx: idx + 2] for idx in range(len(new_view) - 1)]

            pair_counters[new_merge_pair] -= merged_pair_counter
            for curr_pair in new_pairs_split:
                if new_merge_symbol in curr_pair:
                    if curr_pair not in pair_counters:
                        pair_counters[curr_pair] = 0
                    pair_counters[curr_pair] += 1
                    pair_to_pretokens[curr_pair] = pretoken
            
    return vocab, merges


# def test_pretoken_counter():
#     pretoken_counter = pretokenize_chunk(path="data/tsv2-test.txt", start=0, end=1110, special_tokens="<|endoftext|>")
    


if __name__ == "__main__":
    # test_pretoken_counter()
    path = "data/tsv2-test.txt"
    special_tokens=["<|endoftext|>"]

    vocab_size = 258

    vocab, merges = main(path, special_tokens, vocab_size)

    print(vocab)
    print('=' * 100)
    print(merges)