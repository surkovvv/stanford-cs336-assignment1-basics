# print(chr(0))  # >> 
# print(chr(0).__repr__())  >> '\x00' == NULL
# print("this is a test" + chr(0) + " string")  >> this is a test string

# str1 = "макс verstapen"
# print(len(str1.encode("utf-8")))  >> 18
# print(len(str1.encode("utf-16"))) >> 30
# print(len(str1.encode("utf-32"))) >> 60
# supposely, it's more compact in terms of memory.

# def decode_utf_bytes_to_str_wrong(bytestring: bytes):
#     return "".join([bytes([b]).decode("utf-8") for b in bytestring])

# print(decode_utf_bytes_to_str_wrong("hello".encode('utf-8'))) >> 'hello'
# print(decode_utf_bytes_to_str_wrong("сосал?".encode('utf-8')))

# Traceback (most recent call last):
#   File "/devbox/home/n.surkov/stanford-cs336-assignment1-basics/cs336_basics/tokenizer.py", line 15, in <module>
#     print(decode_utf_bytes_to_str_wrong("сосал?".encode('utf-8')))
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/devbox/home/n.surkov/stanford-cs336-assignment1-basics/cs336_basics/tokenizer.py", line 12, in decode_utf_bytes_to_str_wrong
#     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
#                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd1 in position 0: unexpected end of data

# This function incorrectly assumes that each byte in a UTF-8 encoded string represents a standalone character, while some characters are represented by multiple bytes that must be decoded together.

# print(len("я".encode('utf-8'))) >> 2
# print(b'\xB7\xDC'.decode('utf-8')) >> Traceback (most recent call last):
#   File "/Users/tr3n1ttty/code projects/preps/cs 336/stanford-cs336-assignment1-basics/cs336_basics/tokenizer.py", line 29, in <module>
#     print(b'\xB7\xDC'.decode('utf-8'))
#           ~~~~~~~~~~~~~~~~~~^^^^^^^^^
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb7 in position 0: invalid start byte

# how it was invented: gpt said to me, that each correct seq with len = 2 in Unicode(according to Unicode spec)
#  must be presented as 110xxxxx 10xxxxxx => I've switched first and second symbols and give a random x's


import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# print(re.findall(PAT, "some text that i'll pre-tokenize"))
# # ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
# print(re.findall(PAT, "привет, хочу разобраться как работает этот регексп"))
# print(re.findall(PAT, "давай-ка я найду сложный текст, который будет... будет! ну, крутым короче будет"))

def bpe_example(corpus: list[list[str]], n_merges: int = 5):
    # vocab init
    all_bytes = bytes(range(256))
    vocab = [all_bytes[i : i + 1] for i in range(256)] + ["<endoftext>".encode("utf-8")]
    # pre-tokenization
    frequency_table: dict[tuple[bytes, ...], int] = {}
    for text in corpus:
        for pre_token in text.split(): # re.finditer(PAT, text):
            encoded_word = pre_token.encode('utf-8')  # -> bytes, but not 1 by 1
            tuple_of_bytes = tuple([encoded_word[i: i + 1] for i in range(len(encoded_word))])
            # for byte_id in range(len(tuple_of_bytes) - 1):  # assert?
            #     if (tuple_of_bytes[byte_id], tuple_of_bytes[byte_id + 1]) not in frequency_table:
            #         frequency_table[(tuple_of_bytes[byte_id], tuple_of_bytes[byte_id + 1])] = 0
            #     frequency_table[(tuple_of_bytes[byte_id], tuple_of_bytes[byte_id + 1])] += 1

            if tuple_of_bytes not in frequency_table:
                frequency_table[tuple_of_bytes] = 0

            frequency_table[tuple_of_bytes] += 1
    for n_iter in range(n_merges):
        pairs_counter = {}
        for unique_word_tuple in frequency_table:
            if len(unique_word_tuple) < 2:
                continue
            for idx in range(len(unique_word_tuple) - 1):
                pair = (unique_word_tuple[idx], unique_word_tuple[idx + 1])
                if pair not in pairs_counter:
                    pairs_counter[pair] = 0
                pairs_counter[pair] += frequency_table[unique_word_tuple]

        # print(pairs_counter)
        new_merge_pair = sorted(pairs_counter, key=lambda x: (pairs_counter[x], x))[-1]
        new_merge_symbol = b''.join(new_merge_pair)
        vocab.append(new_merge_symbol)

        previous_keys = list(frequency_table.keys())
        for unique_word_tuple in previous_keys:
            if len(unique_word_tuple) < 2:
                continue
            ids_to_replace = []
            for idx in range(len(unique_word_tuple) - 1):
                pair = (unique_word_tuple[idx], unique_word_tuple[idx + 1])
                if pair == new_merge_pair:
                    ids_to_replace.append(idx)

            new_tuple = tuple()
            prev_idx = 0
            for idx in ids_to_replace:
                new_tuple += unique_word_tuple[prev_idx:idx] + (new_merge_symbol, )
                # print(new_tuple)
                prev_idx = idx + 2
            
            if len(new_tuple):
                frequency_table[new_tuple] = frequency_table[unique_word_tuple]
                del frequency_table[unique_word_tuple]

        print(frequency_table)

    return vocab

    
corpus = [
    "low low low low low", 
    "lower lower widest widest widest",
    "newest newest newest newest newest newest"
]

vocab = bpe_example(corpus)
# # print(b''.join((b's', b'et')))
# test_tuple = ('h', 'e', 'r', 'e', 'i', 'a', 'm')  
# # change e r -> er
# print(test_tuple[:1] + ('er',) + test_tuple[1 + 2:])
# # change h e -> he
# print(test_tuple[:0] + ('he',) + test_tuple[0 + 2:])
# # change a m -> am
# print(test_tuple[:5] + ('am',) + test_tuple[5 + 2:])
# print(vocab[-6:])

import os

from pretokenization_example import find_chunk_boundaries


def pretokenize_chunk() -> tokens:
    pass


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    raise NotImplementedError