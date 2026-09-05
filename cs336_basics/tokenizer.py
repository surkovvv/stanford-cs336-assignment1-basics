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

from pretokenization_example import main
import pickle


def train_tokenizer(path, vocab_size, special_tokens, output_path):
    vocab, merges = main(path, special_tokens, vocab_size)
    with open(output_path, "wb") as f:
        pickle.dump(
            {
            "vocab": vocab,
            "merges": merges,
            },
            f,
        )

if __name__ == "__main__":
    # path = "data/TinyStoriesV2-GPT4-train.txt" # "data/TinyStoriesV2-GPT4-train.txt"
    # vocab_size = 10_000
    # special_tokens = ["<|endoftext|>"]
    # output_path = "data/results/TinyStoriesV2-train-bpe_tokenizer.pkl"
    # train_tokenizer(path, vocab_size, special_tokens, output_path)

#     uv run python -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(25)"
# Sat Sep  5 20:58:35 2026    profile.prof

#          492998063 function calls (492997434 primitive calls) in 76.148 seconds

#    Ordered by: cumulative time
#    List reduced from 864 to 25 due to restriction <25>

#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#      46/1    0.000    0.000   76.148   76.148 {built-in method builtins.exec}
#         1    0.000    0.000   76.148   76.148 cs336_basics/tokenizer.py:1(<module>)
#         1    0.012    0.012   76.132   76.132 cs336_basics/tokenizer.py:130(train_tokenizer)
#         1    0.985    0.985   76.118   76.118 /Users/tr3n1ttty/code projects/preps/cs 336/stanford-cs336-assignment1-basics/cs336_basics/pretokenization_example.py:74(main)
#     15227   29.677    0.002   54.562    0.004 {built-in method builtins.max}
#     30/26    0.000    0.000   40.329    1.551 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/connection.py:395(_recv)
#     78/74   20.018    0.257   40.329    0.545 {built-in method posix.read}
#        13    0.000    0.000   39.781    3.060 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/connection.py:251(recv)
#     15/13    0.000    0.000   39.758    3.058 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/connection.py:434(_recv_bytes)
# 490180425   24.885    0.000   24.885    0.000 /Users/tr3n1ttty/code projects/preps/cs 336/stanford-cs336-assignment1-basics/cs336_basics/pretokenization_example.py:121(<lambda>)
#         1    0.000    0.000   20.406   20.406 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/pool.py:738(__exit__)
#         1    0.000    0.000   20.398   20.398 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/pool.py:654(terminate)
#        19    0.000    0.000   20.380    1.073 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/util.py:276(__call__)
#         1    0.000    0.000   20.380   20.380 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/pool.py:680(_terminate_pool)
#         1    0.000    0.000   20.314   20.314 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/pool.py:671(_help_stuff_finish)
#         1    0.002    0.002   20.313   20.313 {method 'acquire' of '_multiprocessing.SemLock' objects}
#       3/1    0.000    0.000   20.311   20.311 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/threading.py:1001(_bootstrap)
#       3/1    0.000    0.000   20.311   20.311 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/threading.py:1028(_bootstrap_inner)
#       3/1    0.000    0.000   20.311   20.311 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/threading.py:984(run)
#         1    0.000    0.000   20.311   20.311 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/pool.py:573(_handle_results)
#         1    0.000    0.000   20.310   20.310 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/pool.py:527(_handle_tasks)
#         1    0.000    0.000   20.309   20.309 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/pool.py:369(starmap)
#         1    0.000    0.000   20.309   20.309 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/pool.py:767(get)
#        24    0.000    0.000    0.849    0.035 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/pool.py:500(_wait_for_updates)
#        51    0.000    0.000    0.837    0.016 /Users/tr3n1ttty/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/lib/python3.13/multiprocessing/connection.py:1160(wait)

    path = "data/owt_train.txt"
    vocab_size = 32_000
    special_tokens = ["<|endoftext|>"]
    output_path = "data/results/owt-train-bpe_tokenizer.pkl"
    train_tokenizer(path, vocab_size, special_tokens, output_path)