# print(chr(0))  >> 
# print(chr(0).__repr__()) >> '\x00' == NULL
# print("this is a test" + chr(0) + " string") >> this is a test string

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


PAT = r"""(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

import regex as re

print(re.findall(PAT, "some text that i'll pre-tokenize"))
# ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
print(re.findall(PAT, "привет, хочу разобраться как работает этот регексп"))
print(re.findall(PAT, "давай-ка я найду сложный текст, который будет... будет! ну, крутым короче будет"))
