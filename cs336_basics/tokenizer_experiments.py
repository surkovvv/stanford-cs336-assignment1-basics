from cs336_basics.tokenizer import Tokenizer
import time


def sample_n_texts(path, n: int = 10, separator: str = "<|endoftext|>") -> list[str]:
    # to simplyfiy - just first n
    with open(path) as f:
        stories = []
        current_story = []
        for line in f:
            if separator in line:
                parts = line.split(separator)
                current_story.append(parts[0])
                stories.append("".join(current_story).strip())

                if len(stories) == n:
                    break

                current_story = [parts[1]] if len(parts) > 1 else []
            else:
                current_story.append(line)

    # print(stories[0])

    return stories


def calc_compression_ratio_for_n_texts(texts: list[str], tokenizer: Tokenizer) -> float:
    sum_bytes = 0.0
    sum_tokens = 0
    for text in texts:
        sum_bytes += len(text.encode("utf-8"))
        sum_tokens += len(tokenizer.encode(text))

    ratio = sum_bytes / sum_tokens
    return ratio

def calc_throughput(texts: list[str], tokenizer: Tokenizer) -> float:
    total_num_of_bytes = 0

    start_time = time.perf_counter()
    for text in texts:
        total_num_of_bytes += len(text.encode("utf-8"))
        tokenizer.encode(text)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"Время выполнения: {execution_time:.6f} сек.")
    throughtput = total_num_of_bytes / execution_time

    return throughtput


if __name__ == "__main__":
    print("TinyStories: ")

    ts_vocab_path = "data/results/TinyStoriesV2-train-bpe_tokenizer-vocab.pkl"
    ts_merges_path = "data/results/TinyStoriesV2-train-bpe_tokenizer-merges.pkl"

    ts_tokenizer = Tokenizer.from_files(
        vocab_filepath=ts_vocab_path,
        merges_filepath=ts_merges_path,
        special_tokens=["<|endoftext|>"]
    )

    ts_file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    ts_texts = sample_n_texts(ts_file_path, n=10)
    comp_ratio = calc_compression_ratio_for_n_texts(ts_texts, ts_tokenizer)
    print("TinyStories compression ratio: ", comp_ratio)  # >> 4.0371761171610965

    print("=" * 99)
    print("owt: ")
    
    owt_vocab_path = "data/results/owt-train-bpe_tokenizer-vocab.pkl"
    owt_merges_path = "data/results/owt-train-bpe_tokenizer-merges.pkl"

    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath=owt_vocab_path,
        merges_filepath=owt_merges_path,
        special_tokens=["<|endoftext|>"]
    )

    owt_file_path = "data/owt_valid.txt"
    owt_texts = sample_n_texts(owt_file_path, n=10)
    comp_ratio = calc_compression_ratio_for_n_texts(owt_texts, owt_tokenizer)
    print("owt compression ratio: ", comp_ratio)  # >> 4.503748661192431

    print("=" * 100)
    comp_ratio = calc_compression_ratio_for_n_texts(owt_texts, ts_tokenizer)
    print("owt compression ratio w TS tokenizer: ", comp_ratio)  # >> 3.4030213110331804
    
    print("=" * 100)
    TRIALS = 1
    ts_throughputs = []
    ts_texts = sample_n_texts(ts_file_path, n=100)
    for trial in range(TRIALS):
        ts_throughput = calc_throughput(ts_texts, ts_tokenizer)
        ts_throughputs.append(ts_throughput)
    
    print("TS tokenizer mean throughput: ", sum(ts_throughputs) / TRIALS)  # >> 52248.6957402083

    owt_throughputs = []
    owt_texts = sample_n_texts(owt_file_path, n=100)
    for trial in range(TRIALS):
        owt_throughput = calc_throughput(owt_texts, owt_tokenizer)
        owt_throughputs.append(owt_throughput)
    
    print("OWT tokenizer mean throughput: ", sum(owt_throughputs) / TRIALS) # >> 6999.8404919530385

    print(calc_throughput(owt_texts, ts_tokenizer))