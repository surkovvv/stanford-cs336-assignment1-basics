import pickle

def save_separately(input_file: str, output_file_prefix: str):
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    vocab = data['vocab']
    merges = data['merges']

    with open(output_file_prefix + '-vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    with open(output_file_prefix + '-merges.pkl', 'wb') as f:
        pickle.dump(merges, f)


if __name__ == "__main__":
    print("separeting TinyStories...")
    save_separately("TinyStoriesV2-train-bpe_tokenizer.pkl", "TinyStoriesV2-train-bpe_tokenizer")
    print("TinyStories separated!")

    print("separeting owt...")
    save_separately("owt-train-bpe_tokenizer.pkl", "owt-train-bpe_tokenizer")
    print("owt separated!")