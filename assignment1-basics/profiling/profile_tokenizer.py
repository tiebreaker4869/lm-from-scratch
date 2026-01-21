from cs336_basics.bpe_tokenizer import train_bpe
from argparse import ArgumentParser
import time


def main():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/TinyStories-valid.txt")
    parser.add_argument("--vocab_size", type=int, default=5000)
    args = parser.parse_args()
    start_time = time.time()
    _, _ = train_bpe(args.data, args.vocab_size, special_tokens=["<|endoftext|>"])
    end_time = time.time()
    print(f"BPE training takes {end_time - start_time:.4f} s.")
    

if __name__ == "__main__":
    main()