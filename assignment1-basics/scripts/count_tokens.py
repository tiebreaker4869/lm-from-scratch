import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Count tokens in a tokenized bin file")
    parser.add_argument('file', type=str, help='Path to the bin file')
    args = parser.parse_args()

    tokens = np.memmap(args.file, dtype=np.uint16, mode='r')
    num_tokens = len(tokens)
    print(f"{num_tokens:,} tokens ({num_tokens / 1e6:.2f}M)")

if __name__ == "__main__":
    main()
