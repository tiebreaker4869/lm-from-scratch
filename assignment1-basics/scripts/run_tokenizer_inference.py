from cs336_basics.bpe_tokenizer import BPETokenizer
from argparse import ArgumentParser
import numpy as np
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_dir', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    args = parser.parse_args()
    
    special_tokens = []
    
    vocab_path = os.path.join(args.tokenizer_dir, 'vocab.json')
    merges_path = os.path.join(args.tokenizer_dir, 'merges.txt')
    special_tokens_path = os.path.join(args.tokenizer_dir, 'special_tokens.txt')
    
    
    with open(special_tokens_path, 'r') as f:
        for line in f:
            special_token = line.strip('\n')
            special_tokens.append(special_token)
    
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)
    
    dtype = np.uint16
    
    with open(args.input_path, 'r') as f1, open(args.output_path, 'wb') as f2:
        for line in f1:
            cleaned_line = line.strip()
            if not line:
                continue
            tokens = tokenizer.encode(cleaned_line)
            f2.write(np.array(tokens, dtype=dtype).tobytes())

if __name__ == "__main__":
    main()