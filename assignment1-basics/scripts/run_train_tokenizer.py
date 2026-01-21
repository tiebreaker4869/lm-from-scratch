from cs336_basics.bpe_tokenizer import train_bpe
from cs336_basics.common import gpt2_bytes_to_unicode
from argparse import ArgumentParser
import os
import json

def main():
    parser = ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--special_tokens", type=str, nargs="*")
    args = parser.parse_args()
    
    vocab, merges = train_bpe(args.corpus, args.vocab_size, args.special_tokens)
    
    byte_to_unicode = gpt2_bytes_to_unicode()
    
    output_vocab_path = os.path.join(args.output_dir, "vocab.json")
    output_merge_path = os.path.join(args.output_dir, "merges.txt")
    output_special_tokens_path = os.path.join(args.output_dir, "special_tokens.txt")
    
    serialized_vocab = dict()
    for token, bs in vocab.items():
        bs = [byte_to_unicode[int(b)] for b in bs]
        serialized_word = "".join(bs)
        serialized_vocab[serialized_word] = token
    
    with open(output_vocab_path, "w") as f:
        json.dump(serialized_vocab, f)
    
    with open(output_merge_path, "w") as f:
        for b1, b2 in merges:
            b1 = [byte_to_unicode[int(b)] for b in b1]
            b2 = [byte_to_unicode[int(b)] for b in b2]
            s1, s2 = "".join(b1), "".join(b2)
            f.write(f"{s1} {s2}\n")
    with open(output_special_tokens_path, "w") as f:
        for special_token in args.special_tokens:
            f.write(f"{special_token}\n")

if __name__ == "__main__":
    main()