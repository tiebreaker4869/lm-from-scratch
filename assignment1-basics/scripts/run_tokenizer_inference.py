from cs336_basics.bpe_tokenizer import BPETokenizer
from argparse import ArgumentParser
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm

# Global tokenizer for worker processes
_tokenizer = None

def _init_worker(vocab_path, merges_path, special_tokens):
    global _tokenizer
    _tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)

def _encode_line(line):
    cleaned_line = line.strip()
    if not cleaned_line:
        return []
    return _tokenizer.encode(cleaned_line)

def main():
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_dir', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--batch_size', type=int, default=10000, help='Lines per batch for parallel processing')

    args = parser.parse_args()

    special_tokens = []

    vocab_path = os.path.join(args.tokenizer_dir, 'vocab.json')
    merges_path = os.path.join(args.tokenizer_dir, 'merges.txt')
    special_tokens_path = os.path.join(args.tokenizer_dir, 'special_tokens.txt')

    with open(special_tokens_path, 'r') as f:
        for line in f:
            special_token = line.strip('\n')
            special_tokens.append(special_token)

    num_workers = args.num_workers or os.cpu_count()
    dtype = np.uint16

    file_size = os.path.getsize(args.input_path)

    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(vocab_path, merges_path, special_tokens)
    ) as pool:
        with open(args.input_path, 'r') as f_in, open(args.output_path, 'wb') as f_out:
            pbar = tqdm(total=file_size, desc="Tokenizing", unit="B", unit_scale=True)
            batch = []
            batch_bytes = 0
            for line in f_in:
                batch.append(line)
                batch_bytes += len(line.encode('utf-8'))
                if len(batch) >= args.batch_size:
                    all_tokens = []
                    for tokens in pool.imap(_encode_line, batch):
                        if tokens:
                            all_tokens.extend(tokens)
                    if all_tokens:
                        f_out.write(np.array(all_tokens, dtype=dtype).tobytes())
                    pbar.update(batch_bytes)
                    batch = []
                    batch_bytes = 0
            # Process remaining lines
            if batch:
                all_tokens = []
                for tokens in pool.imap(_encode_line, batch):
                    if tokens:
                        all_tokens.extend(tokens)
                if all_tokens:
                    f_out.write(np.array(all_tokens, dtype=dtype).tobytes())
                pbar.update(batch_bytes)
            pbar.close()

if __name__ == "__main__":
    main()