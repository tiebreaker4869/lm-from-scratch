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

def _encode_chunk(lines):
    """Encode a chunk of lines, return flat list of tokens."""
    all_tokens = []
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:
            all_tokens.extend(_tokenizer.encode(cleaned_line))
    return all_tokens

def main():
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_dir', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Lines per chunk for each worker')

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

    def chunk_generator(file_handle, chunk_size):
        """Yield chunks of lines from file."""
        chunk = []
        chunk_bytes = 0
        for line in file_handle:
            chunk.append(line)
            chunk_bytes += len(line.encode('utf-8'))
            if len(chunk) >= chunk_size:
                yield chunk, chunk_bytes
                chunk = []
                chunk_bytes = 0
        if chunk:
            yield chunk, chunk_bytes

    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(vocab_path, merges_path, special_tokens)
    ) as pool:
        with open(args.input_path, 'r') as f_in, open(args.output_path, 'wb') as f_out:
            pbar = tqdm(total=file_size, desc="Tokenizing", unit="B", unit_scale=True)
            chunks_with_bytes = list(chunk_generator(f_in, args.chunk_size))
            chunks = [c for c, _ in chunks_with_bytes]
            bytes_list = [b for _, b in chunks_with_bytes]

            for tokens, chunk_bytes in zip(pool.imap(_encode_chunk, chunks), bytes_list):
                if tokens:
                    f_out.write(np.array(tokens, dtype=dtype).tobytes())
                pbar.update(chunk_bytes)
            pbar.close()

if __name__ == "__main__":
    main()