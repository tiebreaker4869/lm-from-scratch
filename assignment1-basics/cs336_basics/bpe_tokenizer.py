from .tokenizer import Tokenizer, pretokenize
from dataclasses import dataclass
import regex as re
from typing import Iterable, Iterator
import json
from collections import defaultdict
from multiprocessing import Pool
from cs336_basics.common import gpt2_bytes_to_unicode
import os
from functools import partial

@dataclass(frozen=True)
class BPETokenizerParams:
    """All params needed to specify a bpe tokenizer."""
    vocab: dict[int, bytes] # index -> bytes
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None
    
class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams):
        self.vocab = params.vocab
        self.special_tokens = params.special_tokens
        self.special_token_to_idx = dict()
        self.special_tokens_regex = None
        if self.special_tokens:
            self._init_special_tokens()
        self.bytes_to_idx = {bs : idx for idx, bs in self.vocab.items()}
        self._init_merges(params.merges)
        
    def _init_merges(self, merges: list[tuple[bytes, bytes]]):
        # pair -> (priority, merged_idx), lower priority = earlier merge
        self.merge_rules: dict[tuple[int, int], tuple[int, int]] = {}
        for priority, (b1, b2) in enumerate(merges):
            idx1, idx2 = self.bytes_to_idx[b1], self.bytes_to_idx[b2]
            merge_idx = self.bytes_to_idx[b1 + b2]
            self.merge_rules[(idx1, idx2)] = (priority, merge_idx)

    def _init_special_tokens(self):
        # initialize special token regex
        escaped_tokens = [re.escape(st) for st in self.special_tokens]
        escaped_tokens.sort(key = len, reverse = True)
        self.special_tokens_regex = "(" + "|".join(escaped_tokens) + ")"
        
        # add new special tokens
        special_tokens_set = set(self.special_tokens)
        bytes_special_tokens_set = set({s.encode("utf-8") for s in special_tokens_set})
        in_vocab_special_tokens = set({t for t in self.vocab.values()if t in bytes_special_tokens_set})
        in_vocab_special_tokens = set({bs.decode("utf-8") for bs in in_vocab_special_tokens})
        nxt_idx = max(self.vocab.keys()) + 1
        new_special_tokens = special_tokens_set - in_vocab_special_tokens
        for new_special_token in new_special_tokens:
            self.vocab[nxt_idx] = new_special_token.encode("utf-8")
            nxt_idx += 1
        
        # add reverse mapping for special tokens
        for token, bstring in self.vocab.items():
            if bstring in bytes_special_tokens_set:
                string = bstring.decode("utf-8")
                self.special_token_to_idx[string] = token

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab = dict()
        merges = []
        unicode_to_byte = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, mode="r") as f:
            content = f.read()
            vocab = json.loads(content)
            vocab = {v: b"".join([bytes([unicode_to_byte[ch]]) for ch in k]) for k, v in vocab.items()}
        with open(merges_filepath, mode="r") as f:
            for line in f:
                cleaned_line = line.strip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    word1, word2 = cleaned_line.split(" ")
                    merges.append((bytes([unicode_to_byte[ch] for ch in word1]), bytes([unicode_to_byte[ch] for ch in word2])))
        params = BPETokenizerParams(vocab, merges, special_tokens)
        tokenizer = BPETokenizer(params)
        return tokenizer
        
    def encode(self, string: str) -> list[int]:
        splitted_iter = re.splititer(self.special_tokens_regex, string) if self.special_tokens else [string]
        indices = []
        for chunk in splitted_iter:
            if not chunk:
                continue
            if chunk in self.special_token_to_idx:
                indices.append(self.special_token_to_idx[chunk])
            else:
                pretokens = pretokenize(chunk)
                for pretoken in pretokens:
                    if not pretoken:
                        continue
                    bs = pretoken.encode("utf-8")
                    bs_list = [bs[i:i+1] for i in range(len(bs))]
                    idxs = [self.bytes_to_idx[b] for b in bs_list]
                    idxs = self._encode_pretoken(idxs)
                    indices.extend(idxs)
        return indices
    def decode(self, tokens: list[int]) -> str:
        bytes_list = [self.vocab[token] for token in tokens]
        string = b"".join(bytes_list).decode("utf-8", errors='replace')
        return string
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            tokens = self.encode(string)
            for token in tokens:
                yield token

    def _encode_pretoken(self, idxs: list[int]) -> list[int]:
        """Encode a pretoken using greedy BPE merging."""
        while len(idxs) >= 2:
            # Find the pair with lowest priority (earliest in merge list)
            best_priority = float('inf')
            best_pos = -1

            for i in range(len(idxs) - 1):
                pair = (idxs[i], idxs[i + 1])
                if pair in self.merge_rules:
                    priority, _ = self.merge_rules[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pos = i

            if best_pos == -1:
                break  # No more merges possible

            # Apply the merge
            _, new_idx = self.merge_rules[(idxs[best_pos], idxs[best_pos + 1])]
            idxs = idxs[:best_pos] + [new_idx] + idxs[best_pos + 2:]

        return idxs

def get_initial_stats(pretoken_freq):
    stats = defaultdict(int)
    pair_to_words = defaultdict(set)
    
    for word, freq in pretoken_freq.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            stats[pair] += freq
            pair_to_words[pair].add(word)
            
    return stats, pair_to_words

def stream_chunks(input_path, max_chunk_size=7 * 1024 * 1024):
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        current_batch = []
        current_batch_size = 0
        for line in f:
            current_batch.append(line)
            current_batch_size += len(line)
            if current_batch_size >= max_chunk_size:
                yield current_batch
                current_batch = []
                current_batch_size = 0
        if current_batch:
            yield current_batch

def process_chunks(lines: list[str], special_tokens_set, special_tokens_regex) -> defaultdict[tuple[bytes, ...], int]:
    pretoken_freq = defaultdict(int)
    text = "".join(lines)
    
    if special_tokens_regex:
        chunks = re.split(special_tokens_regex, text)
        chunks = [c for c in chunks if c and c not in special_tokens_set]
    else:
        chunks = [text]

    for chunk in chunks:
        for pretoken in pretokenize(chunk):
            bs = pretoken.encode("utf-8")
            token_tuple = tuple(bs[i:i+1] for i in range(len(bs)))
            pretoken_freq[token_tuple] += 1
    return pretoken_freq

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str] | None = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = special_tokens or []
    escaped_special_tokens = [re.escape(t) for t in special_tokens]
    escaped_special_tokens.sort(key=len, reverse=True)
    special_tokens_regex = ("(" + "|".join(escaped_special_tokens) + ")") if escaped_special_tokens else None
    
    vocab = {idx: bytes([idx]) for idx in range(256)}
    merges = []
    nxt_idx = 256
    
    for token in special_tokens:
        vocab[nxt_idx] = token.encode("utf-8")
        nxt_idx += 1

    pretoken_freq = defaultdict(int)

    special_tokens_set = set(special_tokens)
    
    num_procs = os.cpu_count()

    worker_func = partial(process_chunks, special_tokens_set=special_tokens_set, special_tokens_regex=special_tokens_regex)
    
    with Pool(processes=num_procs) as pool:
        for result in pool.imap_unordered(worker_func, stream_chunks(input_path)):
            for k, v in result.items():
                pretoken_freq[k] += v        

    # stats: (pair) -> total_count
    # pair_to_words: (pair) -> set(word_tuples)
    stats, pair_to_words = get_initial_stats(pretoken_freq)

    while len(vocab) < vocab_size:
        if not stats:
            break
            
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]), default=(None, 0))[0]
        
        if not best_pair or stats[best_pair] <= 0:
            break

        p1, p2 = best_pair
        new_token = p1 + p2
        merges.append(best_pair)
        vocab[nxt_idx] = new_token
        
        words_to_update = list(pair_to_words[best_pair])
        
        for word in words_to_update:
            count = pretoken_freq[word]
            
            for i in range(len(word) - 1):
                old_p = (word[i], word[i+1])
                stats[old_p] -= count
                if stats[old_p] <= 0:
                    del stats[old_p]
                pair_to_words[old_p].discard(word)

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == p1 and word[i+1] == p2:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            
            del pretoken_freq[word]
            pretoken_freq[new_word] += count
            
            for i in range(len(new_word) - 1):
                new_p = (new_word[i], new_word[i+1])
                stats[new_p] += count
                pair_to_words[new_p].add(new_word)

        nxt_idx += 1
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]
        
    return vocab, merges