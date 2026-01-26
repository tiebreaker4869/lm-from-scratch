import torch
from torch import nn
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.models import softmax
import random

def lm_decode(model: nn.Module, tokenizer: Tokenizer, prompt: str, stop_tokens: list[str] = ["<|endoftext|>"], max_tokens: int | None = None, temperature: float = 1.0, top_p: float = 1.0) -> str:
    tokens = tokenizer.encode(prompt)
    stop_token_indices = set([tokenizer.encode(t)[0] for t in stop_tokens])
    generated_tokens = 0
    last_token = tokens[-1]
    sequence_tensor = torch.tensor(tokens, dtype=torch.long, device=model.device).reshape(1, -1)
    while True:
        if max_tokens is not None and generated_tokens == max_tokens:
            break
        if last_token in stop_token_indices:
            break
        logits = model(sequence_tensor)
        logits /= temperature # (1, seq_len, vocab_size)
        probs = softmax(logits[:,-1,:]).flatten().cpu()
        pairs = [(i, probs[i]) for i in range(probs.shape[0])]
        pairs.sort(key = lambda x: x[1], reverse = True)
        cumsum = 0.0
        cutoff = len(pairs)
        for i, (_, p) in enumerate(pairs):
            cumsum += p
            if cumsum > top_p:
                cutoff = i + 1
                break
        pairs = pairs[:cutoff]
        indices = [x[0] for x in pairs]
        probs_sort = [x[1] for x in pairs]
        next_token = random.choices(indices, weights=probs_sort)[0]
        tokens.append(next_token)
        generated_tokens += 1
        last_token = next_token
        sequence_tensor = torch.cat([sequence_tensor, torch.tensor([[last_token]], dtype=torch.long, device=model.device)], dim=-1)
    return tokenizer.decode(tokens)
