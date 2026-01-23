from jaxtyping import Float, Int, Int64
from torch import Tensor
import torch
import math
from typing import Iterable
import numpy.typing as npt
import numpy as np
import os
from typing import BinaryIO, IO

default_rng = np.random.default_rng()

def cross_entropy(logits: Float[Tensor, '... vocab_size'], 
                  targets: Int[Tensor, '...']) -> Float[Tensor, ""]:
    max_val = torch.max(logits, dim=-1, keepdim=True).values
    logits_stable = logits - max_val
    
    log_sum_exp = torch.logsumexp(logits_stable, dim=-1, keepdim=True)
    
    log_probs_correct = torch.gather(
        logits_stable, 
        dim=-1, 
        index=targets.unsqueeze(-1)
    ).squeeze(-1)
    
    loss = -log_probs_correct + log_sum_exp.squeeze(-1)
    
    return torch.mean(loss)

def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return t / T_w * alpha_max
    if t > T_c:
        return alpha_min
    return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (alpha_max - alpha_min)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    params_with_grad = [param for param in parameters if param.grad is not None]
    grad = torch.concat([param.grad for param in params_with_grad])
    norm = torch.linalg.norm(grad, ord=2)
    for param in params_with_grad:
        param.grad = param.grad * (max_l2_norm / (norm + eps))
        
def get_next_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str, rng = default_rng) -> tuple[Int64[Tensor, 'bs seq_len'], Int64[Tensor, 'bs seq_len']]:
   sampled_start = rng.choice(len(dataset) - context_length, batch_size)
   sequences = torch.tensor([dataset[start:start+context_length] for start in sampled_start], dtype=torch.long, device=device)
   targets = torch.tensor([dataset[start+1:start+1+context_length] for start in sampled_start], dtype=torch.long, device=device)
   return (sequences, targets)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    all_states = {
        'model_weights': model.state_dict(),
        'optimizer_states': optimizer.state_dict(),
        'iteration': iteration
    }
    with open(out, 'wb') as f:
        torch.save(all_states, f)
    
def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    iteration = -1
    with open(src, 'rb') as f:
        all_states = torch.load(src)
        model.load_state_dict(all_states['model_weights'])
        optimizer.load_state_dict(all_states['optimizer_states'])
        iteration = all_states['iteration']
    return iteration

def load_as_array(bin_path, dtype=np.uint16):
    file_size = os.path.getsize(bin_path)
    element_size = np.dtype(dtype).itemsize
    total_len = file_size // element_size
    
    data = np.memmap(bin_path, dtype=dtype, mode='r', shape=(total_len,))
    return data