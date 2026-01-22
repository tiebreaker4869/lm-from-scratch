from jaxtyping import Float, Int
from torch import Tensor
import torch

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