import torch

from torch import nn, Tensor
from einops import einsum, reduce, repeat
from jaxtyping import Float, Int

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        variance = 2 / (in_features + out_features)
        std = variance ** 0.5
        nn.init.trunc_normal_(self.w.data, mean=0, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: Float[Tensor, '... d_in']) -> Float[Tensor, '... d_out']:
        # (.., in) @ (out, in) -> (.., out)
        out = einsum(x, self.w, '... d_in, d_out d_in -> ... d_out')
        return out
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(Embedding, self).__init__()
        self.embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        nn.init.trunc_normal_(self.embedding.data, mean=0, std=1, a=-3, b=3)
        
    def forward(self, x: Int[Tensor, '...']) -> Float[Tensor, '... d_model']:
        return self.embedding[x]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(RMSNorm, self).__init__()
        self.g = nn.Parameter(torch.ones((d_model, ), device=device, dtype=dtype))
        self.device = device
        self.dtype = dtype
        self.eps = eps
        
    def forward(self, x: Float[Tensor, '... d_model']) -> Float[Tensor, '... d_model']:
        in_type = x.dtype
        x = x.to(torch.float32)
        x_sq = x ** 2
        sq_sum = reduce(x_sq, '... d_model -> ... 1', 'mean')
        rms = (sq_sum + self.eps) ** 0.5
        x = x / rms * self.g
        x = x.to(in_type)
        return x
    
class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()
    def forward(self, x: Float[Tensor, '...']) -> Float[Tensor, '...']:
        return x / (1 + torch.exp(-x))
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
    
    @staticmethod
    def _silu(x: Float[Tensor, '...']) -> Float[Tensor, '...']:
        return x * torch.sigmoid(x)

    def forward(self, x: Float[Tensor, '... d_model']) -> Float[Tensor, '... d_model']:
        up_proj = self.w3(x)
        gate = self._silu(self.w1(x))
        gated = gate * up_proj
        down_proj = self.w2(gated)
        return down_proj

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        inv_freq = self._precompute_inv_freq(theta, d_k).to(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def _precompute_inv_freq(self, theta: float, d_k: int):
        pow_series = torch.arange(0, d_k, 2).float()
        inv_freq = 1.0 / (theta ** (pow_series / d_k))
        return inv_freq

    def forward(self, x: Float[Tensor, '... seq_len d_k'], 
                token_positions: Int[Tensor, '... seq_len']) -> Float[Tensor, '... seq_len d_k']:
        x_even = x[..., 0::2]  # shape: (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]   # shape: (..., seq_len, d_k/2)
        
        # token_positions: (..., seq_len)
        # inv_freq: (d_k/2,)
        # angles: (..., seq_len, d_k/2)
        angles = token_positions.unsqueeze(-1).float() * self.inv_freq
        
        cos = angles.cos()
        sin = angles.sin()
        
        x_out_even = x_even * cos - x_odd * sin
        x_out_odd = x_even * sin + x_odd * cos
        
        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x_out_even
        x_out[..., 1::2] = x_out_odd
        
        return x_out

def softmax(x: Float[Tensor, '...'], dim: int = -1) -> Float[Tensor, '...']:
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x = x - max_val
    x = torch.exp(x)
    out = x / torch.sum(x, dim=dim, keepdim=True)
    return out