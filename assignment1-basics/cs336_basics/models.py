import torch

from torch import nn, Tensor
from einops import einsum, reduce, rearrange
from jaxtyping import Float, Int, Bool

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

        # x_even has shape (..., seq_len, d_k/2)
        # angles has shape (..., seq_len, d_k/2) but may have fewer leading dims
        # Insert dims at position -3 (before seq_len dim) to match x's middle dims
        while angles.ndim < x_even.ndim:
            angles = angles.unsqueeze(-3)

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

def scaled_dot_product_attention(q: Float[Tensor, 'batch_size ... q_seq_len d_k'], k: Float[Tensor, 'batch_size ... kv_seq_len d_k'], v: Float[Tensor, 'batch_size ... kv_seq_len d_v'], mask: Bool[Tensor, 'q_seq_len kv_seq_len'] | None = None) -> Float[Tensor, 'batch_size ... q_seq_len d_v']:
    d_k = q.shape[-1]
    scores = einsum(q, k, 'bs ... q_seq_len d_k, bs ... kv_seq_len d_k -> bs ... q_seq_len kv_seq_len') / (d_k ** 0.5)
    if mask is not None:
        while mask.ndim < scores.ndim:
            mask = mask.unsqueeze(0)
        scores = scores.masked_fill(~mask, float('-inf'))
    probs = softmax(scores, dim=-1)
    out = einsum(probs, v, 'bs ... q_seq_len kv_seq_len, bs ... kv_seq_len d_v -> bs ... q_seq_len d_v')
    return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.w_qkv = Linear(d_model, 3 * d_model, device, dtype)
        self.w_out = Linear(d_model, d_model, device, dtype)
        self.device = device
        self.dtype = dtype
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta, head_dim, max_seq_len, device) if self.theta else None
    def forward(self, x: Float[Tensor, 'bs ... seq_len d_model'], token_positions: Int[Float, 'bs ... seq_len'] | None = None) -> Float[Tensor, 'bs ... seq_len d_model']:
        x = self.w_qkv(x)
        q, k, v = torch.split(x, self.d_model, dim=-1)
        q = rearrange(q, 'bs ... seq_len (num_heads d_h) -> bs ... num_heads seq_len d_h', num_heads=self.num_heads)
        k = rearrange(k, 'bs ... seq_len (num_heads d_h) -> bs ... num_heads seq_len d_h', num_heads=self.num_heads)
        v = rearrange(v, 'bs ... seq_len (num_heads d_h) -> bs ... num_heads seq_len d_h', num_heads=self.num_heads)
        if self.rope and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        seq_len = q.shape[-2]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device), diagonal=0).bool()
        out = scaled_dot_product_attention(q, k, v, causal_mask)
        out = rearrange(out, 'bs ... num_heads seq_len d_h -> bs ... seq_len (num_heads d_h)')
        out = self.w_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float | None = None, max_seq_len: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(TransformerBlock, self).__init__()
        self.rmsnorm_mha = RMSNorm(d_model, device=device, dtype=dtype)
        self.rmsnorm_ffn = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.theta = theta
        self.max_seq_len = max_seq_len
    def forward(self, x: Float[Tensor, 'bs ... seq_len d_model']) -> Float[Tensor, 'bs ... seq_len d_model']:
        seq_len = x.shape[-2]
        token_positions = torch.arange(0, seq_len, device=self.device).broadcast_to(x.shape[:-1]) if self.theta else None
        x = x + self.mha(self.rmsnorm_mha(x), token_positions)
        x = x + self.ffn(self.rmsnorm_ffn(x))
        return x

class MiniLM(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, theta: float, vocab_size: int, context_length: int, num_layers: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(MiniLM, self).__init__()
        self.embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len=context_length, device=device, dtype=dtype) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.final_proj = Linear(d_model, vocab_size, device, dtype)
        self.device = device
    def forward(self, x: Int[Tensor, '... seq_len']) -> Float[Tensor, '... seq_len vocab_size']:
        out = self.embeddings(x)
        for block in self.blocks:
            out = block(out)
        out = self.final_norm(out)
        out = self.final_proj(out)
        return out

class TransformerBlockNoLayerNorm(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float | None = None, max_seq_len: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(TransformerBlockNoLayerNorm, self).__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.theta = theta
        self.max_seq_len = max_seq_len
    def forward(self, x: Float[Tensor, 'bs ... seq_len d_model']) -> Float[Tensor, 'bs ... seq_len d_model']:
        seq_len = x.shape[-2]
        token_positions = torch.arange(0, seq_len, device=self.device).broadcast_to(x.shape[:-1]) if self.theta else None
        x = x + self.mha(x, token_positions)
        x = x + self.ffn(x)
        return x
    
class MiniLMNoLN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, theta: float, vocab_size: int, context_length: int, num_layers: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(MiniLMNoLN, self).__init__()
        self.embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([TransformerBlockNoLayerNorm(d_model, num_heads, d_ff, theta, max_seq_len=context_length, device=device, dtype=dtype) for _ in range(num_layers)])
        self.final_proj = Linear(d_model, vocab_size, device, dtype)
        self.device = device
    def forward(self, x: Int[Tensor, '... seq_len']) -> Float[Tensor, '... seq_len vocab_size']:
        out = self.embeddings(x)
        for block in self.blocks:
            out = block(out)
        out = self.final_proj(out)
        return out
    
def get_model(model: str):
    mappings = {
        'MiniLM': MiniLM,
        'MiniLMNoLN': MiniLMNoLN
    }
    if model not in mappings:
        raise NotImplementedError
    return mappings[model]