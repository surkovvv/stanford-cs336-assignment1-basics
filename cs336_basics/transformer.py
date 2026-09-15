import torch.nn as nn
import torch
import math
from einops import einsum, reduce, rearrange


class Linear(nn.Module):
    def __init__(self, 
        in_features: int,
        out_features: int, 
        device: torch.device | None  = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        weights = torch.empty((out_features, in_features), dtype=dtype, device=device)
        std = math.sqrt(2 / (in_features + out_features))
        init_weights = nn.init.trunc_normal_(weights, std=std, a=-3 * std, b=3 * std)
        self.W = nn.Parameter(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = einsum(self.W, x, '... d_out d_in, ... d_in -> ... d_out')
        return result

# my_linear = Linear(in_features=10, out_features=5)
# x = torch.ones(10)

# result = my_linear(x)
# print(result.size())

class Embedding(nn.Module):
    def __init__(self, 
        num_embeddings: int, 
        embeddings_dim: int,
        device: torch.device | None  = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        embeddings = torch.zeros((num_embeddings, embeddings_dim), dtype=dtype, device=device)
        init_embeddings = nn.init.trunc_normal_(embeddings, a=-3, b=3)
        self.embeddings = nn.Parameter(init_embeddings)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        assert token_ids.dtype == torch.long
        result_embeddings = self.embeddings.data[token_ids]
        return result_embeddings


class RMSNorm(nn.Module):
    def __init__(self, 
        d_model: int, 
        eps: float = 1e-5, 
        device: torch.device | None  = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.eps = eps
        g_weights = torch.ones(d_model, device=device, dtype=dtype)
        self.g = nn.Parameter(g_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)  # old me
        rms = torch.sqrt(reduce(x * x, '... d_model -> ... 1', 'mean') + self.eps)  # new me
        rmsnorm = x / rms * self.g

        rmsnorm = rmsnorm.to(in_dtype)
        return rmsnorm


class SwiGLUFFN(nn.Module):
    def __init__(self, 
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None  = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 8 * (d_model // 3)
        
        self.W_silu = nn.Parameter(torch.empty((d_ff, d_model), dtype=dtype, device=device))
        self.W_inner = nn.Parameter(torch.empty((d_ff, d_model), dtype=dtype, device=device))
        self.W_outer = nn.Parameter(torch.empty((d_model, d_ff), dtype=dtype, device=device))

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        result = x * torch.sigmoid(x)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inside_silu = einsum(self.W_silu, x, 'd_ff d_model, ... d_model -> ... d_ff')
        silu_res = self.silu(inside_silu)
        near_silu = einsum(self.W_inner, x, 'd_ff d_model, ... d_model -> ... d_ff')
        inside = silu_res * near_silu
        result = einsum(self.W_outer, inside, 'd_model d_ff, ... d_ff -> ... d_model')
        return result

