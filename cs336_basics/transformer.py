import torch.nn as nn
import torch
import math
from einops import einsum


class Linear(nn.Module):
    def __init__(self, 
        in_features: int,
        out_features: int, 
        device: torch.device | None  = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        weights = torch.zeros((out_features, in_features), dtype=dtype, device=device)
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
