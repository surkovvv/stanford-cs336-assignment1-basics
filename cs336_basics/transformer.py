import torch.nn as nn
import torch
import math
from einops import einsum, reduce, rearrange


def create_and_init_weights(
    in_features: int, 
    out_features: int, 
    device: torch.device | None  = None, 
    dtype: torch.dtype | None = None
    ) -> nn.Parameter:
    weights = torch.empty((out_features, in_features), dtype=dtype, device=device)
    std = math.sqrt(2 / (in_features + out_features))
    init_weights = nn.init.trunc_normal_(weights, std=std, a=-3 * std, b=3 * std)
    params = nn.Parameter(init_weights)
    return params


class Linear(nn.Module):
    def __init__(self, 
        in_features: int,
        out_features: int, 
        device: torch.device | None  = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.W = create_and_init_weights(in_features, out_features, device, dtype)

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
        result_embeddings = self.embeddings[token_ids]
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
        
        self.W_silu = create_and_init_weights(d_model, d_ff, dtype=dtype, device=device)
        self.W_inner = create_and_init_weights(d_model, d_ff, dtype=dtype, device=device)
        self.W_outer = create_and_init_weights(d_ff, d_model, dtype=dtype, device=device)

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


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()

        assert d_k % 2 == 0
        i = torch.arange(end=max_seq_len, device=device).view(max_seq_len, 1)
        ks = torch.arange(start=1, end=d_k//2 + 1, device=device)
        denominator = torch.pow(theta, (2 * ks - 2) / d_k).view(1, d_k // 2)
        thetas =  i / denominator

        sins = torch.sin(thetas)
        coss = torch.cos(thetas)

        self.register_buffer("sin_cached", sins, persistent=False)
        self.register_buffer("cos_cached", coss, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2] == token_positions.shape[-1]
        assert (token_positions.min() >= 0) and (token_positions.max() < self.sin_cached.shape[0])
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        selected_sin = self.sin_cached[token_positions, :]
        selected_cos = self.cos_cached[token_positions, :]

        x_even_rotated = x_even * selected_cos - x_odd * selected_sin
        x_odd_rotated = x_even * selected_sin + x_odd * selected_cos

        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_even_rotated
        x_rotated[..., 1::2] = x_odd_rotated

        return x_rotated


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_elem_among_dim = x.max(dim=dim, keepdim=True).values
    extracted_exp = torch.exp(x - max_elem_among_dim)
    result = extracted_exp / torch.sum(extracted_exp, dim=dim, keepdim=True)
    return result


def sdpa(
    queries: torch.Tensor, 
    keys: torch.Tensor, 
    values: torch.Tensor, 
    mask: torch.Tensor | None = None
    ) -> torch.Tensor:
    d_k = keys.shape[-1]
    presoftmax_numerator = einsum(queries, keys, "... q_seq_len d_k, ... k_seq_len d_k -> ... q_seq_len k_seq_len")
    presoftmax = presoftmax_numerator / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(presoftmax.device)
        presoftmax = torch.masked_fill(presoftmax, ~mask, -torch.inf)

    softmaxed = softmax(presoftmax, dim=-1)
    result = einsum(softmaxed, values, "... q_seq_len k_seq_len, ... k_seq_len d_v -> ... q_seq_len d_v")
    return result


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
        d_model: int, 
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None  = None, 
        dtype: torch.dtype | None = None
        ):
        super().__init__()

        assert d_model % num_heads == 0

        d_k = d_v = d_model // num_heads

        self.num_heads = num_heads

        self.W_q = create_and_init_weights(in_features=d_model, out_features=num_heads*d_k, dtype=dtype, device=device)
        self.W_k = create_and_init_weights(in_features=d_model, out_features=num_heads*d_k, dtype=dtype, device=device)
        self.W_v = create_and_init_weights(in_features=d_model, out_features=num_heads*d_v, dtype=dtype, device=device)
        self.W_o = create_and_init_weights(in_features=num_heads*d_v, out_features=d_model, dtype=dtype, device=device)

        self.use_rope = False

        condition1 = theta is not None and max_seq_len is not None
        condition2 = theta is None and max_seq_len is None
        assert condition1 or condition2

        if max_seq_len is not None and theta is not None:
            self.use_rope = True
            self.max_seq_len = max_seq_len
            self.rope = RotaryPositionalEmbedding(
                theta=theta, 
                d_k=d_k,
                max_seq_len=max_seq_len,
                device=device
            )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[1]
        queries = einsum(x, self.W_q, "... d_model, hd d_model -> ... hd")
        keys = einsum(x, self.W_k, "... d_model, hd d_model -> ... hd")
        values = einsum(x, self.W_v, "... d_model, hd d_model -> ... hd")

        queries_slised = rearrange(queries, "b s (h d) -> b h s d", h=self.num_heads)
        keys_slised = rearrange(keys, "b s (h d) -> b h s d", h=self.num_heads)
        values_slised = rearrange(values, "b s (h d) -> b h s d", h=self.num_heads)

        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len)

            queries_slised = self.rope(queries_slised, token_positions)
            keys_slised = self.rope(keys_slised, token_positions)

        sdpa_result = sdpa(queries_slised, keys_slised, values_slised, mask=mask)
        sdpa_result_unslised = rearrange(sdpa_result, "b h s d -> b s (h d)", h=self.num_heads)

        output = einsum(self.W_o, sdpa_result_unslised, "d_model hd, ... hd -> ... d_model")
        return output


class TransformerBlock(nn.Module):
    def __init__(self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None  = None, 
        dtype: torch.dtype | None = None
        ):
        super().__init__()

        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ffn = SwiGLUFFN(d_model, d_ff, device, dtype)
        self.attention_prenorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn_prenorm = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.mhsa(self.attention_prenorm(x))
        z = y + self.ffn(self.ffn_prenorm(y))
        return z


class TransformerLM(nn.Module):
    def __init__(self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int,
        theta: float,
        vocab_size: int, 
        context_length: int,
        num_layers: int,
        device: torch.device | None  = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embeddings_dim=d_model,
            device=device,
            dtype=dtype
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, 
                num_heads, 
                d_ff,
                context_length,
                theta,
                device,
                dtype
            )
            for _ in range(num_layers)
        ])

        self.last_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        embs = self.embedding(in_indices)

        layer_output = embs
        for layer in self.layers:
            layer_output = layer(layer_output)

        normalized_layers_output = self.last_norm(layer_output)
        vocab_logits = self.lm_head(normalized_layers_output)
        return vocab_logits
