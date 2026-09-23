import torch
from torch.optim import Optimizer
from typing import Callable
import math


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: [..., seq_len, vocab_size]
    x: [..., seq_len] - as I guess, token_ids
    ... represent "any additional batch dimensions"
    """
    max_elem_for_each_logit = torch.max(logits, dim=-1, keepdim=True).values  # means max by vocab_size
    logits_minus_max = logits - max_elem_for_each_logit

    log_sum_exps = torch.log(torch.sum(torch.exp(logits_minus_max), dim=-1))
    next_token_logits = torch.gather(logits_minus_max, dim=-1, index=targets[..., None]).squeeze(dim=-1)
    under_mean = next_token_logits - log_sum_exps
    result = - torch.mean(under_mean)
    return result


class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid lr argument: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, clousure: Callable | None = None):
        loss = None if clousure is None else clousure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss

torch.manual_seed(666)
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1)

# for t in range(100):
#     opt.zero_grad()
#     loss = (weights ** 2).mean()
#     print(loss.cpu().item())
#     loss.backward()
#     opt.step()

def try_different_lrs(lr, n=10):
    opt = SGD([weights], lr=1)

    for t in range(n):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()

# try_different_lrs(lr=1e1)  # >> 17.85977554321289 -> 14.779157638549805, decrease only
# print('~' * 66)
# try_different_lrs(lr=1e2)  # >> 14.592806816101074 -> 12.075705528259277, decrease only
# print('~' * 66)
# try_different_lrs(lr=1e3)  # >> 11.923442840576172 -> 9.866779327392578, decrease only

class AdamW(Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float], eps: float, weight_decay: float):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, clousure: Callable | None = None):
        loss = None if clousure is None else clousure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            beta1 = betas[0]
            beta2 = betas[1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                if t == 1:
                    m = torch.zeros_like(p)
                    v = torch.zeros_like(p)
                else:
                    m = state["m"]
                    v = state["v"]
                
                adjusted_lr = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr * weight_decay * p.data
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                p.data -= adjusted_lr * m / (torch.sqrt(v) + eps)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss

# Resource accounting task:
"""
First of all, peak memory = activations + weights + gradients + optimizer states.
For simplicity, we use fp32 for each tensor.
Also, for simplicity we'll calc the memory consumptions for the activations from:
transformer layer: [
rms norms 
qkv projections + qkT multiplication + softmax + weighted sum of values + output proj
W1, W2, SiLU, element-wise product, W3
]
final rms norm
lm head(output embeddings)
cross-entropy on logits

each param has precision_in_bytes * (1 from weight + 1 from grad + 1 from first momentum + 1 from second momentum)
= 4 * precision_in_bytes(fp32 has 4) = 16 bytes
let's look at all the params(we've done this before, actually)
We have: embeddings -> layers * num_layers -> final_norm -> lm_head
embeddings: [vocab_size, d_model]
layer:
    attention_prenorm: g vector, [d_model]
    mha: q, k, v, o projs, [suppose d_q=d_k=d_v=d_model // num_heads], 4 * [d_model, d_model]
    ffn_prenorm: [d_model]
    ffn: w1: [d_model, d_ff], w2: [d_ff, d_model], w3: [d_model, d_ff]
final_norm: [d_model]
lm_head: [d_model, vocab_size]

So, in total we have:
vocab_size * d_model +
(2 * d_model + 4 * d_model * d_model + 3 * d_model * d_ff) * num_layers +
d_model + 
vocab_size * d_model
params
(we have also rope sin and cos cached tensors..)
also we spend memory for the activations:
(let bsd = batch_size * seq_len * d_model)
rms_norm: bsd (* 2 * num_layers + 1)
qkv_projections: bsd * 3 (* num_layers)
qkT_mm: b * num_heads * seq_len * seq_len (* num_layers)
softmax: b * num_heads * seq_len * seq_len (* num_layers)
weighted_sum: b * num_heads * seq_len * d_head (* num_layers)
output_proj: bsd (* num_layers)
w1_proj: batch_size * seq_len * d_ff (* num_layers)
silu: batch_size * seq_len * d_ff (* num_layers)
w3: batch_size * seq_len * d_ff (* num_layers)
elem_wise product: batch_size * seq_len * d_ff (* num_layers)
w2: bsd (* num_layers)
output_embedding: batch_size * seq_len * vocab_size
cross-entropy on logits: 1
"""
from cs336_basics.resource_accounting import ModelConfig, gpt2_xl_config

def calc_total_number_of_params(config: ModelConfig) -> int:
    vocab_size = config.vocab_size
    d_model = config.d_model
    num_layers = config.num_layers
    d_ff = config.d_ff if config.d_ff is not None else 8 * d_model // 3

    total_params = (
        2 * vocab_size * d_model + 
        (2 * d_model + 4 * d_model ** 2 + 3 * d_model * d_ff) * num_layers + 
        d_model
    )
    return total_params


def calc_total_number_of_activations(config: ModelConfig, batch_size: int = 1) -> int:
    vocab_size = config.vocab_size
    d_model = config.d_model
    num_layers = config.num_layers
    num_heads = config.num_heads
    d_ff = config.d_ff if config.d_ff is not None else 8 * d_model // 3
    seq_len = config.context_length

    sd = seq_len * d_model

    total_number_of_activations = batch_size * (
        sd * (2 * num_layers + 1) +
        (6 * sd + 2 * num_heads * seq_len * seq_len + 4 * seq_len * d_ff) * num_layers + 
        seq_len * vocab_size 
    ) + 1
    return total_number_of_activations


number_of_params = calc_total_number_of_params(gpt2_xl_config)
print("Number of params for XL gpt2: ", number_of_params)
total_memory_gpt2_xl_bytes = 4 * 4 * number_of_params
print("Memory needed for XL gpt2 params(weight + grad + optimizer), fp32 ", total_memory_gpt2_xl_bytes, " bytes")

number_of_activations_bs1 = calc_total_number_of_activations(gpt2_xl_config, batch_size=1)
print("Number of activations for XL gpt2, bs=1: ", number_of_activations_bs1)
print("Memory needed for XL gpt2 activations, fp32, bs=1 ", 4 * number_of_activations_bs1, " bytes")

bs = 32
number_of_activations_bs32 = calc_total_number_of_activations(gpt2_xl_config, batch_size=bs)
print(f"Number of activations for XL gpt2, bs={bs}: ", number_of_activations_bs32)
print(f"Memory needed for XL gpt2 activations, fp32, bs={bs} ", 4 * number_of_activations_bs32, " bytes")

space_total = 80 * 10 ** 9  # 80GB
space_for_activations = space_total - total_memory_gpt2_xl_bytes
vocab_size = gpt2_xl_config.vocab_size
d_model = gpt2_xl_config.d_model
num_layers = gpt2_xl_config.num_layers
num_heads = gpt2_xl_config.num_heads
d_ff = gpt2_xl_config.d_ff if gpt2_xl_config.d_ff is not None else 8 * d_model // 3
seq_len = gpt2_xl_config.context_length

sd = seq_len * d_model
a = (
    sd * (2 * num_layers + 1) +
    (6 * sd + 2 * num_heads * seq_len * seq_len + 4 * seq_len * d_ff) * num_layers + 
    seq_len * vocab_size 
)
b = 4* number_of_params + 1
print("a = ", 4 * a, " bytes")
print("b = ", 4 * b, " bytes")
max_batch_size = (space_total // 4 - b) // a
print("max batch size for gpt XL:", max_batch_size)  # >> 3