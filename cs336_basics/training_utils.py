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

try_different_lrs(lr=1e1)  # >> 17.85977554321289 -> 14.779157638549805, decrease only
print('~' * 66)
try_different_lrs(lr=1e2)  # >> 14.592806816101074 -> 12.075705528259277, decrease only
print('~' * 66)
try_different_lrs(lr=1e3)  # >> 11.923442840576172 -> 9.866779327392578, decrease only