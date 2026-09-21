import torch


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