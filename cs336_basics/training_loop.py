import torch
import numpy as np
import torch.nn as nn
from typing import IO, BinaryIO
import os
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from tqdm import trange

from cs336_basics.transformer import TransformerLM, softmax
from cs336_basics.training_utils import AdamW, cross_entropy, gradient_clipping, cosine_lr_schedulling
from cs336_basics.tokenizer import Tokenizer
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
import weave
import wandb

DTYPES_MAPPING = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def data_loading(
    x: np.ndarray, 
    batch_size: int, 
    context_len: int, 
    device: torch.device,
    rng: np.random.Generator | None = None,
    seed: int = 666
) -> tuple[torch.Tensor, torch.Tensor]:
    x_len = x.shape[0]
    if rng is None:
        rng = np.random.default_rng(seed=seed)

    starts = rng.integers(low=0, high=x_len - context_len, size=batch_size)
    inputs = []
    targets = []
    for start in starts:
        inputs.append(torch.tensor(x[start: start + context_len], dtype=torch.long, device=device))
        targets.append(torch.tensor(x[start + 1: start + context_len + 1], dtype=torch.long, device=device))

    inputs_tensor = torch.stack(inputs, dim=0)
    targets_tensor = torch.stack(targets, dim=0)

    return inputs_tensor, targets_tensor


def save_checkpoint(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | BinaryIO | IO[bytes],
    rng: np.random.Generator | None = None
) -> None:
    dict_to_save = {}
    dict_to_save["model_params"] = model.state_dict()
    dict_to_save["optimizer_params"] = optimizer.state_dict()
    dict_to_save["iteration"] = iteration
    if rng is not None:
        dict_to_save["rng_state"] = rng.bit_generator.state

    torch.save(dict_to_save, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator | None = None
) -> int:
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model_params"])
    optimizer.load_state_dict(state_dict["optimizer_params"])
    iteration = state_dict["iteration"]

    if rng is not None and "rng_state" in state_dict:
        rng.bit_generator.state = state_dict["rng_state"]

    return iteration

"""
Model init params:
    d_model: int, 
    num_heads: int, 
    d_ff: int,
    theta: float,
    vocab_size: int, 
    context_length: int,
    num_layers: int,
    device: torch.device | None  = None, 
    dtype: torch.dtype | None = None

Optimizer init params:
    lr: float, 
    betas: tuple[float, float], 
    eps: float, 
    weight_decay: float

+ we need some Data-related params
    input_path to dataset(s)
    checkpoints_save_path
    checkpoints_load_path

+ we need extra(optinal) params for logging into wandb
"""

@dataclass 
class ModelParams:
    d_model: int
    num_heads: int
    d_ff: int
    theta: float
    vocab_size: int 
    context_length: int
    num_layers: int
    mode: str

@dataclass
class OptimizerParams:
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float

@dataclass
class LRSchedulerParams:
    lr_min: float
    lr_max: float
    T_w: int
    T_c: int

@dataclass
class TrainingParams:
    batch_size: int
    seed: int
    num_steps: int
    max_grad_norm: float

@dataclass
class DataParams:
    train_path: str
    val_path: str
    tokenizer_path: str

@dataclass
class RunParams:
    run_name: str
    device: str
    dtype: str
    log_every: int
    validate_every: int
    save_every: int
    val_batch_size: int
    regular_val_steps_ratio: float

@dataclass
class LoggingParams:
    use_console: bool = False
    # w&b params
    use_wandb: bool = True
    project: str | None = None

@dataclass
class Config:
    model: ModelParams
    optimizer: OptimizerParams
    lr_scheduler: LRSchedulerParams
    training_params: TrainingParams
    data: DataParams
    run_params: RunParams
    logging: LoggingParams

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

@torch.inference_mode(True)
def calc_val_loss(
    model: nn.Module, 
    val_path: str,
    context_length: int,
    batch_size: int,
    device: str | torch.device,
    seed: int = 666,
    num_steps: int = 25
    ) -> float:
    val_np_array = np.load(val_path, mmap_mode="r")
    losses = []
    rng = np.random.default_rng(seed=seed)

    for _ in range(num_steps):
        input_batch, target_batch = data_loading(
            val_np_array,
            batch_size=batch_size,
            context_len=context_length,
            device=device,
            rng=rng,
        )

        logits = model(input_batch)
        loss = cross_entropy(logits, target_batch)
        losses.append(loss)

    mean_loss_over_batches = sum(losses) / len(losses)
    return mean_loss_over_batches


def top_p_sampling(dist: torch.Tensor, top_p: float) -> int:
    values, indices = torch.sort(dist, descending=True)
    cummulative_prob = 0
    subset_of_idx = []
    for val, idx in zip(values, indices):
        cummulative_prob += val
        subset_of_idx.append(idx.item())

        if cummulative_prob > top_p:
            break

    keep = torch.zeros_like(dist, dtype=torch.bool)
    keep[subset_of_idx] = True
    dist[~keep] = 0
    dist /= cummulative_prob

    token_id = torch.multinomial(dist, num_samples=1).item()
    return token_id


@torch.no_grad()
def decode(
    model: nn.Module,
    input_prompt: torch.Tensor,
    stop_token_id: int,
    max_new_tokens: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    context_length: int | None = None
    ):
    if context_length is None:
        new_tokens_left = max_new_tokens
    else:
        new_tokens_left = min(context_length, max_new_tokens)
    current_input = input_prompt

    while new_tokens_left > 0:
        logits = model(current_input)
        temperatured_dist = softmax(logits, dim=-1, temp=temperature)
        new_generated_token = top_p_sampling(temperatured_dist[..., -1, :].squeeze(), top_p)
        current_input = torch.cat((current_input, current_input.new_tensor([new_generated_token]).unsqueeze(0)), dim=1)

        new_tokens_left -= 1
        if new_generated_token == stop_token_id:
            break

    return current_input


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    torch.manual_seed(cfg.training_params.seed)

    train_np_array = np.load(cfg.data.train_path, mmap_mode="r")  # same as we saved
    val_np_array = np.load(cfg.data.val_path, mmap_mode="r")
    val_array_len = len(val_np_array)
    total_steps_for_current_config = val_array_len // (cfg.model.context_length * cfg.run_params.val_batch_size)
    regular_val_steps = int(cfg.run_params.regular_val_steps_ratio * total_steps_for_current_config)
    del val_np_array
    run = wandb.init(project=cfg.logging.project) if cfg.logging.use_wandb else None

    device = torch.device(cfg.run_params.device)
    if cfg.run_params.dtype in DTYPES_MAPPING:
        dtype = DTYPES_MAPPING[cfg.run_params.dtype]
    else:
        raise ValueError(f"This dtype {cfg.run_params.dtype} is not supported! Availible dtypes: {list(DTYPES_MAPPING.keys())}")

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_name = f"{run_dir.parent.name}_{run_dir.name}"
    checkpoint_dir = run_dir.parent.parent / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = TransformerLM(**cfg.model, device=device, dtype=dtype)
    if cfg.logging.use_wandb:
        run.watch(model, log="all", log_freq=cfg.run_params.validate_every)
    
    optimizer = AdamW(model.parameters(), **cfg.optimizer)
    rng = np.random.default_rng(seed=cfg.training_params.seed)

    for step in trange(cfg.training_params.num_steps):
        input_batch, target_batch = data_loading(
            train_np_array,
            batch_size=cfg.training_params.batch_size,
            context_len=cfg.model.context_length,
            device=device,
            rng=rng,
            seed=cfg.training_params.seed
        )

        logits = model(input_batch)
        optimizer.zero_grad()
        loss = cross_entropy(logits, target_batch)
        loss.backward()

        current_lr = cosine_lr_schedulling(step, **cfg.lr_scheduler)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        gradient_clipping(model.parameters(), max_grad_norm=cfg.training_params.max_grad_norm)
        optimizer.step()

        if step % cfg.run_params.log_every == 0:
            print(f"Step {step + 1} loss=", loss.item())
            if run is not None:
                run.log({"train/loss": loss.item(), "step": step + 1})
                run.log({"current lr": current_lr, "step": step + 1})
            else:
                print(f"train/loss: {loss.item()}, step: {step + 1}")
                print("current lr = ", current_lr)
        
        if step % cfg.run_params.validate_every == 0:
            num_steps = regular_val_steps
            model.eval()
            mean_val_loss = calc_val_loss(
                model, 
                cfg.data.val_path, 
                context_length=cfg.model.context_length, 
                batch_size=cfg.run_params.val_batch_size, 
                device=device,
                seed=cfg.training_params.seed,
                num_steps=num_steps
            )
            model.train()

            if run is not None:
                run.log({"val/loss": mean_val_loss.item(), "step": step + 1})
            else:
                print(f"val/loss: {mean_val_loss.item()}, step: {step + 1}")

        if step % cfg.run_params.save_every == 0:
            # example: outputs/checkpoints/2026-09-28_09-46-51/step_1000.pt
            checkpoint_path = checkpoint_dir / f"step_{step}.pt"
            save_checkpoint(model, optimizer, iteration=step, out=checkpoint_path, rng=rng)
            print(f"Step: {step + 1} model checkpoint was saved! Path: ", checkpoint_path)

    checkpoint_path = checkpoint_dir / f"step_{step}.pt"
    save_checkpoint(model, optimizer, iteration=step, out=checkpoint_path, rng=rng)
    print(f"Step: {step + 1} model checkpoint was saved! Path: ", checkpoint_path)

    num_steps = total_steps_for_current_config
    model.eval()
    mean_val_loss = calc_val_loss(
        model, 
        cfg.data.val_path, 
        context_length=cfg.model.context_length, 
        batch_size=cfg.run_params.val_batch_size, 
        device=device,
        seed=cfg.training_params.seed,
        num_steps=num_steps
    )
    run.log({"final_full_val_loss": mean_val_loss.item(), "step": step + 1})
    print(f"step: {step + 1} FULL val loss: ", mean_val_loss.item())
    
    if run is not None:
        run.finish()
    

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_generate(cfg):
    path_to_checkpoint = "outputs/checkpoints/2026-09-30_21-42-50/step_4999.pt"
    device = torch.device(cfg.run_params.device)

    model = TransformerLM(**cfg.model, device=device)
    state_dict = torch.load(path_to_checkpoint, weights_only=False)
    model.load_state_dict(state_dict["model_params"])
    model.eval()

    vocab_filepath = "./data/results/TinyStoriesV2-train-bpe_tokenizer-vocab.pkl"
    merges_filepath = "./data/results/TinyStoriesV2-train-bpe_tokenizer-merges.pkl"

    with open(vocab_filepath, "rb") as f:
        import pickle
        tokenizer_vocab = pickle.load(f)
    
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath)

    stop_token_id  = next(
        idx for idx, token in tokenizer_vocab.items()
        if token == b"<|endoftext|>"
    )

    token_ids = tokenizer.encode(" ")
    input_prompt = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    all_text = decode(model, input_prompt, stop_token_id, 30, 0.6, 0.9, context_length=cfg.model.context_length)
    token_ids_all_text = [token_id.item() for token_id in all_text.squeeze()]
    print("generated text: ", tokenizer.decode(token_ids_all_text))


if __name__ == "__main__":
    main()
    # run_generate()
    # test_dist = torch.tensor([0.6, 0.3, 0.1])
    # for _ in range(100):
    #     res = top_p_sampling(test_dist, top_p=0.8)
    #     print(res)
