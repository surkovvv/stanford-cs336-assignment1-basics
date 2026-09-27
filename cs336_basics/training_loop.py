import torch
import numpy as np
import torch.nn as nn
from typing import IO, BinaryIO
import os
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from tqdm import trange

from cs336_basics.transformer import TransformerLM
from cs336_basics.training_utils import AdamW, cross_entropy, gradient_clipping


def data_loading(
    x: np.ndarray, 
    batch_size: int, 
    context_len: int, 
    device: torch.device,
    seed: int = 666
) -> tuple[torch.Tensor, torch.Tensor]:
    x_len = x.shape[0]
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
    out: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    dict_to_save = {}
    dict_to_save["model_params"] = model.state_dict()
    dict_to_save["optimizer_params"] = optimizer.state_dict()
    dict_to_save["iteration"] = iteration

    torch.save(dict_to_save, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer
) -> int:
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model_params"])
    optimizer.load_state_dict(state_dict["optimizer_params"])
    iteration = state_dict["iteration"]

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
    T_warmup: int
    T: int

@dataclass
class TrainingParams:
    batch_size: int
    seed: int
    num_epochs: int
    optimizer_params: OptimizerParams
    lr_scheduler_params: LRSchedulerParams

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

@dataclass
class LoggingParams:
    use_console: bool = True
    # w&b params


# cs = ConfigStore.instance()
# cs.store(name="model", node=ModelParams)
# cs.store(name="optimizer", node=OptimizerParams)
# cs.store(group="db", name="base_mysql", node=MySQLConfig)
# cs.store(group="db", name="base_postgresql", node=PostGreSQLConfig)


def train_step():
    pass


def train():
    pass


def validate():
    pass


def parse_args():
    pass


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    print(cfg)

    # print(cfg.data.train_path)
    train_np_array = np.memmap(cfg.data.train_path, dtype=np.uint16)  # same as we saved
    # print(len(train_np_array))  # >> 555917109
    # print(max(train_np_array))  # >> 31999

    with open("/Users/tr3n1ttty/code projects/preps/cs 336/stanford-cs336-assignment1-basics/data/results/TinyStoriesV2-train-bpe_tokenizer-vocab.pkl", "rb") as f:
        import pickle
        tokenizer_vocab = pickle.load(f)

    print("tokenizer vocab size: ", len(tokenizer_vocab))

    device = torch.device(cfg.run_params.device)
    # print("dtype: ", cfg.run_params.dtype)
    # dtype = torch.dtype(cfg.run_params.dtype)

    model = TransformerLM(**cfg.model, device=device)  # dtype=dtype,
    optimizer = AdamW(model.parameters(), **cfg.optimizer)
    # lr_scheduler = cosine_lr_schedulling()

    for step in trange(cfg.training_params.num_steps):
        input_batch, target_batch = data_loading(
            train_np_array,
            batch_size=cfg.training_params.batch_size,
            context_len=cfg.model.context_length,
            device=device,
            seed=cfg.training_params.seed
        )

        logits = model(input_batch)
        optimizer.zero_grad()
        loss = cross_entropy(logits, target_batch)
        loss.backward()
        gradient_clipping(model.parameters(), max_grad_norm=cfg.training_params.max_grad_norm)
        optimizer.step()
        print(f"Step {step + 1} loss=", loss.detach().data)
    

if __name__ == "__main__":
    main()
