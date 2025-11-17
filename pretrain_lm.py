"""
Pretraining Script for HRM Tiny Language Models

This script trains tiny language models using the Hierarchical Reasoning Model (HRM)
architecture with Adaptive Computation Time (ACT).

Usage:
    # Single GPU training
    python pretrain_lm.py data_path=data/text-lm

    # Multi-GPU training (8 GPUs)
    OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain_lm.py data_path=data/text-lm

    # Tiny model variant
    python pretrain_lm.py arch=hrm_lm_tiny data_path=data/text-lm

    # Small model variant
    python pretrain_lm.py arch=hrm_lm_small data_path=data/text-lm
"""

from typing import Optional, Any, Sequence, Dict
from dataclasses import dataclass
import os
import math
import yaml
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from lm_dataset import LMDataset, LMDatasetConfig
from dataset.common import PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class LMPretrainConfig(pydantic.BaseModel):
    """Configuration for language model pretraining."""
    # Architecture
    arch: ArchConfig

    # Data
    data_path: str

    # Training hyperparameters
    global_batch_size: int
    epochs: int

    # Learning rate schedule
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    # Optimizer
    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding (for compatibility, not used in LM)
    puzzle_emb_lr: float = 1e-4
    puzzle_emb_weight_decay: float = 0.1

    # Experiment tracking
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Evaluation and checkpointing
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: list = []


@dataclass
class TrainState:
    """Training state container."""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    carry: Any
    step: int
    total_steps: int


def create_dataloader(
    config: LMPretrainConfig,
    split: str,
    rank: int,
    world_size: int,
    **kwargs
):
    """Create dataloader for language model data."""
    dataset = LMDataset(
        LMDatasetConfig(
            seed=config.seed,
            dataset_path=config.data_path,
            rank=rank,
            num_replicas=world_size,
            **kwargs
        ),
        split=split
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )

    return dataloader, dataset.metadata


def create_model(
    config: LMPretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    world_size: int
):
    """Create HRM language model."""
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
    )

    # Load model and loss classes
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    # Instantiate model
    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)

        # Compile model for efficiency (if not disabled)
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)

        # Synchronize parameters across GPUs
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Create optimizer
    optimizer = AdamATan2(
        model.parameters(),
        lr=0,  # Set by scheduler
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )

    return model, optimizer


def cosine_schedule_with_warmup(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0
):
    """Cosine learning rate schedule with warmup."""
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


def init_train_state(
    config: LMPretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    world_size: int
) -> TrainState:
    """Initialize training state."""
    # Estimate total training steps
    total_steps = int(
        config.epochs * train_metadata.total_groups *
        train_metadata.mean_puzzle_examples / config.global_batch_size
    )

    # Create model and optimizer
    model, optimizer = create_model(config, train_metadata, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizer=optimizer,
        carry=None
    )


def save_train_state(config: LMPretrainConfig, train_state: TrainState):
    """Save model checkpoint."""
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(config.checkpoint_path, f"step_{train_state.step}")
    torch.save(train_state.model.state_dict(), checkpoint_file)


def compute_lr(config: LMPretrainConfig, train_state: TrainState) -> float:
    """Compute learning rate for current step."""
    return cosine_schedule_with_warmup(
        current_step=train_state.step,
        base_lr=config.lr,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def train_batch(
    config: LMPretrainConfig,
    train_state: TrainState,
    batch: Dict[str, torch.Tensor],
    global_batch_size: int,
    rank: int,
    world_size: int
):
    """Train on a single batch."""
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return None

    # Move batch to GPU
    batch = {k: v.cuda() for k, v in batch.items()}

    # Initialize carry if needed
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

    # Forward pass
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry,
        batch=batch,
        return_keys=[]
    )

    # Backward pass (scaled by batch size)
    ((1 / global_batch_size) * loss).backward()

    # All-reduce gradients across GPUs
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Update learning rate and step optimizer
    lr = compute_lr(config, train_state)
    for param_group in train_state.optimizer.param_groups:
        param_group['lr'] = lr

    train_state.optimizer.step()
    train_state.optimizer.zero_grad()

    # Reduce metrics across GPUs
    if len(metrics) and rank == 0:
        metric_keys = sorted(metrics.keys())
        metric_values = torch.stack([metrics[k] for k in metric_keys])

        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        metric_values = metric_values.cpu().numpy()
        reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

        # Normalize metrics
        count = max(reduced_metrics["count"], 1)
        reduced_metrics = {
            f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
            for k, v in reduced_metrics.items()
        }
        reduced_metrics["train/lr"] = lr

        return reduced_metrics

    return None


def evaluate(
    config: LMPretrainConfig,
    train_state: TrainState,
    eval_loader: DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int
):
    """Evaluate model on test set."""
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        all_preds = {}
        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0] * len(set_ids)

        for set_name, batch, global_batch_size in eval_loader:
            # Move to GPU
            batch = {k: v.cuda() for k, v in batch.items()}

            # Initialize carry
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)

            # Run until all sequences halt
            while True:
                carry, _, metrics, preds, all_finish = train_state.model(
                    carry=carry,
                    batch=batch,
                    return_keys=config.eval_save_outputs
                )

                if all_finish:
                    break

            # Save predictions if requested
            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())

            del carry, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = sorted(metrics.keys())
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics)),
                    dtype=torch.float32,
                    device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

        # Save predictions
        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(
                all_preds,
                os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        # Reduce and log metrics
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Normalize metrics
                for set_name, metrics in reduced_metrics.items():
                    count = metrics.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}

                return reduced_metrics

        return None


def save_code_and_config(config: LMPretrainConfig):
    """Save code and configuration for reproducibility."""
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy source code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]

    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Save config as YAML
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log to W&B
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> LMPretrainConfig:
    """Load and synchronize config across all ranks."""
    objects = [None]

    if rank == 0:
        config = LMPretrainConfig(**hydra_config)

        # Auto-generate names if not provided
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_path).capitalize()} HRM-LM"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]


@hydra.main(config_path="config", config_name="cfg_lm_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    """Main training loop."""
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load synchronized config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Set random seed
    torch.random.manual_seed(config.seed + RANK)

    # Create dataloaders
    train_epochs_per_iter = config.eval_interval if config.eval_interval else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, \
        "eval_interval must divide epochs evenly"

    train_loader, train_metadata = create_dataloader(
        config, "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE
    )

    eval_loader, eval_metadata = create_dataloader(
        config, "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE
    )

    # Initialize training
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Setup logging
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True)
        )
        wandb.log({"num_params": sum(p.numel() for p in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    # Training loop
    for iter_id in range(total_iters):
        if RANK == 0:
            print(f"Epoch {iter_id * train_epochs_per_iter}/{config.epochs}")

        # Training
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(
                config, train_state, batch, global_batch_size,
                rank=RANK, world_size=WORLD_SIZE
            )

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)

        # Evaluation
        train_state.model.eval()
        metrics = evaluate(
            config, train_state, eval_loader, eval_metadata,
            rank=RANK, world_size=WORLD_SIZE
        )

        if RANK == 0 and metrics is not None:
            wandb.log(metrics, step=train_state.step)

        # Checkpointing
        if RANK == 0 and (config.checkpoint_every_eval or iter_id == total_iters - 1):
            save_train_state(config, train_state)

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
