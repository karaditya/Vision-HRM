from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
import os
import math
import json
import numpy as np

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from tqdm import tqdm

from adam_atan2 import AdamATan2
from dataset.common import CIFARDatasetMetadata
from utils.functions import load_model_class

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig

class VisionConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = 1
    eval_save_outputs: List[str] = []

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int

def create_dataloader(data_path: str, split: str, global_batch_size: int, rank: int, world_size: int):
    """Create dataloader for the preprocessed CIFAR dataset."""
    with open(os.path.join(data_path, split, "dataset.json"), "r") as f:
        metadata = CIFARDatasetMetadata(**json.load(f))
    
    inputs = np.load(os.path.join(data_path, split, "all__inputs.npy"))
    labels = np.load(os.path.join(data_path, split, "all__labels.npy"))
    
    dataset = TensorDataset(
        torch.from_numpy(inputs).float(),
        torch.from_numpy(labels).long()
    )
    
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=(split == "train")
        )
    
    loader = DataLoader(
        dataset,
        batch_size=global_batch_size // world_size,
        shuffle=(sampler is None and split == "train"),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    return loader, metadata

def create_model(config: VisionConfig, train_metadata: CIFARDatasetMetadata, world_size: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        num_classes=train_metadata.num_classes,
        image_size=train_metadata.image_size,
        patch_size=train_metadata.patch_size,
        num_channels=train_metadata.num_channels
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model: nn.Module = model_cls(**model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
        
        if device == "cuda" and "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)

        if world_size > 1:
            for param in model.parameters():
                dist.broadcast(param.data, src=0)

    optimizers = [torch.optim.AdamW(
        model.parameters(), lr=0, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2)
    )]
    print("Using AdamW optimizer")

    return model, optimizers, [config.lr]

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, 
    num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))))

def init_train_state(config: VisionConfig, train_metadata: CIFARDatasetMetadata, world_size: int):
    total_steps = int(config.epochs * train_metadata.num_train_examples / config.global_batch_size)
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size)
    return TrainState(step=0, total_steps=total_steps, model=model, optimizers=optimizers, optimizer_lrs=optimizer_lrs, carry=None)

def train_batch(train_state: TrainState, batch: Dict[str, torch.Tensor], config: VisionConfig, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps: return

    model_device = next(train_state.model.parameters()).device
    batch = {k: v.to(model_device) for k, v in batch.items()}

    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
    
    # The loss from the head is already a mean over the batch.
    # We don't need to divide by global_batch_size again.
    loss.backward()

    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = cosine_schedule_with_warmup_lr_lambda(
            current_step=train_state.step, base_lr=base_lr, num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=train_state.total_steps, min_ratio=config.lr_min_ratio
        )
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()

    if metrics:
        metric_keys = sorted(metrics.keys())
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.all_reduce(metric_values)
            metric_values /= world_size

        if rank == 0:
            reduced_metrics = {f"train/{k}": v.item() for k, v in zip(metric_keys, metric_values)}
            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics

def evaluate(train_state: TrainState, eval_loader: DataLoader, rank: int, world_size: int):
    train_state.model.eval()
    all_metrics = []
    with torch.inference_mode():
        for batch in eval_loader:
            model_device = next(train_state.model.parameters()).device
            batch = {k: v.to(model_device) for k, v in batch.items()}
            _, _, metrics, _, _ = train_state.model(carry=None, batch=batch, return_keys=[])
            all_metrics.append(metrics)

    metric_keys = sorted(all_metrics[0].keys())
    metric_values = torch.stack([torch.stack([m[k] for k in metric_keys]) for m in all_metrics])
    
    # Average over all batches
    metric_values = metric_values.mean(dim=0)

    if world_size > 1:
        dist.all_reduce(metric_values)
        metric_values /= world_size

    if rank == 0:
        return {f"test/{k}": v.item() for k, v in zip(metric_keys, metric_values)}

def save_train_state(config: VisionConfig, train_state: TrainState):
    if config.checkpoint_path:
        os.makedirs(config.checkpoint_path, exist_ok=True)
        torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt"))

@hydra.main(config_path="config", config_name="cfg_vision_pretrain", version_base=None)
def main(hydra_config: DictConfig):
    RANK, WORLD_SIZE = 0, 1
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    config = VisionConfig(**hydra_config)
    if config.run_name is None: config.run_name = f"{config.arch.name.split('.')[-1]}-{coolname.generate_slug(2)}"
    if config.checkpoint_path is None: config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

    torch.manual_seed(config.seed + RANK)
    np.random.seed(config.seed + RANK)

    train_loader, train_metadata = create_dataloader(config.data_path, "train", config.global_batch_size, RANK, WORLD_SIZE)
    eval_loader, _ = create_dataloader(config.data_path, "test", config.global_batch_size, RANK, WORLD_SIZE)
    train_state = init_train_state(config, train_metadata, WORLD_SIZE)

    if RANK == 0:
        print(f"Starting run {config.run_name} in project {config.project_name}")
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump())
        wandb.log({"num_params": sum(p.numel() for p in train_state.model.parameters())})

    for epoch in range(config.epochs):
        if RANK == 0: print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        if hasattr(train_loader.sampler, 'set_epoch'): train_loader.sampler.set_epoch(epoch)
        
        train_state.model.train()
        pbar = tqdm.tqdm(train_loader, disable=(RANK != 0))
        for inputs, labels in pbar:
            metrics = train_batch(train_state, {"inputs": inputs, "labels": labels}, config, RANK, WORLD_SIZE)
            if RANK == 0 and metrics:
                pbar.set_postfix({k.split('/')[-1]: f"{v:.3f}" for k, v in metrics.items()})
                wandb.log(metrics, step=train_state.step)

        if (epoch + 1) % config.eval_interval == 0:
            eval_metrics = evaluate(train_state, eval_loader, RANK, WORLD_SIZE)
            if RANK == 0 and eval_metrics:
                print("Evaluation metrics:", {k: f"{v:.4f}" for k,v in eval_metrics.items()})
                wandb.log(eval_metrics, step=train_state.step)
                if config.checkpoint_every_eval:
                    save_train_state(config, train_state)

    if dist.is_initialized(): dist.destroy_process_group()
    if RANK == 0: wandb.finish()

if __name__ == "__main__":
    main()