
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
from torch.utils.data.distributed import DistributedSampler

import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from torch.optim import AdamW
from dataset.build_cifar_dataset import CIFARDatasetMetadata
from utils.functions import load_model_class, get_model_source_path


from dataset.common import PreprocessedCIFARDataset 





class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str
    loss: LossConfig

class PretrainVisionConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    # Architecture config
    arch: ArchConfig

    # Path to dataset
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

    # Misc
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int






# Create dataloader function that loads pre-processed CIFAR images and labels 
def create_dataloader(data_path: str, split: str, global_batch_size: int, rank: int, world_size: int):
    """
    Creates a DataLoader using the PreprocessedCIFARDataset.
    """
    # The data directory for the specific split
    split_data_dir = os.path.join(data_path, split)

    # Instantiate our new, efficient dataset class
    dataset = PreprocessedCIFARDataset(data_dir=split_data_dir)
    
    # Get metadata directly from the dataset object
    metadata = dataset.metadata

    sampler: Optional[DistributedSampler] = None
    if world_size > 1:
        sampler = DistributedSampler(
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



# Create model function
def create_model(config: PretrainVisionConfig, train_metadata: CIFARDatasetMetadata, world_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract model configuration from arch
    model_cfg = dict(config.arch.model_extra , # type: ignore
                     
        batch_size=config.global_batch_size // world_size,

        image_size=train_metadata.image_size,
        patch_size=train_metadata.patch_size,
        num_classes=train_metadata.num_classes
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)
    optimizers = [AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )]

    print("Using AdamW optimizer")
    return model, optimizers, [config.lr]

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int,
    num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, min_ratio + (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

def init_train_state(config: PretrainVisionConfig, train_metadata: CIFARDatasetMetadata, world_size: int):
    total_steps = int(config.epochs * train_metadata.num_train_examples / config.global_batch_size)
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size)
    # Initialize carry
    batch_size = config.global_batch_size // world_size
    dummy_batch = {
        "inputs": torch.zeros((batch_size, train_metadata.patches_per_image * train_metadata.patch_dim)),
        "labels": torch.zeros((batch_size,), dtype=torch.long)
    }
    carry = model.initial_carry(dummy_batch) # type: ignore
    return TrainState(step=0, total_steps=total_steps, model=model, optimizers=optimizers, optimizer_lrs=optimizer_lrs, carry=carry)

def detach_carry(carry: Any) -> Any:
    """Detach tensors in carry to prevent gradient tracking."""
    if isinstance(carry, torch.Tensor):
        return carry.detach()
    elif isinstance(carry, (list, tuple)):
        return type(carry)(detach_carry(x) for x in carry)
    elif isinstance(carry, dict):
        return {k: detach_carry(v) for k, v in carry.items()}
    return carry


def train_batch(train_state: TrainState, batch: tuple, config: PretrainVisionConfig, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    model_device = next(train_state.model.parameters()).device
    inputs, labels = batch
    batch_dict = {"inputs": inputs.to(model_device), "labels": labels.to(model_device)}

    # Reinitialize carry every batch to avoid backprop-through-graph across steps
    train_state.carry = train_state.model.initial_carry(batch_dict)  # type: ignore

    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry,
        batch=batch_dict,
        return_keys=config.eval_save_outputs
    )
    # Backward
    (loss / config.global_batch_size).backward()

    if world_size > 1 and dist.is_initialized():
        for p in train_state.model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)

    

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = base_lr * cosine_schedule_with_warmup_lr_lambda(
            current_step=train_state.step, base_lr=base_lr, num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=train_state.total_steps, min_ratio=config.lr_min_ratio
        )
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()

    if metrics:
        keys = sorted(metrics.keys())
        vals = torch.stack([metrics[k] for k in keys])
        if world_size > 1 and dist.is_initialized():
            dist.all_reduce(vals)
            vals /= world_size
        if rank == 0:
            out = {f"train/{k}": v.item() for k, v in zip(keys, vals)}
            out["train/lr"] = lr_this_step # type: ignore
            return out

def evaluate(train_state: TrainState, eval_loader: DataLoader, rank: int, world_size: int):
    train_state.model.eval()
    all_metrics = []
    with torch.inference_mode():
        for batch in eval_loader:
            model_device = next(train_state.model.parameters()).device
            inputs, labels = batch
            batch_dict = {"inputs": inputs.to(model_device), "labels": labels.to(model_device)}
            carry = train_state.model.initial_carry(batch_dict) # type: ignore
            _, _, metrics, _, _ = train_state.model(carry=carry, batch=batch_dict, return_keys=[])
            all_metrics.append(metrics)

    if not all_metrics:
        return {}

    keys = sorted(all_metrics[0].keys())
    vals = torch.stack([torch.stack([m[k] for k in keys]) for m in all_metrics]).mean(dim=0)

    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(vals)
        vals /= world_size

    if rank == 0:
        return {f"test/{k}": v.item() for k, v in zip(keys, vals)}

def save_train_state(config: PretrainVisionConfig, train_state: TrainState):
    if config.checkpoint_path:
        os.makedirs(config.checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(config.checkpoint_path, f"{config.run_name}_step_{train_state.step}.pt")
        torch.save(train_state.model.state_dict(), checkpoint_file)

@hydra.main(config_path="config", config_name="cfg_vision_pretrain", version_base=None)
def main(hydra_config: DictConfig):
    RANK, WORLD_SIZE = 0, 1
    if "LOCAL_RANK" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    config = PretrainVisionConfig(**OmegaConf.to_container(hydra_config, resolve=True)) # type: ignore
    if config.run_name is None:
        config.run_name = f"{config.arch.name.split('.')[-1]}-{coolname.generate_slug(2)}"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name) # type: ignore

    torch.manual_seed(config.seed + RANK)
    np.random.seed(config.seed + RANK)

    train_loader, train_metadata = create_dataloader(config.data_path, "train", config.global_batch_size, RANK, WORLD_SIZE)
    eval_loader, _ = create_dataloader(config.data_path, "test", config.global_batch_size, RANK, WORLD_SIZE)
    train_state = init_train_state(config, train_metadata, WORLD_SIZE)

    if RANK == 0:
        print(f"Starting run {config.run_name} in project {config.project_name}")
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump())
        wandb.log({"num_params": sum(p.numel() for p in train_state.model.parameters() if p.requires_grad)})

    for epoch in range(config.epochs):
        if RANK == 0:
            print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch) # type: ignore

        train_state.model.train()
        pbar = tqdm(train_loader, disable=(RANK != 0))
        for batch in pbar:
            metrics = train_batch(train_state, batch, config, RANK, WORLD_SIZE)
            if RANK == 0 and metrics:
                pbar.set_postfix({k.split('/')[-1]: f"{v:.3f}" for k, v in metrics.items()})
                wandb.log(metrics, step=train_state.step)

        if (epoch + 1) % (config.eval_interval or 1) == 0:
            eval_metrics = evaluate(train_state, eval_loader, RANK, WORLD_SIZE)
            if RANK == 0 and eval_metrics:
                print("Evaluation metrics:", {k: f"{v:.4f}" for k, v in eval_metrics.items()})
                wandb.log(eval_metrics, step=train_state.step)
                if config.checkpoint_every_eval:
                    save_train_state(config, train_state)

    if dist.is_initialized():
        dist.destroy_process_group()
    if RANK == 0:
        wandb.finish()

if __name__ == "__main__":
    main()
