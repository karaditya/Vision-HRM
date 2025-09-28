# # filename="hrm_vision_v1.py"

# from typing import Dict, Tuple, Any, Optional
# from dataclasses import dataclass
# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor

# from models.common import trunc_normal_init_
# from models.layers import SwiGLU, RotaryEmbedding, CastedEmbedding, CastedLinear, CosSin, apply_rotary_pos_emb, rms_norm


# @dataclass
# class HierarchicalReasoningModel_VisionV1InnerCarry:
#     """Inner carry state for vision HRM."""
#     H_hidden: Tensor
#     L_hidden: Tensor


# @dataclass
# class HierarchicalReasoningModel_VisionV1Carry:
#     """Carry state for vision HRM with ACT."""
#     inner_carry: HierarchicalReasoningModel_VisionV1InnerCarry
#     steps: Tensor
#     halted: Tensor
#     current_data: Dict[str, Tensor]


# @dataclass
# class HierarchicalReasoningModel_VisionV1Config:
#     # Model dimensions
#     hidden_size: int
#     num_heads: int
#     expansion: float
    
#     # Architecture
#     H_layers: int
#     L_layers: int
#     H_cycles: int
#     L_cycles: int
    
#     # Vision-specific
#     num_classes: int
#     patch_size: int
#     image_size: int
    
#     # ACT parameters
#     halt_exploration_prob: float
#     halt_max_steps: int
    
#     # Training
#     batch_size: int
#     forward_dtype: str = "bfloat16"
    
#     # Positional encoding
#     pos_encodings: str = "rope"
#     rope_theta: float = 10000.0
#     rms_norm_eps: float = 1e-5


# class VisionPatchEmbedding(nn.Module):
#     """CNN-based patch embedding with inductive bias."""
    
#     def __init__(self, config: HierarchicalReasoningModel_VisionV1Config):
#         super().__init__()
#         self.config = config
        
#         # CNN-based patch embedding (3-block conv)
#         embed_dim = config.hidden_size
#         self.patch_embed = nn.Sequential(
#             nn.Conv2d(3, embed_dim//4, kernel_size=3, stride=1, padding=1),
#             nn.GELU(),
#             nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=3, stride=1, padding=1),
#             nn.GELU(),
#             nn.Conv2d(embed_dim//2, embed_dim, kernel_size=4, stride=4)  # 4x4 patches for 32x32 -> 8x8=64 patches
#         )
        
#         # Multi-class tokens (2 instead of 1)
#         self.cls_tokens = nn.Parameter(torch.zeros(1, 2, config.hidden_size))
        
#     def forward(self, x: Tensor) -> Tensor:
#         batch_size = x.shape[0]
        
#         # Reshape input to image format
#         x = x.view(batch_size, self.config.image_size, self.config.image_size, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
#         x = x.float() / 255.0  # Normalize to [0, 1]
        
#         # CNN patch embedding
#         x = self.patch_embed(x)  # (B, hidden_size, 8, 8)
#         x = x.flatten(2).transpose(1, 2)  # (B, 64, hidden_size)
        
#         # Add multi-class tokens
#         cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)
#         x = torch.cat([cls_tokens, x], dim=1)  # (B, 66, hidden_size)
        
#         return x


# class MultiHeadLatentAttention(nn.Module):
#     """Compressed attention with rotary embeddings support."""
    
#     def __init__(self, config: HierarchicalReasoningModel_VisionV1Config):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_heads
#         self.head_dim = config.hidden_size // config.num_heads
#         self.compression_dim = 64  # Key hyperparameter
        
#         # Compressed query projections
#         self.q_down = CastedLinear(config.hidden_size, self.compression_dim, bias=False)
#         self.q_up = CastedLinear(self.compression_dim, config.hidden_size, bias=False)
        
#         # Standard K, V projections
#         self.k_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
#         self.v_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
#         self.out_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        
#     def forward(self, x: Tensor, cos_sin: Optional[CosSin] = None) -> Tensor:
#         B, N, C = x.shape
        
#         # Compressed queries
#         q_compressed = self.q_down(x)  # (B, N, compression_dim)
#         q = self.q_up(q_compressed)    # (B, N, hidden_size)
        
#         # Standard K, V
#         k = self.k_proj(x)
#         v = self.v_proj(x)
        
#         # Apply rotary if provided
#         if cos_sin is not None:
#             cos, sin = cos_sin
#             q, k = apply_rotary_pos_emb(q.view(B, N, self.num_heads, self.head_dim), 
#                                         k.view(B, N, self.num_heads, self.head_dim), 
#                                         cos, sin)
#             q = q.view(B, N, C)
#             k = k.view(B, N, C)
        
#         # Multi-head reshape
#         q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
#         k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
#         v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
#         # Scaled dot-product attention
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         attn = F.softmax(scores, dim=-1)
        
#         out = torch.matmul(attn, v)
#         out = out.transpose(1, 2).contiguous().view(B, N, C)
        
#         return self.out_proj(out)


# class HierarchicalReasoningModel_VisionV1Block(nn.Module):
#     def __init__(self, config: HierarchicalReasoningModel_VisionV1Config):
#         super().__init__()
#         self.config = config
        
#         # Use MLA with rotary support
#         self.attention = MultiHeadLatentAttention(config)
        
#         # MLP
#         self.mlp = SwiGLU(
#             hidden_size=config.hidden_size,
#             expansion=config.expansion
#         )

#         # Norms (RMSNorm for stability)
#         self.norm_eps = config.rms_norm_eps

#     def forward(self, hidden_states: Tensor, cos_sin: Optional[CosSin] = None) -> Tensor:
#         # Post Norm
#         # Self Attention
#         hidden_states = rms_norm(hidden_states + self.attention(hidden_states, cos_sin), variance_epsilon=self.norm_eps)
#         # Fully Connected
#         hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
#         return hidden_states


# class HierarchicalReasoningModel_VisionV1ReasoningModule(nn.Module):
#     def __init__(self, layers: list[HierarchicalReasoningModel_VisionV1Block]):
#         super().__init__()
#         self.layers = nn.ModuleList(layers)

#     def forward(self, hidden_states: Tensor, input_injection: Tensor, cos_sin: Optional[CosSin] = None) -> Tensor:
#         # Input injection (add)
#         hidden_states = hidden_states + input_injection
#         # Layers
#         for layer in self.layers:
#             hidden_states = layer(hidden_states=hidden_states, cos_sin=cos_sin)
#         return hidden_states


# class VisionClassificationHead(nn.Module):
#     def __init__(self, config: HierarchicalReasoningModel_VisionV1Config):
#         super().__init__()
#         self.config = config
        
#         # Global average pooling + classification head
#         self.norm = nn.LayerNorm(config.hidden_size * 2)
#         self.classifier = CastedLinear(config.hidden_size * 2, config.num_classes, bias=True)  # *2 for multi-class tokens
        
#     def forward(self, x: Tensor) -> Tensor:
#         # Use both class tokens (first 2 tokens) for classification
#         cls_tokens = x[:, :2, :]  # (batch_size, 2, hidden_size)
#         cls_tokens = cls_tokens.flatten(1)  # (batch_size, 2*hidden_size)
        
#         # Normalize and classify
#         cls_tokens = self.norm(cls_tokens)
#         logits = self.classifier(cls_tokens)
        
#         return logits


# class HierarchicalReasoningModel_VisionV1_Inner(nn.Module):
#     def __init__(self, config: HierarchicalReasoningModel_VisionV1Config) -> None:
#         super().__init__()
#         self.config = config
#         self.forward_dtype = getattr(torch, self.config.forward_dtype)
#         self.embed_scale = math.sqrt(self.config.hidden_size)
        
#         # Vision input embedding
#         self.patch_embedding = VisionPatchEmbedding(config)
        
#         # Positional encodings
#         self.seq_len_tokens = 2 + (config.image_size // 4) ** 2  # 2 cls + patches (32/4=8, 8*8=64, total 66)
#         if self.config.pos_encodings == "rope":
#             self.rotary_emb = RotaryEmbedding(
#                 dim=self.config.hidden_size // self.config.num_heads,
#                 max_position_embeddings=self.seq_len_tokens,
#                 base=self.config.rope_theta
#             )
#         elif self.config.pos_encodings == "learned":
#             self.embed_pos = CastedEmbedding(
#                 self.seq_len_tokens,
#                 self.config.hidden_size,
#                 init_std=1.0 / self.embed_scale,
#                 cast_to=self.forward_dtype
#             )
#         else:
#             raise NotImplementedError()
        
#         # Reasoning Layers
#         self.H_level = HierarchicalReasoningModel_VisionV1ReasoningModule(
#             layers=[HierarchicalReasoningModel_VisionV1Block(self.config) for _ in range(self.config.H_layers)]
#         )
#         self.L_level = HierarchicalReasoningModel_VisionV1ReasoningModule(
#             layers=[HierarchicalReasoningModel_VisionV1Block(self.config) for _ in range(self.config.L_layers)]
#         )
        
#         # Classification head (supervise on z_H)
#         self.classification_head = VisionClassificationHead(self.config)
        
#         # Q head for ACT
#         self.q_head = CastedLinear(self.config.hidden_size * 2, 2, bias=True)  # On flattened cls tokens
        
#         # Initial states (truncated normal init)
#         self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1.0), persistent=True)
#         self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1.0), persistent=True)
        
#         # Q head special init
#         with torch.no_grad():
#             self.q_head.weight.zero_()
#             self.q_head.bias.fill_(-5)  # type: ignore

#     def empty_carry(self, batch_size: int) -> HierarchicalReasoningModel_VisionV1InnerCarry:
#         """Create empty carry state."""
#         device = next(self.parameters()).device
#         dtype = self.forward_dtype
        
#         return HierarchicalReasoningModel_VisionV1InnerCarry(
#             H_hidden=torch.empty((batch_size, self.seq_len_tokens, self.config.hidden_size), device=device, dtype=dtype),
#             L_hidden=torch.empty((batch_size, self.seq_len_tokens, self.config.hidden_size), device=device, dtype=dtype),
#         )
    
#     def reset_carry(self, reset_flag: Tensor, carry: HierarchicalReasoningModel_VisionV1InnerCarry) -> HierarchicalReasoningModel_VisionV1InnerCarry:
#         return HierarchicalReasoningModel_VisionV1InnerCarry(
#             H_hidden=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.H_hidden),
#             L_hidden=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.L_hidden),
#         )
    
#     def forward(self, carry: HierarchicalReasoningModel_VisionV1InnerCarry, batch: Dict[str, Tensor]) -> Tuple[HierarchicalReasoningModel_VisionV1InnerCarry, Tensor, Tensor]:
#         """Forward pass for vision HRM inner."""
#         cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        
#         # Input encoding
#         input_embeddings = self.patch_embedding(batch["inputs"])
        
#         # Add learned positions if applicable
#         if self.config.pos_encodings == "learned":
#             pos_ids = torch.arange(self.seq_len_tokens, device=input_embeddings.device).unsqueeze(0)
#             input_embeddings = input_embeddings + self.embed_pos(pos_ids)
        
#         # Forward iterations
#         with torch.no_grad():
#             z_H, z_L = carry.H_hidden, carry.L_hidden

#             for _H_step in range(self.config.H_cycles):
#                 for _L_step in range(self.config.L_cycles):
#                     if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
#                         z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin)

#                 if not (_H_step == self.config.H_cycles - 1):
#                     z_H = self.H_level(z_H, z_L, cos_sin)

#         assert not z_H.requires_grad and not z_L.requires_grad

#         # 1-step grad
#         z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin)
#         z_H = self.H_level(z_H, z_L, cos_sin)

#         # Classification on z_H
#         logits = self.classification_head(z_H)
        
#         # Q logits on z_H cls tokens
#         cls_tokens = z_H[:, :2, :].flatten(1)
#         q_logits = self.q_head(cls_tokens).to(torch.float32)
        
#         new_carry = HierarchicalReasoningModel_VisionV1InnerCarry(H_hidden=z_H.detach(), L_hidden=z_L.detach())
        
#         return new_carry, logits, q_logits


# class HierarchicalReasoningModel_VisionV1(nn.Module):
#     """Vision HRM with ACT wrapper."""
    
#     def __init__(self, config_dict: dict):
#         super().__init__()
#         self.config = HierarchicalReasoningModel_VisionV1Config(**config_dict)
#         self.inner = HierarchicalReasoningModel_VisionV1_Inner(self.config)
    
#     def initial_carry(self, batch: Dict[str, Tensor]):
#         """Initialize carry state."""
#         batch_size = batch["inputs"].shape[0]
        
#         return HierarchicalReasoningModel_VisionV1Carry(
#             inner_carry=self.inner.empty_carry(batch_size),
#             steps=torch.zeros((batch_size,), dtype=torch.int32),
#             halted=torch.ones((batch_size,), dtype=torch.bool),
#             current_data={k: torch.empty_like(v) for k, v in batch.items()}
#         )
    
#     def forward(self, carry: HierarchicalReasoningModel_VisionV1Carry, batch: Dict[str, Tensor], return_keys: Optional[list] = None) -> Tuple[HierarchicalReasoningModel_VisionV1Carry, Dict[str, Tensor]]:
#         """Forward pass for vision HRM with ACT."""
#         # Update data, reset halted sequences
#         new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
#         new_steps = torch.where(carry.halted, 0, carry.steps)
        
#         new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}
        
#         # Forward inner model
#         new_inner_carry, logits, q_logits = self.inner(new_inner_carry, new_current_data)
        
#         q_halt_logits = q_logits[..., 0]
#         q_continue_logits = q_logits[..., 1]
        
#         outputs = {
#             "logits": logits,
#             "q_halt_logits": q_halt_logits,
#             "q_continue_logits": q_continue_logits
#         }
        
#         with torch.no_grad():
#             # Step
#             new_steps = new_steps + 1
#             is_last_step = new_steps >= self.config.halt_max_steps
            
#             halted = is_last_step
            
#             # If training and ACT enabled
#             if self.training and (self.config.halt_max_steps > 1):
#                 # Halt signal
#                 halted = halted | (q_halt_logits > q_continue_logits)
                
#                 # Exploration
#                 min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                
#                 halted = halted & (new_steps >= min_halt_steps)
                
#                 # Compute target Q
#                 _, _, next_q_logits = self.inner(new_inner_carry, new_current_data)
#                 next_q_halt = next_q_logits[..., 0]
#                 next_q_continue = next_q_logits[..., 1]
                
#                 outputs["target_q_continue"] = torch.where(is_last_step, next_q_halt, torch.maximum(next_q_halt, next_q_continue))
        
#         new_carry = HierarchicalReasoningModel_VisionV1Carry(new_inner_carry, new_steps, halted, new_current_data)
        
#         return new_carry, outputs


# # filename="vision_losses.py"

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, Any, Tuple, Optional


# IGNORE_LABEL_ID = -100


# class VisionClassificationLossHead(nn.Module):
#     """Loss head for vision classification tasks with deep supervision."""
    
#     def __init__(self, model: nn.Module, loss_type: str = "cross_entropy", **kwargs):
#         super().__init__()
#         self.model = model
#         self.loss_type = loss_type
        
#     def forward(self, carry: Any, batch: Dict[str, torch.Tensor], return_keys: Any = None) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], bool]:
#         """Forward pass with loss computation."""
#         # Forward through model (ACT wrapper)
#         new_carry, outputs = self.model(carry, batch, return_keys)
        
#         labels = new_carry.current_data["labels"]
#         if labels.dim() > 1:
#             labels = labels[:, 0]  # Assume scalar labels
        
#         # LM loss (classification)
#         lm_loss = F.cross_entropy(outputs["logits"], labels.long())
        
#         with torch.no_grad():
#             is_correct = (torch.argmax(outputs["logits"], dim=-1) == labels.long())
#             valid_metrics = new_carry.halted
            
#             metrics = {
#                 "count": valid_metrics.sum().float(),
#                 "accuracy": torch.where(valid_metrics, is_correct.float(), 0).sum(),
#                 "exact_accuracy": (valid_metrics & is_correct).sum().float(),
#                 "steps": torch.where(valid_metrics, new_carry.steps.float(), 0).sum(),
#             }
        
#         # Q halt loss
#         q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], is_correct.float(), reduction="sum")
        
#         # Q continue loss (if present)
#         q_continue_loss = torch.tensor(0.0, device=lm_loss.device)
#         if "target_q_continue" in outputs:
#             q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
        
#         metrics.update({
#             "lm_loss": lm_loss.detach(),
#             "q_halt_loss": q_halt_loss.detach(),
#             "q_continue_loss": q_continue_loss.detach(),
#         })
        
#         total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        
#         # Filter outputs for return
#         detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs} if return_keys else {}
        
#         all_finish = new_carry.halted.all()
        
#         return new_carry, total_loss, metrics, detached_outputs, all_finish

#     def initial_carry(self, batch: Dict[str, torch.Tensor]):
#         """Proxy to the underlying model's initial_carry."""
#         return self.model.initial_carry(batch)  # type: ignore


# #  filename="pretrain_vision.py"

# from typing import Optional, Any, Sequence, List, Dict
# from dataclasses import dataclass
# import os
# import math
# import json
# import numpy as np

# import torch
# import torch.distributed as dist
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.data.distributed import DistributedSampler

# import wandb
# import coolname
# import hydra
# import pydantic
# from omegaconf import DictConfig, OmegaConf
# from tqdm import tqdm

# from torch.optim import AdamW
# from adam_atan2 import AdamATan2 # type: ignore
# from dataset.build_cifar_dataset import CIFARDatasetMetadata
# from utils.functions import load_model_class, get_model_source_path


# from dataset.common import PreprocessedCIFARDataset 





# class LossConfig(pydantic.BaseModel):
#     model_config = pydantic.ConfigDict(extra="allow")

#     name: str

# class ArchConfig(pydantic.BaseModel):
#     model_config = pydantic.ConfigDict(extra="allow")

#     name: str
#     loss: LossConfig

# class PretrainVisionConfig(pydantic.BaseModel):
#     model_config = pydantic.ConfigDict(extra="allow")

#     # Architecture config
#     arch: ArchConfig

#     # Path to dataset
#     data_path: str

#     # Hyperparams
#     global_batch_size: int
#     epochs: int

#     lr: float
#     lr_min_ratio: float
#     lr_warmup_steps: int

#     weight_decay: float
#     beta1: float
#     beta2: float

#     # Names
#     project_name: Optional[str] = None
#     run_name: Optional[str] = None
#     checkpoint_path: Optional[str] = None

#     # Misc
#     seed: int = 0
#     checkpoint_every_eval: bool = False
#     eval_interval: Optional[int] = None
#     eval_save_outputs: List[str] = []


# @dataclass
# class TrainState:
#     model: nn.Module
#     optimizers: Sequence[torch.optim.Optimizer]
#     optimizer_lrs: Sequence[float]
#     carry: Any

#     step: int
#     total_steps: int


# def cosine_schedule_with_warmup_lr_lambda(
#     current_step: int, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0
# ):
#     if current_step < num_warmup_steps:
#         return float(current_step) / float(max(1, num_warmup_steps))
#     progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#     return min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


# def create_dataloader(data_path: str, split: str, global_batch_size: int, rank: int, world_size: int):
#     """
#     Creates a DataLoader using the PreprocessedCIFARDataset.
#     """
#     # The data directory for the specific split
#     split_data_dir = os.path.join(data_path, split)

#     # Instantiate our new, efficient dataset class
#     dataset = PreprocessedCIFARDataset(data_dir=split_data_dir)
    
#     # Get metadata directly from the dataset object
#     metadata = dataset.metadata

#     sampler: Optional[DistributedSampler] = None
#     if world_size > 1:
#         sampler = DistributedSampler(
#             dataset, num_replicas=world_size, rank=rank, shuffle=(split == "train")
#         )

#     loader = DataLoader(
#         dataset,
#         batch_size=global_batch_size // world_size,
#         shuffle=(sampler is None and split == "train"),
#         sampler=sampler,
#         num_workers=4,
#         pin_memory=True,
#         drop_last=True
#     )
#     return loader, metadata


# def create_model(config: PretrainVisionConfig, train_metadata: CIFARDatasetMetadata, world_size: int):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Extract model configuration from arch
#     model_cfg = dict(config.arch.model_extra , # type: ignore
                     
#         batch_size=config.global_batch_size // world_size,

#         image_size=train_metadata.image_size,
#         patch_size=train_metadata.patch_size,
#         num_classes=train_metadata.num_classes
#     )

#     # Instantiate model with loss head
#     model_cls = load_model_class(config.arch.name)
#     loss_head_cls = load_model_class(config.arch.loss.name)

#     with torch.device(device):
#         model: nn.Module = model_cls(model_cfg)
#         model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
#         if "DISABLE_COMPILE" not in os.environ:
#             model = torch.compile(model, dynamic=False)  # type: ignore

#         # Broadcast parameters from rank 0
#         if world_size > 1:
#             with torch.no_grad():
#                 for param in list(model.parameters()) + list(model.buffers()):
#                     dist.broadcast(param, src=0)


#     # Optimizers - Try AdamATan2, fallback to AdamW if CUDA fails
#     optimizers = []
#     use_adamw = os.environ.get("FORCE_ADAMW", "false").lower() == "true"
    
#     if device == "cuda" and not use_adamw:
#         try:
#             print("Attempting to create AdamATan2 optimizer...")
#             # Create a dummy optimizer to test if CUDA kernels work
#             dummy_param = nn.Parameter(torch.randn(1, device=device))
#             test_optimizer = AdamATan2([dummy_param], lr=1e-3)
#             dummy_param.grad = torch.zeros_like(dummy_param)
#             test_optimizer.step()  # This will fail if CUDA kernels don't work
            
#             # If we get here, it works
#             optimizers = [
#                 AdamATan2(
#                     model.parameters(),
#                     lr=0,  # Needs to be set by scheduler
#                     weight_decay=config.weight_decay,
#                     betas=(config.beta1, config.beta2)
#                 )
#             ]
#             print("Successfully created AdamATan2 optimizer")
            
#         except Exception as e:
#             print(f"AdamATan2 failed with error: {type(e).__name__}: {e}")
#             print("Falling back to AdamW optimizer")
#             use_adamw = True
#     else:
#         use_adamw = True
        
#     if use_adamw:
#         optimizers = [
#             AdamW(
#                 model.parameters(),
#                 lr=0,  # Needs to be set by scheduler
#                 weight_decay=config.weight_decay,
#                 betas=(config.beta1, config.beta2)
#             )
#         ]
#         print("Using AdamW optimizer")
    
#     optimizer_lrs = [
#         config.lr
#     ]
#     return model, optimizers, optimizer_lrs


# def init_train_state(config: PretrainVisionConfig, train_metadata: CIFARDatasetMetadata, world_size: int):
#     total_steps = (len(PreprocessedCIFARDataset(os.path.join(config.data_path, "train"))) * config.epochs) // config.global_batch_size
#     model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size)
#     return TrainState(
#         model=model,
#         optimizers=optimizers,
#         optimizer_lrs=optimizer_lrs,
#         carry=None,
#         step=0,
#         total_steps=total_steps
#     )


# def save_train_state(config: PretrainVisionConfig, train_state: TrainState):
#     if config.checkpoint_path is None:
#         return
#     os.makedirs(config.checkpoint_path, exist_ok=True)
#     torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}.pth"))


# def detach_carry(carry: HierarchicalReasoningModel_VisionV1Carry) -> HierarchicalReasoningModel_VisionV1Carry:
#     """Detach tensors in carry."""
#     return HierarchicalReasoningModel_VisionV1Carry(
#         inner_carry=HierarchicalReasoningModel_VisionV1InnerCarry(
#             H_hidden=carry.inner_carry.H_hidden.detach(),
#             L_hidden=carry.inner_carry.L_hidden.detach()
#         ),
#         steps=carry.steps,
#         halted=carry.halted,
#         current_data=carry.current_data  # No need to detach data
#     )


# def train_batch(train_state: TrainState, batch: Tuple[Tensor, Tensor], config: PretrainVisionConfig, rank: int, world_size: int):
#     train_state.step += 1
#     if train_state.step > train_state.total_steps:
#         return None

#     model_device = next(train_state.model.parameters()).device
#     inputs, labels = batch
#     batch_dict = {"inputs": inputs.to(model_device), "labels": labels.to(model_device)}

#     # Init carry
#     carry = train_state.model.initial_carry(batch_dict)  # type: ignore

#     all_finish = False
#     metrics_list = []

#     while not all_finish:
#         carry, loss, step_metrics, _, all_finish = train_state.model(
#             carry=carry,
#             batch=batch_dict,
#             return_keys=config.eval_save_outputs
#         )
#         (loss / config.global_batch_size).backward()
#         metrics_list.append(step_metrics)
#         if not all_finish:
#             carry = detach_carry(carry)

#     if world_size > 1 and dist.is_initialized():
#         for p in train_state.model.parameters():
#             if p.grad is not None:
#                 dist.all_reduce(p.grad)
#                 p.grad /= world_size

#     lr_this_step = None
#     for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
#         lr_multiplier = cosine_schedule_with_warmup_lr_lambda(
#             train_state.step, config.lr_warmup_steps, train_state.total_steps, config.lr_min_ratio
#         )
#         lr_this_step = base_lr * lr_multiplier
#         for param_group in optim.param_groups:
#             param_group['lr'] = lr_this_step
#         optim.step()
#         optim.zero_grad()

#     # Aggregate metrics over steps (since only non-zero when halted)
#     if metrics_list:
#         keys = sorted(metrics_list[0].keys())
#         vals = torch.stack([torch.stack([m[k] for m in metrics_list]) for k in keys], dim=1).sum(dim=1)
#         if world_size > 1 and dist.is_initialized():
#             dist.all_reduce(vals)
#         if rank == 0:
#             out = {f"train/{k}": v.item() for k, v in zip(keys, vals)}
#             out["train/lr"] = lr_this_step  # type: ignore
#             return out
#     return None


# def evaluate(train_state: TrainState, eval_loader: DataLoader, rank: int, world_size: int):
#     train_state.model.eval()
#     all_metrics = []
#     with torch.inference_mode():
#         for batch in eval_loader:
#             model_device = next(train_state.model.parameters()).device
#             inputs, labels = batch
#             batch_dict = {"inputs": inputs.to(model_device), "labels": labels.to(model_device)}
#             carry = train_state.model.initial_carry(batch_dict)  # type: ignore
#             halted = carry.halted
#             step_metrics_list = []
#             while not halted.all():
#                 carry, _, step_metrics, _, _ = train_state.model(carry=carry, batch=batch_dict, return_keys=[])
#                 step_metrics_list.append(step_metrics)
#                 halted = carry.halted
#             # Aggregate as in train
#             keys = sorted(step_metrics_list[0].keys())
#             vals = torch.stack([torch.stack([m[k] for m in step_metrics_list]) for k in keys], dim=1).sum(dim=1)
#             all_metrics.append(vals)

#     if not all_metrics:
#         return {}

#     vals = torch.stack(all_metrics).sum(dim=0)
#     if world_size > 1 and dist.is_initialized():
#         dist.all_reduce(vals)

#     if rank == 0:
#         keys = sorted(step_metrics_list[0].keys())
#         return {f"test/{k}": v.item() for k, v in zip(keys, vals)}



# @hydra.main(config_path="config", config_name="cfg_vision_pretrain", version_base=None)
# def main(hydra_config: DictConfig):
#     RANK, WORLD_SIZE = 0, 1
#     if "LOCAL_RANK" in os.environ:
#         backend = "nccl" if torch.cuda.is_available() else "gloo"
#         dist.init_process_group(backend=backend)
#         RANK = dist.get_rank()
#         WORLD_SIZE = dist.get_world_size()
#         if torch.cuda.is_available():
#             torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

#     config = PretrainVisionConfig(**OmegaConf.to_container(hydra_config, resolve=True)) # type: ignore
#     if config.run_name is None:
#         config.run_name = f"{config.arch.name.split('.')[-1]}-{coolname.generate_slug(2)}"
#     if config.checkpoint_path is None:
#         config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name) # type: ignore

#     torch.manual_seed(config.seed + RANK)
#     np.random.seed(config.seed + RANK)

#     train_loader, train_metadata = create_dataloader(config.data_path, "train", config.global_batch_size, RANK, WORLD_SIZE)
#     eval_loader, _ = create_dataloader(config.data_path, "test", config.global_batch_size, RANK, WORLD_SIZE)
#     train_state = init_train_state(config, train_metadata, WORLD_SIZE)

#     if RANK == 0:
#         print(f"Starting run {config.run_name} in project {config.project_name}")
#         wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump())
#         wandb.log({"num_params": sum(p.numel() for p in train_state.model.parameters() if p.requires_grad)})

#     for epoch in range(config.epochs):
#         if RANK == 0:
#             print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
#         if hasattr(train_loader.sampler, 'set_epoch'):
#             train_loader.sampler.set_epoch(epoch) # type: ignore

#         train_state.model.train()
#         pbar = tqdm(train_loader, disable=(RANK != 0))
#         for batch in pbar:
#             metrics = train_batch(train_state, batch, config, RANK, WORLD_SIZE)
#             if RANK == 0 and metrics:
#                 metrics["epoch"] = epoch + 1
#                 pbar.set_postfix({k.split('/')[-1]: f"{v:.3f}" for k, v in metrics.items()})
#                 wandb.log(metrics, step=train_state.step)

#         if (epoch + 1) % (config.eval_interval or 1) == 0:
#             eval_metrics = evaluate(train_state, eval_loader, RANK, WORLD_SIZE)
#             if RANK == 0 and eval_metrics:
#                 eval_metrics["epoch"] = epoch + 1
#                 print("Evaluation metrics:", {k: f"{v:.4f}" for k, v in eval_metrics.items()})
#                 wandb.log(eval_metrics, step=train_state.step)
#                 if config.checkpoint_every_eval:
#                     save_train_state(config, train_state)

#     if dist.is_initialized():
#         dist.destroy_process_group()
#     if RANK == 0:
#         wandb.finish()

# if __name__ == "__main__":
#     main()


from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.common import trunc_normal_init_
from models.layers import SwiGLU, RotaryEmbedding, CastedEmbedding, CastedLinear, CosSin, apply_rotary_pos_emb, rms_norm


@dataclass
class HierarchicalReasoningModel_VisionV1InnerCarry:
    """Inner carry state for vision HRM."""
    H_hidden: Tensor
    L_hidden: Tensor


@dataclass
class HierarchicalReasoningModel_VisionV1Carry:
    """Carry state for vision HRM with ACT."""
    inner_carry: HierarchicalReasoningModel_VisionV1InnerCarry
    steps: Tensor
    halted: Tensor
    current_data: Dict[str, Tensor]


@dataclass
class HierarchicalReasoningModel_VisionV1Config:
    # Model dimensions
    hidden_size: int
    num_heads: int
    expansion: float
    
    # Architecture
    H_layers: int
    L_layers: int
    H_cycles: int
    L_cycles: int
    
    # Vision-specific
    num_classes: int
    patch_size: int
    image_size: int
    
    # ACT parameters
    halt_exploration_prob: float
    halt_max_steps: int
    
    # Training
    batch_size: int
    forward_dtype: str = "bfloat16"
    
    # Positional encoding
    pos_encodings: str = "rope"
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5


class VisionPatchEmbedding(nn.Module):
    """CNN-based patch embedding with inductive bias."""
    
    def __init__(self, config: HierarchicalReasoningModel_VisionV1Config):
        super().__init__()
        self.config = config
        
        # CNN-based patch embedding (3-block conv)
        embed_dim = config.hidden_size
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim//4, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=4, stride=4)  # 4x4 patches for 32x32 -> 8x8=64 patches
        )
        
        # Multi-class tokens (2 instead of 1)
        self.cls_tokens = nn.Parameter(torch.zeros(1, 2, config.hidden_size))
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        
        # Reshape input to image format
        x = x.view(batch_size, self.config.image_size, self.config.image_size, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
        x = x.float() / 255.0  # Normalize to [0, 1]
        
        # CNN patch embedding
        x = self.patch_embed(x)  # (B, hidden_size, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, hidden_size)
        
        # Add multi-class tokens
        cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 66, hidden_size)
        
        return x


class MultiHeadLatentAttention(nn.Module):
    """Compressed attention with rotary embeddings support."""
    
    def __init__(self, config: HierarchicalReasoningModel_VisionV1Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.compression_dim = 64  # Key hyperparameter
        
        # Compressed query projections
        self.q_down = CastedLinear(config.hidden_size, self.compression_dim, bias=False)
        self.q_up = CastedLinear(self.compression_dim, config.hidden_size, bias=False)
        
        # Standard K, V projections
        self.k_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        
    def forward(self, x: Tensor, cos_sin: Optional[CosSin] = None) -> Tensor:
        B, N, C = x.shape
        
        # Compressed queries
        q_compressed = self.q_down(x)  # (B, N, compression_dim)
        q = self.q_up(q_compressed)    # (B, N, hidden_size)
        
        # Standard K, V
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Apply rotary if provided
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q.view(B, N, self.num_heads, self.head_dim), 
                                        k.view(B, N, self.num_heads, self.head_dim), 
                                        cos, sin)
            q = q.view(B, N, C)
            k = k.view(B, N, C)
        
        # Multi-head reshape
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        
        return self.out_proj(out)


class HierarchicalReasoningModel_VisionV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_VisionV1Config):
        super().__init__()
        self.config = config
        
        # Use MLA with rotary support
        self.attention = MultiHeadLatentAttention(config)
        
        # MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion
        )

        # Norms (RMSNorm for stability)
        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states: Tensor, cos_sin: Optional[CosSin] = None) -> Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.attention(hidden_states, cos_sin), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_VisionV1ReasoningModule(nn.Module):
    def __init__(self, layers: list[HierarchicalReasoningModel_VisionV1Block]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: Tensor, input_injection: Tensor, cos_sin: Optional[CosSin] = None) -> Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, cos_sin=cos_sin)
        return hidden_states


class VisionClassificationHead(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_VisionV1Config):
        super().__init__()
        self.config = config
        
        # Global average pooling + classification head
        self.norm = nn.LayerNorm(config.hidden_size * 2)
        self.classifier = CastedLinear(config.hidden_size * 2, config.num_classes, bias=True)  # *2 for multi-class tokens
        
    def forward(self, x: Tensor) -> Tensor:
        # Use both class tokens (first 2 tokens) for classification
        cls_tokens = x[:, :2, :]  # (batch_size, 2, hidden_size)
        cls_tokens = cls_tokens.flatten(1)  # (batch_size, 2*hidden_size)
        
        # Normalize and classify
        cls_tokens = self.norm(cls_tokens)
        logits = self.classifier(cls_tokens)
        
        return logits


class HierarchicalReasoningModel_VisionV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_VisionV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        
        # Vision input embedding
        self.patch_embedding = VisionPatchEmbedding(config)
        
        # Positional encodings
        self.seq_len_tokens = 2 + (config.image_size // 4) ** 2  # 2 cls + patches (32/4=8, 8*8=64, total 66)
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.seq_len_tokens,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.seq_len_tokens,
                self.config.hidden_size,
                init_std=1.0 / self.embed_scale,
                cast_to=self.forward_dtype
            )
        else:
            raise NotImplementedError()
        
        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_VisionV1ReasoningModule(
            layers=[HierarchicalReasoningModel_VisionV1Block(self.config) for _ in range(self.config.H_layers)]
        )
        self.L_level = HierarchicalReasoningModel_VisionV1ReasoningModule(
            layers=[HierarchicalReasoningModel_VisionV1Block(self.config) for _ in range(self.config.L_layers)]
        )
        
        # Classification head (supervise on z_H)
        self.classification_head = VisionClassificationHead(self.config)
        
        # Q head for ACT
        self.q_head = CastedLinear(self.config.hidden_size * 2, 2, bias=True)  # On flattened cls tokens
        
        # Initial states (truncated normal init)
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1.0), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1.0), persistent=True)
        
        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def empty_carry(self, batch_size: int) -> HierarchicalReasoningModel_VisionV1InnerCarry:
        """Create empty carry state."""
        device = next(self.parameters()).device
        dtype = self.forward_dtype
        
        return HierarchicalReasoningModel_VisionV1InnerCarry(
            H_hidden=torch.empty((batch_size, self.seq_len_tokens, self.config.hidden_size), device=device, dtype=dtype),
            L_hidden=torch.empty((batch_size, self.seq_len_tokens, self.config.hidden_size), device=device, dtype=dtype),
        )
    
    def reset_carry(self, reset_flag: Tensor, carry: HierarchicalReasoningModel_VisionV1InnerCarry) -> HierarchicalReasoningModel_VisionV1InnerCarry:
        device = carry.H_hidden.device
        reset_flag = reset_flag.to(device)
        
        return HierarchicalReasoningModel_VisionV1InnerCarry(
            H_hidden=torch.where(reset_flag.view(-1, 1, 1), self.H_init.to(device), carry.H_hidden),
            L_hidden=torch.where(reset_flag.view(-1, 1, 1), self.L_init.to(device), carry.L_hidden),
        )
    
    def forward(self, carry: HierarchicalReasoningModel_VisionV1InnerCarry, batch: Dict[str, Tensor]) -> Tuple[HierarchicalReasoningModel_VisionV1InnerCarry, Tensor, Tensor]:
        """Forward pass for vision HRM inner."""
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        
        # Input encoding
        input_embeddings = self.patch_embedding(batch["inputs"])
        
        # Add learned positions if applicable
        if self.config.pos_encodings == "learned":
            pos_ids = torch.arange(self.seq_len_tokens, device=input_embeddings.device).unsqueeze(0)
            input_embeddings = input_embeddings + self.embed_pos(pos_ids)
        
        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.H_hidden, carry.L_hidden

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, cos_sin)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin)

        # Classification on z_H
        logits = self.classification_head(z_H)
        
        # Q logits on z_H cls tokens
        cls_tokens = z_H[:, :2, :].flatten(1)
        q_logits = self.q_head(cls_tokens).to(torch.float32)
        
        new_carry = HierarchicalReasoningModel_VisionV1InnerCarry(H_hidden=z_H.detach(), L_hidden=z_L.detach())
        
        return new_carry, logits, q_logits


class HierarchicalReasoningModel_VisionV1(nn.Module):
    """Vision HRM with ACT wrapper."""
    
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_VisionV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_VisionV1_Inner(self.config)
    
    def initial_carry(self, batch: Dict[str, Tensor]):
        """Initialize carry state."""
        batch_size = batch["inputs"].shape[0]
        device = next(self.parameters()).device
        
        return HierarchicalReasoningModel_VisionV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
    
    def forward(self, carry: HierarchicalReasoningModel_VisionV1Carry, batch: Dict[str, Tensor], return_keys: Optional[list] = None) -> Tuple[HierarchicalReasoningModel_VisionV1Carry, Dict[str, Tensor]]:
        """Forward pass for vision HRM with ACT."""
        # Update data, reset halted sequences
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)
        
        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}
        
        # Forward inner model
        new_inner_carry, logits, q_logits = self.inner(new_inner_carry, new_current_data)
        
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step
            
            # If training and ACT enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                halted = halted | (q_halt_logits > q_continue_logits)
                
                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                
                halted = halted & (new_steps >= min_halt_steps)
                
                # Compute target Q
                _, _, next_q_logits = self.inner(new_inner_carry, new_current_data)
                next_q_halt = next_q_logits[..., 0]
                next_q_continue = next_q_logits[..., 1]
                
                outputs["target_q_continue"] = torch.where(is_last_step, next_q_halt, torch.maximum(next_q_halt, next_q_continue))
        
        new_carry = HierarchicalReasoningModel_VisionV1Carry(new_inner_carry, new_steps, halted, new_current_data)
        
        return new_carry, outputs
