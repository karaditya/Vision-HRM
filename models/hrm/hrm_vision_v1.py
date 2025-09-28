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

        device = next(self.parameters()).device  # Get the device's device
        reset_flag = reset_flag.to(device)  # Move reset_flag to the model's device to avoid torch dynamo errors

        return HierarchicalReasoningModel_VisionV1InnerCarry(
            H_hidden=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.H_hidden),
            L_hidden=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.L_hidden),
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
        device = next(self.parameters()).device  # Use model's device

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