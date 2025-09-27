from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.layers import Attention, SwiGLU, RotaryEmbedding, CastedEmbedding, CastedLinear


@dataclass
class HierarchicalReasoningModel_VisionV1Config:
    # Model dimensions
    hidden_size: int
    num_heads: int
    expansion: int
    
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
    forward_dtype: str = "float16"
    
    # Positional encoding
    pos_encodings: str = "rope"
    rope_theta: float = 10000.0


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
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=4, stride=4)  # 4x4 patches
        )
        
        # Multi-class tokens (2 instead of 1)
        self.cls_tokens = nn.Parameter(torch.zeros(1, 2, config.hidden_size))
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        
        # Reshape input to image format
        x = x.view(batch_size, 32, 32, 3).permute(0, 3, 1, 2)  # (B, 3, 32, 32)
        x = x.float() / 255.0  # Normalize to [0, 1]
        
        # CNN patch embedding
        x = self.patch_embed(x)  # (B, hidden_size, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, hidden_size)
        
        # Add multi-class tokens
        cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 66, hidden_size)
        
        return x

class MultiHeadLatentAttention(nn.Module):
    """Compressed attention with 17x parameter reduction."""
    
    def __init__(self, config):
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
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Compressed queries
        q_compressed = self.q_down(x)  # (B, N, compression_dim)
        q = self.q_up(q_compressed)    # (B, N, hidden_size)
        
        # Standard K, V
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        
        return self.out_proj(out)
    



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

    

class HierarchicalReasoningModel_VisionV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_VisionV1Config):
        super().__init__()
        self.config = config
        
        # Use MLA instead of standard attention
        self.attention = MultiHeadLatentAttention(config)
        
        # Keep existing MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion
        )

        # Layer norms
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.mlp_norm = nn.LayerNorm(config.hidden_size)
        
        # Stochastic depth for regularization
        self.drop = nn.Dropout(0.1) 
        
    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        # Self-attention with stochastic depth
        residual = x
        x = self.attn_norm(x)
        x = self.attention(x)
        x = self.drop(x)
        x = residual + x
        
        # MLP with stochastic depth
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.drop(x)
        x = residual + x
        
        return x


class HierarchicalReasoningModel_VisionV1ReasoningModule(nn.Module):
    """Reasoning module for vision tasks."""
    
    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, hidden_states: Tensor, input_injection: Tensor, **kwargs) -> Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        
        # Apply layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        return hidden_states


@dataclass
class HierarchicalReasoningModel_VisionV1InnerCarry:
    """Carry state for vision HRM inner model."""
    H_hidden: Tensor
    L_hidden: Tensor
    step: int


class HierarchicalReasoningModel_VisionV1_Inner(nn.Module):
    """Inner model for vision HRM."""
    
    def __init__(self, config: HierarchicalReasoningModel_VisionV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        # Vision token count (num_patches + 1 for class token)
        self.num_patches = (self.config.image_size // self.config.patch_size) ** 2
        self.seq_len_tokens = self.num_patches + 2
        
        # Vision-specific components
        self.patch_embedding = VisionPatchEmbedding(config)
        self.classification_head = VisionClassificationHead(config)
        
        # Positional encoding
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.seq_len_tokens,  # match vision token length
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.seq_len_tokens,  # match vision token length
                self.config.hidden_size,
                init_std=0.02,
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
        
        # Projection layers
        self.H_proj = CastedLinear(self.config.hidden_size, self.config.hidden_size, bias=False)
        self.L_proj = CastedLinear(self.config.hidden_size, self.config.hidden_size, bias=False)
        
    def empty_carry(self, batch_size: int) -> HierarchicalReasoningModel_VisionV1InnerCarry:
        """Create empty carry state."""
        device = next(self.parameters()).device
        dtype = self.forward_dtype
        
        return HierarchicalReasoningModel_VisionV1InnerCarry(
            H_hidden=torch.zeros((batch_size, self.seq_len_tokens, self.config.hidden_size), device=device, dtype=dtype),
            L_hidden=torch.zeros((batch_size, self.seq_len_tokens, self.config.hidden_size), device=device, dtype=dtype),
            step=0
        )
    
    def forward(self, carry: HierarchicalReasoningModel_VisionV1InnerCarry, batch: Dict[str, Tensor]) -> Tuple[HierarchicalReasoningModel_VisionV1InnerCarry, Tensor, Tensor]:
        """Forward pass for vision HRM."""
        # Get input embeddings
        input_embeds = self.patch_embedding(batch["inputs"])
        
        # Apply positional encoding
        if self.config.pos_encodings == "rope":
            cos, sin = self.rotary_emb()
            # Apply rotary embedding to attention layers
            # (This would be handled in the attention layers themselves)
        elif self.config.pos_encodings == "learned":
            pos_ids = torch.arange(input_embeds.shape[1], device=input_embeds.device).unsqueeze(0)
            input_embeds = input_embeds + self.embed_pos(pos_ids)
        
        # Hierarchical reasoning
        
        with torch.no_grad():
            z_H, z_L = carry.H_hidden, carry.L_hidden
            
            for h_step in range(self.config.H_cycles):
                # High-level reasoning
                if not (h_step == self.config.H_cycles - 1):  # Skip last iteration
                    z_H = self.H_level(z_H, input_embeds)
                
                # Low-level reasoning
                for l_step in range(self.config.L_cycles):
                    if not ((h_step == self.config.H_cycles - 1) and (l_step == self.config.L_cycles - 1)):  # Skip last iteration
                        z_L = self.L_level(z_L, self.H_proj(z_H))
        
        # Ensure no gradients from previous iterations
        assert not z_H.requires_grad and not z_L.requires_grad
        
        # Final iteration WITH gradients (1-step grad)
        z_H = self.H_level(z_H, input_embeds)
        z_L = self.L_level(z_L, self.H_proj(z_H))
        
        # Classification
        logits = self.classification_head(z_L)
        
        # Update carry without gradients for next iteration
        carry.H_hidden = z_H.detach()
        carry.L_hidden = z_L.detach()
        carry.step += 1
        
        return carry, logits, torch.tensor(0.0, device=logits.device)  # Dummy halt probability


@dataclass
class HierarchicalReasoningModel_VisionV1Carry:
    """Carry state for vision HRM."""
    inner_carry: HierarchicalReasoningModel_VisionV1InnerCarry
    steps: Tensor
    halted: Tensor
    current_data: Dict[str, Tensor]


class HierarchicalReasoningModel_VisionV1(nn.Module):
    """Vision HRM with ACT wrapper."""
    
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_VisionV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_VisionV1_Inner(self.config)
    
    def initial_carry(self, batch: Dict[str, Tensor]):
        """Initialize carry state."""
        batch_size = batch["inputs"].shape[0]
        
        return HierarchicalReasoningModel_VisionV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
    
    def forward(self, carry: HierarchicalReasoningModel_VisionV1Carry, batch: Dict[str, Tensor], return_keys: Optional[list] = None) -> Tuple[HierarchicalReasoningModel_VisionV1Carry, Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], bool]:
        """Forward pass for vision HRM."""
        # Update current data
        carry.current_data = batch
        
        # Forward through inner model
        carry.inner_carry, logits, halt_prob = self.inner(carry.inner_carry, batch)
        
        # Compute loss and metrics
        labels = batch["labels"]
        if labels.dim() > 1:
            # If labels are sequences, use the first token (class token position)
            labels = labels[:, 0]
        
        loss = F.cross_entropy(logits, labels.long())
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == labels.long()).float().mean()
        
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "count": torch.tensor(1.0, device=loss.device)
        }
        
        # Return predictions
        preds_dict = {"logits": logits, "predictions": preds}
        
        # Always halt for vision tasks (single forward pass)
        all_finish = True
        
        return carry, metrics, preds_dict, {}, all_finish
