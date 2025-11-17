"""
HRM Language Model Architecture v1

A causal (autoregressive) language model based on the Hierarchical Reasoning Model.
Optimized for training tiny language models with limited compute and data.

Key features:
- Autoregressive (causal) attention for language modeling
- Hierarchical reasoning with H-level (slow, abstract) and L-level (fast, detailed) modules
- Adaptive Computation Time (ACT) for dynamic depth
- Efficient training with small parameter count
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding,
    CosSin, CastedEmbedding, CastedLinear
)


@dataclass
class HRMLMInnerCarry:
    """Internal state for HRM language model."""
    z_H: torch.Tensor  # High-level hidden state
    z_L: torch.Tensor  # Low-level hidden state


@dataclass
class HRMLMCarry:
    """Complete carry state for HRM language model with ACT."""
    inner_carry: HRMLMInnerCarry
    steps: torch.Tensor  # Number of reasoning steps taken
    halted: torch.Tensor  # Whether sequence has halted
    current_data: Dict[str, torch.Tensor]  # Current input data


class HRMLMConfig(BaseModel):
    """Configuration for HRM Language Model."""
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int = 0  # For compatibility, usually 0 for LM

    # Hierarchical reasoning cycles
    H_cycles: int = 2  # Number of high-level reasoning cycles
    L_cycles: int = 2  # Number of low-level reasoning cycles per H-cycle

    # Layer depths
    H_layers: int = 4  # Layers in high-level module
    L_layers: int = 4  # Layers in low-level module

    # Model dimensions
    hidden_size: int = 256
    expansion: float = 4.0  # MLP expansion ratio
    num_heads: int = 4

    # Positional encodings
    pos_encodings: str = "rope"  # "rope" or "learned"
    rope_theta: float = 10000.0

    # Normalization
    rms_norm_eps: float = 1e-5

    # Adaptive Computation Time (ACT)
    halt_max_steps: int = 8  # Maximum reasoning steps
    halt_exploration_prob: float = 0.1  # Exploration probability for Q-learning

    # Computation
    forward_dtype: str = "bfloat16"  # "float32", "float16", or "bfloat16"

    # Language modeling specific
    causal: bool = True  # Always True for language modeling


class HRMLMBlock(nn.Module):
    """Transformer block for HRM language model."""

    def __init__(self, config: HRMLMConfig):
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=config.causal  # Causal attention for LM
        )

        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with post-normalization."""
        # Self-attention with residual and normalization
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps
        )

        # MLP with residual and normalization
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps
        )

        return hidden_states


class HRMLMReasoningModule(nn.Module):
    """Hierarchical reasoning module (H-level or L-level)."""

    def __init__(self, layers: List[HRMLMBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with input injection.

        Args:
            hidden_states: Current hidden states
            input_injection: Information to inject from other level
        """
        # Input injection (additive)
        hidden_states = hidden_states + input_injection

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HRMLMInner(nn.Module):
    """Inner HRM language model without ACT wrapper."""

    def __init__(self, config: HRMLMConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # Embedding layer
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            config.vocab_size,
            config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype
        )

        # Output head
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        # Q-value head for ACT (predicts whether to halt)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # Positional encodings
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len,
                base=config.rope_theta
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len,
                config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype
            )
        else:
            raise ValueError(f"Unknown pos_encodings: {config.pos_encodings}")

        # Hierarchical reasoning modules
        self.H_level = HRMLMReasoningModule(
            layers=[HRMLMBlock(config) for _ in range(config.H_layers)]
        )
        self.L_level = HRMLMReasoningModule(
            layers=[HRMLMBlock(config) for _ in range(config.L_layers)]
        )

        # Initial states for H and L levels
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )

        # Initialize Q-head to predict halt=False initially (helps bootstrapping)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # Bias towards not halting initially

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute input embeddings with positional information."""
        # Token embeddings
        embedding = self.embed_tokens(input_ids.to(torch.int32))

        # Add positional embeddings
        if self.config.pos_encodings == "learned":
            # Scale by 1/sqrt(2) to maintain variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale embeddings
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> HRMLMInnerCarry:
        """Create empty carry state."""
        return HRMLMInnerCarry(
            z_H=torch.empty(
                batch_size, self.config.seq_len, self.config.hidden_size,
                dtype=self.forward_dtype
            ),
            z_L=torch.empty(
                batch_size, self.config.seq_len, self.config.hidden_size,
                dtype=self.forward_dtype
            )
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: HRMLMInnerCarry
    ) -> HRMLMInnerCarry:
        """Reset carry state for halted sequences."""
        return HRMLMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L)
        )

    def forward(
        self,
        carry: HRMLMInnerCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[HRMLMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through HRM language model.

        Returns:
            new_carry: Updated carry state
            logits: Language model logits [batch, seq_len, vocab_size]
            q_logits: (q_halt_logits, q_continue_logits) for ACT
        """
        # Positional information
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None
        )

        # Input embeddings
        input_embeddings = self._input_embeddings(batch["inputs"])

        # Hierarchical reasoning with gradient only on last step
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            # Run H_cycles and L_cycles (all but last step without gradient)
            for h_step in range(self.config.H_cycles):
                for l_step in range(self.config.L_cycles):
                    if not ((h_step == self.config.H_cycles - 1) and
                            (l_step == self.config.L_cycles - 1)):
                        # L-level update: incorporate H-level and input
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if h_step < self.config.H_cycles - 1:
                    # H-level update: incorporate L-level information
                    z_H = self.H_level(z_H, z_L, **seq_info)

        # Final step with gradient for training
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # Language model output logits
        new_carry = HRMLMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        logits = self.lm_head(z_H)

        # Q-values for halting decision (use first token as summary)
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, logits, (q_logits[..., 0], q_logits[..., 1])


class HRMLM(nn.Module):
    """
    HRM Language Model with Adaptive Computation Time (ACT).

    This is the complete model with ACT wrapper for dynamic computation depth.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HRMLMConfig(**config_dict)
        self.inner = HRMLMInner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> HRMLMCarry:
        """Create initial carry state for a batch."""
        batch_size = batch["inputs"].shape[0]

        return HRMLMCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Start halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(
        self,
        carry: HRMLMCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[HRMLMCarry, Dict[str, torch.Tensor]]:
        """
        Forward pass with ACT.

        Args:
            carry: Current carry state
            batch: Input batch with "inputs" and "labels"

        Returns:
            new_carry: Updated carry state
            outputs: Dictionary with "logits", "q_halt_logits", "q_continue_logits"
        """
        # Reset carry for halted sequences, update data
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }

        # Forward through inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = \
            self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        # ACT halting logic
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # During training with ACT enabled
            if self.training and self.config.halt_max_steps > 1:
                # Halt when Q(halt) > Q(continue)
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration: force minimum number of steps
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) *
                    torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q-value for bootstrapping (PQN-style)
                next_q_halt_logits, next_q_continue_logits = self.inner(
                    new_inner_carry, new_current_data
                )[-1]

                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits)
                    )
                )

        return HRMLMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
