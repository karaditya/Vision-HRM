"""
HRM Language Model - Adapted for Text Generation and RAG Tasks
Hierarchical Reasoning Model for natural language processing
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from models.layers import (
    Attention,
    SwiGLUFeedForward,
    RMSNorm,
    RoPE,
)


@dataclass
class HRMLanguageConfig:
    """Configuration for HRM Language Model"""
    vocab_size: int = 32000  # Vocabulary size
    max_seq_len: int = 2048  # Maximum sequence length
    hidden_size: int = 512  # Hidden dimension
    num_heads: int = 8  # Number of attention heads
    num_layers: int = 6  # Number of transformer layers

    # HRM-specific: Hierarchical reasoning
    H_cycles: int = 2  # High-level reasoning cycles
    L_cycles: int = 4  # Low-level reasoning cycles

    # Training config
    dropout: float = 0.1
    use_rope: bool = True  # Use Rotary Position Embeddings
    forward_dtype: str = "bfloat16"  # float16 or bfloat16

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"


class HierarchicalModule(nn.Module):
    """Single hierarchical reasoning module (H-level or L-level)"""

    def __init__(self, config: HRMLanguageConfig, level: str = "L"):
        super().__init__()
        self.config = config
        self.level = level  # "H" (abstract) or "L" (detailed)

        # Attention and FFN layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn_norm': RMSNorm(config.hidden_size),
                'attention': Attention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    use_rope=config.use_rope,
                    max_seq_len=config.max_seq_len,
                ),
                'ffn_norm': RMSNorm(config.hidden_size),
                'ffn': SwiGLUFeedForward(
                    hidden_size=config.hidden_size,
                    dropout=config.dropout,
                ),
            })
            for _ in range(config.num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            mask: (batch_size, seq_len) - causal mask
        """
        for layer in self.layers:
            # Pre-norm attention
            attn_out = layer['attention'](layer['attn_norm'](x), mask=mask)
            x = x + attn_out

            # Pre-norm FFN
            ffn_out = layer['ffn'](layer['ffn_norm'](x))
            x = x + ffn_out

        return x


class HRMLanguageModel(nn.Module):
    """
    Hierarchical Reasoning Model for Language Tasks
    Uses two-level hierarchy: H (abstract planning) and L (detailed processing)
    """

    def __init__(self, config: HRMLanguageConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Hierarchical modules
        self.H_module = HierarchicalModule(config, level="H")  # Abstract/planning
        self.L_module = HierarchicalModule(config, level="L")  # Detailed/execution

        # Cross-hierarchy communication
        self.H_to_L_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.L_to_H_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Output head
        self.output_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights (optional: weight tying between embedding and output)
        # self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_len) - token indices
            labels: (batch_size, seq_len) - target tokens for loss computation

        Returns:
            dict with 'logits', 'loss' (if labels provided), 'hidden_states'
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get token embeddings
        x = self.token_embedding(input_ids)  # (B, L, D)

        # Create causal mask
        mask = self.create_causal_mask(seq_len, device)

        # Hierarchical reasoning cycles
        H_state = x  # Initialize H-level with input embeddings

        for h_cycle in range(self.config.H_cycles):
            # H-level: Abstract reasoning
            H_state = self.H_module(H_state, mask=mask)

            # L-level: Detailed processing with H-level guidance
            L_state = x + self.H_to_L_proj(H_state)  # Inject H-level info

            for l_cycle in range(self.config.L_cycles):
                L_state = self.L_module(L_state, mask=mask)

            # Update H-level with L-level insights
            H_state = H_state + self.L_to_H_proj(L_state)

        # Final output from L-level (detailed level)
        hidden_states = L_state

        # Language modeling head
        logits = self.lm_head(self.output_norm(hidden_states))  # (B, L, vocab_size)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate text autoregressively

        Args:
            input_ids: (batch_size, seq_len) - prompt tokens
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold

        Returns:
            (batch_size, seq_len + max_new_tokens) - generated tokens
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            outputs = self(idx_cond)
            logits = outputs['logits'][:, -1, :]  # (B, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params


def create_hrm_language_model(config: Optional[HRMLanguageConfig] = None) -> HRMLanguageModel:
    """Factory function to create HRM language model"""
    if config is None:
        config = HRMLanguageConfig()

    model = HRMLanguageModel(config)

    print(f"Created HRM Language Model:")
    print(f"  Total parameters: {model.get_num_params(non_embedding=False):,}")
    print(f"  Non-embedding parameters: {model.get_num_params(non_embedding=True):,}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  H-cycles: {config.H_cycles}, L-cycles: {config.L_cycles}")

    return model


if __name__ == "__main__":
    # Test the model
    config = HRMLanguageConfig(
        vocab_size=32000,
        max_seq_len=512,
        hidden_size=512,
        num_heads=8,
        num_layers=4,
    )

    model = create_hrm_language_model(config)

    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids, labels=labels)
    print(f"\nForward pass test:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")

    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"\nGeneration test:")
    print(f"  Prompt shape: {prompt.shape}")
    print(f"  Generated shape: {generated.shape}")
