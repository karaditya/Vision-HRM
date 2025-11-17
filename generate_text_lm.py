"""
Text Generation Script for HRM Language Models

This script loads a trained HRM language model and generates text.

Usage:
    # Generate text with default settings
    python generate_text_lm.py --checkpoint checkpoints/path/to/model --prompt "Hello"

    # Generate with custom parameters
    python generate_text_lm.py \
        --checkpoint checkpoints/path/to/model \
        --prompt "The quick brown" \
        --max-length 100 \
        --temperature 0.8 \
        --top-k 40
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from utils.functions import load_model_class


class TextGenerator:
    """Text generator for HRM language models."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Load config
        config_path = os.path.join(checkpoint_path, "all_config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Load vocabulary
        data_path = self.config["data_path"]
        vocab_path = os.path.join(data_path, "vocab.json")
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
            self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # Load model
        self._load_model()

    def _load_model(self):
        """Load model from checkpoint."""
        # Get latest checkpoint
        checkpoint_files = sorted(
            Path(self.checkpoint_path).glob("step_*"),
            key=lambda p: int(p.stem.split("_")[1])
        )
        if not checkpoint_files:
            raise ValueError(f"No checkpoint found in {self.checkpoint_path}")

        checkpoint_file = checkpoint_files[-1]
        print(f"Loading checkpoint: {checkpoint_file}")

        # Build model config
        arch_config = self.config["arch"]
        model_cfg = {
            **arch_config,
            "batch_size": 1,
            "vocab_size": len(self.vocab),
            "seq_len": self.config.get("max_seq_len", 128),
            "num_puzzle_identifiers": 0,
        }

        # Remove nested dicts if present
        if "loss" in model_cfg:
            del model_cfg["loss"]
        if "name" in model_cfg:
            model_name = model_cfg["name"]
            del model_cfg["name"]
        else:
            model_name = "hrm.hrm_lm_v1@HRMLM"

        # Load model class
        model_cls = load_model_class(model_name)

        # Instantiate model
        with torch.device(self.device):
            self.model = model_cls(model_cfg)
            self.model.eval()

        # Load weights
        state_dict = torch.load(checkpoint_file, map_location=self.device)
        # Remove "model." prefix if present (from ACTLossHead wrapper)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        tokens = [self.vocab.get("<bos>", 1)]  # BOS token
        for char in text:
            tokens.append(self.vocab.get(char, self.vocab.get("<unk>", 3)))
        return torch.tensor(tokens, dtype=torch.int32, device=self.device)

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text."""
        chars = []
        for token in tokens.tolist():
            if token in [self.vocab.get("<pad>", 0), self.vocab.get("<bos>", 1), self.vocab.get("<eos>", 2)]:
                continue
            chars.append(self.inv_vocab.get(token, "<unk>"))
        return "".join(chars)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling threshold
        """
        # Encode prompt
        tokens = self.encode(prompt)
        generated_tokens = tokens.tolist()

        print(f"Generating with prompt: '{prompt}'")
        print(f"Initial tokens: {tokens.tolist()}")

        # Initialize carry
        batch = {
            "inputs": tokens.unsqueeze(0),
            "labels": tokens.unsqueeze(0),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.int32, device=self.device)
        }

        carry = self.model.initial_carry(batch)

        # Generate tokens
        for _ in range(max_length):
            # Prepare input (last token)
            current_input = torch.tensor(
                [generated_tokens[-1]],
                dtype=torch.int32,
                device=self.device
            ).unsqueeze(0)

            batch["inputs"] = current_input
            batch["labels"] = current_input

            # Forward pass
            carry, outputs = self.model(carry, batch)

            # Get logits for last position
            logits = outputs["logits"][0, -1]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float("inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float("inf")

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Stop if EOS token
            if next_token == self.vocab.get("<eos>", 2):
                break

            generated_tokens.append(next_token)

        # Decode generated tokens
        return self.decode(torch.tensor(generated_tokens))


def main():
    parser = argparse.ArgumentParser(description="Generate text with HRM language model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--prompt", type=str, default="",
                        help="Input prompt for generation")
    parser.add_argument("--max-length", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Create generator
    generator = TextGenerator(args.checkpoint, device=args.device)

    # Generate text
    generated_text = generator.generate(
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    print("\n" + "="*80)
    print("GENERATED TEXT:")
    print("="*80)
    print(generated_text)
    print("="*80)


if __name__ == "__main__":
    main()
