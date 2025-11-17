"""
Text Dataset Builder for HRM Language Models

This script builds a text dataset for training tiny language models with HRM.
It supports various text sources including text files, JSON datasets, and common NLP datasets.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
from common import PuzzleDatasetMetadata

# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


class SimpleTokenizer:
    """Simple character or BPE-style tokenizer for tiny language models."""

    def __init__(self, vocab_size: int = 256, tokenizer_type: str = "char"):
        self.vocab_size = vocab_size
        self.tokenizer_type = tokenizer_type
        self.vocab = {}
        self.inv_vocab = {}

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        if self.tokenizer_type == "char":
            # Character-level tokenizer
            chars = set()
            for text in texts:
                chars.update(text)

            # Reserve special tokens
            self.vocab = {
                PAD_TOKEN: 0,
                BOS_TOKEN: 1,
                EOS_TOKEN: 2,
                UNK_TOKEN: 3,
            }

            # Add characters
            for i, char in enumerate(sorted(chars)):
                if len(self.vocab) >= self.vocab_size:
                    break
                self.vocab[char] = len(self.vocab)

        elif self.tokenizer_type == "byte":
            # Byte-level tokenizer (handles any text)
            self.vocab = {
                PAD_TOKEN: 0,
                BOS_TOKEN: 1,
                EOS_TOKEN: 2,
                UNK_TOKEN: 3,
            }
            # All bytes 0-255
            for i in range(256):
                self.vocab[chr(i)] = len(self.vocab)

        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Built vocabulary with {len(self.vocab)} tokens")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if add_special_tokens:
            tokens = [self.vocab[BOS_TOKEN]]
        else:
            tokens = []

        for char in text:
            tokens.append(self.vocab.get(char, self.vocab[UNK_TOKEN]))

        if add_special_tokens:
            tokens.append(self.vocab[EOS_TOKEN])

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for token in tokens:
            if token in [self.vocab[PAD_TOKEN], self.vocab[BOS_TOKEN], self.vocab[EOS_TOKEN]]:
                continue
            chars.append(self.inv_vocab.get(token, UNK_TOKEN))
        return "".join(chars)


def load_text_data(data_path: str, file_pattern: str = "*.txt") -> List[str]:
    """Load text data from files."""
    texts = []
    data_path = Path(data_path)

    if data_path.is_file():
        # Single file
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split into paragraphs or sentences
            texts = [p.strip() for p in content.split('\n\n') if p.strip()]
    else:
        # Directory of files
        for file_path in data_path.glob(file_pattern):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                texts.extend([p.strip() for p in content.split('\n\n') if p.strip()])

    return texts


def load_json_data(json_path: str, text_field: str = "text") -> List[str]:
    """Load text data from JSON/JSONL files."""
    texts = []
    json_path = Path(json_path)

    if json_path.suffix == '.jsonl':
        # JSONL format (one JSON per line)
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if text_field in data:
                    texts.append(data[text_field])
    else:
        # Standard JSON format
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and text_field in item:
                        texts.append(item[text_field])
                    elif isinstance(item, str):
                        texts.append(item)
            elif isinstance(data, dict) and text_field in data:
                texts.append(data[text_field])

    return texts


def create_sequences(texts: List[str], tokenizer: SimpleTokenizer,
                     max_seq_len: int, stride: Optional[int] = None) -> List[List[int]]:
    """Create fixed-length sequences from texts."""
    if stride is None:
        stride = max_seq_len  # No overlap by default

    sequences = []
    for text in tqdm(texts, desc="Creating sequences"):
        tokens = tokenizer.encode(text, add_special_tokens=True)

        # Split into chunks with stride
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + max_seq_len]
            if len(chunk) == max_seq_len:  # Only keep full sequences
                sequences.append(chunk)

    return sequences


def build_dataset(
    output_dir: str,
    texts: List[str],
    tokenizer: SimpleTokenizer,
    max_seq_len: int = 128,
    train_split: float = 0.9,
    num_aug: int = 1,
    stride: Optional[int] = None,
    subsample_size: Optional[int] = None
):
    """Build HRM-compatible dataset from text sequences."""

    os.makedirs(output_dir, exist_ok=True)

    # Create sequences
    sequences = create_sequences(texts, tokenizer, max_seq_len, stride)

    if subsample_size is not None and subsample_size < len(sequences):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(sequences), subsample_size, replace=False)
        sequences = [sequences[i] for i in indices]

    print(f"Created {len(sequences)} sequences of length {max_seq_len}")

    # Split train/test
    split_idx = int(len(sequences) * train_split)
    train_sequences = sequences[:split_idx]
    test_sequences = sequences[split_idx:]

    print(f"Train: {len(train_sequences)}, Test: {len(test_sequences)}")

    # Build dataset for each split
    for split_name, split_sequences in [("train", train_sequences), ("test", test_sequences)]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Create augmented versions
        all_inputs = []
        all_labels = []
        all_puzzle_identifiers = []
        puzzle_indices = [0]
        group_indices = [0]

        for puzzle_id, seq in enumerate(tqdm(split_sequences, desc=f"Building {split_name}")):
            for aug_id in range(num_aug):
                # For language modeling: predict next token
                # Input: tokens[:-1], Label: tokens[1:]
                inputs = seq[:-1]
                labels = seq[1:]

                all_inputs.append(inputs)
                all_labels.append(labels)
                all_puzzle_identifiers.append(puzzle_id)
                puzzle_indices.append(puzzle_indices[-1] + 1)

            group_indices.append(group_indices[-1] + 1)

        # Convert to numpy arrays
        inputs_array = np.array(all_inputs, dtype=np.int32)
        labels_array = np.array(all_labels, dtype=np.int32)
        puzzle_identifiers_array = np.array(all_puzzle_identifiers, dtype=np.int32)
        puzzle_indices_array = np.array(puzzle_indices, dtype=np.int64)
        group_indices_array = np.array(group_indices, dtype=np.int64)

        # Save as numpy arrays
        set_name = "default"
        np.save(os.path.join(split_dir, f"{set_name}__inputs.npy"), inputs_array)
        np.save(os.path.join(split_dir, f"{set_name}__labels.npy"), labels_array)
        np.save(os.path.join(split_dir, f"{set_name}__puzzle_identifiers.npy"), puzzle_identifiers_array)
        np.save(os.path.join(split_dir, f"{set_name}__puzzle_indices.npy"), puzzle_indices_array)
        np.save(os.path.join(split_dir, f"{set_name}__group_indices.npy"), group_indices_array)

        # Create metadata
        metadata = PuzzleDatasetMetadata(
            vocab_size=len(tokenizer.vocab),
            seq_len=max_seq_len - 1,  # -1 because we predict next token
            num_puzzle_identifiers=len(split_sequences),
            pad_id=tokenizer.vocab[PAD_TOKEN],
            blank_identifier_id=0,
            ignore_label_id=-100,
            sets=[set_name],
            total_groups=len(split_sequences),
            mean_puzzle_examples=num_aug
        )

        with open(os.path.join(split_dir, "dataset.json"), 'w') as f:
            json.dump(metadata.model_dump(), f, indent=2)

    # Save tokenizer vocab
    with open(os.path.join(output_dir, "vocab.json"), 'w') as f:
        json.dump(tokenizer.vocab, f, indent=2)

    print(f"Dataset saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build text dataset for HRM language models")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to text file(s) or directory containing text files")
    parser.add_argument("--data-format", type=str, default="txt", choices=["txt", "json", "jsonl"],
                        help="Format of input data")
    parser.add_argument("--text-field", type=str, default="text",
                        help="Field name for text in JSON data")
    parser.add_argument("--output-dir", type=str, default="data/text-lm",
                        help="Output directory for dataset")
    parser.add_argument("--tokenizer-type", type=str, default="char", choices=["char", "byte"],
                        help="Type of tokenizer to use")
    parser.add_argument("--vocab-size", type=int, default=512,
                        help="Maximum vocabulary size")
    parser.add_argument("--max-seq-len", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride for creating overlapping sequences (default: max_seq_len)")
    parser.add_argument("--train-split", type=float, default=0.9,
                        help="Fraction of data for training")
    parser.add_argument("--num-aug", type=int, default=1,
                        help="Number of augmentations per sequence")
    parser.add_argument("--subsample-size", type=int, default=None,
                        help="Subsample to this many sequences (for small experiments)")

    args = parser.parse_args()

    # Load texts
    print(f"Loading texts from {args.data_path}...")
    if args.data_format in ["json", "jsonl"]:
        texts = load_json_data(args.data_path, args.text_field)
    else:
        texts = load_text_data(args.data_path)

    print(f"Loaded {len(texts)} text samples")

    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size, tokenizer_type=args.tokenizer_type)
    tokenizer.build_vocab(texts)

    # Build dataset
    build_dataset(
        output_dir=args.output_dir,
        texts=texts,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        train_split=args.train_split,
        num_aug=args.num_aug,
        stride=args.stride,
        subsample_size=args.subsample_size
    )


if __name__ == "__main__":
    main()
