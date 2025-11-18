"""
Training Script for HRM Language Model
Trains the hierarchical reasoning model on text data for language modeling tasks
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from pathlib import Path
from typing import List, Dict
import json
from tqdm import tqdm
import numpy as np

from models.hrm.hrm_language_v1 import HRMLanguageModel, HRMLanguageConfig


class TextDataset(Dataset):
    """Simple text dataset for language model training"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 512,
        train: bool = True,
    ):
        """
        Args:
            data_path: Path to text file or directory
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            train: Whether this is training set
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.train = train

        # Load text data
        self.texts = self._load_texts(data_path)
        print(f"Loaded {len(self.texts)} text samples")

    def _load_texts(self, data_path: str) -> List[str]:
        """Load text data from file or directory"""
        texts = []
        path = Path(data_path)

        if path.is_file():
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by double newline (paragraphs) or lines
                texts = [p.strip() for p in content.split('\n\n') if p.strip()]
                if len(texts) < 10:  # If too few paragraphs, split by line
                    texts = [line.strip() for line in content.split('\n') if line.strip()]
        elif path.is_dir():
            for file_path in path.glob('**/*.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    texts.extend([p.strip() for p in content.split('\n\n') if p.strip()])

        return texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        tokens = self.tokenizer.encode(text, max_length=self.max_seq_len)

        # Ensure minimum length
        if len(tokens) < 2:
            tokens = tokens + [self.tokenizer.pad_token_id] * (2 - len(tokens))

        # Truncate to max length
        tokens = tokens[:self.max_seq_len]

        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)

        # Labels are same as input_ids (for next-token prediction)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class SimpleTokenizer:
    """Simple character-level or word-level tokenizer"""

    def __init__(self, vocab_size: int = 5000, level: str = 'char'):
        """
        Args:
            vocab_size: Size of vocabulary
            level: 'char' or 'word' level tokenization
        """
        self.vocab_size = vocab_size
        self.level = level
        self.vocab = {}
        self.inv_vocab = {}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

        # Special tokens
        self.special_tokens = {
            '<PAD>': self.pad_token_id,
            '<UNK>': self.unk_token_id,
            '<BOS>': self.bos_token_id,
            '<EOS>': self.eos_token_id,
        }

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from text corpus"""
        from collections import Counter

        if self.level == 'char':
            # Character-level
            all_chars = set(''.join(texts))
            chars = sorted(list(all_chars))
            self.vocab = {**self.special_tokens}
            for i, char in enumerate(chars):
                if len(self.vocab) < self.vocab_size:
                    self.vocab[char] = len(self.vocab)
        else:
            # Word-level
            words = []
            for text in texts:
                words.extend(text.lower().split())

            word_counts = Counter(words)
            most_common = word_counts.most_common(self.vocab_size - len(self.special_tokens))

            self.vocab = {**self.special_tokens}
            for word, _ in most_common:
                self.vocab[word] = len(self.vocab)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Built {self.level}-level vocabulary with {len(self.vocab)} tokens")

    def encode(self, text: str, max_length: int = None) -> List[int]:
        """Encode text to token IDs"""
        if self.level == 'char':
            tokens = [self.vocab.get(char, self.unk_token_id) for char in text]
        else:
            words = text.lower().split()
            tokens = [self.vocab.get(word, self.unk_token_id) for word in words]

        # Add BOS token
        tokens = [self.bos_token_id] + tokens

        if max_length is not None:
            tokens = tokens[:max_length]

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        if self.level == 'char':
            chars = [self.inv_vocab.get(tid, '<UNK>') for tid in token_ids]
            text = ''.join([c for c in chars if c not in self.special_tokens])
        else:
            words = [self.inv_vocab.get(tid, '<UNK>') for tid in token_ids]
            text = ' '.join([w for w in words if w not in self.special_tokens])

        return text

    def save(self, path: str):
        """Save tokenizer"""
        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'vocab_size': self.vocab_size,
                'level': self.level,
            }, f)

    @classmethod
    def load(cls, path: str):
        """Load tokenizer"""
        with open(path, 'r') as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data['vocab_size'], level=data['level'])
        tokenizer.vocab = data['vocab']
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
        return tokenizer


def collate_fn(batch):
    """Collate function for DataLoader with padding"""
    # Find max length in batch
    max_len = max(item['input_ids'].size(0) for item in batch)

    input_ids_list = []
    labels_list = []

    for item in batch:
        input_ids = item['input_ids']
        labels = item['labels']

        # Pad to max length
        padding_len = max_len - input_ids.size(0)
        if padding_len > 0:
            input_ids = torch.cat([input_ids, torch.zeros(padding_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((padding_len,), -100, dtype=torch.long)])  # -100 is ignored in loss

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        'input_ids': torch.stack(input_ids_list),
        'labels': torch.stack(labels_list),
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Track loss
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)

    return avg_loss, perplexity


def main(args):
    """Main training function"""
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build or load tokenizer
    tokenizer_path = output_dir / 'tokenizer.json'
    if tokenizer_path.exists():
        print("Loading existing tokenizer...")
        tokenizer = SimpleTokenizer.load(str(tokenizer_path))
    else:
        print("Building tokenizer...")
        # Load all texts for vocab building
        path = Path(args.data_path)
        if path.is_file():
            with open(path, 'r', encoding='utf-8') as f:
                all_texts = [line.strip() for line in f if line.strip()]
        else:
            all_texts = []
            for file_path in path.glob('**/*.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_texts.extend([line.strip() for line in f if line.strip()])

        tokenizer = SimpleTokenizer(vocab_size=args.vocab_size, level=args.tokenizer_level)
        tokenizer.build_vocab(all_texts[:10000])  # Use subset for vocab building
        tokenizer.save(str(tokenizer_path))

    # Create datasets
    train_dataset = TextDataset(
        args.data_path,
        tokenizer,
        max_seq_len=args.max_seq_len,
        train=True,
    )

    # Split into train/val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Create model
    config = HRMLanguageConfig(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=args.max_seq_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        H_cycles=args.h_cycles,
        L_cycles=args.l_cycles,
        dropout=args.dropout,
    )

    model = HRMLanguageModel(config)
    model.to(device)

    print(f"\nModel size: {model.get_num_params():,} parameters")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        val_loss, val_perplexity = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, checkpoint_path)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRM Language Model")

    # Data
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='outputs/hrm_lm', help='Output directory')

    # Tokenizer
    parser.add_argument('--vocab_size', type=int, default=5000, help='Vocabulary size')
    parser.add_argument('--tokenizer_level', type=str, default='char', choices=['char', 'word'], help='Tokenization level')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')

    # Model
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--h_cycles', type=int, default=2, help='H-level reasoning cycles')
    parser.add_argument('--l_cycles', type=int, default=3, help='L-level reasoning cycles')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    main(args)
