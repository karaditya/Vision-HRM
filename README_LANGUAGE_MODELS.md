# Training Tiny Language Models with HRM

This guide explains how to use the Hierarchical Reasoning Model (HRM) to train tiny language models with limited data and compute resources.

## Overview

The HRM architecture brings powerful hierarchical reasoning capabilities to language modeling:

- **Hierarchical Processing**: Two-level reasoning (H-level for abstract planning, L-level for detailed computation)
- **Adaptive Computation Time (ACT)**: Dynamic depth based on input complexity
- **Parameter Efficient**: Achieves strong performance with <50M parameters
- **Data Efficient**: Works well with limited training data (1K-10K examples)
- **Fast Training**: Optimized for training on consumer GPUs

## Quick Start

### 1. Prepare Your Text Data

Create a simple text file or use an existing corpus:

```bash
# Example: Create a simple text dataset
cat > sample_text.txt << 'EOF'
The quick brown fox jumps over the lazy dog.
This is a sample sentence for language model training.
HRM enables efficient reasoning with hierarchical processing.
Language models learn patterns from text data.
EOF
```

### 2. Build the Dataset

```bash
# Character-level tokenization (recommended for tiny models)
python dataset/build_text_dataset.py \
    --data-path sample_text.txt \
    --data-format txt \
    --output-dir data/text-lm \
    --tokenizer-type char \
    --vocab-size 256 \
    --max-seq-len 128 \
    --num-aug 10 \
    --subsample-size 1000

# Byte-level tokenization (handles any text)
python dataset/build_text_dataset.py \
    --data-path sample_text.txt \
    --output-dir data/text-lm-byte \
    --tokenizer-type byte \
    --max-seq-len 128
```

### 3. Train a Tiny Language Model

```bash
# Ultra-tiny model (~5M params) - Single GPU
python pretrain_lm.py \
    arch=hrm_lm_tiny \
    data_path=data/text-lm \
    global_batch_size=64 \
    epochs=5000 \
    eval_interval=500

# Standard tiny model (~15M params)
python pretrain_lm.py \
    arch=hrm_lm_v1 \
    data_path=data/text-lm \
    global_batch_size=128

# Small model (~50M params) - 4 GPUs
OMP_NUM_THREADS=8 torchrun --nproc-per-node 4 pretrain_lm.py \
    arch=hrm_lm_small \
    data_path=data/text-lm \
    global_batch_size=512
```

## Architecture Variants

### Ultra-Tiny (`hrm_lm_tiny`)
- **Parameters**: ~5M
- **Hidden Size**: 128
- **Layers**: 2 H-layers, 2 L-layers
- **Use Case**: Proof of concept, extreme resource constraints
- **Training Time**: ~30 minutes on RTX 4070

### Standard Tiny (`hrm_lm_v1`)
- **Parameters**: ~15M
- **Hidden Size**: 256
- **Layers**: 3 H-layers, 3 L-layers
- **Use Case**: Small-scale projects, experimentation
- **Training Time**: ~2 hours on RTX 4070

### Small (`hrm_lm_small`)
- **Parameters**: ~50M
- **Hidden Size**: 512
- **Layers**: 6 H-layers, 6 L-layers
- **Use Case**: Production tiny models, better performance
- **Training Time**: ~8 hours on RTX 4070

## Dataset Preparation

### Supported Input Formats

1. **Plain Text Files** (`.txt`)
   ```bash
   python dataset/build_text_dataset.py \
       --data-path /path/to/corpus.txt \
       --data-format txt
   ```

2. **JSON Files** (`.json`)
   ```bash
   python dataset/build_text_dataset.py \
       --data-path /path/to/data.json \
       --data-format json \
       --text-field "content"
   ```

3. **JSONL Files** (`.jsonl`)
   ```bash
   python dataset/build_text_dataset.py \
       --data-path /path/to/data.jsonl \
       --data-format jsonl \
       --text-field "text"
   ```

4. **Directory of Files**
   ```bash
   python dataset/build_text_dataset.py \
       --data-path /path/to/text_files/ \
       --data-format txt
   ```

### Tokenization Options

**Character-level** (Recommended for tiny models):
- Vocab size: 100-500
- Best for: Small, domain-specific corpora
- Advantages: Smaller vocab, never encounters unknown tokens

```bash
--tokenizer-type char --vocab-size 256
```

**Byte-level**:
- Vocab size: ~260 (fixed)
- Best for: Multi-lingual or diverse text
- Advantages: Universal, handles any UTF-8 text

```bash
--tokenizer-type byte
```

### Dataset Builder Parameters

```bash
python dataset/build_text_dataset.py \
    --data-path <path>              # Input text file(s)
    --data-format txt               # txt, json, or jsonl
    --text-field text               # JSON field containing text
    --output-dir data/text-lm       # Output directory
    --tokenizer-type char           # char or byte
    --vocab-size 512                # Max vocabulary size
    --max-seq-len 128               # Sequence length
    --stride 64                     # Overlap between sequences (default: max-seq-len)
    --train-split 0.9               # Train/test split ratio
    --num-aug 10                    # Augmentation factor
    --subsample-size 1000           # Limit dataset size (optional)
```

## Training Configuration

### Basic Training

```bash
# Minimal training command
python pretrain_lm.py data_path=data/text-lm

# With custom hyperparameters
python pretrain_lm.py \
    data_path=data/text-lm \
    global_batch_size=256 \
    lr=5e-4 \
    epochs=10000 \
    eval_interval=1000
```

### Advanced Training

```bash
# Custom learning rate schedule
python pretrain_lm.py \
    data_path=data/text-lm \
    lr=1e-3 \
    lr_min_ratio=0.1 \
    lr_warmup_steps=1000 \
    weight_decay=0.1

# Adaptive Computation Time (ACT) tuning
python pretrain_lm.py \
    data_path=data/text-lm \
    arch.halt_max_steps=16 \
    arch.halt_exploration_prob=0.2

# Custom model dimensions
python pretrain_lm.py \
    data_path=data/text-lm \
    arch.hidden_size=384 \
    arch.num_heads=6 \
    arch.H_layers=4 \
    arch.L_layers=4
```

### Multi-GPU Training

```bash
# 4 GPUs
OMP_NUM_THREADS=8 torchrun --nproc-per-node 4 pretrain_lm.py \
    data_path=data/text-lm \
    global_batch_size=1024

# 8 GPUs
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain_lm.py \
    data_path=data/text-lm \
    global_batch_size=2048
```

## Monitoring Training

The training script automatically logs to [Weights & Biases](https://wandb.ai/):

```bash
# Login to W&B first
wandb login

# Training will log:
# - train/lm_loss: Language modeling loss
# - train/accuracy: Token-level accuracy
# - train/exact_accuracy: Sequence-level accuracy
# - train/q_halt_loss: ACT halting loss
# - train/steps: Average reasoning steps
# - train/lr: Current learning rate
```

## Checkpoints and Evaluation

Checkpoints are saved in `checkpoints/<project_name>/<run_name>/`:

```bash
# Resume from checkpoint
python pretrain_lm.py \
    checkpoint=checkpoints/Text-lm-HRM-LM/HRMLM-cool-name/step_50000

# Evaluate only
python evaluate.py \
    checkpoint=checkpoints/Text-lm-HRM-LM/HRMLM-cool-name/step_50000
```

## Example Use Cases

### 1. Code Completion Model

```bash
# Prepare code dataset
python dataset/build_text_dataset.py \
    --data-path /path/to/source_code/ \
    --output-dir data/code-lm \
    --tokenizer-type byte \
    --max-seq-len 256 \
    --subsample-size 10000

# Train
python pretrain_lm.py \
    data_path=data/code-lm \
    arch=hrm_lm_small \
    epochs=20000
```

### 2. Domain-Specific Text Model

```bash
# Medical text, legal documents, etc.
python dataset/build_text_dataset.py \
    --data-path domain_corpus.txt \
    --output-dir data/domain-lm \
    --tokenizer-type char \
    --vocab-size 256 \
    --max-seq-len 128

# Train with domain adaptation
python pretrain_lm.py \
    data_path=data/domain-lm \
    lr=1e-3 \
    weight_decay=0.05
```

### 3. Tiny Math/Logic Model

```bash
# Mathematical expressions or logical statements
python dataset/build_text_dataset.py \
    --data-path math_data.txt \
    --output-dir data/math-lm \
    --tokenizer-type char \
    --vocab-size 128 \
    --max-seq-len 64

# Train with higher ACT steps for reasoning
python pretrain_lm.py \
    data_path=data/math-lm \
    arch.halt_max_steps=32 \
    arch.H_cycles=4 \
    arch.L_cycles=4
```

## Architecture Details

### Hierarchical Reasoning

HRM uses two reasoning levels:

1. **H-Level (High-Level Module)**
   - Slow, abstract planning
   - Captures long-range dependencies
   - Guides overall generation strategy

2. **L-Level (Low-Level Module)**
   - Fast, detailed computation
   - Handles local token predictions
   - Receives guidance from H-level

### Adaptive Computation Time (ACT)

The model dynamically adjusts reasoning depth:

- **Simple inputs**: Fewer reasoning cycles (faster)
- **Complex inputs**: More reasoning cycles (better quality)
- **Q-learning**: Learns when to halt via reinforcement learning

## Performance Tips

### For Limited GPU Memory

```bash
# Reduce batch size
python pretrain_lm.py \
    data_path=data/text-lm \
    global_batch_size=32 \
    arch=hrm_lm_tiny

# Use smaller sequences
python dataset/build_text_dataset.py \
    --max-seq-len 64 \
    ...
```

### For Faster Training

```bash
# Reduce ACT steps
python pretrain_lm.py \
    arch.halt_max_steps=4 \
    arch.H_cycles=1 \
    arch.L_cycles=2

# Disable compilation (if causing issues)
DISABLE_COMPILE=1 python pretrain_lm.py ...
```

### For Better Quality

```bash
# More data augmentation
python dataset/build_text_dataset.py \
    --num-aug 100 \
    --stride 32 \
    ...

# Larger model with more reasoning
python pretrain_lm.py \
    arch=hrm_lm_small \
    arch.halt_max_steps=32 \
    epochs=50000
```

## Troubleshooting

### Out of Memory

- Reduce `global_batch_size`
- Use smaller architecture (`hrm_lm_tiny`)
- Reduce `max_seq_len`
- Use single GPU instead of multi-GPU

### Slow Training

- Increase `global_batch_size` (if memory allows)
- Reduce `halt_max_steps`
- Disable compilation: `DISABLE_COMPILE=1`
- Use fewer workers: `num_workers=1`

### Poor Performance

- Increase training data (`--num-aug`)
- Use larger model (`hrm_lm_small`)
- Increase `halt_max_steps` for more reasoning
- Adjust learning rate (`lr=1e-3` to `lr=5e-4`)

## File Structure

```
Vision-HRM/
├── dataset/
│   └── build_text_dataset.py      # Dataset builder
├── models/
│   └── hrm/
│       └── hrm_lm_v1.py           # Language model architecture
├── config/
│   ├── cfg_lm_pretrain.yaml       # Training config
│   └── arch/
│       ├── hrm_lm_tiny.yaml       # Ultra-tiny model
│       ├── hrm_lm_v1.yaml         # Standard tiny model
│       └── hrm_lm_small.yaml      # Small model
├── lm_dataset.py                  # Dataset loader
├── pretrain_lm.py                 # Training script
└── README_LANGUAGE_MODELS.md      # This file
```

## Citation

If you use HRM for language modeling in your research, please cite:

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model},
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734},
}
```

## Support

- **Discord**: [https://discord.gg/sapient](https://discord.gg/sapient)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Docs**: See main [README.md](README.md) for general HRM documentation
