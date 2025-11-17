# HRM Language Model Examples

This directory contains example scripts and data for training tiny language models with HRM.

## Quick Start

The fastest way to get started is with the quick start script:

```bash
# Make script executable
chmod +x examples/quickstart_tiny_lm.sh

# Run the complete workflow
bash examples/quickstart_tiny_lm.sh
```

This script will:
1. Use the sample text data
2. Build a character-level dataset
3. Train an ultra-tiny language model (~5M params)
4. Generate sample text

**Estimated time**: 10-30 minutes on a modern GPU

## Sample Data

`sample_text_data.txt` contains sample text about AI and language models. You can:

1. **Use it as-is** for testing the workflow
2. **Replace it** with your own text data
3. **Expand it** with more text for better results

### Using Your Own Data

```bash
# Simple text file
cat > my_data.txt << 'EOF'
Your text here...
More text...
EOF

# Build dataset
python dataset/build_text_dataset.py \
    --data-path my_data.txt \
    --output-dir data/my-lm \
    --tokenizer-type char \
    --max-seq-len 128

# Train
python pretrain_lm.py data_path=data/my-lm
```

## Example Workflows

### 1. Ultra-Tiny Model (5M params)

**Use case**: Quick experiments, proof of concept

```bash
# Prepare data
python dataset/build_text_dataset.py \
    --data-path examples/sample_text_data.txt \
    --output-dir data/ultra-tiny-lm \
    --tokenizer-type char \
    --vocab-size 128 \
    --max-seq-len 64 \
    --num-aug 10

# Train
python pretrain_lm.py \
    arch=hrm_lm_tiny \
    data_path=data/ultra-tiny-lm \
    global_batch_size=32 \
    epochs=5000 \
    lr=1e-3
```

### 2. Standard Tiny Model (15M params)

**Use case**: Small-scale projects, better quality

```bash
# Prepare data
python dataset/build_text_dataset.py \
    --data-path your_corpus.txt \
    --output-dir data/standard-lm \
    --tokenizer-type char \
    --vocab-size 256 \
    --max-seq-len 128 \
    --num-aug 50

# Train
python pretrain_lm.py \
    arch=hrm_lm_v1 \
    data_path=data/standard-lm \
    global_batch_size=128 \
    epochs=10000
```

### 3. Character-Level Code Model

**Use case**: Code completion, code generation

```bash
# Collect Python files
find /path/to/code -name "*.py" -exec cat {} \; > code_corpus.txt

# Build dataset
python dataset/build_text_dataset.py \
    --data-path code_corpus.txt \
    --output-dir data/code-lm \
    --tokenizer-type byte \
    --max-seq-len 256 \
    --stride 128 \
    --num-aug 20

# Train with more reasoning steps
python pretrain_lm.py \
    arch=hrm_lm_v1 \
    data_path=data/code-lm \
    arch.halt_max_steps=16 \
    arch.H_cycles=3 \
    arch.L_cycles=3 \
    epochs=20000
```

### 4. Domain-Specific Model

**Use case**: Medical, legal, technical text

```bash
# Prepare domain corpus
python dataset/build_text_dataset.py \
    --data-path domain_text.txt \
    --output-dir data/domain-lm \
    --tokenizer-type char \
    --vocab-size 512 \
    --max-seq-len 128 \
    --num-aug 100

# Train with domain adaptation
python pretrain_lm.py \
    arch=hrm_lm_small \
    data_path=data/domain-lm \
    lr=5e-4 \
    weight_decay=0.05 \
    epochs=30000
```

## Comparing Architectures

Train multiple models to compare:

```bash
# Ultra-tiny
python pretrain_lm.py arch=hrm_lm_tiny data_path=data/lm project_name="LM-Comparison" run_name="ultra-tiny"

# Standard tiny
python pretrain_lm.py arch=hrm_lm_v1 data_path=data/lm project_name="LM-Comparison" run_name="standard"

# Small
python pretrain_lm.py arch=hrm_lm_small data_path=data/lm project_name="LM-Comparison" run_name="small"
```

Then compare in W&B dashboard!

## Text Generation Examples

After training, generate text with different settings:

```bash
# Creative generation (high temperature)
python generate_text_lm.py \
    --checkpoint checkpoints/path/to/model \
    --prompt "The future of" \
    --temperature 1.2 \
    --max-length 100

# Focused generation (low temperature)
python generate_text_lm.py \
    --checkpoint checkpoints/path/to/model \
    --prompt "Language models" \
    --temperature 0.5 \
    --max-length 100

# Diverse generation (nucleus sampling)
python generate_text_lm.py \
    --checkpoint checkpoints/path/to/model \
    --prompt "Neural networks" \
    --temperature 0.8 \
    --top-p 0.9 \
    --top-k 0
```

## Advanced: Multi-GPU Training

For larger models or datasets:

```bash
# 4 GPUs
OMP_NUM_THREADS=8 torchrun --nproc-per-node 4 pretrain_lm.py \
    arch=hrm_lm_small \
    data_path=data/lm \
    global_batch_size=512

# 8 GPUs
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain_lm.py \
    arch=hrm_lm_small \
    data_path=data/lm \
    global_batch_size=1024
```

## Tips for Success

### Data Preparation
- **More data is better**: Aim for 10K+ sequences if possible
- **Use augmentation**: Set `--num-aug` to 10-100
- **Overlap sequences**: Use `--stride` less than `--max-seq-len`

### Training
- **Start small**: Begin with `hrm_lm_tiny` to verify everything works
- **Monitor metrics**: Watch W&B for `train/accuracy` and `eval/exact_accuracy`
- **Be patient**: Tiny models need 5K-20K epochs to converge

### Generation
- **Temperature**: 0.5-0.7 for focused, 0.8-1.2 for creative
- **Top-k**: 20-50 for diverse but coherent text
- **Top-p**: 0.9-0.95 for balanced sampling

## Troubleshooting

### "Out of Memory"
- Reduce `global_batch_size` to 16 or 32
- Use `hrm_lm_tiny` architecture
- Reduce `max_seq_len` to 64

### "Training is slow"
- Reduce `halt_max_steps` to 4
- Use smaller architecture
- Check GPU utilization with `nvidia-smi`

### "Model not learning"
- Increase learning rate to 1e-3
- Add more training data
- Increase `num_aug` parameter
- Check that text data is diverse

## File Structure

```
examples/
├── README_EXAMPLES.md          # This file
├── quickstart_tiny_lm.sh       # Quick start script
└── sample_text_data.txt        # Sample training data
```

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes, model sizes
2. **Use your own data**: Train on domain-specific text
3. **Compare architectures**: See which works best for your use case
4. **Share results**: Join our Discord to discuss your experiments

Happy training! 🚀
