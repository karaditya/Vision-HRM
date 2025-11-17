#!/bin/bash
# Quick Start Script for Training a Tiny Language Model with HRM
#
# This script demonstrates the complete workflow:
# 1. Prepare sample text data
# 2. Build dataset
# 3. Train a tiny language model
# 4. Generate text from the trained model
#
# Usage: bash examples/quickstart_tiny_lm.sh

set -e  # Exit on error

echo "========================================="
echo "HRM Tiny Language Model Quick Start"
echo "========================================="
echo ""

# Configuration
DATA_PATH="examples/sample_text_data.txt"
DATASET_DIR="data/quickstart-lm"
CHECKPOINT_DIR="checkpoints/quickstart-lm"
EPOCHS=2000
EVAL_INTERVAL=500
BATCH_SIZE=32

# Step 1: Check if sample data exists
echo "Step 1: Checking sample data..."
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Sample data not found at $DATA_PATH"
    echo "Please ensure examples/sample_text_data.txt exists"
    exit 1
fi
echo "✓ Sample data found"
echo ""

# Step 2: Build dataset
echo "Step 2: Building dataset..."
python dataset/build_text_dataset.py \
    --data-path "$DATA_PATH" \
    --data-format txt \
    --output-dir "$DATASET_DIR" \
    --tokenizer-type char \
    --vocab-size 256 \
    --max-seq-len 64 \
    --stride 32 \
    --train-split 0.9 \
    --num-aug 20 \
    --subsample-size 500

echo "✓ Dataset built successfully"
echo ""

# Step 3: Train model
echo "Step 3: Training tiny language model..."
echo "This will take approximately 10-30 minutes depending on your GPU"
echo ""

python pretrain_lm.py \
    arch=hrm_lm_tiny \
    data_path="$DATASET_DIR" \
    global_batch_size="$BATCH_SIZE" \
    epochs="$EPOCHS" \
    eval_interval="$EVAL_INTERVAL" \
    lr=5e-4 \
    checkpoint_every_eval=true \
    project_name="Quickstart-Tiny-LM" \
    run_name="quickstart-demo"

echo ""
echo "✓ Training completed"
echo ""

# Step 4: Find the checkpoint
echo "Step 4: Locating trained model checkpoint..."
CHECKPOINT_PATH=$(find checkpoints/Quickstart-Tiny-LM -type d -name "quickstart-demo" | head -n 1)

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: Could not find checkpoint"
    exit 1
fi

echo "✓ Found checkpoint at: $CHECKPOINT_PATH"
echo ""

# Step 5: Generate sample text
echo "Step 5: Generating sample text..."
echo ""

echo "--- Generation with prompt: 'The' ---"
python generate_text_lm.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --prompt "The" \
    --max-length 50 \
    --temperature 0.8 \
    --top-k 40

echo ""
echo "--- Generation with prompt: 'Language' ---"
python generate_text_lm.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --prompt "Language" \
    --max-length 50 \
    --temperature 0.8 \
    --top-k 40

echo ""
echo "========================================="
echo "Quick Start Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Check W&B dashboard for training metrics"
echo "2. Experiment with different prompts:"
echo "   python generate_text_lm.py --checkpoint $CHECKPOINT_PATH --prompt 'Your text'"
echo "3. Try training with more data or different architectures:"
echo "   python pretrain_lm.py arch=hrm_lm_v1 data_path=$DATASET_DIR"
echo ""
