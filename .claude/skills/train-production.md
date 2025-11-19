# Train Production Model

Train a production-quality model on standard datasets.

## Purpose
- Train on real datasets (WikiText, OpenWebText, C4)
- Production-quality model for demos
- Configurable model size

## Usage

Set environment variables then run:
```bash
export DATASET=wikitext        # or openwebtext, c4
export MODEL_SIZE=medium       # small, medium, large
export OUTPUT_NAME=prod_model

/train-production
```

## Commands

```bash
# Set defaults if not provided
DATASET=${DATASET:-wikitext}
MODEL_SIZE=${MODEL_SIZE:-medium}
OUTPUT_NAME=${OUTPUT_NAME:-production_model}

echo "=================================="
echo "Training Production Model"
echo "=================================="
echo "Dataset: $DATASET"
echo "Model size: $MODEL_SIZE"
echo "Output: outputs/$OUTPUT_NAME"
echo ""

# Map model size to config
case $MODEL_SIZE in
    tiny)
        HIDDEN=128
        LAYERS=3
        HEADS=4
        BATCH=4
        ;;
    small)
        HIDDEN=256
        LAYERS=4
        HEADS=8
        BATCH=16
        ;;
    medium)
        HIDDEN=512
        LAYERS=6
        HEADS=8
        BATCH=32
        ;;
    large)
        HIDDEN=768
        LAYERS=8
        HEADS=12
        BATCH=64
        ;;
    *)
        echo "❌ Invalid MODEL_SIZE: $MODEL_SIZE"
        echo "   Use: tiny, small, medium, or large"
        exit 1
        ;;
esac

echo "Configuration:"
echo "  Hidden size: $HIDDEN"
echo "  Layers: $LAYERS"
echo "  Heads: $HEADS"
echo "  Batch size: $BATCH"
echo ""

# Check if dataset library is installed
python -c "import datasets" 2>/dev/null || {
    echo "❌ 'datasets' library not installed"
    echo "   Run: pip install datasets"
    exit 1
}

# Train model
python train_language_model.py \
    --dataset $DATASET \
    --output_dir outputs/$OUTPUT_NAME \
    --vocab_size 10000 \
    --tokenizer_level char \
    --hidden_size $HIDDEN \
    --num_layers $LAYERS \
    --num_heads $HEADS \
    --h_cycles 2 \
    --l_cycles 4 \
    --batch_size $BATCH \
    --num_epochs 10 \
    --learning_rate 0.0003 \
    --use_streaming

# Verify
if [ -f "outputs/$OUTPUT_NAME/best_model.pt" ]; then
    echo ""
    echo "=================================="
    echo "✓ Production Model Ready!"
    echo "=================================="
    echo "Location: outputs/$OUTPUT_NAME/"
    echo ""
    echo "Model size:"
    ls -lh outputs/$OUTPUT_NAME/best_model.pt | awk '{print "  " $5}'
    echo ""
    echo "Next steps:"
    echo "  - Test RAG: export MODEL_PATH=outputs/$OUTPUT_NAME && /test-rag"
    echo "  - Benchmark: /benchmark"
    echo "  - Deploy: /deploy-docker"
    echo ""
else
    echo "❌ Training failed"
    exit 1
fi
```

## Examples

### Small model for testing
```bash
export DATASET=wikitext
export MODEL_SIZE=small
/train-production
```

### Medium model for production
```bash
export DATASET=openwebtext
export MODEL_SIZE=medium
export OUTPUT_NAME=medical_model
/train-production
```

### Large model for research
```bash
export DATASET=c4
export MODEL_SIZE=large
/train-production
```

## Training Time Estimates

- Tiny: 30-60 min (CPU)
- Small: 1-2 hours (GPU)
- Medium: 4-8 hours (GPU)
- Large: 12-24 hours (GPU)

## Success Criteria

- [x] Dataset downloads successfully
- [x] Training completes
- [x] Validation loss < training loss + 0.5
- [x] Model file created
- [x] Ready for RAG deployment
