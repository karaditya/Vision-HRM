# Quick Train Skill

Quickly train a small test model to verify everything works.

## Purpose
- Verify installation is correct
- Test training pipeline
- Create a small model for testing RAG
- ~10 minutes on CPU

## Steps

1. Check if sample data exists, create if needed
2. Train tiny model (5M params, 5 epochs)
3. Verify model was created
4. Report results

## Usage

In Claude Code, run: `/quick-train`

Or manually:
```bash
bash .claude/skills/quick-train.md
```

## Commands

```bash
# Create sample data if needed
if [ ! -f "data/sample.txt" ]; then
    echo "Creating sample data..."
    mkdir -p data
    wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/sample.txt 2>/dev/null || \
    curl -o data/sample.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt 2>/dev/null || \
    echo "Sample text for testing. Machine learning is amazing. Neural networks learn patterns." > data/sample.txt
fi

echo "✓ Sample data ready"

# Train tiny model
echo "Training tiny model (this will take ~10 minutes)..."

python train_language_model.py \
    --data_path data/sample.txt \
    --output_dir outputs/quick_test \
    --vocab_size 500 \
    --tokenizer_level char \
    --hidden_size 128 \
    --num_heads 4 \
    --num_layers 3 \
    --h_cycles 1 \
    --l_cycles 2 \
    --batch_size 4 \
    --num_epochs 5 \
    --learning_rate 0.001 \
    --num_workers 0

# Verify output
if [ -f "outputs/quick_test/best_model.pt" ]; then
    echo ""
    echo "=================================="
    echo "✓ Training Complete!"
    echo "=================================="
    echo "Model: outputs/quick_test/best_model.pt"
    echo "Tokenizer: outputs/quick_test/tokenizer.json"
    echo ""
    echo "Next steps:"
    echo "  - Test RAG: /test-rag"
    echo "  - Benchmark: /benchmark"
    echo ""
else
    echo "❌ Training failed - model file not found"
    exit 1
fi
```

## Expected Output

- Training loss should decrease
- Final loss < 3.0
- Model file created
- Tokenizer file created

## Success Criteria

- [x] Sample data downloaded/created
- [x] Training completes without errors
- [x] Loss decreases over epochs
- [x] Model file exists
- [x] Can be used for RAG testing
