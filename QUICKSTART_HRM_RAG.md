# 🚀 Quick Start: Train HRM Language Model + Build RAG System

This guide walks you through training a Hierarchical Reasoning Model (HRM) for language tasks and deploying it in a Retrieval-Augmented Generation (RAG) system.

**⏱️ Estimated Time**: 30-60 minutes (depending on model size)

---

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **RAM**: 8GB minimum (16GB+ recommended)
- **GPU**: Optional but recommended (CUDA-compatible)
- **Storage**: 2GB for models + data

### Install Dependencies

```bash
# Core dependencies
pip install torch numpy tqdm

# For standard datasets (OpenWebText, C4, The Pile, etc.)
pip install datasets

# Optional for REST API
pip install flask flask-cors
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed, CUDA available: {torch.cuda.is_available()}')"
```

---

## 🎯 Step-by-Step Guide

### Step 1: Prepare Your Training Data

The model can train on **any text data**. Choose one:

#### Option A: Use Standard Datasets (Recommended for Production)

**NEW**: Train on the same datasets used by ChatGPT, LLaMA, and DeepSeek!

```bash
# See all available datasets
python train_language_model.py --list_datasets

# Use WikiText (good for testing, 500MB)
python train_language_model.py \
    --dataset wikitext \
    --output_dir outputs/wikitext_model \
    --hidden_size 256 \
    --num_epochs 10

# Use OpenWebText (GPT-2 style, 40GB)
python train_language_model.py \
    --dataset openwebtext \
    --output_dir outputs/openwebtext_model \
    --hidden_size 512 \
    --num_epochs 3 \
    --use_streaming

# Use C4 (Large-scale, 750GB)
python train_language_model.py \
    --dataset c4 \
    --output_dir outputs/c4_model \
    --hidden_size 768 \
    --use_streaming
```

**📚 See [DATASETS_GUIDE.md](DATASETS_GUIDE.md) for complete documentation on all datasets.**

#### Option B: Use Your Own Data

Create a text file with your content:
```bash
# Single file
cat > my_data.txt << 'EOF'
Your text content here...
Multiple paragraphs work well.

Add more content to improve training.
EOF
```

#### Option C: Use Sample Data (for testing)

```bash
# Download sample data (Shakespeare, Wikipedia, etc.)
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data.txt
```

#### Option D: Use Multiple Files

```bash
# Create a directory with multiple text files
mkdir training_data
# Add your .txt files to this directory
```

**💡 Tip**: More data = better model. Aim for at least 1MB of text (ideally 10MB+).

---

### Step 2: Train the HRM Language Model

#### Quick Start (Small Model for Testing)

```bash
python train_language_model.py \
    --data_path data.txt \
    --output_dir outputs/hrm_tiny \
    --vocab_size 1000 \
    --tokenizer_level char \
    --max_seq_len 256 \
    --hidden_size 128 \
    --num_heads 4 \
    --num_layers 3 \
    --h_cycles 2 \
    --l_cycles 2 \
    --batch_size 8 \
    --num_epochs 10 \
    --learning_rate 0.001 \
    --num_workers 0
```

**⏱️ Training time**: ~10-20 minutes on CPU

#### Production Model (Better Quality)

```bash
python train_language_model.py \
    --data_path data.txt \
    --output_dir outputs/hrm_production \
    --vocab_size 5000 \
    --tokenizer_level char \
    --max_seq_len 512 \
    --hidden_size 512 \
    --num_heads 8 \
    --num_layers 6 \
    --h_cycles 2 \
    --l_cycles 4 \
    --batch_size 16 \
    --num_epochs 20 \
    --learning_rate 0.0003 \
    --num_workers 4
```

**⏱️ Training time**: ~2-4 hours on GPU

#### Understanding the Parameters

| Parameter | Description | Small | Medium | Large |
|-----------|-------------|-------|--------|-------|
| `vocab_size` | Number of unique tokens | 1000 | 5000 | 10000 |
| `tokenizer_level` | char or word | char | char | word |
| `max_seq_len` | Max sequence length | 256 | 512 | 1024 |
| `hidden_size` | Model dimension | 128 | 512 | 768 |
| `num_heads` | Attention heads | 4 | 8 | 12 |
| `num_layers` | Layers per module | 3 | 6 | 8 |
| `h_cycles` | High-level reasoning cycles | 1-2 | 2 | 2-3 |
| `l_cycles` | Low-level processing cycles | 2 | 3-4 | 4-5 |
| `batch_size` | Batch size | 4-8 | 16-32 | 32-64 |

**💡 HRM Cycles Explained**:
- **H-cycles**: Abstract, strategic thinking (like planning)
- **L-cycles**: Detailed, tactical execution (like implementation)
- More cycles = better reasoning but slower inference

#### Monitor Training

Watch for these signs:
```
✅ Good: Train loss steadily decreasing
✅ Good: Val loss following train loss (gap < 0.5)
❌ Bad: Val loss increasing while train loss decreases (overfitting)
❌ Bad: Loss not decreasing (learning rate too low/high)
```

**Expected Losses**:
- Initial: 5-8
- After 5 epochs: 2-4
- After 10+ epochs: 1-2

#### Training Output

After training, you'll have:
```
outputs/hrm_tiny/
├── best_model.pt          # Best checkpoint (use this!)
├── tokenizer.json         # Vocabulary
└── checkpoint_epoch_5.pt  # Periodic checkpoints
```

---

### Step 3: Prepare Documents for RAG

Create documents you want to query:

```bash
mkdir my_documents

# Example: Create a document about AI
cat > my_documents/ai_basics.txt << 'EOF'
Artificial Intelligence Basics

Artificial intelligence (AI) is the simulation of human intelligence by machines.
It encompasses machine learning, where systems learn from data, and deep learning,
which uses neural networks with multiple layers.

Key applications include:
- Natural language processing
- Computer vision
- Robotics
- Autonomous vehicles

Machine learning algorithms can be supervised (trained with labeled data),
unsupervised (finding patterns in unlabeled data), or reinforcement learning
(learning through trial and error).
EOF

# Add more documents...
cat > my_documents/neural_networks.txt << 'EOF'
Neural Networks Explained

Neural networks are computing systems inspired by biological brains.
They consist of interconnected nodes (neurons) organized in layers.
...
EOF
```

**💡 Tips for Good RAG Documents**:
- Keep paragraphs focused on single topics
- Use clear, factual language
- Break long documents into multiple files
- Aim for 500-5000 words per document

---

### Step 4: Build RAG Knowledge Base

#### Using Python Script

Create `build_rag.py`:

```python
from rag_system import create_rag_system

# Load your trained model
rag = create_rag_system(
    model_checkpoint='outputs/hrm_tiny/best_model.pt',
    tokenizer_path='outputs/hrm_tiny/tokenizer.json',
    device='cuda'  # or 'cpu'
)

# Add documents
import glob
for doc_path in glob.glob('my_documents/*.txt'):
    print(f"Adding {doc_path}...")
    rag.add_document(doc_path)

# Save knowledge base
rag.save('rag_data')
print("✓ RAG knowledge base built successfully!")
```

Run it:
```bash
python build_rag.py
```

#### Using Command Line

```bash
# Add a single document
python rag_system.py \
    --model_checkpoint outputs/hrm_tiny/best_model.pt \
    --tokenizer outputs/hrm_tiny/tokenizer.json \
    --document my_documents/ai_basics.txt \
    --save_dir rag_data
```

---

### Step 5: Query Your RAG System

#### Option A: Interactive Mode (Recommended)

```bash
python rag_inference_api.py \
    --model_checkpoint outputs/hrm_tiny/best_model.pt \
    --tokenizer outputs/hrm_tiny/tokenizer.json \
    --rag_dir rag_data \
    --interactive
```

**Interactive commands**:
```
You: What is machine learning?
Assistant: Machine learning is a type of AI where systems learn from data...

You: /add my_documents/new_doc.txt
> Added document: my_documents/new_doc.txt

You: /stats
> Knowledge base chunks: 45
> Model parameters: 2,345,678

You: /quit
> Goodbye!
```

#### Option B: Single Query

```bash
python rag_inference_api.py \
    --model_checkpoint outputs/hrm_tiny/best_model.pt \
    --tokenizer outputs/hrm_tiny/tokenizer.json \
    --rag_dir rag_data \
    --query "What are neural networks?" \
    --top_k 3 \
    --max_tokens 150
```

#### Option C: Python API

```python
from rag_system import create_rag_system

# Load model and RAG system
rag = create_rag_system(
    model_checkpoint='outputs/hrm_tiny/best_model.pt',
    tokenizer_path='outputs/hrm_tiny/tokenizer.json',
)
rag.load('rag_data')

# Query
result = rag.query(
    question="What is artificial intelligence?",
    top_k=3,
    max_new_tokens=150,
    temperature=0.7
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")

# Show sources
for i, source in enumerate(result['sources'], 1):
    print(f"\n[{i}] {source['text'][:200]}...")
```

#### Option D: REST API

**Start the server**:
```python
# api_server.py
from rag_inference_api import RAGInterface, create_rest_api

interface = RAGInterface(
    model_checkpoint='outputs/hrm_tiny/best_model.pt',
    tokenizer_path='outputs/hrm_tiny/tokenizer.json',
    rag_save_dir='rag_data'
)

app = create_rest_api(interface)
app.run(host='0.0.0.0', port=5000)
```

```bash
python api_server.py
```

**Use the API**:
```bash
# Query endpoint
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is AI?",
    "top_k": 3,
    "temperature": 0.7
  }'

# Add document
curl -X POST http://localhost:5000/add_document \
  -H "Content-Type: application/json" \
  -d '{"file_path": "my_documents/new_doc.txt"}'

# Check stats
curl http://localhost:5000/stats
```

---

## 🎬 Complete End-to-End Example

Run this single command to see everything in action:

```bash
python example_usage.py
```

This will:
1. ✅ Create sample training data
2. ✅ Train a small HRM model
3. ✅ Build a RAG knowledge base
4. ✅ Run example queries
5. ✅ Show you how to use the system

---

## 🔧 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Install PyTorch
```bash
pip install torch
# Or with CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Problem: "CUDA out of memory"
**Solutions**:
- Reduce `batch_size` (try 4 or 2)
- Reduce `max_seq_len` (try 256 or 128)
- Reduce `hidden_size` (try 256 or 128)
- Use CPU instead: set `device='cpu'`

### Problem: Training loss not decreasing
**Solutions**:
- Increase `learning_rate` (try 0.001 or 0.003)
- Add more training data
- Train for more epochs
- Check data quality (remove noise/corrupted text)

### Problem: Model generates gibberish
**Solutions**:
- Train longer (try 20-50 epochs)
- Increase model size (`hidden_size`, `num_layers`)
- Use character-level tokenization for small datasets
- Add more diverse training data

### Problem: RAG returns irrelevant answers
**Solutions**:
- Reduce `chunk_size` in RAG system (try 256)
- Increase `top_k` when querying (try 5-7)
- Add more relevant documents
- Improve document quality (clear, factual content)

### Problem: Slow inference
**Solutions**:
- Use GPU if available
- Reduce `max_new_tokens` (try 50-100)
- Reduce `l_cycles` to 2-3
- Use smaller model (`hidden_size=256`)

---

## 📊 Expected Results

### Training (10 epochs, small model)
```
Epoch 1/10: Loss: 4.23
Epoch 5/10: Loss: 2.15
Epoch 10/10: Loss: 1.42
Val Loss: 1.58, Perplexity: 4.85
✓ Saved best model
```

### RAG Query Example
```
Question: What is machine learning?
Answer: Machine learning is a type of artificial intelligence where systems
learn from data to improve their performance on tasks without being explicitly
programmed for each scenario.

Confidence: 87%
Sources used: 3
```

---

## 🎓 Next Steps

### Improve Model Quality
1. **More Data**: Train on 10MB+ text for better results
2. **Bigger Model**: Use `hidden_size=512+` for production
3. **Better Tokenization**: Switch to BPE/SentencePiece
4. **Fine-tuning**: Train on domain-specific data

### Enhance RAG System
1. **Better Embeddings**: Use sentence-transformers
2. **Reranking**: Add cross-encoder reranking
3. **Hybrid Search**: Combine dense + keyword search
4. **Chunk Optimization**: Tune chunk size and overlap

### Production Deployment
1. **Model Optimization**: Quantization, pruning
2. **Caching**: Cache embeddings and results
3. **Load Balancing**: Multi-GPU inference
4. **Monitoring**: Add logging and metrics

---

## 📚 Additional Resources

- **Full Documentation**: See `README_HRM_RAG.md`
- **Architecture Details**: See `models/hrm/hrm_language_v1.py`
- **Training Script**: See `train_language_model.py`
- **RAG Implementation**: See `rag_system.py`

---

## ❓ FAQ

**Q: How much data do I need?**
A: Minimum 1MB text, recommend 10MB+ for good results.

**Q: Can I use word-level tokenization?**
A: Yes, add `--tokenizer_level word`. Better for large datasets (10MB+).

**Q: How long does training take?**
A: Small model: 10-30 min (CPU), Medium: 1-2 hours (GPU), Large: 4-8 hours (GPU).

**Q: Can I train on multiple GPUs?**
A: Not yet implemented. Use single GPU for now.

**Q: How to save GPU memory?**
A: Reduce batch_size, max_seq_len, or hidden_size.

**Q: Can I use this commercially?**
A: Check the project license file.

**Q: How does this compare to GPT/LLaMA?**
A: Much smaller (2M-150M vs 7B-70B params). Good for learning, research, and specialized tasks.

---

## ✅ Checklist

Before training:
- [ ] PyTorch installed and working
- [ ] Training data prepared (1MB+ text)
- [ ] Output directory specified
- [ ] GPU available (optional but recommended)

After training:
- [ ] Training completed without errors
- [ ] `best_model.pt` exists in output directory
- [ ] Validation loss reasonable (< 3.0)
- [ ] Model generates coherent text

Before RAG:
- [ ] Documents prepared in text format
- [ ] Model checkpoint path correct
- [ ] Tokenizer path correct

After RAG setup:
- [ ] Documents added to knowledge base
- [ ] RAG database saved
- [ ] Queries return relevant answers

---

**🎉 Congratulations!** You now have a working HRM language model and RAG system!

**Need Help?** Open an issue or check the full documentation in `README_HRM_RAG.md`.
