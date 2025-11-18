# HRM Language Model + RAG System

Complete end-to-end system for training a **Hierarchical Reasoning Model (HRM)** for language tasks and using it in a **Retrieval-Augmented Generation (RAG)** platform for document-based question answering.

## 🎯 Overview

This project implements:

1. **HRM Language Model**: A hierarchical reasoning architecture adapted for text generation
   - Two-level hierarchy: H-level (abstract planning) and L-level (detailed processing)
   - Inspired by human cognitive processes
   - Suitable for complex reasoning and language tasks

2. **RAG System**: Complete retrieval-augmented generation pipeline
   - Document processing and chunking
   - Vector-based semantic search
   - Context-aware answer generation
   - Interactive CLI and REST API interfaces

## 📋 Key Features

- ✅ **Self-contained**: No dependency on external LLM APIs (like OpenAI)
- ✅ **Efficient**: Small model suitable for CPU/single GPU
- ✅ **Hierarchical Reasoning**: Multi-timescale processing like human cognition
- ✅ **RAG Pipeline**: Complete document retrieval and generation system
- ✅ **Flexible**: Easy to scale up model size and customize
- ✅ **Production-ready**: Includes training, inference, and API code

## 🏗️ Architecture

### HRM Language Model

```
Input Tokens → Token Embedding
                ↓
        ┌──────────────┐
        │  H-Level     │  ← High-level abstract reasoning
        │  (Planning)  │     (slow, strategic)
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │  L-Level     │  ← Low-level detailed processing
        │  (Execution) │     (fast, tactical)
        └──────┬───────┘
               ↓
         Language Head → Output Tokens
```

### RAG System

```
1. DOCUMENT INGESTION
   Documents → Chunking → Embedding → Vector DB

2. QUERY PROCESSING
   User Question → Embedding → Similarity Search → Top-K Chunks

3. GENERATION
   Question + Context Chunks → HRM Model → Answer
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd Vision-HRM

# Install dependencies
pip install torch numpy tqdm einops

# Optional for REST API
pip install flask flask-cors
```

### Complete Example

Run the full end-to-end example:

```bash
python example_usage.py
```

This will:
1. Create sample training data
2. Train a small HRM language model
3. Build a RAG knowledge base with sample documents
4. Run example queries

## 📚 Step-by-Step Guide

### Step 1: Train HRM Language Model

```bash
python train_language_model.py \
    --data_path your_text_data.txt \
    --output_dir outputs/my_hrm_model \
    --vocab_size 5000 \
    --tokenizer_level char \
    --max_seq_len 512 \
    --hidden_size 256 \
    --num_heads 8 \
    --num_layers 4 \
    --h_cycles 2 \
    --l_cycles 3 \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 0.0003
```

**Training Data Format**: Plain text file(s) or directory containing `.txt` files

**Key Parameters**:
- `hidden_size`: Model dimension (128-512 for small, 768+ for large)
- `num_layers`: Transformer layers per module (3-6 typical)
- `h_cycles`: High-level reasoning iterations (1-3)
- `l_cycles`: Low-level processing iterations (2-5)
- `tokenizer_level`: 'char' (character-level) or 'word' (word-level)

**Output**:
- `best_model.pt`: Best checkpoint based on validation loss
- `tokenizer.json`: Vocabulary and tokenizer config
- `checkpoint_epoch_N.pt`: Periodic checkpoints

### Step 2: Build RAG Knowledge Base

```bash
# Using Python API
from rag_system import create_rag_system

# Load trained model
rag = create_rag_system(
    model_checkpoint='outputs/my_hrm_model/best_model.pt',
    tokenizer_path='outputs/my_hrm_model/tokenizer.json',
    device='cuda'
)

# Add documents
rag.add_document('path/to/document1.txt')
rag.add_document('path/to/document2.txt')

# Save database
rag.save('rag_data')
```

### Step 3: Query the System

#### Option A: Python API

```python
result = rag.query(
    question="What is machine learning?",
    top_k=3,              # Number of chunks to retrieve
    max_new_tokens=150,   # Max answer length
    temperature=0.7,      # Sampling temperature
)

print(result['answer'])
print(f"Confidence: {result['confidence']:.2%}")
```

#### Option B: Interactive CLI

```bash
python rag_inference_api.py \
    --model_checkpoint outputs/my_hrm_model/best_model.pt \
    --tokenizer outputs/my_hrm_model/tokenizer.json \
    --rag_dir rag_data \
    --interactive
```

Commands:
- `/add <file>`: Add document to knowledge base
- `/stats`: Show system statistics
- `/quit`: Exit
- Type any question to get answers

#### Option C: Single Query

```bash
python rag_inference_api.py \
    --model_checkpoint outputs/my_hrm_model/best_model.pt \
    --tokenizer outputs/my_hrm_model/tokenizer.json \
    --rag_dir rag_data \
    --query "What is deep learning?" \
    --top_k 3
```

#### Option D: REST API

```python
# In rag_inference_api.py, use main_api() function
# Or create custom Flask app:

from rag_inference_api import create_rest_api, RAGInterface

interface = RAGInterface(
    model_checkpoint='outputs/my_hrm_model/best_model.pt',
    tokenizer_path='outputs/my_hrm_model/tokenizer.json',
)

app = create_rest_api(interface)
app.run(host='0.0.0.0', port=5000)
```

API Endpoints:
- `GET /health`: Health check
- `GET /stats`: System statistics
- `POST /add_document`: Add document (`{"file_path": "..."}`)
- `POST /query`: Query system (`{"question": "...", "top_k": 3}`)

## 📁 Project Structure

```
Vision-HRM/
├── models/hrm/
│   ├── hrm_language_v1.py       # HRM language model architecture
│   ├── hrm_vision_v1.py         # Original vision model (for reference)
│   └── hrm_act_v1.py            # Base HRM with ACT
│
├── train_language_model.py      # Training script
├── rag_system.py                 # RAG implementation
├── rag_inference_api.py          # CLI and API interfaces
├── example_usage.py              # End-to-end example
│
├── outputs/                      # Trained models
├── rag_data/                     # RAG vector database
└── sample_data/                  # Example data
```

## 🔧 Configuration

### Model Sizes

**Tiny** (Demo/Testing):
```python
hidden_size=128, num_layers=3, num_heads=4
# ~5M parameters, runs on CPU
```

**Small** (Research):
```python
hidden_size=256, num_layers=4, num_heads=8
# ~15M parameters, GPU recommended
```

**Medium** (Production):
```python
hidden_size=512, num_layers=6, num_heads=8
# ~60M parameters, requires GPU
```

**Large** (High Performance):
```python
hidden_size=768, num_layers=8, num_heads=12
# ~150M parameters, multi-GPU recommended
```

### HRM-Specific Parameters

- `H_cycles`: Number of high-level reasoning iterations (1-3)
  - More cycles = more abstract planning
  - Increases computation time

- `L_cycles`: Number of low-level processing iterations (2-5)
  - More cycles = more detailed processing
  - Good for complex reasoning tasks

### RAG Parameters

- `chunk_size`: Words per document chunk (256-512)
- `chunk_overlap`: Overlap between chunks (64-128)
- `top_k`: Retrieved chunks per query (3-5)
- `temperature`: Generation randomness (0.5-1.0)

## 📊 Performance Tips

### Training

1. **Start Small**: Train on a small dataset first to verify setup
2. **Monitor Loss**: Watch for overfitting (train loss << val loss)
3. **Adjust Learning Rate**: Decrease if loss spikes, increase if too slow
4. **Use Character-Level**: For small datasets, char-level works better
5. **Gradient Clipping**: Already enabled (max_norm=1.0)

### Inference

1. **Batch Processing**: Process multiple queries together
2. **Cache Embeddings**: Reuse document embeddings
3. **Adjust top_k**: Fewer chunks = faster, more = better context
4. **GPU Memory**: Reduce max_seq_len if OOM errors occur

### RAG Quality

1. **Chunk Size**: Smaller chunks = more precise, larger = more context
2. **Overlap**: More overlap = better continuity across chunks
3. **Diverse Documents**: Add varied content for better coverage
4. **Reranking**: Implement cross-encoder reranking for better retrieval

## 🆚 Comparison with LlamaIndex

| Feature | This Project | LlamaIndex |
|---------|-------------|------------|
| **Type** | Complete Model + RAG | RAG Framework Only |
| **Model** | Custom HRM (trained) | External APIs (OpenAI, etc.) |
| **Dependencies** | Minimal (PyTorch) | Many (APIs, services) |
| **Cost** | One-time training | Per-query API costs |
| **Privacy** | Fully local | Data sent to APIs |
| **Customization** | Full control | Limited to framework |
| **Learning Curve** | Moderate | Low |

**When to use this**: Research, privacy-sensitive, learning, full control

**When to use LlamaIndex**: Quick prototypes, production with external APIs

## 🔬 Advanced Usage

### Custom Tokenizer

Replace `SimpleTokenizer` with proper BPE tokenizer:

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Train BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
tokenizer.train(files=["data.txt"], trainer=trainer)
```

### Better Embeddings

Use sentence-transformers for embeddings:

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(texts)
```

### Distributed Training

```bash
torchrun --nproc_per_node=4 train_language_model.py \
    --data_path data.txt \
    --output_dir outputs/distributed \
    --batch_size 32
```

### Fine-tuning

Load pre-trained checkpoint and continue training:

```python
checkpoint = torch.load('outputs/base_model/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Continue training on new domain
train(model, new_data_loader, ...)
```

## 🐛 Troubleshooting

**Problem**: OOM (Out of Memory)
- Reduce `batch_size`, `max_seq_len`, or `hidden_size`
- Use `forward_dtype='float16'` in config

**Problem**: Poor generation quality
- Train longer (more epochs)
- Increase model size
- Use better tokenization (BPE instead of char)
- Add more diverse training data

**Problem**: RAG returns irrelevant results
- Reduce `chunk_size` for more precise chunks
- Increase `top_k` to get more context
- Improve document quality/diversity
- Use better embeddings (sentence-transformers)

**Problem**: Slow inference
- Use GPU (cuda)
- Reduce `max_new_tokens`
- Decrease `L_cycles` in config
- Batch multiple queries

## 📖 References

1. **Hierarchical Reasoning Model**: Inspired by cognitive neuroscience
2. **RAG**: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
3. **Transformers**: Vaswani et al., "Attention Is All You Need"

## 🤝 Contributing

Feel free to:
- Add new tokenizers (BPE, SentencePiece)
- Implement better retrieval (dense, hybrid, reranking)
- Add multi-modal support (images, tables)
- Optimize performance (quantization, pruning)
- Create benchmarks and evaluations

## 📄 License

See main project LICENSE file.

## 🙏 Acknowledgments

Built on top of the Vision-HRM project's hierarchical reasoning architecture.

---

**Questions?** Open an issue or check the example scripts!
