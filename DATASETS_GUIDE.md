# Standard Pretraining Datasets Guide

Complete guide for training HRM Language Model on standard datasets used by ChatGPT, LLaMA, DeepSeek, and other large language models.

---

## 🎯 Supported Datasets

The training script now supports the **same datasets** used for pretraining models like:
- **ChatGPT/GPT-4**: Common Crawl, WebText, Books
- **LLaMA**: RedPajama, The Pile, C4
- **DeepSeek**: OpenWebText, Wikipedia, Code datasets

---

## 📚 Available Datasets

### List All Datasets

```bash
python train_language_model.py --list_datasets
```

### Dataset Registry

| Dataset | Size | Description | Use Case |
|---------|------|-------------|----------|
| **openwebtext** | 40GB | Web text from Reddit URLs | General language |
| **c4** | 750GB | Colossal Clean Crawled Corpus | Large-scale pretraining |
| **the_pile** | 825GB | Diverse text (22 sources) | General purpose |
| **wikipedia** | 20GB | Wikipedia articles | Factual knowledge |
| **bookcorpus** | 5GB | Books from various genres | Long-form text |
| **the_stack** | 3TB | Source code (30+ languages) | Code generation |
| **wikitext** | 500MB | Quality Wikipedia subset | Testing/small models |
| **tiny_shakespeare** | 1MB | Shakespeare works | Quick testing |

---

## 🚀 Quick Start Examples

### 1. Small Dataset (Testing)

Start with WikiText for quick testing:

```bash
# Install HuggingFace datasets
pip install datasets

# Train on WikiText-103
python train_language_model.py \
    --dataset wikitext \
    --output_dir outputs/wikitext_model \
    --hidden_size 256 \
    --num_layers 4 \
    --batch_size 16 \
    --num_epochs 10 \
    --use_streaming
```

**Training time**: ~1-2 hours on GPU

### 2. Medium Dataset (Production)

Train on OpenWebText (similar to GPT-2):

```bash
python train_language_model.py \
    --dataset openwebtext \
    --output_dir outputs/openwebtext_model \
    --hidden_size 512 \
    --num_heads 8 \
    --num_layers 6 \
    --batch_size 32 \
    --num_epochs 3 \
    --use_streaming \
    --learning_rate 0.0003
```

**Training time**: ~12-24 hours on GPU
**Requirements**: Good GPU (16GB+ VRAM), streaming enabled

### 3. Large Dataset (Research)

Train on The Pile (full pretraining):

```bash
python train_language_model.py \
    --dataset the_pile \
    --output_dir outputs/pile_model \
    --hidden_size 768 \
    --num_heads 12 \
    --num_layers 8 \
    --batch_size 64 \
    --num_epochs 1 \
    --use_streaming \
    --learning_rate 0.0001
```

**Training time**: Days to weeks
**Requirements**: Multi-GPU setup, 32GB+ VRAM

### 4. Code Dataset

Train on code (like Codex/CodeLLaMA):

```bash
python train_language_model.py \
    --dataset the_stack \
    --output_dir outputs/code_model \
    --hidden_size 512 \
    --num_layers 6 \
    --batch_size 16 \
    --num_epochs 2 \
    --use_streaming
```

---

## 📥 Using Custom Datasets

### HuggingFace Datasets

```bash
# Any HuggingFace dataset
python train_language_model.py \
    --data_path "your-username/your-dataset" \
    --dataset_type huggingface \
    --output_dir outputs/custom_hf \
    --use_streaming
```

### JSONL Files

```bash
# Prepare JSONL file
cat > data.jsonl << 'EOF'
{"text": "First document content..."}
{"text": "Second document content..."}
{"text": "Third document content..."}
EOF

# Train
python train_language_model.py \
    --data_path data.jsonl \
    --dataset_type jsonl \
    --output_dir outputs/jsonl_model
```

### Parquet Files

```bash
python train_language_model.py \
    --data_path data.parquet \
    --dataset_type parquet \
    --output_dir outputs/parquet_model \
    --use_streaming
```

### Multiple Text Files

```bash
# Create directory with text files
mkdir training_data
# Add your .txt files

# Train
python train_language_model.py \
    --data_path training_data/ \
    --dataset_type text \
    --output_dir outputs/multi_text
```

---

## 🎛️ Dataset-Specific Configurations

### For Web Text (OpenWebText, C4)

```bash
python train_language_model.py \
    --dataset openwebtext \
    --vocab_size 10000 \
    --tokenizer_level word \  # Word-level better for web text
    --max_seq_len 1024 \
    --batch_size 32 \
    --use_streaming
```

### For Books (BookCorpus)

```bash
python train_language_model.py \
    --dataset bookcorpus \
    --vocab_size 8000 \
    --tokenizer_level word \
    --max_seq_len 2048 \  # Longer for books
    --batch_size 16 \
    --use_streaming
```

### For Code (The Stack)

```bash
python train_language_model.py \
    --dataset the_stack \
    --vocab_size 15000 \  # More tokens for code
    --tokenizer_level char \  # Char-level for code
    --max_seq_len 512 \
    --batch_size 24 \
    --use_streaming
```

### For Wikipedia

```bash
python train_language_model.py \
    --dataset wikipedia \
    --vocab_size 10000 \
    --tokenizer_level word \
    --max_seq_len 512 \
    --batch_size 32
```

---

## 🔄 Streaming vs In-Memory

### When to Use Streaming

✅ **Use `--use_streaming` when**:
- Dataset > 1GB
- Limited RAM
- Training on HuggingFace datasets
- Want to start training immediately

**Example**:
```bash
python train_language_model.py \
    --dataset c4 \
    --use_streaming \  # Required for 750GB dataset!
    --batch_size 32
```

### When to Use In-Memory

✅ **Don't use streaming when**:
- Dataset < 1GB
- Have enough RAM
- Want faster epoch iterations
- Training on local text files

**Example**:
```bash
python train_language_model.py \
    --data_path small_dataset.txt \
    --batch_size 16  # No --use_streaming
```

---

## 📊 Recommended Configurations by Dataset Size

### Tiny (< 100MB) - Testing

```bash
python train_language_model.py \
    --dataset tiny_shakespeare \
    --vocab_size 500 \
    --hidden_size 128 \
    --num_layers 3 \
    --batch_size 8 \
    --num_epochs 20
```

### Small (100MB - 1GB) - Quick Training

```bash
python train_language_model.py \
    --dataset wikitext \
    --vocab_size 5000 \
    --hidden_size 256 \
    --num_layers 4 \
    --batch_size 16 \
    --num_epochs 10
```

### Medium (1GB - 50GB) - Production

```bash
python train_language_model.py \
    --dataset openwebtext \
    --vocab_size 10000 \
    --hidden_size 512 \
    --num_layers 6 \
    --batch_size 32 \
    --num_epochs 3 \
    --use_streaming
```

### Large (50GB+) - Research

```bash
python train_language_model.py \
    --dataset the_pile \
    --vocab_size 15000 \
    --hidden_size 768 \
    --num_layers 8 \
    --batch_size 64 \
    --num_epochs 1 \
    --use_streaming
```

---

## 🛠️ Advanced Usage

### Mixed Datasets

```python
# Create custom script for multiple datasets
from dataset_loader import StreamingTextDataset
from torch.utils.data import ChainDataset

# Combine datasets
wiki = StreamingTextDataset("wikipedia", tokenizer, streaming=True)
books = StreamingTextDataset("bookcorpus", tokenizer, streaming=True)
code = StreamingTextDataset("the_stack", tokenizer, streaming=True)

# Chain them (requires modification to handle IterableDataset)
# This is advanced - see PyTorch docs
```

### Custom Data Processing

```python
from dataset_loader import StreamingTextDataset

class CustomDataset(StreamingTextDataset):
    def _load_huggingface_dataset(self):
        for text in super()._load_huggingface_dataset():
            # Custom preprocessing
            text = text.lower()  # Lowercase
            text = clean_text(text)  # Your cleaning function
            yield text
```

### Distributed Training Preparation

```bash
# For multi-GPU training (future feature)
python train_language_model.py \
    --dataset c4 \
    --use_streaming \
    --batch_size 64 \  # Total across GPUs
    --num_workers 0 \  # Important for streaming
    --output_dir outputs/distributed
```

---

## 📋 Dataset Comparison

### Similar to ChatGPT/GPT-4

```bash
# Use combination of:
--dataset openwebtext  # Web text
--dataset bookcorpus   # Books
--dataset c4           # Common Crawl
```

### Similar to LLaMA

```bash
# Use:
--dataset the_pile     # Primary dataset
--dataset c4           # Additional web data
```

### Similar to Code Models (Codex, CodeLLaMA)

```bash
# Use:
--dataset the_stack    # Source code
--dataset openwebtext  # Natural language
```

---

## ⚡ Performance Tips

### 1. Optimize Batch Size

```bash
# Find optimal batch size
for bs in 16 32 64; do
    python train_language_model.py \
        --dataset wikitext \
        --batch_size $bs \
        --num_epochs 1
done
```

### 2. Use Streaming for Large Data

```bash
# Always stream large datasets
python train_language_model.py \
    --dataset c4 \
    --use_streaming \  # Saves RAM
    --num_workers 0    # Best for streaming
```

### 3. Cache Tokenizer

```bash
# First run builds tokenizer
python train_language_model.py --dataset wikitext ...

# Subsequent runs reuse it (faster startup)
python train_language_model.py --dataset wikitext ...
```

---

## 🐛 Troubleshooting

### "Dataset not found"

**Solution**:
```bash
# Install datasets library
pip install datasets

# Check available datasets
python train_language_model.py --list_datasets
```

### "Out of memory"

**Solutions**:
```bash
# 1. Enable streaming
--use_streaming

# 2. Reduce batch size
--batch_size 8

# 3. Reduce sequence length
--max_seq_len 256

# 4. Use smaller model
--hidden_size 256 --num_layers 4
```

### "Download too slow"

**Solutions**:
```bash
# 1. Use smaller dataset first
--dataset wikitext  # Instead of c4

# 2. Download manually
from datasets import load_dataset
dataset = load_dataset("openwebtext")
dataset.save_to_disk("openwebtext_local")

# Then use local path
--data_path openwebtext_local
```

### "Invalid field name"

**Solution**:
```python
# Check dataset structure
from datasets import load_dataset
ds = load_dataset("your-dataset")
print(ds['train'][0].keys())  # See available fields

# Use correct field in code
# Edit dataset_loader.py if needed
```

---

## 📦 Installation Requirements

### Minimal (Text files only)

```bash
pip install torch numpy tqdm
```

### Standard (HuggingFace datasets)

```bash
pip install torch numpy tqdm datasets
```

### Full (All formats)

```bash
pip install torch numpy tqdm datasets pyarrow
```

---

## 🎯 Example: Training Like GPT-2

GPT-2 was trained on WebText. Here's a similar setup:

```bash
# Install dependencies
pip install torch datasets

# Train on OpenWebText (open version of WebText)
python train_language_model.py \
    --dataset openwebtext \
    --vocab_size 50000 \
    --tokenizer_level word \
    --max_seq_len 1024 \
    --hidden_size 768 \
    --num_heads 12 \
    --num_layers 12 \
    --h_cycles 2 \
    --l_cycles 4 \
    --batch_size 32 \
    --num_epochs 3 \
    --learning_rate 0.00025 \
    --use_streaming \
    --output_dir outputs/gpt2_style
```

**Expected**:
- **Model size**: ~124M parameters
- **Training time**: ~48-72 hours on V100 GPU
- **VRAM needed**: 16GB+

---

## 📚 Further Reading

- **The Pile**: [paper](https://arxiv.org/abs/2101.00027)
- **C4**: [paper](https://arxiv.org/abs/1910.10683)
- **GPT-2**: [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **HuggingFace Datasets**: [docs](https://huggingface.co/docs/datasets/)

---

## ✅ Quick Reference

| Task | Command |
|------|---------|
| List datasets | `python train_language_model.py --list_datasets` |
| Train on WikiText | `python train_language_model.py --dataset wikitext` |
| Train on local file | `python train_language_model.py --data_path data.txt` |
| Use streaming | Add `--use_streaming` flag |
| Check progress | Watch terminal output, loss should decrease |

---

**🎉 You can now train on the same datasets as ChatGPT, LLaMA, and other SOTA models!**
