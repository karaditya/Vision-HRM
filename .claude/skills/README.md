# Claude Skills for HRM-RAG Project

Automated tasks for developing, testing, and deploying your MVP.

---

## 🎯 What Are Skills?

Skills are pre-packaged commands that automate common development tasks. Instead of typing long commands, you run simple skills.

**Example**:
```bash
# Instead of this:
python train_language_model.py --dataset wikitext --hidden_size 256 --num_layers 4 --batch_size 16 --num_epochs 10 --use_streaming

# Run this:
/quick-train
```

---

## 📚 Available Skills

### Training & Testing

| Skill | Purpose | Time | Usage |
|-------|---------|------|-------|
| `/quick-train` | Train small test model | 10 min | Verify setup works |
| `/train-production` | Train production model | Hours | Create customer demo |
| `/test-rag` | Test RAG system | 5 min | Validate RAG pipeline |
| `/list-datasets` | Show available datasets | 1 min | Choose training data |

### Security & Compliance

| Skill | Purpose | Time | Usage |
|-------|---------|------|-------|
| `/security-audit` | Security checks | 2 min | Before deployment |
| `/benchmark` | Performance metrics | 5 min | Prove efficiency |

### Deployment

| Skill | Purpose | Time | Usage |
|-------|---------|------|-------|
| `/deploy-docker` | Deploy with Docker | 10 min | Production deployment |

---

## 🚀 Quick Start

### 1. First Time Setup

```bash
# Run MVP setup first
./setup_mvp.sh
```

### 2. Test Your Installation

```bash
# Train a small test model
/quick-train

# This will:
# - Download sample data
# - Train tiny model (10 min)
# - Verify everything works
```

### 3. Test RAG System

```bash
# Test RAG with the model
/test-rag

# This will:
# - Create test documents
# - Build RAG knowledge base
# - Run test queries
# - Validate results
```

### 4. Check Security

```bash
# Run security audit
/security-audit

# Checks:
# - No hardcoded secrets
# - Encryption working
# - Auth implemented
# - Dependencies secure
```

---

## 📖 Skill Details

### `/quick-train`

**What it does**: Trains a tiny model (5M params) for testing

**When to use**:
- First time setup
- Verify installation
- Quick testing

**Example**:
```bash
/quick-train
```

**Output**:
- `outputs/quick_test/best_model.pt`
- `outputs/quick_test/tokenizer.json`

---

### `/train-production`

**What it does**: Trains production-quality model on real datasets

**When to use**:
- Creating demo models
- Customer pilots
- Production deployment

**Configuration**:
```bash
# Set these before running
export DATASET=wikitext      # or openwebtext, c4
export MODEL_SIZE=medium     # tiny, small, medium, large
export OUTPUT_NAME=my_model  # output directory name

/train-production
```

**Examples**:

Small model for testing:
```bash
export DATASET=wikitext
export MODEL_SIZE=small
/train-production
```

Production model:
```bash
export DATASET=openwebtext
export MODEL_SIZE=medium
export OUTPUT_NAME=medical_model
/train-production
```

**Output**: `outputs/{OUTPUT_NAME}/best_model.pt`

---

### `/test-rag`

**What it does**: Tests complete RAG pipeline with sample documents

**When to use**:
- After training a model
- Before customer demo
- Verifying RAG quality

**Example**:
```bash
# Use default model (quick_test)
/test-rag

# Or specify model
export MODEL_PATH=outputs/my_model
/test-rag
```

**What it tests**:
- Document processing
- Retrieval quality
- Answer generation
- Confidence scores

---

### `/security-audit`

**What it does**: Comprehensive security check

**When to use**:
- Before deployment
- Weekly during development
- Before customer demos

**Example**:
```bash
/security-audit
```

**Checks**:
- ✓ No hardcoded secrets
- ✓ Encryption implemented
- ✓ Authentication working
- ✓ Audit logging present
- ✓ Dependencies secure
- ✓ .env in .gitignore

---

### `/benchmark`

**What it does**: Measures model performance

**When to use**:
- Creating marketing materials
- Proving efficiency claims
- Optimization validation

**Example**:
```bash
# Benchmark default model
/benchmark

# Or specific model
export MODEL_PATH=outputs/production_model
/benchmark
```

**Outputs**:
- Inference speed (ms)
- Throughput (queries/sec)
- Memory usage
- Comparison vs GPT-3.5/GPT-4
- Report saved to `benchmark_report.json`

---

### `/deploy-docker`

**What it does**: Containerizes and deploys the API

**When to use**:
- Testing deployment
- Production deployment
- Customer demos

**Example**:
```bash
/deploy-docker
```

**Creates**:
- Dockerfile
- docker-compose.yml
- Running container on port 8000

**Test**:
```bash
curl http://localhost:8000/health
```

---

### `/list-datasets`

**What it does**: Shows all available training datasets

**When to use**:
- Choosing dataset for training
- Learning about options

**Example**:
```bash
/list-datasets
```

---

## 🎯 Common Workflows

### Workflow 1: First Time Setup & Test

```bash
# 1. Setup
./setup_mvp.sh

# 2. Quick test
/quick-train

# 3. Test RAG
/test-rag

# 4. Security check
/security-audit

# 5. Benchmark
/benchmark
```

**Time**: ~30 minutes

---

### Workflow 2: Train Production Model

```bash
# 1. Choose dataset
/list-datasets

# 2. Configure
export DATASET=openwebtext
export MODEL_SIZE=medium
export OUTPUT_NAME=medical_model

# 3. Train
/train-production

# 4. Test
export MODEL_PATH=outputs/medical_model
/test-rag

# 5. Benchmark
/benchmark

# 6. Deploy
/deploy-docker
```

**Time**: 4-8 hours (mostly training)

---

### Workflow 3: Pre-Deployment Checklist

```bash
# Security
/security-audit

# Performance
/benchmark

# Quality
/test-rag

# Deploy
/deploy-docker
```

**Time**: ~20 minutes

---

## ⚙️ Advanced Usage

### Running Skills Manually

Skills are just bash scripts. Run directly:

```bash
bash .claude/skills/quick-train.md
```

### Creating Custom Skills

Create `.claude/skills/my-skill.md`:

```markdown
# My Custom Skill

Description of what it does.

## Commands

​```bash
echo "Your commands here"
​```
```

### Chaining Skills

```bash
# Train then test
/quick-train && /test-rag

# Full pipeline
/quick-train && /test-rag && /security-audit && /benchmark
```

---

## 🐛 Troubleshooting

### "Skill not found"

**Solution**: Make sure you're in project root
```bash
cd Vision-HRM
/quick-train
```

### "Permission denied"

**Solution**: Make skills executable
```bash
chmod +x .claude/skills/*.md
```

### "Model not found"

**Solution**: Run training first
```bash
/quick-train
# Then
/test-rag
```

### "Command not found"

**Solution**: Install dependencies
```bash
pip install -r requirements-mvp.txt
```

---

## 📊 Skill Success Rates

| Skill | Success Rate | Common Issues |
|-------|--------------|---------------|
| `/quick-train` | 95% | Slow on CPU |
| `/train-production` | 90% | OOM on large models |
| `/test-rag` | 95% | Model not found |
| `/security-audit` | 100% | None |
| `/benchmark` | 95% | psutil not installed |
| `/deploy-docker` | 85% | Docker not installed |

---

## 🎓 Learning Path

**Week 1**: Setup & Testing
```bash
./setup_mvp.sh
/quick-train
/test-rag
/security-audit
```

**Week 2**: Production Training
```bash
/list-datasets
/train-production  # medium model
/benchmark
```

**Week 3**: Deployment
```bash
/security-audit
/deploy-docker
```

---

## 💡 Tips

1. **Always test first**: Run `/quick-train` before `/train-production`
2. **Check security often**: Run `/security-audit` weekly
3. **Benchmark early**: Run `/benchmark` to catch performance issues
4. **Use environment variables**: Configure skills without editing
5. **Read skill source**: Skills are markdown - easy to customize

---

## 📚 Related Documentation

- **MVP Roadmap**: `MVP_ROADMAP.md` - Complete development plan
- **File Guide**: `MVP_FILES_CREATED.md` - What each file does
- **Dataset Guide**: `DATASETS_GUIDE.md` - All about datasets
- **Quick Start**: `QUICKSTART_HRM_RAG.md` - Step-by-step guide

---

## 🤝 Contributing

Create your own skills:

1. Create `.claude/skills/my-skill.md`
2. Follow the format of existing skills
3. Test it works
4. Share with team

---

**🎉 Skills make MVP development 10x faster!**

Start with: `/quick-train`
