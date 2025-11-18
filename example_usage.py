"""
Complete Example: Training HRM Language Model and Using it for RAG
This script demonstrates the end-to-end workflow
"""

import os
import sys
from pathlib import Path
import torch

print("="*70)
print("HRM Language Model + RAG System - Complete Example")
print("="*70)

# ============================================================================
# STEP 1: Prepare Sample Training Data
# ============================================================================
print("\n[STEP 1] Preparing sample training data...")

sample_text = """
Machine learning is a subset of artificial intelligence that focuses on
developing algorithms and statistical models that enable computers to learn
from and make predictions or decisions based on data.

Neural networks are computing systems inspired by biological neural networks.
They consist of interconnected nodes called neurons that process information
in layers. Deep learning uses neural networks with multiple layers.

The Hierarchical Reasoning Model (HRM) is a novel architecture that mimics
human cognitive processes. It uses hierarchical processing with high-level
abstract reasoning and low-level detailed computations.

Training neural networks requires large datasets, computational resources,
and careful optimization. Common techniques include backpropagation, gradient
descent, and various regularization methods to prevent overfitting.

Retrieval-Augmented Generation (RAG) combines information retrieval with
language generation. It retrieves relevant documents and uses them as context
to generate more accurate and grounded responses.
"""

# Create sample data directory
data_dir = Path("sample_data")
data_dir.mkdir(exist_ok=True)

train_file = data_dir / "train.txt"
with open(train_file, 'w') as f:
    # Repeat the text to have more training data
    for _ in range(50):
        f.write(sample_text + "\n\n")

print(f"✓ Created training data: {train_file}")
print(f"  Size: {train_file.stat().st_size / 1024:.1f} KB")

# ============================================================================
# STEP 2: Train HRM Language Model
# ============================================================================
print("\n[STEP 2] Training HRM Language Model...")
print("Note: Using small model for demonstration. For production, use larger config.")

training_command = f"""
python train_language_model.py \\
    --data_path {train_file} \\
    --output_dir outputs/hrm_lm_demo \\
    --vocab_size 500 \\
    --tokenizer_level char \\
    --max_seq_len 256 \\
    --hidden_size 128 \\
    --num_heads 4 \\
    --num_layers 3 \\
    --h_cycles 2 \\
    --l_cycles 3 \\
    --batch_size 4 \\
    --num_epochs 5 \\
    --learning_rate 0.001 \\
    --num_workers 0
"""

print("\nTraining command:")
print(training_command)

print("\n" + "-"*70)
print("To run training, execute the above command.")
print("For this demo, we'll simulate having a trained model.")
print("-"*70)

# Check if model exists
model_checkpoint = Path("outputs/hrm_lm_demo/best_model.pt")
tokenizer_path = Path("outputs/hrm_lm_demo/tokenizer.json")

if model_checkpoint.exists() and tokenizer_path.exists():
    print(f"\n✓ Found trained model at: {model_checkpoint}")
else:
    print(f"\n⚠ Model not found. Please run the training command above first.")
    print("\nTo quickly test with a small model, run:")
    print("  python train_language_model.py --data_path sample_data/train.txt \\")
    print("         --output_dir outputs/hrm_lm_demo --num_epochs 5 --batch_size 4 \\")
    print("         --hidden_size 128 --num_layers 3")
    sys.exit(0)

# ============================================================================
# STEP 3: Prepare Documents for RAG
# ============================================================================
print("\n[STEP 3] Preparing documents for RAG knowledge base...")

# Create sample documents
docs_dir = Path("sample_documents")
docs_dir.mkdir(exist_ok=True)

doc1 = docs_dir / "ml_basics.txt"
with open(doc1, 'w') as f:
    f.write("""
Machine Learning Fundamentals

Machine learning is a field of study that gives computers the ability to learn
without being explicitly programmed. It focuses on developing algorithms that
can learn from and make predictions on data.

Key Concepts:
- Supervised Learning: Learning from labeled data
- Unsupervised Learning: Finding patterns in unlabeled data
- Reinforcement Learning: Learning through trial and error

Common algorithms include linear regression, decision trees, support vector
machines, and neural networks. The choice of algorithm depends on the problem
type, data characteristics, and computational resources available.
""")

doc2 = docs_dir / "neural_networks.txt"
with open(doc2, 'w') as f:
    f.write("""
Neural Networks and Deep Learning

Neural networks are computational models inspired by the human brain. They
consist of layers of interconnected nodes (neurons) that process information.

Architecture:
- Input Layer: Receives the raw data
- Hidden Layers: Process and transform the data
- Output Layer: Produces the final prediction

Deep learning refers to neural networks with many hidden layers. These deep
architectures can learn hierarchical representations of data, making them
powerful for tasks like image recognition, natural language processing, and
speech recognition.

Training involves backpropagation and gradient descent to adjust the weights
of connections between neurons to minimize prediction error.
""")

doc3 = docs_dir / "rag_systems.txt"
with open(doc3, 'w') as f:
    f.write("""
Retrieval-Augmented Generation (RAG)

RAG is a technique that enhances language models by combining them with
information retrieval systems. Instead of relying solely on the model's
parametric knowledge, RAG retrieves relevant documents and uses them as
context for generation.

How RAG Works:
1. Query Processing: User question is embedded into a vector
2. Retrieval: Most similar documents are found using vector search
3. Context Building: Retrieved documents are formatted as context
4. Generation: Language model generates answer based on context

Benefits:
- More accurate and factual responses
- Ability to cite sources
- Easy to update knowledge without retraining
- Reduced hallucinations

RAG systems are widely used in question answering, document search, and
knowledge base applications.
""")

print(f"✓ Created {len(list(docs_dir.glob('*.txt')))} sample documents:")
for doc in docs_dir.glob('*.txt'):
    print(f"  - {doc.name}")

# ============================================================================
# STEP 4: Build RAG Knowledge Base
# ============================================================================
print("\n[STEP 4] Building RAG knowledge base...")

from rag_system import create_rag_system

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create RAG system
rag = create_rag_system(
    model_checkpoint=str(model_checkpoint),
    tokenizer_path=str(tokenizer_path),
    device=device,
)

# Add documents
print("\nAdding documents to RAG system...")
for doc_path in docs_dir.glob('*.txt'):
    try:
        rag.add_document(str(doc_path))
        print(f"✓ Added: {doc_path.name}")
    except Exception as e:
        print(f"✗ Error adding {doc_path.name}: {e}")

# Save RAG database
rag_dir = "rag_data_demo"
rag.save(rag_dir)
print(f"\n✓ Saved RAG database to: {rag_dir}")

# ============================================================================
# STEP 5: Query the RAG System
# ============================================================================
print("\n[STEP 5] Querying the RAG system...")
print("="*70)

# Example queries
queries = [
    "What is machine learning?",
    "How do neural networks work?",
    "What are the benefits of RAG systems?",
    "Explain supervised learning",
]

for i, question in enumerate(queries, 1):
    print(f"\n{i}. Question: {question}")
    print("-"*70)

    try:
        result = rag.query(
            question,
            top_k=2,
            max_new_tokens=100,
            temperature=0.7,
        )

        print(f"Answer: {result['answer']}")
        print(f"\nConfidence: {result['confidence']:.2%}")
        print(f"Chunks used: {result['num_chunks_used']}")

        # Show top source
        if result['sources']:
            top_source = result['sources'][0]
            print(f"\nTop source (score: {top_source['score']:.3f}):")
            print(f"  {top_source['text'][:150]}...")

    except Exception as e:
        print(f"Error: {e}")

    print("="*70)

# ============================================================================
# STEP 6: Interactive Mode
# ============================================================================
print("\n[STEP 6] Interactive Mode Available")
print("="*70)

print("\nTo start interactive mode, run:")
print(f"  python rag_inference_api.py \\")
print(f"    --model_checkpoint {model_checkpoint} \\")
print(f"    --tokenizer {tokenizer_path} \\")
print(f"    --rag_dir {rag_dir} \\")
print(f"    --interactive")

print("\nOr for REST API mode:")
print(f"  python rag_inference_api.py \\")
print(f"    --model_checkpoint {model_checkpoint} \\")
print(f"    --tokenizer {tokenizer_path} \\")
print(f"    --rag_dir {rag_dir}")
print("  # Then start API separately")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY - Complete RAG System Built Successfully!")
print("="*70)

stats = rag.get_stats()
print(f"\n✓ HRM Language Model trained and loaded")
print(f"  - Parameters: {stats['model_params']:,}")
print(f"  - Embedding dim: {stats['embedding_dim']}")

print(f"\n✓ RAG Knowledge Base created")
print(f"  - Documents: {len(list(docs_dir.glob('*.txt')))}")
print(f"  - Chunks: {stats['num_chunks']}")
print(f"  - Location: {rag_dir}/")

print(f"\n✓ System ready for queries")

print("\n" + "="*70)
print("Next Steps:")
print("="*70)
print("1. Add more documents: rag.add_document('your_file.txt')")
print("2. Query interactively: python rag_inference_api.py --interactive ...")
print("3. Build REST API: Use Flask endpoints in rag_inference_api.py")
print("4. Scale up: Train larger model with more data and bigger architecture")
print("5. Improve: Add better tokenization (BPE), embeddings, reranking")
print("="*70)
