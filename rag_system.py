"""
RAG (Retrieval-Augmented Generation) System
Uses HRM Language Model for document-based question answering
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import pickle
from dataclasses import dataclass
import re

from models.hrm.hrm_language_v1 import HRMLanguageModel, HRMLanguageConfig


@dataclass
class DocumentChunk:
    """Represents a chunk of a document"""
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = None
    chunk_id: int = 0
    source: str = ""


class SimpleEmbedder:
    """
    Simple embedding model using averaged token embeddings from HRM model
    For production, use sentence-transformers or similar
    """

    def __init__(self, model: HRMLanguageModel, tokenizer, device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_dim = model.config.hidden_size

    @torch.no_grad()
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts

        Args:
            texts: List of text strings

        Returns:
            embeddings: (num_texts, embedding_dim) numpy array
        """
        self.model.eval()
        embeddings = []

        for text in texts:
            # Tokenize
            tokens = self.tokenizer.encode(text, max_length=512)
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

            # Get hidden states
            outputs = self.model(input_ids)
            hidden_states = outputs['hidden_states']  # (1, seq_len, hidden_dim)

            # Mean pooling (excluding padding)
            mask = (input_ids != self.tokenizer.pad_token_id).float().unsqueeze(-1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)  # (1, hidden_dim)

            embeddings.append(pooled.cpu().numpy()[0])

        return np.array(embeddings)


class VectorDatabase:
    """Simple in-memory vector database for similarity search"""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks to database"""
        self.chunks.extend(chunks)

        # Stack embeddings
        chunk_embeddings = np.array([chunk.embedding for chunk in chunks])

        if self.embeddings is None:
            self.embeddings = chunk_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, chunk_embeddings])

        print(f"Added {len(chunks)} chunks. Total: {len(self.chunks)}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for most similar chunks using cosine similarity

        Args:
            query_embedding: (embedding_dim,) query vector
            top_k: Number of results to return

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        db_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity
        similarities = np.dot(db_norms, query_norm)

        # Get top-k
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = [(self.chunks[idx], float(similarities[idx])) for idx in top_indices]
        return results

    def save(self, path: str):
        """Save database to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'embedding_dim': self.embedding_dim,
            }, f)

    @classmethod
    def load(cls, path: str):
        """Load database from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        db = cls(embedding_dim=data['embedding_dim'])
        db.chunks = data['chunks']
        db.embeddings = data['embeddings']
        return db


class DocumentProcessor:
    """Process documents into chunks for RAG"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self, file_path: str) -> str:
        """Load document from file"""
        path = Path(file_path)

        if path.suffix == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        elif path.suffix == '.pdf':
            # For PDF support, would need PyPDF2 or pdfplumber
            raise NotImplementedError("PDF support requires additional libraries (PyPDF2)")
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def chunk_text(self, text: str, source: str = "") -> List[DocumentChunk]:
        """
        Split text into overlapping chunks

        Args:
            text: Input text
            source: Source identifier (filename, URL, etc.)

        Returns:
            List of DocumentChunk objects
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()

        # Split into sentences (simple split on periods)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source=source,
                    metadata={'num_words': len(chunk_text.split())}
                ))
                chunk_id += 1

                # Start new chunk with overlap
                overlap_words = int(self.chunk_overlap)
                overlap_text = ' '.join(chunk_text.split()[-overlap_words:])
                current_chunk = [overlap_text, sentence]
                current_length = len(overlap_text.split()) + sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(DocumentChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source=source,
                metadata={'num_words': len(chunk_text.split())}
            ))

        return chunks


class RAGSystem:
    """
    Complete RAG system with document processing, retrieval, and generation
    """

    def __init__(
        self,
        model: HRMLanguageModel,
        tokenizer,
        device: str = 'cuda',
        chunk_size: int = 256,
        chunk_overlap: int = 64,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Components
        self.embedder = SimpleEmbedder(model, tokenizer, device)
        self.vector_db = VectorDatabase(embedding_dim=model.config.hidden_size)
        self.doc_processor = DocumentProcessor(chunk_size, chunk_overlap)

    def add_document(self, file_path: str):
        """
        Add a document to the RAG system

        Args:
            file_path: Path to document file
        """
        print(f"Processing document: {file_path}")

        # Load and chunk document
        text = self.doc_processor.load_document(file_path)
        chunks = self.doc_processor.chunk_text(text, source=file_path)

        print(f"Created {len(chunks)} chunks")

        # Embed chunks
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed(chunk_texts)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        # Add to vector database
        self.vector_db.add_chunks(chunks)

        print(f"Added document to RAG system")

    def query(
        self,
        question: str,
        top_k: int = 3,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        show_sources: bool = True,
    ) -> Dict[str, any]:
        """
        Query the RAG system

        Args:
            question: User question
            top_k: Number of relevant chunks to retrieve
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            show_sources: Whether to return source chunks

        Returns:
            Dictionary with 'answer', 'sources', 'confidence'
        """
        print(f"\nQuery: {question}")

        # 1. Embed query
        query_embedding = self.embedder.embed([question])[0]

        # 2. Retrieve relevant chunks
        results = self.vector_db.search(query_embedding, top_k=top_k)

        if not results:
            return {
                'answer': "No relevant information found in the knowledge base.",
                'sources': [],
                'confidence': 0.0,
            }

        # 3. Build context from retrieved chunks
        context_parts = []
        sources = []

        for i, (chunk, score) in enumerate(results):
            context_parts.append(f"[{i+1}] {chunk.text}")
            sources.append({
                'text': chunk.text,
                'score': score,
                'source': chunk.source,
                'chunk_id': chunk.chunk_id,
            })

        context = "\n\n".join(context_parts)

        # 4. Create prompt
        prompt = self._create_prompt(context, question)

        print(f"Retrieved {len(results)} relevant chunks")
        print(f"Context length: {len(prompt.split())} words")

        # 5. Generate answer
        answer = self._generate_answer(prompt, max_new_tokens, temperature)

        # Calculate average confidence from retrieval scores
        avg_confidence = np.mean([score for _, score in results])

        return {
            'answer': answer,
            'sources': sources if show_sources else [],
            'confidence': float(avg_confidence),
            'num_chunks_used': len(results),
        }

    def _create_prompt(self, context: str, question: str) -> str:
        """Create RAG prompt"""
        prompt = f"""Context information is below:
---
{context}
---

Given the context information above, answer the following question.
If the answer is not in the context, say "I don't have enough information to answer this question."

Question: {question}

Answer:"""
        return prompt

    def _generate_answer(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate answer using HRM model"""
        # Tokenize prompt
        input_tokens = self.tokenizer.encode(prompt, max_length=1536)
        input_ids = torch.tensor([input_tokens], dtype=torch.long).to(self.device)

        # Generate
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
        )

        # Decode (only the generated part)
        generated_tokens = generated_ids[0][len(input_tokens):].tolist()
        answer = self.tokenizer.decode(generated_tokens)

        # Clean up answer
        answer = answer.strip()

        # Stop at first newline or end marker
        for stop_seq in ['\n\n', 'Question:', 'Context:']:
            if stop_seq in answer:
                answer = answer.split(stop_seq)[0].strip()

        return answer

    def save(self, save_dir: str):
        """Save RAG system state"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save vector database
        self.vector_db.save(str(save_path / 'vector_db.pkl'))

        print(f"Saved RAG system to {save_dir}")

    def load(self, save_dir: str):
        """Load RAG system state"""
        save_path = Path(save_dir)

        # Load vector database
        db_path = save_path / 'vector_db.pkl'
        if db_path.exists():
            self.vector_db = VectorDatabase.load(str(db_path))
            print(f"Loaded RAG system from {save_dir}")
            print(f"Knowledge base contains {len(self.vector_db.chunks)} chunks")
        else:
            print(f"No saved database found at {save_dir}")

    def get_stats(self) -> Dict:
        """Get RAG system statistics"""
        return {
            'num_chunks': len(self.vector_db.chunks),
            'embedding_dim': self.vector_db.embedding_dim,
            'model_params': self.model.get_num_params(),
        }


def create_rag_system(
    model_checkpoint: str,
    tokenizer_path: str,
    device: str = 'cuda',
) -> RAGSystem:
    """
    Factory function to create RAG system from trained model

    Args:
        model_checkpoint: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        device: Device to run on

    Returns:
        RAGSystem instance
    """
    # Load tokenizer
    from train_language_model import SimpleTokenizer
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    # Load model
    checkpoint = torch.load(model_checkpoint, map_location=device)
    config = checkpoint['config']

    model = HRMLanguageModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from {model_checkpoint}")
    print(f"Model has {model.get_num_params():,} parameters")

    # Create RAG system
    rag = RAGSystem(model, tokenizer, device=device)

    return rag


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--document', type=str, help='Document to add')
    parser.add_argument('--query', type=str, help='Query to run')
    parser.add_argument('--save_dir', type=str, default='rag_data')

    args = parser.parse_args()

    # Create RAG system
    rag = create_rag_system(
        args.model_checkpoint,
        args.tokenizer,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Load existing database if available
    if Path(args.save_dir).exists():
        rag.load(args.save_dir)

    # Add document
    if args.document:
        rag.add_document(args.document)
        rag.save(args.save_dir)

    # Run query
    if args.query:
        result = rag.query(args.query)
        print(f"\nAnswer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"\nSources used: {result['num_chunks_used']}")
