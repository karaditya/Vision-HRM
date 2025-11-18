"""
Inference API for HRM RAG System
Provides both REST API and CLI interface for document querying
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict
import torch

from rag_system import create_rag_system


class RAGInterface:
    """Interactive interface for RAG system"""

    def __init__(
        self,
        model_checkpoint: str,
        tokenizer_path: str,
        rag_save_dir: str = 'rag_data',
        device: str = None,
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Initializing RAG system on {device}...")

        # Create RAG system
        self.rag = create_rag_system(model_checkpoint, tokenizer_path, device)

        # Load existing database
        self.rag_save_dir = rag_save_dir
        if Path(rag_save_dir).exists():
            self.rag.load(rag_save_dir)

        print("RAG system ready!")
        stats = self.rag.get_stats()
        print(f"Knowledge base: {stats['num_chunks']} chunks")

    def add_documents(self, file_paths: List[str]):
        """Add multiple documents to RAG system"""
        for file_path in file_paths:
            if not Path(file_path).exists():
                print(f"Warning: File not found: {file_path}")
                continue

            try:
                self.rag.add_document(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Save updated database
        self.rag.save(self.rag_save_dir)

    def query(
        self,
        question: str,
        top_k: int = 3,
        max_tokens: int = 150,
        temperature: float = 0.7,
    ) -> Dict:
        """Query the RAG system"""
        result = self.rag.query(
            question,
            top_k=top_k,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        return result

    def interactive_mode(self):
        """Run interactive CLI mode"""
        print("\n" + "="*60)
        print("HRM RAG System - Interactive Mode")
        print("="*60)
        print("Commands:")
        print("  /add <file_path>  - Add document to knowledge base")
        print("  /stats            - Show system statistics")
        print("  /quit or /exit    - Exit interactive mode")
        print("  <question>        - Ask a question")
        print("="*60 + "\n")

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    parts = user_input.split(maxsplit=1)
                    command = parts[0].lower()

                    if command in ['/quit', '/exit']:
                        print("Goodbye!")
                        break

                    elif command == '/stats':
                        stats = self.rag.get_stats()
                        print(f"\nSystem Statistics:")
                        print(f"  Knowledge base chunks: {stats['num_chunks']}")
                        print(f"  Embedding dimension: {stats['embedding_dim']}")
                        print(f"  Model parameters: {stats['model_params']:,}")

                    elif command == '/add':
                        if len(parts) < 2:
                            print("Usage: /add <file_path>")
                        else:
                            file_path = parts[1]
                            self.add_documents([file_path])
                            print(f"Added document: {file_path}")

                    else:
                        print(f"Unknown command: {command}")

                else:
                    # Process as question
                    result = self.query(user_input)

                    print(f"\nAssistant: {result['answer']}")
                    print(f"\n[Confidence: {result['confidence']:.2%} | Sources: {result['num_chunks_used']}]")

                    # Optionally show sources
                    if result['sources']:
                        show_sources = input("\nShow sources? (y/n): ").lower()
                        if show_sources == 'y':
                            print("\nSources:")
                            for i, source in enumerate(result['sources'], 1):
                                print(f"\n[{i}] Score: {source['score']:.3f}")
                                print(f"    {source['text'][:200]}...")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
            except Exception as e:
                print(f"Error: {e}")


def main_cli():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="HRM RAG System - Inference Interface")

    # Model config
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to tokenizer')
    parser.add_argument('--rag_dir', type=str, default='rag_data',
                       help='Directory for RAG database')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')

    # Actions
    parser.add_argument('--add_documents', type=str, nargs='+',
                       help='Add documents to knowledge base')
    parser.add_argument('--query', type=str,
                       help='Single query to run')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive mode')

    # Query parameters
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of chunks to retrieve')
    parser.add_argument('--max_tokens', type=int, default=150,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')

    args = parser.parse_args()

    # Create interface
    interface = RAGInterface(
        model_checkpoint=args.model_checkpoint,
        tokenizer_path=args.tokenizer,
        rag_save_dir=args.rag_dir,
        device=args.device,
    )

    # Add documents
    if args.add_documents:
        interface.add_documents(args.add_documents)

    # Run query
    if args.query:
        result = interface.query(
            args.query,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        print(f"\nQuestion: {args.query}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nConfidence: {result['confidence']:.2%}")
        print(f"Sources used: {result['num_chunks_used']}")

        if result['sources']:
            print("\nTop sources:")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"\n[{i}] Score: {source['score']:.3f}")
                print(f"    {source['text'][:150]}...")

    # Interactive mode
    if args.interactive:
        interface.interactive_mode()


# REST API (optional - requires Flask)
def create_rest_api(interface: RAGInterface):
    """Create REST API for RAG system (requires Flask)"""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("Flask not installed. Install with: pip install flask flask-cors")
        return None

    app = Flask(__name__)
    CORS(app)

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy'})

    @app.route('/stats', methods=['GET'])
    def stats():
        return jsonify(interface.rag.get_stats())

    @app.route('/add_document', methods=['POST'])
    def add_document():
        data = request.json
        file_path = data.get('file_path')

        if not file_path:
            return jsonify({'error': 'file_path required'}), 400

        try:
            interface.add_documents([file_path])
            return jsonify({'success': True, 'message': f'Added {file_path}'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/query', methods=['POST'])
    def query():
        data = request.json
        question = data.get('question')

        if not question:
            return jsonify({'error': 'question required'}), 400

        try:
            result = interface.query(
                question,
                top_k=data.get('top_k', 3),
                max_tokens=data.get('max_tokens', 150),
                temperature=data.get('temperature', 0.7),
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app


def main_api():
    """REST API entry point"""
    parser = argparse.ArgumentParser(description="HRM RAG System - REST API")

    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--rag_dir', type=str, default='rag_data')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)

    args = parser.parse_args()

    # Create interface
    interface = RAGInterface(
        model_checkpoint=args.model_checkpoint,
        tokenizer_path=args.tokenizer,
        rag_save_dir=args.rag_dir,
    )

    # Create and run API
    app = create_rest_api(interface)
    if app:
        print(f"\nStarting REST API on {args.host}:{args.port}")
        print(f"Endpoints:")
        print(f"  GET  /health        - Health check")
        print(f"  GET  /stats         - System statistics")
        print(f"  POST /add_document  - Add document")
        print(f"  POST /query         - Query RAG system")
        print()

        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main_cli()
