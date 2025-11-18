"""
Dataset Loader for Standard Pretraining Datasets
Supports HuggingFace datasets, The Pile, C4, OpenWebText, Wikipedia, etc.
"""

import os
import json
from pathlib import Path
from typing import Iterator, List, Dict, Optional, Union
import torch
from torch.utils.data import IterableDataset, Dataset


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large-scale text corpora
    Supports HuggingFace datasets, JSONL, and text files
    """

    def __init__(
        self,
        data_source: Union[str, List[str]],
        tokenizer,
        max_seq_len: int = 512,
        dataset_type: str = "auto",  # auto, huggingface, jsonl, text
        text_field: str = "text",  # Field name for text in JSONL/HF datasets
        streaming: bool = True,
        split: str = "train",
    ):
        """
        Args:
            data_source: Path to data or HuggingFace dataset name
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            dataset_type: Type of dataset (auto-detected if 'auto')
            text_field: Field name containing text in structured data
            streaming: Whether to stream data (recommended for large datasets)
            split: Dataset split (train/validation/test)
        """
        super().__init__()
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_field = text_field
        self.streaming = streaming
        self.split = split

        # Auto-detect dataset type
        if dataset_type == "auto":
            self.dataset_type = self._detect_dataset_type(data_source)
        else:
            self.dataset_type = dataset_type

        print(f"Initialized {self.dataset_type} dataset: {data_source}")

    def _detect_dataset_type(self, source: str) -> str:
        """Auto-detect dataset type from source"""
        if isinstance(source, str):
            path = Path(source)

            if not path.exists():
                # Assume it's a HuggingFace dataset name
                return "huggingface"
            elif path.is_file():
                if path.suffix == '.jsonl':
                    return "jsonl"
                elif path.suffix in ['.txt', '.text']:
                    return "text"
                elif path.suffix == '.parquet':
                    return "parquet"
            elif path.is_dir():
                # Check for common dataset structures
                if list(path.glob("*.jsonl")):
                    return "jsonl"
                elif list(path.glob("*.txt")):
                    return "text"
                elif list(path.glob("*.parquet")):
                    return "parquet"

        return "text"  # Default fallback

    def _load_huggingface_dataset(self):
        """Load HuggingFace dataset"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets not installed. "
                "Install with: pip install datasets"
            )

        print(f"Loading HuggingFace dataset: {self.data_source}")

        # Load dataset with streaming if enabled
        dataset = load_dataset(
            self.data_source,
            split=self.split,
            streaming=self.streaming,
            trust_remote_code=True,
        )

        # Iterate over dataset
        for example in dataset:
            # Extract text from the specified field
            if isinstance(example, dict):
                text = example.get(self.text_field, "")
                if not text and "content" in example:
                    text = example["content"]
                elif not text and "article" in example:
                    text = example["article"]
            else:
                text = str(example)

            if text and len(text.strip()) > 0:
                yield text

    def _load_jsonl_dataset(self):
        """Load JSONL dataset"""
        paths = []

        if isinstance(self.data_source, str):
            path = Path(self.data_source)
            if path.is_file():
                paths = [path]
            elif path.is_dir():
                paths = sorted(path.glob("*.jsonl"))
        else:
            paths = [Path(p) for p in self.data_source]

        print(f"Loading {len(paths)} JSONL file(s)")

        for file_path in paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        text = data.get(self.text_field, "")

                        # Try alternative field names
                        if not text:
                            for field in ["content", "article", "body", "document"]:
                                if field in data:
                                    text = data[field]
                                    break

                        if text and len(text.strip()) > 0:
                            yield text
                    except json.JSONDecodeError:
                        continue

    def _load_text_dataset(self):
        """Load plain text dataset"""
        paths = []

        if isinstance(self.data_source, str):
            path = Path(self.data_source)
            if path.is_file():
                paths = [path]
            elif path.is_dir():
                paths = sorted(path.glob("*.txt"))
        else:
            paths = [Path(p) for p in self.data_source]

        print(f"Loading {len(paths)} text file(s)")

        for file_path in paths:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read in chunks or by paragraphs
                content = f.read()

                # Split by double newlines (paragraphs)
                paragraphs = content.split('\n\n')

                for para in paragraphs:
                    para = para.strip()
                    if para and len(para) > 50:  # Skip very short paragraphs
                        yield para

    def _load_parquet_dataset(self):
        """Load Parquet dataset"""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "PyArrow not installed. Install with: pip install pyarrow"
            )

        paths = []
        if isinstance(self.data_source, str):
            path = Path(self.data_source)
            if path.is_file():
                paths = [path]
            elif path.is_dir():
                paths = sorted(path.glob("*.parquet"))
        else:
            paths = [Path(p) for p in self.data_source]

        print(f"Loading {len(paths)} Parquet file(s)")

        for file_path in paths:
            table = pq.read_table(str(file_path))
            df = table.to_pandas()

            if self.text_field in df.columns:
                for text in df[self.text_field]:
                    if text and len(str(text).strip()) > 0:
                        yield str(text)

    def __iter__(self):
        """Iterate over dataset and yield tokenized examples"""
        # Select appropriate loader
        if self.dataset_type == "huggingface":
            text_iterator = self._load_huggingface_dataset()
        elif self.dataset_type == "jsonl":
            text_iterator = self._load_jsonl_dataset()
        elif self.dataset_type == "parquet":
            text_iterator = self._load_parquet_dataset()
        else:  # text
            text_iterator = self._load_text_dataset()

        # Tokenize and yield examples
        for text in text_iterator:
            # Tokenize
            tokens = self.tokenizer.encode(text, max_length=self.max_seq_len)

            # Skip if too short
            if len(tokens) < 2:
                continue

            # Truncate to max length
            tokens = tokens[:self.max_seq_len]

            # Convert to tensor
            input_ids = torch.tensor(tokens, dtype=torch.long)
            labels = input_ids.clone()

            yield {
                'input_ids': input_ids,
                'labels': labels,
            }


class PretrainingDatasetRegistry:
    """Registry of common pretraining datasets with download instructions"""

    DATASETS = {
        # General text corpora
        "openwebtext": {
            "hf_name": "openwebtext",
            "text_field": "text",
            "description": "40GB of web text from Reddit URLs",
            "size": "40GB",
        },
        "c4": {
            "hf_name": "c4",
            "text_field": "text",
            "description": "Colossal Clean Crawled Corpus (750GB)",
            "size": "750GB",
            "config": "en",
        },
        "the_pile": {
            "hf_name": "EleutherAI/pile",
            "text_field": "text",
            "description": "825GB diverse text dataset",
            "size": "825GB",
        },
        "wikipedia": {
            "hf_name": "wikipedia",
            "text_field": "text",
            "description": "Wikipedia articles",
            "size": "20GB",
            "config": "20220301.en",
        },
        "bookcorpus": {
            "hf_name": "bookcorpus",
            "text_field": "text",
            "description": "Books dataset",
            "size": "5GB",
        },

        # Code datasets
        "the_stack": {
            "hf_name": "bigcode/the-stack",
            "text_field": "content",
            "description": "3TB of source code",
            "size": "3TB",
        },
        "codeparrot": {
            "hf_name": "codeparrot/github-code",
            "text_field": "code",
            "description": "GitHub code dataset",
            "size": "800GB",
        },

        # Smaller datasets for testing
        "tiny_shakespeare": {
            "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            "description": "1MB Shakespeare text (for testing)",
            "size": "1MB",
        },
        "wikitext": {
            "hf_name": "wikitext",
            "text_field": "text",
            "description": "Quality Wikipedia text (100MB-500MB)",
            "size": "500MB",
            "config": "wikitext-103-v1",
        },
    }

    @classmethod
    def list_datasets(cls):
        """List all available datasets"""
        print("\n" + "="*70)
        print("Available Pretraining Datasets")
        print("="*70)

        for name, info in cls.DATASETS.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']}")
            if 'hf_name' in info:
                print(f"  HuggingFace: {info['hf_name']}")
            if 'url' in info:
                print(f"  URL: {info['url']}")

        print("\n" + "="*70)

    @classmethod
    def get_dataset_info(cls, name: str) -> Dict:
        """Get information about a specific dataset"""
        return cls.DATASETS.get(name, {})

    @classmethod
    def create_dataset(
        cls,
        name: str,
        tokenizer,
        max_seq_len: int = 512,
        streaming: bool = True,
        split: str = "train",
    ):
        """Create a dataset from the registry"""
        if name not in cls.DATASETS:
            raise ValueError(
                f"Dataset '{name}' not found. "
                f"Available: {', '.join(cls.DATASETS.keys())}"
            )

        info = cls.DATASETS[name]

        # Handle direct URLs
        if 'url' in info:
            # Download if needed
            import urllib.request
            filepath = Path(f"data/{name}.txt")
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if not filepath.exists():
                print(f"Downloading {name}...")
                urllib.request.urlretrieve(info['url'], filepath)

            return StreamingTextDataset(
                str(filepath),
                tokenizer,
                max_seq_len,
                dataset_type="text",
                streaming=False,
            )

        # Handle HuggingFace datasets
        hf_name = info.get('hf_name')
        config = info.get('config')

        # Adjust HF name with config if needed
        if config:
            hf_name = f"{hf_name}"  # Config handled in load_dataset

        return StreamingTextDataset(
            hf_name,
            tokenizer,
            max_seq_len,
            dataset_type="huggingface",
            text_field=info.get('text_field', 'text'),
            streaming=streaming,
            split=split,
        )


def download_dataset_cli():
    """CLI tool to download and prepare datasets"""
    import argparse

    parser = argparse.ArgumentParser(description="Download pretraining datasets")
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--dataset', type=str, help='Dataset name to download')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')

    args = parser.parse_args()

    if args.list:
        PretrainingDatasetRegistry.list_datasets()
    elif args.dataset:
        print(f"Downloading {args.dataset}...")
        info = PretrainingDatasetRegistry.get_dataset_info(args.dataset)

        if 'url' in info:
            import urllib.request
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f"{args.dataset}.txt"

            print(f"Downloading to {filepath}...")
            urllib.request.urlretrieve(info['url'], filepath)
            print(f"✓ Downloaded: {filepath}")
        elif 'hf_name' in info:
            print(f"HuggingFace dataset: {info['hf_name']}")
            print(f"Use in training with: --dataset {args.dataset}")
            print(f"\nOr download manually:")
            print(f"  from datasets import load_dataset")
            print(f"  dataset = load_dataset('{info['hf_name']}')")
        else:
            print(f"Unknown dataset source for {args.dataset}")
    else:
        parser.print_help()


if __name__ == "__main__":
    # Example usage
    print("Dataset Loader Examples\n")

    # List all datasets
    PretrainingDatasetRegistry.list_datasets()

    # Example: Load tiny shakespeare
    print("\n" + "="*70)
    print("Example: Loading tiny_shakespeare dataset")
    print("="*70)

    from train_language_model import SimpleTokenizer

    # Create simple tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000, level='char')
    tokenizer.build_vocab(["This is a test sentence."])

    # Create dataset
    try:
        dataset = PretrainingDatasetRegistry.create_dataset(
            'tiny_shakespeare',
            tokenizer,
            max_seq_len=256,
            streaming=False,
        )

        print("Dataset created successfully!")
        print("Iterating over first 3 examples...")

        for i, example in enumerate(dataset):
            if i >= 3:
                break
            print(f"\nExample {i+1}:")
            print(f"  Input shape: {example['input_ids'].shape}")
            print(f"  First 50 tokens: {example['input_ids'][:50].tolist()}")

    except Exception as e:
        print(f"Error: {e}")
        print("This is expected if dependencies are not installed.")
