"""
Language Model Dataset Loader for HRM

Simplified dataset loader specifically designed for causal language modeling tasks.
Compatible with the HRM training pipeline.
"""

import os
import json
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pydantic
import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata


class LMDatasetConfig(pydantic.BaseModel):
    """Configuration for language model dataset."""
    seed: int
    dataset_path: str
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead
    rank: int
    num_replicas: int


class LMDataset(IterableDataset):
    """
    Language Model Dataset for HRM training.

    Loads pre-tokenized text data and prepares it for autoregressive language modeling.
    The dataset is compatible with the HRM ACT (Adaptive Computation Time) framework.
    """

    def __init__(self, config: LMDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split
        self.metadata = self._load_metadata()

        # Validate batch size
        assert self.config.global_batch_size % self.config.num_replicas == 0, \
            f"Global batch size {self.config.global_batch_size} must be divisible by {self.config.num_replicas} replicas"

        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self) -> PuzzleDatasetMetadata:
        """Load dataset metadata."""
        metadata_path = os.path.join(self.config.dataset_path, self.split, "dataset.json")
        with open(metadata_path, "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        """Lazy load dataset arrays on first access."""
        if self._data is not None:
            return

        # Memory-mapped arrays for efficiency
        field_mmap_modes = {
            "inputs": "r",  # Memory-mapped read-only
            "labels": "r",
            "puzzle_identifiers": None,  # Keep in memory
            "puzzle_indices": None,
            "group_indices": None
        }

        self._data = {}
        for set_name in self.metadata.sets:
            self._data[set_name] = {
                field: np.load(
                    os.path.join(self.config.dataset_path, self.split, f"{set_name}__{field}.npy"),
                    mmap_mode=mmap_mode
                )
                for field, mmap_mode in field_mmap_modes.items()
            }

    def _collate_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Collate batch data into tensors."""
        # Convert to int32
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Handle ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad if necessary
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size

            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }

            batch = {
                k: np.pad(
                    v,
                    ((0, pad_size),) + ((0, 0),) * (v.ndim - 1),
                    constant_values=pad_values[k]
                )
                for k, v in batch.items()
            }

        # Convert to PyTorch tensors
        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _iter_test(self):
        """Iterate over test set (deterministic, no shuffling)."""
        for set_name, dataset in self._data.items():
            total_examples = len(dataset["inputs"])

            start_index = 0
            while start_index < total_examples:
                # Compute batch boundaries
                end_index = min(total_examples, start_index + self.config.global_batch_size)

                # Compute local boundaries for this rank
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)

                # Map example indices to puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1

                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and \
                            i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1
                    puzzle_indices.append(puzzle_index)

                # Collate batch
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start:local_end],
                    "labels": dataset["labels"][local_start:local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                })

                yield set_name, batch, end_index - start_index

                start_index += self.config.global_batch_size

    def _sample_batch(
        self,
        rng: np.random.Generator,
        group_order: np.ndarray,
        puzzle_indices: np.ndarray,
        group_indices: np.ndarray,
        start_index: int
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Sample a batch from the dataset."""
        batch = []
        batch_puzzle_indices = []
        current_size = 0

        while (start_index < group_order.size) and (current_size < self.config.global_batch_size):
            # Pick a group and sample a puzzle from it
            group_id = group_order[start_index]
            puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
            start_index += 1

            # Get puzzle range
            puzzle_start = puzzle_indices[puzzle_id]
            puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

            append_size = min(puzzle_size, self.config.global_batch_size - current_size)

            # Add to batch
            batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
            batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

            current_size += append_size

        return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)

    def _iter_train(self):
        """Iterate over training set with shuffling."""
        for set_name, dataset in self._data.items():
            self._iters += 1

            # Create RNG for reproducibility
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            # Shuffle groups across epochs
            group_order = np.concatenate([
                rng.permutation(dataset["group_indices"].size - 1)
                for _ in range(self.config.epochs_per_iter)
            ])

            start_index = 0
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = self._sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index
                )

                global_effective_batch_size = batch_puzzle_indices.size

                # Drop incomplete final batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                # Select data for current rank
                local_slice = slice(
                    self.config.rank * self.local_batch_size,
                    (self.config.rank + 1) * self.local_batch_size
                )
                batch_indices = batch_indices[local_slice]
                batch_puzzle_indices = batch_puzzle_indices[local_slice]

                # Collate batch
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                })

                yield set_name, batch, global_effective_batch_size

    def __iter__(self):
        """Iterate over dataset."""
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, \
            "Multi-worker data loading is not currently supported"

        self._lazy_load_dataset()

        # Choose iteration mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
