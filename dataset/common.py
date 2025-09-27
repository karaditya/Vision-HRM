from typing import List, Optional, Tuple

import pydantic
import numpy as np
import os
import json

import torch
from dataset.build_cifar_dataset import CIFARDatasetMetadata
from torch.utils.data import Dataset


# *****   FOR CIFAR VISION TASK *****

class PreprocessedCIFARDataset(Dataset):
    """
    A PyTorch Dataset that loads pre-processed data from .npy files.

    This is designed to be a fast and memory-efficient way to load data
    that has already been processed by a script like build_cifar_dataset.py.
    It uses numpy's memory-mapping to avoid loading the entire dataset into RAM.
    """
    def __init__(self, data_dir: str):
        """
        Initializes the dataset.

        Args:
            data_dir (str): The directory containing 'all__inputs.npy', 
                            'all__labels.npy', and 'dataset_metadata.json'.
        """
        self.data_dir = data_dir
        
        # Load metadata
        with open(os.path.join(self.data_dir, "dataset_metadata.json"), "r") as f:
            self.metadata = CIFARDatasetMetadata(**json.load(f))

        # Load the numpy arrays using memory-mapping for efficiency
        # This treats the file on disk like an in-memory array, saving RAM.
        self.inputs = np.load(os.path.join(self.data_dir, "all__inputs.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(self.data_dir, "all__labels.npy"), mmap_mode='r')

        # Ensure the number of inputs and labels match
        assert len(self.inputs) == len(self.labels), "Mismatch between number of inputs and labels."

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A tuple containing the input tensor and the label tensor.
        """
        # Slicing a memory-mapped array loads only the requested data into memory
        input_data = torch.from_numpy(self.inputs[idx].copy()).float()
        label_data = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return input_data, label_data



# ***** ------------------------------------------------------------ *****




# *****   FOR PUZZLE SOLVING TASK *****

# Global list mapping each dihedral transform id to its inverse.
# Index corresponds to the original tid, and the value is its inverse.
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


class PuzzleDatasetMetadata(pydantic.BaseModel):
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    
    total_groups: int
    mean_puzzle_examples: float

    sets: List[str]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""
    
    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)       # horizontal flip
    elif tid == 5:
        return np.flipud(arr)       # vertical flip
    elif tid == 6:
        return arr.T                # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        return arr
    
    
def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])

# ***** ------------------------------------------------------------ *****