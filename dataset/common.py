from typing import List, Optional, Tuple

import pydantic
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


# Dataset class for CIFAR-10 and CIFAR-100
class CIFARDataset(Dataset):
    """
    Simplified CIFAR dataset with proper augmentations for vision models.
    """
    def __init__(
        self,
        root: str = './data',
        train: bool = True,
        cifar100: bool = False,
        image_size: int = 32,
        patch_size: int = 4
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.train = train
        
        # Define transforms
        if train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])

        # Load dataset
        dataset_class = torchvision.datasets.CIFAR100 if cifar100 else torchvision.datasets.CIFAR10
        self.dataset = dataset_class(
            root=root,
            train=train,
            download=True,
            transform=self.transform
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        return image, label

    def get_num_classes(self) -> int:
        return 100 if isinstance(self.dataset, torchvision.datasets.CIFAR100) else 10

    def image_to_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image tensor to patches."""
        B, C, H, W = image.shape
        P = self.patch_size
        
        # Reshape to patches: (B, C, H/P, P, W/P, P)
        patches = image.view(B, C, H//P, P, W//P, P)
        
        # Permute to: (B, H/P, W/P, P, P, C)
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        
        # Merge patch spatial dims: (B, N, P*P*C)
        patches = patches.reshape(B, -1, P*P*C)
        
        return patches

class CIFARDatasetMetadata(pydantic.BaseModel):
    """Metadata for CIFAR dataset processing with patch-based vision models."""
    
    # Dataset properties
    num_classes: int
    num_train_examples: int
    num_test_examples: int
    split: str  # 'train' or 'test'
    
    # Image properties 
    image_size: int = 32
    patch_size: int = 4
    num_channels: int = 3
    
    # Normalization stats
    mean: List[float] = [0.4914, 0.4822, 0.4465]
    std: List[float] = [0.2023, 0.1994, 0.2010]
    
    # Augmentation settings
    use_augmentation: bool = True
    crop_padding: int = 4
    rotation_degrees: int = 15
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    
    @property
    def patches_per_image(self) -> int:
        """Number of patches per image."""
        return (self.image_size // self.patch_size) ** 2
    
    @property
    def patch_dim(self) -> int:
        """Dimension of each flattened patch."""
        return self.patch_size * self.patch_size * self.num_channels
    
    @property
    def total_examples(self) -> int:
        """Total number of examples in the current split."""
        return self.num_train_examples if self.split == 'train' else self.num_test_examples

    def get_transforms(self) -> transforms.Compose:
        """Get the transforms for current split."""
        if self.split == 'train' and self.use_augmentation:
            return transforms.Compose([
                transforms.RandomCrop(self.image_size, padding=self.crop_padding),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(self.rotation_degrees),
                transforms.ColorJitter(
                    brightness=self.color_jitter_brightness,
                    contrast=self.color_jitter_contrast,
                    saturation=self.color_jitter_saturation
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

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
