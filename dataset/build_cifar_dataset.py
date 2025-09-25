from typing import List
import os
from tqdm import tqdm
import json
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from pydantic import BaseModel

import hydra
from omegaconf import DictConfig



# Configuration for data processing
class DataProcessConfig(BaseModel):
    dataset_name: str 
    output_dir: str 
    
    seed: int 
    num_aug: int 
    image_size: int 
    patch_size: int 
    num_channels: int 


    crop_padding: int 
    rotation_degrees: int 
    translate: List[float]
    color_jitter_brightness: float
    color_jitter_contrast: float 
    color_jitter_saturation: float 

    # Normalization stats
    mean: List[float] 
    std: List[float]




# Metadata for CIFAR dataset
class CIFARDatasetMetadata(BaseModel):
    """Metadata for CIFAR dataset processing with patch-based vision models."""
    
    # Dataset properties
    num_classes: int
    num_train_examples: int
    num_test_examples: int
    seq_len : int
    split: str  # 'train' or 'test'
    
    # Image properties 
    image_size: int 
    patch_size: int 
    num_channels: int 
    
    # Normalization stats
    mean: List[float] 
    std: List[float]
    
    # Augmentation settings
    use_augmentation: bool = True
    crop_padding: int 
    rotation_degrees: int
    translate: List[float] 
    color_jitter_brightness: float 
    color_jitter_contrast: float 
    color_jitter_saturation: float 
    
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



# Processor class for CIFAR dataset
class CIFARProcessor:
    def __init__(self, config: DataProcessConfig):
        self.config = config
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(config.image_size, padding=config.crop_padding),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(config.rotation_degrees),
            transforms.ColorJitter(brightness=config.color_jitter_brightness, contrast=config.color_jitter_contrast),
            transforms.RandomAffine(degrees=0, translate=(config.translate[0], config.translate[1])),
            transforms.ToTensor(),
            transforms.Normalize((config.mean[0], config.mean[1], config.mean[2]), 
                               (config.std[0], config.std[1], config.std[2]))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((config.mean[0], config.mean[1], config.mean[2]), 
                               (config.std[0], config.std[1], config.std[2]))
        ])

    
    
    def get_num_classes(self) -> int:
        """Get number of classes in the dataset."""
        return 100 if self.config.dataset_name == "CIFAR100" else 10
    
    def image_to_patches(self, image: torch.Tensor) -> np.ndarray:
        """Convert image tensor to patches."""
        
        # Convert to patches
        C, H, W = image.shape  # Use the numpy variable
        P = self.config.patch_size

        # Reshape to (C, H/P, P, W/P, P)
        patches = image.view(C, H // P, P, W // P, P)
        # Transpose to (H/P, W/P, P, P, C)
        patches = patches.permute(1, 3, 2, 4, 0)
        # Reshape to (Num_Patches, Patch_Dim) where Patch_Dim = P*P*C
        patches = patches.reshape(-1, P * P * C)

        return patches.cpu().numpy()

    def process_dataset(self):
        np.random.seed(self.config.seed)
        
        # Load dataset
        if self.config.dataset_name == "CIFAR10":
            dataset_class = torchvision.datasets.CIFAR10
        else:
            dataset_class = torchvision.datasets.CIFAR100

        train_dataset = dataset_class(
            root='./data', train=True, download=True,
            transform=self.transform_train
        )
        test_dataset = dataset_class(
            root='./data', train=False, download=True,
            transform=self.transform_test
        )

        # Process datasets
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        for split, dataset in [("train", train_dataset), ("test", test_dataset)]:
            split_dir = os.path.join(self.config.output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            all_patches = []
            all_labels = []
            
            # Process each image into patch sequences
            for idx in tqdm(range(len(dataset)), desc=f"Processing {split} data"):
                image, label = dataset[idx]
                patches = self.image_to_patches(image)
                
                # Store original - flatten patches into one sequence
                sequence = patches.flatten()  # One sequence per image : len = num_patches * P * P * C
                all_patches.append(sequence)
                all_labels.append(label)  # One label per image 
                
                # For training split only: add augmentations
                if split == "train":
                    for _ in range(self.config.num_aug):
                        aug_image, _ = dataset[idx]
                        aug_patches = self.image_to_patches(aug_image)
                        aug_sequence = aug_patches.flatten()  # One sequence per augmented image
                        all_patches.append(aug_sequence)
                        all_labels.append(label)  # One label per augmented image
            
            # Save as numpy arrays
            inputs = np.array(all_patches)
            labels = np.array(all_labels)
            
            np.save(os.path.join(split_dir, "all__inputs.npy"), inputs)
            np.save(os.path.join(split_dir, "all__labels.npy"), labels)
            
            # Save metadata
            seq_len = self.config.patch_size * self.config.patch_size * self.config.num_channels

            metadata = CIFARDatasetMetadata(
                num_classes=self.get_num_classes(),
                num_train_examples=len(inputs),  
                num_test_examples=len(inputs),
                seq_len=seq_len,
                split=split,
                image_size=self.config.image_size,
                patch_size=self.config.patch_size,
                num_channels=self.config.num_channels,
                use_augmentation=(split == 'train'),
                crop_padding=self.config.crop_padding,
                rotation_degrees=self.config.rotation_degrees,
                translate=self.config.translate,
                color_jitter_brightness=self.config.color_jitter_brightness,
                color_jitter_contrast=self.config.color_jitter_contrast,
                color_jitter_saturation=self.config.color_jitter_saturation,
                mean=self.config.mean,
                std=self.config.std
            )

            with open(os.path.join(split_dir, "dataset_metadata.json"), "w") as f:
                json.dump(metadata.model_dump(), f)

@hydra.main(config_path="../config/data", config_name="cfg_build_cifar", version_base=None)
def main(hydra_cfg: DictConfig):
    
    data_process_config = DataProcessConfig(**hydra_cfg)  # type: ignore

    processor = CIFARProcessor(data_process_config)
    processor.process_dataset()


if __name__ == "__main__":
    main()
