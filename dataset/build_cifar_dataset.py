from typing import Dict
import os
from tqdm import tqdm
import json
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from argdantic import ArgParser
from pydantic import BaseModel
from common import CIFARDatasetMetadata



cli = ArgParser()

class DataProcessConfig(BaseModel):
    dataset_name: str = "CIFAR10"
    output_dir: str = "data/cifar-processed"
    seed: int = 42
    num_aug: int = 4
    image_size: int = 32
    patch_size: int = 4
    num_channels: int = 3

class CIFARProcessor:
    def __init__(self, config: DataProcessConfig):
        self.config = config
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])

    
    
    def get_num_classes(self) -> int:
        """Get number of classes in the dataset."""
        return 100 if self.config.dataset_name == "CIFAR100" else 10
    
    def image_to_patches(self, image: torch.Tensor) -> np.ndarray:
        """Convert image tensor to patches."""
        # Denormalize and convert to numpy
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1).to(image.device)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1).to(image.device)
        image = (image * std + mean) * 255
        image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # Different variable name
        
        # Convert to patches
        H, W, C = image_np.shape  # Use the numpy variable
        P = self.config.patch_size
        patches = image_np.reshape(H//P, P, W//P, P, C)  # Use the numpy variable
        patches = patches.transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(-1, P*P*C)
        return patches

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
                sequence = patches.flatten()  # ✅ One sequence per image
                all_patches.append(sequence)
                all_labels.append(label)  # ✅ One label per image
                
                # For training split only: add augmentations
                if split == "train":
                    for _ in range(self.config.num_aug):
                        aug_image, _ = dataset[idx]
                        aug_patches = self.image_to_patches(aug_image)
                        aug_sequence = aug_patches.flatten()  # ✅ One sequence per augmented image
                        all_patches.append(aug_sequence)
                        all_labels.append(label)  # ✅ One label per augmented image
            
            # Save as numpy arrays
            inputs = np.array(all_patches)
            labels = np.array(all_labels)
            
            np.save(os.path.join(split_dir, "all__inputs.npy"), inputs)
            np.save(os.path.join(split_dir, "all__labels.npy"), labels)
            
            # Save metadata
            seq_len = self.config.patch_size * self.config.patch_size * self.config.num_channels
            metadata = CIFARDatasetMetadata(
                num_classes=self.get_num_classes(),
                num_train_examples=len(inputs),  # Now correct since inputs contains sequences, not patches
                num_test_examples=len(inputs),   # Same here
                split=split,
                image_size=self.config.image_size,
                patch_size=self.config.patch_size,
                num_channels=self.config.num_channels,
                use_augmentation=(split == 'train')
            )
            
            with open(os.path.join(split_dir, "dataset.json"), "w") as f:
                json.dump(metadata.model_dump(), f)

@cli.command(singleton=True)
def main(config: DataProcessConfig):
    processor = CIFARProcessor(config)
    processor.process_dataset()

if __name__ == "__main__":
    cli()
