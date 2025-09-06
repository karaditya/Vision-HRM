from typing import List, Tuple, Dict
from dataclasses import dataclass
import os
import json
import hashlib
import numpy as np

import torchvision
import torchvision.transforms as transforms
from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata, dihedral_transform

cli = ArgParser()


class VisionDataProcessConfig(BaseModel):
    dataset_name: str = "CIFAR10"  # or "CIFAR100"
    output_dir: str = "data/cifar-vision-aug-4"
    seed: int = 42
    num_aug: int = 4 
    image_size: int = 32
    patch_size: int = 4
    num_channels: int = 3
    vocab_size: int = 256


@dataclass
class CIFARVisionPuzzle:
    id: str
    examples: List[Tuple[np.ndarray, int]]  # (patch_sequence, class_label)


def image_to_patch_sequence(image: np.ndarray, patch_size: int, vocab_size: int) -> np.ndarray:
    """Convert image to flattened patch sequence."""
    # Ensure (H, W, C) format
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    H, W, C = image.shape
    assert H % patch_size == 0 and W % patch_size == 0
    
    # Extract patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    patches = image.reshape(num_patches_h, patch_size, num_patches_w, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size * patch_size * C)
    
    # Quantize to vocab tokens
    patches_normalized = patches / 255.0
    patches_quantized = np.clip(patches_normalized * (vocab_size - 1), 0, vocab_size - 1).astype(np.uint8)
    
    return patches_quantized.flatten()


def augment_image(image: np.ndarray, trans_id: int, brightness: float, contrast: float) -> np.ndarray:
    """Apply augmentations to image."""
    # Dihedral transform
    augmented = dihedral_transform(image, trans_id)
    
    # Brightness and contrast
    augmented = augmented * brightness
    augmented = np.clip(augmented, 0, 255)
    
    mean = np.mean(augmented)
    augmented = (augmented - mean) * contrast + mean
    augmented = np.clip(augmented, 0, 255)
    
    return augmented.astype(np.uint8)


def puzzle_hash(puzzle: dict) -> str:
    """Hash puzzle for duplicate detection."""
    hashes = []
    for example_type, example in puzzle.items():
        for seq, label in example.examples:
            seq_hash = hashlib.sha256(seq.tobytes()).hexdigest()
            hashes.append(f"{seq_hash}|{label}")
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def convert_puzzle(results: dict, name: str, puzzle: dict, aug_count: int, 
                  dest_mapping: Dict[str, Tuple[str, str]], config: VisionDataProcessConfig):
    """Convert single CIFAR puzzle to vision format."""
    
    # Convert to vision format
    dests = set(dest_mapping.values())
    converted = {dest: CIFARVisionPuzzle(name, []) for dest in dests}
    
    for example_type, item_example in puzzle.items():
        dest = dest_mapping[example_type]
        for example in item_example['examples']:
            seq = image_to_patch_sequence(example["image"], config.patch_size, config.vocab_size)
            converted[dest].examples.append((seq, example["label"]))

    group = [converted]
    
    # Augmentation
    if aug_count > 0:
        hashes = {puzzle_hash(converted)}
        
        for _ in range(5 * aug_count):  # Try multiple times
            trans_id = np.random.randint(0, 8)
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            
            augmented = {dest: CIFARVisionPuzzle(name, []) for dest in dests}
            
            for dest, puzzle_obj in converted.items():
                for seq, label in puzzle_obj.examples:
                    # Reconstruct image for augmentation
                    patch_dim = config.patch_size ** 2 * config.num_channels
                    num_patches = len(seq) // patch_dim
                    patches_per_side = int(np.sqrt(num_patches))
                    
                    patches = seq.reshape(num_patches, patch_dim)
                    patches_denorm = (patches / (config.vocab_size - 1) * 255).astype(np.uint8)
                    patches_2d = patches_denorm.reshape(num_patches, config.patch_size, config.patch_size, config.num_channels)
                    
                    image = patches_2d.reshape(patches_per_side, patches_per_side, config.patch_size, config.patch_size, config.num_channels)
                    image = image.transpose(0, 2, 1, 3, 4).reshape(config.image_size, config.image_size, config.num_channels)
                    
                    aug_image = augment_image(image, trans_id, brightness, contrast)
                    aug_seq = image_to_patch_sequence(aug_image, config.patch_size, config.vocab_size)
                    
                    augmented[dest].examples.append((aug_seq, label))
            
            h = puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)
                
            if len(group) >= aug_count + 1:
                break

    # Add to results
    for dest in dests:
        dest_split, dest_set = dest
        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([converted[dest] for converted in group])


def load_cifar_dataset(config: VisionDataProcessConfig):
    """Load CIFAR dataset."""
    if config.dataset_name == "CIFAR10":
        dataset_class = torchvision.datasets.CIFAR10
        num_classes = 10
    elif config.dataset_name == "CIFAR100":
        dataset_class = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")
    
    train_dataset = dataset_class(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    # Convert to puzzle format
    train_puzzles = []
    for idx in range(len(train_dataset)):
        image, label = train_dataset[idx]
        image_np = (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        puzzle = {"train": {"examples": [{"image": image_np, "label": label}]}}
        train_puzzles.append((f"train_{idx}", puzzle))
    
    test_puzzles = []
    for idx in range(len(test_dataset)):
        image, label = test_dataset[idx]
        image_np = (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        puzzle = {"test": {"examples": [{"image": image_np, "label": label}]}}
        test_puzzles.append((f"test_{idx}", puzzle))
    
    return train_puzzles, test_puzzles, num_classes


def convert_dataset(config: VisionDataProcessConfig):
    """Convert CIFAR dataset to vision format."""
    np.random.seed(config.seed)
    
    train_puzzles, test_puzzles, num_classes = load_cifar_dataset(config)
    all_puzzles = train_puzzles + test_puzzles
    
    # Map puzzle identifiers
    num_identifiers = 1  # 0 is blank
    identifier_map = {}
    for puzzle_id, _ in all_puzzles:
        if puzzle_id not in identifier_map:
            identifier_map[puzzle_id] = num_identifiers
            num_identifiers += 1

    # Calculate sequence length
    patches_per_side = config.image_size // config.patch_size
    patch_dim = config.patch_size ** 2 * config.num_channels
    seq_len = patches_per_side ** 2 * patch_dim

    results = {}
    
    # Process training puzzles with augmentation
    for name, puzzle in train_puzzles:
        convert_puzzle(results, name, puzzle, config.num_aug,
                      {"train": ("train", "all"), "test": ("train", "all")}, config)
    
    # Process test puzzles without augmentation
    for name, puzzle in test_puzzles:
        convert_puzzle(results, name, puzzle, 0,
                      {"train": ("test", "all"), "test": ("test", "all")}, config)

    # Save processed data
    for split_name, split in results.items():
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)
        
        total_examples = 0
        total_puzzles = 0 
        total_groups = 0
        
        for subset_name, subset in split.items():
            data_dict = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
            data_dict["puzzle_indices"].append(0)
            data_dict["group_indices"].append(0)
            
            example_id = 0
            puzzle_id = 0
            
            for group in subset:
                for puzzle in group:
                    for seq, label in puzzle.examples:
                        # Ensure correct sequence length
                        if len(seq) != seq_len:
                            if len(seq) > seq_len:
                                seq = seq[:seq_len]
                            else:
                                seq = np.concatenate([seq, np.zeros(seq_len - len(seq), dtype=np.uint8)])
                        
                        data_dict["inputs"].append(seq)
                        data_dict["labels"].append(np.array([label], dtype=np.int32))
                        example_id += 1
                        total_examples += 1

                    data_dict["puzzle_indices"].append(example_id)
                    data_dict["puzzle_identifiers"].append(identifier_map[puzzle.id])
                    puzzle_id += 1
                    total_puzzles += 1
                    
                data_dict["group_indices"].append(puzzle_id)
                total_groups += 1
            
            # Save arrays
            for k, v in data_dict.items():
                if k == "inputs":
                    v = np.stack(v, 0)
                elif k == "labels":
                    v = np.stack(v, 0)  # Shape: (num_examples, 1)
                else:
                    v = np.array(v, dtype=np.int32)
                
                np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"), v)
        
        # Save metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=seq_len,
            vocab_size=config.vocab_size,
            pad_id=0,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles,
            sets=list(split.keys())
        )

        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
            
    # Save identifier mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        ids_mapping = {v: k for k, v in identifier_map.items()}
        json.dump([ids_mapping.get(i, "<blank>") for i in range(num_identifiers)], f)

    print(f"Dataset saved to {config.output_dir}")
    print(f"Sequence length: {seq_len}, Classes: {num_classes}")


@cli.command(singleton=True)
def main(config: VisionDataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()