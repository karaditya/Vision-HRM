# import numpy as np
# import matplotlib.pyplot as plt
# import json
# import os
# from pathlib import Path

# def load_dataset_metadata(split_dir):
#     """Load dataset metadata from json file"""
#     with open(os.path.join(split_dir, "dataset.json"), "r") as f:
#         return json.load(f)

# def reconstruct_image_from_patches(patches, patch_size, num_channels):
#     """Reconstruct a single image from its patches"""
#     # Reshape patches into a grid
#     num_patches = patches.shape[0]
#     grid_size = int(np.sqrt(num_patches))
    
#     # Reshape each patch into (patch_size, patch_size, channels)
#     patches = patches.reshape(num_patches, patch_size, patch_size, num_channels)
    
#     # Arrange patches into a grid
#     image = np.zeros((grid_size * patch_size, grid_size * patch_size, num_channels))
#     for idx in range(num_patches):
#         i, j = idx // grid_size, idx % grid_size
#         image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[idx]
    
#     return image.astype(np.uint8)

# def visualize_patches(data_dir="data/cifar-processed", split="train", num_examples=5):
#     """Visualize original images and their patches"""
#     split_dir = Path(data_dir) / split
    
#     # Load metadata
#     metadata = load_dataset_metadata(split_dir)
#     patch_size = metadata['patch_size']
#     num_channels = metadata['num_channels']
    
#     # Load data
#     inputs = np.load(split_dir / "all__inputs.npy")
#     labels = np.load(split_dir / "all__labels.npy")
    
#     # Number of patches per image
#     patches_per_image = (metadata['image_size'] // patch_size) ** 2
    
#     # Create figure
#     fig, axes = plt.subplots(num_examples, 2, figsize=(10, num_examples * 3))
#     fig.suptitle(f'CIFAR {split} set: Original Images vs Patches')
    
#     for idx in range(num_examples):
#         # Get patches for one image
#         image_patches = inputs[idx * patches_per_image:(idx + 1) * patches_per_image]
        
#         # Reconstruct image from patches
#         reconstructed = reconstruct_image_from_patches(image_patches, patch_size, num_channels)
        
#         # Plot original reconstructed image
#         axes[idx, 0].imshow(reconstructed)
#         axes[idx, 0].set_title(f'Image {idx} (Label: {labels[idx * patches_per_image]})')
#         axes[idx, 0].axis('off')
        
#         # Plot patches grid
#         patch_grid = np.zeros((4 * patch_size, 4 * patch_size, num_channels))
#         for p in range(min(16, patches_per_image)):
#             i, j = p // 4, p % 4
#             patch = image_patches[p].reshape(patch_size, patch_size, num_channels)
#             patch_grid[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patch
        
#         axes[idx, 1].imshow(patch_grid.astype(np.uint8))
#         axes[idx, 1].set_title(f'First 16 patches')
#         axes[idx, 1].axis('off')
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     visualize_patches()
import numpy as np
import matplotlib.pyplot as plt

# Load the saved patches and labels
inputs = np.load("data/cifar-processed/train/all__inputs.npy")
labels = np.load("data/cifar-processed/train/all__labels.npy")

# Parameters (must match the ones used in build_cifar_dataset.py)
patch_size = 4
image_size = 32
num_channels = 3
num_patches_per_image = (image_size // patch_size) ** 2

# Pick an image index
img_idx = 0  # change to look at another image

# Extract the patches belonging to this image
start = img_idx * num_patches_per_image
end = start + num_patches_per_image
patches = inputs[start:end]

# Reshape patches back to (num_patches, patch_size, patch_size, C)
patches = patches.reshape(-1, patch_size, patch_size, num_channels)

# Plot patches independently in an 8x8 grid
grid_size = int(np.sqrt(num_patches_per_image))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

for idx, ax in enumerate(axes.flat):
    ax.imshow(patches[idx])
    ax.axis("off")

plt.suptitle(f"Image {img_idx} patches (label={labels[start]})")
plt.tight_layout()
plt.show()

