import numpy as np
import matplotlib.pyplot as plt

# Load the saved patches and labels
inputs = np.load("data/CIFAR10-processed/train/all__inputs.npy") # CIFAR10-processed
labels = np.load("data/CIFAR10-processed/train/all__labels.npy") 

# Parameters (must match the ones used in build_cifar_dataset.py)
patch_size = 4
image_size = 32
num_channels = 3
num_patches_per_image = (image_size // patch_size) ** 2

# Pick an image index
img_idx = 11  # change to look at another image

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
plt.savefig("output.png", dpi=300, bbox_inches="tight")


