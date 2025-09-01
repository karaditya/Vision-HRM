#!/usr/bin/env python3
"""
Visualization script for HRM reasoning tract analysis.

This script helps you understand how the HRM model reasons about images
by visualizing attention maps, hidden state evolution, and class token changes.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from typing import Dict, List, Any, Optional
import argparse

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def visualize_attention_maps(attention_maps: List[Dict], image_size: int = 32, patch_size: int = 4, 
                           save_path: str = "attention_maps.png"):
    """Visualize attention maps from different layers and heads."""
    
    num_patches = (image_size // patch_size) ** 2
    patches_per_side = image_size // patch_size
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('HRM Attention Maps Across Layers', fontsize=16)
    
    for layer_idx, layer_maps in enumerate(attention_maps):
        if layer_idx >= 8:  # Limit to 8 visualizations
            break
            
        level = layer_maps['level']
        layer_num = layer_maps['layer']
        
        # Get attention weights from first head
        if layer_maps['maps']:
            attention_weights = layer_maps['maps'][0]['weights'][0, 0]  # [seq_len, seq_len]
            
            # Focus on class token attention (first row)
            class_token_attention = attention_weights[0, 1:].reshape(patches_per_side, patches_per_side)
            
            # Plot
            row = layer_idx // 4
            col = layer_idx % 4
            ax = axes[row, col]
            
            im = ax.imshow(class_token_attention.numpy(), cmap='viridis', interpolation='nearest')
            ax.set_title(f'{level}-Level Layer {layer_num}\nClass Token Attention')
            ax.set_xticks(range(patches_per_side))
            ax.set_yticks(range(patches_per_side))
            ax.set_xticklabels([f'{i*patch_size}' for i in range(patches_per_side)])
            ax.set_yticklabels([f'{i*patch_size}' for i in range(patches_per_side)])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Attention maps saved to {save_path}")


def visualize_class_token_evolution(class_token_evolution: List[Dict], save_path: str = "class_token_evolution.png"):
    """Visualize how the class token evolves through reasoning cycles."""
    
    # Extract class token representations
    tokens = []
    labels = []
    
    for step in class_token_evolution:
        tokens.append(step['class_token'][0].numpy())  # First sample
        level = step['level']
        cycle = step.get('cycle', 0)
        sub_cycle = step.get('sub_cycle', 0)
        labels.append(f"{level}-{cycle}.{sub_cycle}")
    
    tokens = np.array(tokens)
    
    # PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    tokens_2d = pca.fit_transform(tokens)
    
    # Plot evolution
    plt.figure(figsize=(12, 8))
    
    # Color by level
    colors = ['red' if 'H' in label else 'blue' for label in labels]
    
    plt.scatter(tokens_2d[:, 0], tokens_2d[:, 1], c=colors, s=100, alpha=0.7)
    
    # Add arrows showing evolution
    for i in range(len(tokens_2d) - 1):
        plt.arrow(tokens_2d[i, 0], tokens_2d[i, 1], 
                 tokens_2d[i+1, 0] - tokens_2d[i, 0], 
                 tokens_2d[i+1, 1] - tokens_2d[i, 1],
                 head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.5)
    
    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(label, (tokens_2d[i, 0], tokens_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('Class Token Evolution Through Reasoning Cycles\n(Red: H-Level, Blue: L-Level)')
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class token evolution saved to {save_path}")


def visualize_patch_importance(patch_embeddings: torch.Tensor, attention_maps: List[Dict], 
                             image_size: int = 32, patch_size: int = 4, 
                             save_path: str = "patch_importance.png"):
    """Visualize which patches are most important for classification."""
    
    num_patches = (image_size // patch_size) ** 2
    patches_per_side = image_size // patch_size
    
    # Compute patch importance by averaging attention weights across all layers
    patch_importance = np.zeros(num_patches)
    count = 0
    
    for layer_maps in attention_maps:
        if layer_maps['maps']:
            for map_data in layer_maps['maps']:
                # Class token attention to patches
                attention_weights = map_data['weights'][0, 0]  # [seq_len, seq_len]
                class_token_attention = attention_weights[0, 1:num_patches+1]  # Exclude class token
                patch_importance += class_token_attention.numpy()
                count += 1
    
    if count > 0:
        patch_importance /= count
    
    # Reshape to image grid
    patch_importance = patch_importance.reshape(patches_per_side, patches_per_side)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Create a grid showing patch importance
    im = plt.imshow(patch_importance, cmap='hot', interpolation='nearest')
    
    # Add grid lines
    for i in range(patches_per_side + 1):
        plt.axhline(y=i-0.5, color='white', linewidth=1, alpha=0.5)
        plt.axvline(x=i-0.5, color='white', linewidth=1, alpha=0.5)
    
    plt.title('Patch Importance for Classification\n(Average attention from class token)')
    plt.xlabel('Patch Column')
    plt.ylabel('Patch Row')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Attention Weight')
    
    # Add patch indices
    for i in range(patches_per_side):
        for j in range(patches_per_side):
            plt.text(j, i, f'{i*patches_per_side + j}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Patch importance saved to {save_path}")


def visualize_hidden_state_analysis(H_hidden_states: List[torch.Tensor], L_hidden_states: List[torch.Tensor],
                                  save_path: str = "hidden_state_analysis.png"):
    """Analyze how hidden states evolve through reasoning."""
    
    # Compute statistics for each hidden state
    H_stats = []
    L_stats = []
    
    for h_state in H_hidden_states:
        h_state_np = h_state[0].numpy()  # First sample
        H_stats.append({
            'mean': np.mean(h_state_np),
            'std': np.std(h_state_np),
            'max': np.max(h_state_np),
            'min': np.min(h_state_np)
        })
    
    for l_state in L_hidden_states:
        l_state_np = l_state[0].numpy()  # First sample
        L_stats.append({
            'mean': np.mean(l_state_np),
            'std': np.std(l_state_np),
            'max': np.max(l_state_np),
            'min': np.min(l_state_np)
        })
    
    # Plot statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hidden State Evolution Analysis', fontsize=16)
    
    # H-level statistics
    H_cycles = list(range(len(H_stats)))
    axes[0, 0].plot(H_cycles, [s['mean'] for s in H_stats], 'r-o', label='H-Level')
    axes[0, 0].set_title('Mean Activation')
    axes[0, 0].set_ylabel('Mean')
    axes[0, 0].legend()
    
    axes[0, 1].plot(H_cycles, [s['std'] for s in H_stats], 'r-o', label='H-Level')
    axes[0, 1].set_title('Standard Deviation')
    axes[0, 1].set_ylabel('Std')
    axes[0, 1].legend()
    
    # L-level statistics
    L_cycles = list(range(len(L_stats)))
    axes[1, 0].plot(L_cycles, [s['mean'] for s in L_stats], 'b-o', label='L-Level')
    axes[1, 0].set_title('Mean Activation')
    axes[1, 0].set_xlabel('Reasoning Cycle')
    axes[1, 0].set_ylabel('Mean')
    axes[1, 0].legend()
    
    axes[1, 1].plot(L_cycles, [s['std'] for s in L_stats], 'b-o', label='L-Level')
    axes[1, 1].set_title('Standard Deviation')
    axes[1, 1].set_xlabel('Reasoning Cycle')
    axes[1, 1].set_ylabel('Std')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hidden state analysis saved to {save_path}")


def create_reasoning_report(reasoning_tract: Dict, predictions: Dict, save_dir: str = "reasoning_report"):
    """Create a comprehensive reasoning report with all visualizations."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("Creating reasoning report...")
    
    # Extract components
    attention_maps = reasoning_tract.get('attention_maps', [])
    class_token_evolution = reasoning_tract.get('class_token_evolution', [])
    patch_embeddings = reasoning_tract.get('patch_embeddings')
    H_hidden_states = reasoning_tract.get('H_hidden_states', [])
    L_hidden_states = reasoning_tract.get('L_hidden_states', [])
    
    # Create visualizations
    if attention_maps:
        visualize_attention_maps(attention_maps, save_path=os.path.join(save_dir, "attention_maps.png"))
    
    if class_token_evolution:
        visualize_class_token_evolution(class_token_evolution, save_path=os.path.join(save_dir, "class_token_evolution.png"))
    
    if patch_embeddings is not None and attention_maps:
        visualize_patch_importance(patch_embeddings, attention_maps, save_path=os.path.join(save_dir, "patch_importance.png"))
    
    if H_hidden_states or L_hidden_states:
        visualize_hidden_state_analysis(H_hidden_states, L_hidden_states, save_path=os.path.join(save_dir, "hidden_state_analysis.png"))
    
    # Create summary report
    report = {
        "model_predictions": {
            "logits": predictions.get("logits", []).tolist() if isinstance(predictions.get("logits"), torch.Tensor) else predictions.get("logits", []),
            "predictions": predictions.get("predictions", []).tolist() if isinstance(predictions.get("predictions"), torch.Tensor) else predictions.get("predictions", [])
        },
        "reasoning_summary": {
            "num_h_cycles": len(H_hidden_states),
            "num_l_cycles": len(L_hidden_states),
            "num_attention_layers": len(attention_maps),
            "class_token_steps": len(class_token_evolution)
        },
        "attention_summary": [
            {
                "level": layer_map["level"],
                "layer": layer_map["layer"],
                "num_heads": len(layer_map["maps"]) if layer_map["maps"] else 0
            }
            for layer_map in attention_maps
        ]
    }
    
    with open(os.path.join(save_dir, "report.json"), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Reasoning report saved to {save_dir}/")
    return report


def analyze_single_image(model, dataloader, image_idx: int = 0, save_dir: str = "single_image_analysis"):
    """Analyze the reasoning tract for a single image."""
    
    print(f"Analyzing reasoning for image {image_idx}...")
    
    # Get the image
    for batch_idx, (set_name, batch, global_batch_size) in enumerate(dataloader):
        if batch_idx == 0:  # Just get first batch
            break
    
    # Select specific image
    single_batch = {
        "inputs": batch["inputs"][image_idx:image_idx+1],
        "labels": batch["labels"][image_idx:image_idx+1]
    }
    
    # Move to device
    device = next(model.parameters()).device
    single_batch = {k: v.to(device) for k, v in single_batch.items()}
    
    # Forward pass with reasoning capture
    carry = model.initial_carry(single_batch)
    carry, metrics, predictions, _, _ = model(carry, single_batch, capture_reasoning=True)
    
    # Get reasoning tract
    reasoning_tract = model.get_reasoning_tract()
    
    # Create report
    report = create_reasoning_report(reasoning_tract, predictions, save_dir)
    
    print(f"Analysis complete! Check {save_dir}/ for visualizations.")
    return reasoning_tract, predictions, report


def main():
    parser = argparse.ArgumentParser(description="Visualize HRM reasoning tract")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data-path", type=str, default="data/cifar-aug-1000", help="Path to dataset")
    parser.add_argument("--image-idx", type=int, default=0, help="Index of image to analyze")
    parser.add_argument("--save-dir", type=str, default="reasoning_analysis", help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Load model and data
    print("Loading model and data...")
    
    # This would need to be implemented based on your specific model loading setup
    # model = load_model_from_checkpoint(args.checkpoint)
    # dataloader = create_dataloader(args.data_path)
    
    # For now, we'll create a placeholder
    print("Model loading not implemented in this example.")
    print("To use this script:")
    print("1. Load your trained HRM model")
    print("2. Create a dataloader for your dataset")
    print("3. Call analyze_single_image(model, dataloader, args.image_idx, args.save_dir)")
    
    # Example usage:
    # analyze_single_image(model, dataloader, args.image_idx, args.save_dir)


if __name__ == "__main__":
    main()
