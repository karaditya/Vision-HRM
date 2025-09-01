#!/usr/bin/env python3
"""
Example script showing how to analyze the reasoning tract of HRM.

This demonstrates how to capture and visualize the model's reasoning process
when classifying images.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def example_reasoning_analysis():
    """Example of how to analyze HRM reasoning tract."""
    
    print("=== HRM Reasoning Tract Analysis Example ===\n")
    
    # 1. Load your trained model (replace with your actual model loading)
    print("1. Loading trained HRM model...")
    # model = load_your_trained_model("path/to/checkpoint")
    print("   ✓ Model loaded (placeholder)")
    
    # 2. Prepare a single image for analysis
    print("\n2. Preparing image for analysis...")
    
    # Create dummy image data (replace with actual image loading)
    batch_size = 1
    seq_len = 3072  # 32x32 image as 4x4 patches
    
    # Simulate image data
    dummy_image = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)
    dummy_label = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
    
    batch = {
        "inputs": dummy_image,
        "labels": dummy_label
    }
    
    print("   ✓ Image prepared (dummy data)")
    
    # 3. Run forward pass with reasoning capture
    print("\n3. Running forward pass with reasoning capture...")
    
    # This would be the actual forward pass with your model
    # carry = model.initial_carry(batch)
    # carry, metrics, predictions, _, _ = model(carry, batch, capture_reasoning=True)
    # reasoning_tract = model.get_reasoning_tract()
    
    print("   ✓ Forward pass completed (placeholder)")
    
    # 4. Analyze the reasoning tract
    print("\n4. Analyzing reasoning tract...")
    
    # Example of what the reasoning tract contains:
    example_reasoning_tract = {
        'H_hidden_states': [
            torch.randn(1, 65, 512),  # [batch, seq_len+1, hidden_size]
            torch.randn(1, 65, 512)
        ],
        'L_hidden_states': [
            torch.randn(1, 65, 512),
            torch.randn(1, 65, 512),
            torch.randn(1, 65, 512),
            torch.randn(1, 65, 512)
        ],
        'attention_maps': [
            {
                'level': 'H',
                'layer': 0,
                'maps': [{
                    'weights': torch.randn(1, 8, 65, 65),  # [batch, heads, seq_len, seq_len]
                    'query': torch.randn(1, 8, 65, 64),
                    'key': torch.randn(1, 8, 65, 64),
                    'value': torch.randn(1, 8, 65, 64)
                }]
            },
            {
                'level': 'L',
                'layer': 0,
                'maps': [{
                    'weights': torch.randn(1, 8, 65, 65),
                    'query': torch.randn(1, 8, 65, 64),
                    'key': torch.randn(1, 8, 65, 64),
                    'value': torch.randn(1, 8, 65, 64)
                }]
            }
        ],
        'patch_embeddings': torch.randn(1, 65, 512),
        'class_token_evolution': [
            {
                'cycle': 0,
                'level': 'H',
                'class_token': torch.randn(1, 512)
            },
            {
                'cycle': 0,
                'sub_cycle': 0,
                'level': 'L',
                'class_token': torch.randn(1, 512)
            }
        ]
    }
    
    example_predictions = {
        "logits": torch.randn(1, 10),
        "predictions": torch.randint(0, 10, (1,))
    }
    
    print("   ✓ Reasoning tract captured")
    
    # 5. Create visualizations
    print("\n5. Creating visualizations...")
    
    # Import visualization functions
    from visualize_reasoning import create_reasoning_report
    
    # Create comprehensive report
    report = create_reasoning_report(example_reasoning_tract, example_predictions, "example_analysis")
    
    print("   ✓ Visualizations created")
    
    # 6. Interpret the results
    print("\n6. Interpreting the results...")
    
    print("   What you can learn from the reasoning tract:")
    print("   • Attention Maps: Which image patches the model focuses on")
    print("   • Class Token Evolution: How the model's understanding changes")
    print("   • Hidden State Analysis: How representations evolve")
    print("   • Patch Importance: Which parts of the image are most relevant")
    
    print("\n   Key insights:")
    print("   • H-Level: Abstract, high-level reasoning (e.g., 'this looks like a car')")
    print("   • L-Level: Detailed, low-level reasoning (e.g., 'wheels, windows, body shape')")
    print("   • Attention: Shows which patches contribute to the final decision")
    print("   • Evolution: Tracks how the model's confidence changes through reasoning")
    
    print("\n=== Analysis Complete ===")
    print("Check the 'example_analysis/' directory for visualizations!")


def practical_usage_tips():
    """Provide practical tips for using reasoning tract analysis."""
    
    print("\n=== Practical Usage Tips ===\n")
    
    print("1. **When to Use Reasoning Analysis:**")
    print("   • Debugging model decisions")
    print("   • Understanding model biases")
    print("   • Validating model behavior")
    print("   • Research on interpretability")
    
    print("\n2. **Key Questions to Answer:**")
    print("   • Which image regions does the model attend to?")
    print("   • How does the model's understanding evolve?")
    print("   • What features are most important for classification?")
    print("   • Does the model use hierarchical reasoning as expected?")
    
    print("\n3. **Interpreting Attention Maps:**")
    print("   • Brighter colors = higher attention")
    print("   • Class token attention shows what's important for classification")
    print("   • Compare H-level vs L-level attention patterns")
    print("   • Look for attention to relevant object parts")
    
    print("\n4. **Analyzing Class Token Evolution:**")
    print("   • Red dots = H-level reasoning steps")
    print("   • Blue dots = L-level reasoning steps")
    print("   • Arrows show progression through reasoning")
    print("   • Large jumps indicate significant reasoning changes")
    
    print("\n5. **Common Patterns to Look For:**")
    print("   • H-Level: Broad, global attention patterns")
    print("   • L-Level: Fine-grained, local attention")
    print("   • Convergence: Class token stabilizes near the end")
    print("   • Divergence: Model explores multiple hypotheses")


if __name__ == "__main__":
    example_reasoning_analysis()
    practical_usage_tips()
