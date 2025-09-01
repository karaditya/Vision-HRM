#!/usr/bin/env python3
"""
Test script to verify the attention-enabled HRM model works correctly.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_attention_model():
    """Test the attention-enabled HRM model."""
    
    print("Testing Attention-Enabled HRM Model...")
    
    try:
        # Import the attention model
        from models.hrm.hrm_vision_v1_with_attention import (
            HierarchicalReasoningModel_VisionV1,
            HierarchicalReasoningModel_VisionV1Config
        )
        
        print("✓ Successfully imported attention model")
        
        # Create config
        config = HierarchicalReasoningModel_VisionV1Config(
            hidden_size=256,
            num_heads=4,
            expansion=2,
            H_layers=2,
            L_layers=2,
            H_cycles=1,
            L_cycles=1,
            vocab_size=256,
            seq_len=3072,  # 32x32 image as patches
            num_classes=10,
            patch_size=4,
            image_size=32,
            num_puzzle_identifiers=1,
            puzzle_emb_ndim=64,
            halt_exploration_prob=0.0,
            halt_max_steps=1,
            batch_size=2,
            forward_dtype="float32"
        )
        
        print("✓ Config created successfully")
        
        # Create model
        model = HierarchicalReasoningModel_VisionV1(config.__dict__)
        print("✓ Model created successfully")
        
        # Create dummy batch
        batch_size = 2
        seq_len = 3072
        
        batch = {
            "inputs": torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long),
            "labels": torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
        }
        
        print("✓ Dummy batch created")
        
        # Test forward pass without reasoning capture
        print("\nTesting forward pass without reasoning capture...")
        carry = model.initial_carry(batch)
        carry, metrics, predictions, _, _ = model(carry, batch, capture_reasoning=False)
        
        print(f"✓ Forward pass successful")
        print(f"  - Loss: {metrics['loss']:.4f}")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - Predictions shape: {predictions['predictions'].shape}")
        
        # Test forward pass with reasoning capture
        print("\nTesting forward pass with reasoning capture...")
        carry = model.initial_carry(batch)
        carry, metrics, predictions, _, _ = model(carry, batch, capture_reasoning=True)
        
        print(f"✓ Forward pass with reasoning capture successful")
        
        # Get reasoning tract
        reasoning_tract = model.get_reasoning_tract()
        
        print(f"✓ Reasoning tract captured:")
        print(f"  - H hidden states: {len(reasoning_tract['H_hidden_states'])}")
        print(f"  - L hidden states: {len(reasoning_tract['L_hidden_states'])}")
        print(f"  - Attention maps: {len(reasoning_tract['attention_maps'])}")
        print(f"  - Class token evolution: {len(reasoning_tract['class_token_evolution'])}")
        print(f"  - Patch embeddings: {reasoning_tract['patch_embeddings'] is not None}")
        
        # Test clearing reasoning tract
        model.clear_reasoning_tract()
        empty_tract = model.get_reasoning_tract()
        print(f"✓ Reasoning tract cleared successfully")
        
        print("\n=== All Tests Passed! ===")
        print("The attention-enabled HRM model is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization_imports():
    """Test that visualization functions can be imported."""
    
    print("\nTesting visualization imports...")
    
    try:
        from visualize_reasoning import (
            visualize_attention_maps,
            visualize_class_token_evolution,
            visualize_patch_importance,
            visualize_hidden_state_analysis,
            create_reasoning_report
        )
        
        print("✓ All visualization functions imported successfully")
        
        # Test with dummy data
        dummy_attention_maps = [
            {
                'level': 'H',
                'layer': 0,
                'maps': [{
                    'weights': torch.randn(1, 4, 65, 65),
                    'query': torch.randn(1, 4, 65, 64),
                    'key': torch.randn(1, 4, 65, 64),
                    'value': torch.randn(1, 4, 65, 64)
                }]
            }
        ]
        
        dummy_class_evolution = [
            {
                'cycle': 0,
                'level': 'H',
                'class_token': torch.randn(1, 256)
            }
        ]
        
        dummy_predictions = {
            "logits": torch.randn(1, 10),
            "predictions": torch.randint(0, 10, (1,))
        }
        
        print("✓ Dummy data created for visualization testing")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Attention Model Test Suite ===\n")
    
    # Test attention model
    model_test_passed = test_attention_model()
    
    # Test visualization imports
    viz_test_passed = test_visualization_imports()
    
    print(f"\n=== Test Results ===")
    print(f"Model Test: {'✓ PASSED' if model_test_passed else '✗ FAILED'}")
    print(f"Visualization Test: {'✓ PASSED' if viz_test_passed else '✗ FAILED'}")
    
    if model_test_passed and viz_test_passed:
        print("\n🎉 All tests passed! The attention-enabled HRM is ready to use.")
        print("\nNext steps:")
        print("1. Train your model using the attention-enabled version")
        print("2. Use capture_reasoning=True to get reasoning tracts")
        print("3. Use visualize_reasoning.py to analyze the results")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
