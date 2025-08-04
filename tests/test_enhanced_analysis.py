#!/usr/bin/env python3
"""
Test script for enhanced analysis module
"""

import sys
import os
import torch
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_enhanced_analyzer_import():
    """Test that the enhanced analyzer can be imported."""
    try:
        from evaluation.enhanced_analysis import EnhancedAnalyzer
        print("✅ EnhancedAnalyzer imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import EnhancedAnalyzer: {e}")
        return False

def test_enhanced_analyzer_initialization():
    """Test that the enhanced analyzer can be initialized."""
    try:
        from evaluation.enhanced_analysis import EnhancedAnalyzer
        from models.modular_rlvae import ModularRiemannianFlowVAE as RlVAE
        from omegaconf import DictConfig
        
        # Create a simple mock config
        config = DictConfig({
            'input_dim': [1, 64, 64],  # [C, H, W]
            'latent_dim': 16,
            'n_flows': 4,
            'flow_hidden_size': 128,
            'flow_n_blocks': 2,
            'flow_n_hidden': 2,
            'epsilon': 1e-6,
            'beta': 1.0,
            'posterior': {'type': 'normal'},
            'loop': {'mode': 'open', 'penalty': 0.1},
            'encoder': {'architecture': 'mlp'},
            'decoder': {'architecture': 'mlp'},
            'pretrained': {
                'encoder_path': 'dummy_encoder.pt',
                'decoder_path': 'dummy_decoder.pt',
                'metric_path': 'dummy_metric.pt'
            },
            'sampling': {'type': 'basic', 'use_riemannian': False}
        })
        
        # Create a simple mock model
        model = RlVAE(config)
        
        # Initialize analyzer
        analyzer = EnhancedAnalyzer(
            model=model,
            device="cpu",
            output_dir="test_outputs"
        )
        
        print("✅ EnhancedAnalyzer initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize EnhancedAnalyzer: {e}")
        return False

def test_analysis_methods():
    """Test that analysis methods exist and are callable."""
    try:
        from evaluation.enhanced_analysis import EnhancedAnalyzer
        from models.modular_rlvae import ModularRiemannianFlowVAE as RlVAE
        from omegaconf import DictConfig
        
        # Create a simple mock config
        config = DictConfig({
            'input_dim': [1, 64, 64],  # [C, H, W]
            'latent_dim': 16,
            'n_flows': 4,
            'flow_hidden_size': 128,
            'flow_n_blocks': 2,
            'flow_n_hidden': 2,
            'epsilon': 1e-6,
            'beta': 1.0,
            'posterior': {'type': 'normal'},
            'loop': {'mode': 'open', 'penalty': 0.1},
            'encoder': {'architecture': 'mlp'},
            'decoder': {'architecture': 'mlp'},
            'pretrained': {
                'encoder_path': 'dummy_encoder.pt',
                'decoder_path': 'dummy_decoder.pt',
                'metric_path': 'dummy_metric.pt'
            },
            'sampling': {'type': 'basic', 'use_riemannian': False}
        })
        
        # Create a simple mock model
        model = RlVAE(config)
        
        # Initialize analyzer
        analyzer = EnhancedAnalyzer(
            model=model,
            device="cpu",
            output_dir="test_outputs"
        )
        
        # Check that methods exist
        assert hasattr(analyzer, 'analyze_generation')
        assert hasattr(analyzer, 'analyze_inference')
        assert hasattr(analyzer, 'analyze_geodesic_sampling')
        assert hasattr(analyzer, 'create_master_visualizations')
        assert hasattr(analyzer, 'run_comprehensive_analysis')
        
        print("✅ All analysis methods exist")
        return True
    except Exception as e:
        print(f"❌ Failed to test analysis methods: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Enhanced Analysis Module")
    print("=" * 50)
    
    tests = [
        test_enhanced_analyzer_import,
        test_enhanced_analyzer_initialization,
        test_analysis_methods
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced analysis module is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main()) 