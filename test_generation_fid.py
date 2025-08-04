#!/usr/bin/env python3
"""
Test Script for Generation and FID Evaluation
============================================

This script demonstrates and tests the new generation and FID evaluation features
integrated into the modular RlVAE model. It provides a comprehensive test suite
to verify that all components work correctly together.

Usage:
    python test_generation_fid.py --model-path path/to/model.ckpt --dataset cifar10
    python test_generation_fid.py --config conf/model/stage1_vanilla_vae_mlp_ld32.yaml
"""

import argparse
import sys
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.modular_rlvae import ModularRiemannianFlowVAE
from src.evaluation.fid_scorer import create_fid_scorer
from src.generation.generator import GenerationConfig
from src.inference.inference_pipeline import InferenceConfig
from src.evaluation.evaluator import EvaluationConfig


def get_test_dataset(dataset_name: str = "cifar10", subset_size: int = 1000):
    """Get test dataset for evaluation."""
    print(f"🔍 Loading {dataset_name} dataset (subset: {subset_size})...")
    
    # Standard transform - ensure consistent 64x64 size
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to standard size
        transforms.ToTensor(),
        # Note: Adjust normalization based on your training
    ])
    
    if dataset_name.lower() == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
    elif dataset_name.lower() == "mnist":
        # Convert MNIST to RGB for consistency
        transform_mnist = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform_mnist
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Create subset for testing
    indices = torch.randperm(len(dataset))[:subset_size]
    subset = Subset(dataset, indices)
    
    return DataLoader(subset, batch_size=32, shuffle=False)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> ModularRiemannianFlowVAE:
    """Load model from checkpoint."""
    print(f"📂 Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif 'hyper_parameters' in checkpoint:
        config = checkpoint['hyper_parameters'].get('config')
    else:
        raise ValueError("No config found in checkpoint")
    
    # Create model
    model = ModularRiemannianFlowVAE(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded successfully")
    print(f"   📊 Model info: {model.get_model_summary()}")
    
    return model


def load_model_from_config(config_path: str, device: torch.device) -> ModularRiemannianFlowVAE:
    """Load model from config (for testing architecture without trained weights)."""
    print(f"⚙️ Creating model from config {config_path}...")
    
    with hydra.initialize(config_path="conf", version_base=None):
        cfg = hydra.compose(config_name="config", overrides=[f"model={config_path}"])
    
    model = ModularRiemannianFlowVAE(cfg.model)
    model.to(device)
    model.eval()
    
    print(f"   ✅ Model created successfully (untrained)")
    print(f"   📊 Model info: {model.get_model_summary()}")
    
    return model


def test_generation_methods(model: ModularRiemannianFlowVAE):
    """Test different generation methods."""
    print("\n🎨 Testing Generation Methods")
    print("=" * 50)
    
    generation_methods = ["geodesic", "enhanced", "basic", "standard"]
    sampler_types = ["working"]  # Add more if available
    
    results = {}
    
    for sampler_type in sampler_types:
        print(f"\n📝 Testing {sampler_type} sampler...")
        sampler_results = {}
        
        for method in generation_methods:
            print(f"   🎯 Testing {method} generation...")
            
            try:
                # Test small batch for speed
                gen_result = model.generate_samples(
                    num_samples=16,
                    method=method,
                    sampler_type=sampler_type,
                    sequence_length=1
                )
                
                images = gen_result['images']
                if images.dim() == 5:
                    images = images[:, 0]  # Remove sequence dimension
                
                # Basic checks
                assert images.shape[0] == 16, f"Expected 16 samples, got {images.shape[0]}"
                assert 0 <= images.min() and images.max() <= 1, "Images not in [0,1] range"
                
                sampler_results[method] = {
                    'success': True,
                    'shape': list(images.shape),
                    'value_range': [images.min().item(), images.max().item()],
                    'mean_pixel': images.mean().item(),
                }
                
                print(f"      ✅ Success - Shape: {images.shape}, Range: [{images.min():.3f}, {images.max():.3f}]")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                sampler_results[method] = {'success': False, 'error': str(e)}
        
        results[sampler_type] = sampler_results
    
    return results


def test_fid_computation(model: ModularRiemannianFlowVAE, real_images: torch.Tensor):
    """Test FID score computation."""
    print("\n📊 Testing FID Score Computation")
    print("=" * 50)
    
    try:
        # Test with small sample for speed
        real_sample = real_images[:100]  # Use first 100 real images
        
        print(f"   🔍 Computing FID with {len(real_sample)} real images...")
        
        fid_result = model.compute_fid_score(
            real_images=real_sample,
            num_generated=100,  # Generate 100 samples
            cache_key="test_evaluation",
            sampling_method="geodesic",
            sampler_type="working"
        )
        
        if fid_result and 'fid_score' in fid_result:
            fid_score = fid_result['fid_score']
            print(f"   ✅ FID Score computed successfully: {fid_score:.3f}")
            
            # Additional metrics from FID computation
            if 'real_statistics' in fid_result:
                print(f"   📊 Real images statistics cached")
            if 'generated_statistics' in fid_result:
                print(f"   📊 Generated images statistics computed")
            
            return {
                'success': True,
                'fid_score': fid_score,
                'full_result': fid_result
            }
        else:
            print(f"   ❌ FID computation failed: Invalid result")
            return {'success': False, 'error': 'Invalid FID result'}
            
    except Exception as e:
        print(f"   ❌ FID computation failed: {e}")
        return {'success': False, 'error': str(e)}


def test_reconstruction_evaluation(model: ModularRiemannianFlowVAE, real_images: torch.Tensor):
    """Test reconstruction evaluation."""
    print("\n🔄 Testing Reconstruction Evaluation")
    print("=" * 50)
    
    try:
        # Test with small sample
        test_sample = real_images[:50]
        
        print(f"   🔍 Evaluating reconstruction on {len(test_sample)} images...")
        
        recon_result = model.evaluate_reconstruction(
            test_images=test_sample,
            batch_size=16
        )
        
        metrics = recon_result['reconstruction_metrics']
        
        print(f"   ✅ Reconstruction evaluation successful!")
        print(f"      📊 MSE: {metrics['mse']:.6f}")
        print(f"      📊 PSNR: {metrics['psnr']:.2f} dB")
        print(f"      📊 SSIM: {metrics['ssim']:.4f}")
        print(f"      📊 L1 Loss: {metrics['l1_loss']:.6f}")
        
        # Latent statistics
        if 'latent_statistics' in recon_result:
            latent_stats = recon_result['latent_statistics']
            print(f"      📊 Mean latent norm: {latent_stats['mean_norm']:.4f}")
            print(f"      📊 Std latent norm: {latent_stats['std_norm']:.4f}")
        
        return {
            'success': True,
            'metrics': metrics,
            'full_result': recon_result
        }
        
    except Exception as e:
        print(f"   ❌ Reconstruction evaluation failed: {e}")
        return {'success': False, 'error': str(e)}


def test_inference_pipeline(model: ModularRiemannianFlowVAE, real_images: torch.Tensor):
    """Test inference pipeline functionality."""
    print("\n🧠 Testing Inference Pipeline")
    print("=" * 50)
    
    try:
        # Create inference pipeline
        inference_pipeline = model.create_inference_pipeline()
        
        test_sample = real_images[:20]
        print(f"   🔍 Testing encoding/decoding cycle on {len(test_sample)} images...")
        
        # Test encoding
        config = InferenceConfig(batch_size=8, return_uncertainties=True)
        
        encoding_result = inference_pipeline.encode_images(test_sample, config)
        print(f"      ✅ Encoding successful - Latent shape: {encoding_result['latents'].shape}")
        
        # Test reconstruction
        reconstruction_result = inference_pipeline.reconstruct_from_latents(
            encoding_result['latents'], config
        )
        print(f"      ✅ Reconstruction successful - Image shape: {reconstruction_result['reconstructions'].shape}")
        
        # Test full cycle
        full_result = inference_pipeline.encode_and_reconstruct(test_sample, config)
        metrics = full_result['reconstruction_metrics']
        print(f"      ✅ Full cycle successful - PSNR: {metrics['psnr']:.2f} dB")
        
        # Test latent analysis
        latent_analysis = inference_pipeline.analyze_latent_space(encoding_result['latents'])
        effective_dim = latent_analysis['dimensionality']['effective_dim']
        print(f"      ✅ Latent analysis successful - Effective dim: {effective_dim:.2f}")
        
        return {
            'success': True,
            'encoding_shape': list(encoding_result['latents'].shape),
            'reconstruction_psnr': metrics['psnr'],
            'effective_dimensionality': effective_dim
        }
        
    except Exception as e:
        print(f"   ❌ Inference pipeline test failed: {e}")
        return {'success': False, 'error': str(e)}


def test_comprehensive_evaluation(model: ModularRiemannianFlowVAE, real_images: torch.Tensor):
    """Test comprehensive evaluation."""
    print("\n🎯 Testing Comprehensive Evaluation")
    print("=" * 50)
    
    try:
        # Use small sample for speed
        real_sample = real_images[:100]
        
        print(f"   🔍 Running comprehensive evaluation on {len(real_sample)} images...")
        
        eval_result = model.comprehensive_evaluation(
            real_images=real_sample,
            n_real_samples=50,
            n_generated_samples=50,
            n_reconstruction_samples=30,
            generation_methods=["geodesic", "basic"],
            compute_fid=True,
            analyze_latent_space=True,
            n_interpolations=5,
            measure_timing=True
        )
        
        print(f"   ✅ Comprehensive evaluation completed!")
        
        # Print summary
        summary = eval_result['summary']
        print(f"      📊 Evaluation time: {eval_result['evaluation_time']:.2f}s")
        
        if 'best_fid_score' in summary:
            print(f"      📊 Best FID score: {summary['best_fid_score']:.3f}")
        
        if 'reconstruction_psnr' in summary:
            print(f"      📊 Reconstruction PSNR: {summary['reconstruction_psnr']:.2f} dB")
        
        if 'effective_dimensionality' in summary:
            print(f"      📊 Effective dimensionality: {summary['effective_dimensionality']:.2f}")
        
        return {
            'success': True,
            'summary': summary,
            'evaluation_time': eval_result['evaluation_time']
        }
        
    except Exception as e:
        print(f"   ❌ Comprehensive evaluation failed: {e}")
        return {'success': False, 'error': str(e)}


def run_all_tests(model: ModularRiemannianFlowVAE, real_images: torch.Tensor):
    """Run all tests and compile results."""
    print("\n🚀 Running All Tests")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Generation methods
    results['generation'] = test_generation_methods(model)
    
    # Test 2: FID computation
    results['fid'] = test_fid_computation(model, real_images)
    
    # Test 3: Reconstruction evaluation
    results['reconstruction'] = test_reconstruction_evaluation(model, real_images)
    
    # Test 4: Inference pipeline
    results['inference'] = test_inference_pipeline(model, real_images)
    
    # Test 5: Comprehensive evaluation
    results['comprehensive'] = test_comprehensive_evaluation(model, real_images)
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, test_result in results.items():
        if isinstance(test_result, dict) and 'success' in test_result:
            # Simple success/failure test
            if test_result.get('success', False):
                print(f"   ✅ {test_name.capitalize()}: PASSED")
                passed_tests += 1
            else:
                print(f"   ❌ {test_name.capitalize()}: FAILED - {test_result.get('error', 'Unknown error')}")
        elif isinstance(test_result, dict):
            # Nested structure (like generation tests)
            sub_passed = 0
            sub_total = 0
            for sampler_type, sampler_results in test_result.items():
                if isinstance(sampler_results, dict):
                    for method, method_result in sampler_results.items():
                        sub_total += 1
                        if isinstance(method_result, dict) and method_result.get('success', False):
                            sub_passed += 1
            
            if sub_passed == sub_total and sub_total > 0:
                print(f"   ✅ {test_name.capitalize()}: PASSED ({sub_passed}/{sub_total})")
                passed_tests += 1
            elif sub_total > 0:
                print(f"   ❌ {test_name.capitalize()}: PARTIAL ({sub_passed}/{sub_total})")
            else:
                # Might be a different dict structure, check for 'success' key
                if test_result.get('success', False):
                    print(f"   ✅ {test_name.capitalize()}: PASSED")
                    passed_tests += 1
                else:
                    print(f"   ❌ {test_name.capitalize()}: FAILED - {test_result.get('error', 'Unknown error')}")
        
        total_tests += 1
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests PASSED! Generation and FID evaluation are working correctly.")
    else:
        print("⚠️ Some tests failed. Check the errors above for debugging.")
    
    return results


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Generation and FID Evaluation")
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Model config name (for testing without weights)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"], help="Test dataset")
    parser.add_argument("--subset-size", type=int, default=500, help="Dataset subset size")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ Using device: {device}")
    
    try:
        # Load model
        if args.model_path:
            model = load_model_from_checkpoint(args.model_path, device)
        elif args.config:
            model = load_model_from_config(args.config, device)
        else:
            print("❌ Please provide either --model-path or --config")
            return
        
        # Load test data
        dataloader = get_test_dataset(args.dataset, args.subset_size)
        
        # Collect real images
        real_images = []
        for batch_idx, (images, _) in enumerate(dataloader):
            real_images.append(images)
            if len(real_images) * images.shape[0] >= args.subset_size:
                break
        
        real_images = torch.cat(real_images, dim=0)[:args.subset_size]
        real_images = real_images.to(device)
        
        print(f"📊 Loaded {len(real_images)} real images for testing")
        
        # Run tests
        results = run_all_tests(model, real_images)
        
        # Optional: Save results
        import json
        results_path = Path("test_results_generation_fid.json")
        
        # Convert tensors to lists for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif hasattr(obj, '__dict__'):
                # Handle dataclass/object instances (like EvaluationConfig)
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items() if k != 'config'}  # Skip config objects
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            else:
                return obj
        
        # Remove non-serializable objects before conversion
        results_copy = dict(results)
        if 'comprehensive' in results_copy and 'config' in results_copy['comprehensive']:
            del results_copy['comprehensive']['config']
        
        serializable_results = make_serializable(results_copy)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Test results saved to: {results_path}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 