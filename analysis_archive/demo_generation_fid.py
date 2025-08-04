#!/usr/bin/env python3
"""
Demo: Generation and FID Evaluation with Modular RlVAE
======================================================

This is a simple demonstration script showing how to use the new generation
and FID evaluation features integrated into the modular RlVAE model.

Usage:
    python demo_generation_fid.py
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.modular_rlvae import ModularRiemannianFlowVAE


def demo_generation_and_fid():
    """Demo the new generation and FID features."""
    print("🎨 Demo: Generation and FID Evaluation")
    print("=" * 50)
    
    # 1. Load a trained model (replace with your actual model)
    print("📂 Loading model...")
    # model = torch.load('path/to/your/trained_model.ckpt')
    # For this demo, we'll show the API without actually loading
    print("   (Replace this with your actual model loading)")
    
    # 2. Simple generation
    print("\n🎯 Simple Generation Example:")
    print("""
    # Generate 64 samples using geodesic sampling
    generated = model.generate_samples(
        num_samples=64,
        method="geodesic",
        sampler_type="working"
    )
    
    images = generated['images']  # Shape: [64, 1, C, H, W] or [64, C, H, W]
    print(f"Generated {len(images)} samples")
    """)
    
    # 3. FID score computation
    print("\n📊 FID Score Computation Example:")
    print("""
    # Compute FID score against real images
    real_images = load_your_test_dataset()  # Shape: [N, C, H, W]
    
    fid_result = model.compute_fid_score(
        real_images=real_images,
        num_generated=1000,
        cache_key="my_evaluation"
    )
    
    fid_score = fid_result['fid_score']
    print(f"FID Score: {fid_score:.3f}")
    """)
    
    # 4. Reconstruction evaluation
    print("\n🔄 Reconstruction Evaluation Example:")
    print("""
    # Evaluate reconstruction quality
    test_images = real_images[:100]  # Test subset
    
    recon_result = model.evaluate_reconstruction(
        test_images=test_images,
        batch_size=32
    )
    
    metrics = recon_result['reconstruction_metrics']
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    """)
    
    # 5. Comprehensive evaluation
    print("\n🎯 Comprehensive Evaluation Example:")
    print("""
    # Run complete evaluation suite
    eval_results = model.comprehensive_evaluation(
        real_images=real_images,
        n_generated_samples=1000,
        generation_methods=["geodesic", "enhanced"],
        compute_fid=True,
        analyze_latent_space=True
    )
    
    summary = eval_results['summary']
    print(f"Best FID: {summary['best_fid_score']:.3f}")
    print(f"Reconstruction PSNR: {summary['reconstruction_psnr']:.2f} dB")
    """)
    
    # 6. Advanced usage
    print("\n🔬 Advanced Usage Examples:")
    print("""
    # Create individual components for custom workflows
    generator = model.create_generator()
    inference_pipeline = model.create_inference_pipeline()
    evaluator = model.create_evaluator()
    
    # Custom generation config
    from src.generation.generator import GenerationConfig
    config = GenerationConfig(
        num_samples=100,
        sampling_method="geodesic",
        sampler_type="working",
        sequence_length=10,  # For sequence generation
        temperature=0.8
    )
    result = generator.generate_from_prior(config)
    
    # Custom inference
    from src.inference.inference_pipeline import InferenceConfig
    config = InferenceConfig(
        batch_size=64,
        use_mean=False,  # Use sampling vs posterior mean
        return_uncertainties=True
    )
    encoded = inference_pipeline.encode_images(test_images, config)
    
    # Latent space analysis
    analysis = inference_pipeline.analyze_latent_space(encoded['latents'])
    print(f"Effective dimensionality: {analysis['dimensionality']['effective_dim']}")
    """)
    
    print("\n✅ Demo complete! Check test_generation_fid.py for a full test suite.")


if __name__ == "__main__":
    demo_generation_and_fid() 