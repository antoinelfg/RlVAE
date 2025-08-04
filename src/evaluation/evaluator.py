"""
Comprehensive Evaluation Module for Modular RlVAE
================================================

This module provides a unified evaluation framework that combines FID scores,
reconstruction metrics, generation quality assessment, and latent space analysis
for comprehensive model evaluation.

Key Features:
- FID score computation using pre-trained Inception network
- Reconstruction quality metrics (MSE, PSNR, SSIM)
- Generation quality assessment with multiple sampling methods
- Latent space analysis and geometry evaluation
- Comparative evaluation between different models
- Comprehensive reporting and visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from pathlib import Path
import warnings
from dataclasses import dataclass
import json
import time
from datetime import datetime
import wandb

from src.evaluation.fid_scorer import FIDScorer, create_fid_scorer
from src.generation.generator import RlVAEGenerator, GenerationConfig, create_generator
from src.inference.inference_pipeline import RlVAEInferencePipeline, InferenceConfig, create_inference_pipeline
from src.models.modular_rlvae import ModularRiemannianFlowVAE


@dataclass
class EvaluationConfig:
    """Configuration for comprehensive evaluation."""
    
    # Dataset parameters
    real_data_cache_key: str = "test_dataset"
    n_real_samples: int = 1000
    n_generated_samples: int = 1000
    
    # Generation evaluation
    generation_methods: List[str] = None  # ["geodesic", "enhanced", "basic", "standard"]
    sampler_types: List[str] = None  # ["working", "hmc", "official"]
    
    # FID evaluation
    compute_fid: bool = True
    fid_batch_size: int = 32
    fid_cache_dir: str = "data/fid_cache"
    
    # Reconstruction evaluation
    reconstruction_batch_size: int = 64
    n_reconstruction_samples: int = 500
    
    # Latent space analysis
    analyze_latent_space: bool = True
    latent_analysis_samples: int = 1000
    
    # Interpolation evaluation
    n_interpolations: int = 20
    interpolation_steps: int = 10
    interpolation_methods: List[str] = None  # ["linear", "spherical", "geodesic"]
    
    # Performance evaluation
    measure_timing: bool = True
    memory_profiling: bool = True
    
    # Output
    save_results: bool = True
    results_dir: str = "results/evaluation"
    save_visualizations: bool = True
    
    def __post_init__(self):
        """Set defaults for list fields."""
        if self.generation_methods is None:
            self.generation_methods = ["geodesic", "enhanced", "basic"]
        if self.sampler_types is None:
            self.sampler_types = ["working"]
        if self.interpolation_methods is None:
            self.interpolation_methods = ["linear", "spherical"]


class ModelEvaluator:
    """
    Comprehensive model evaluator for RlVAE models.
    
    Combines generation, inference, and FID scoring for complete evaluation.
    """
    
    def __init__(self, model: ModularRiemannianFlowVAE, device: Optional[torch.device] = None):
        """
        Initialize evaluator with a trained model.
        
        Args:
            model: Trained modular RlVAE model
            device: Device for computation
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        
        # Initialize components
        print("🚀 Initializing evaluation components...")
        
        self.generator = create_generator(model, device)
        self.inference_pipeline = create_inference_pipeline(model, device)
        self.fid_scorer = None  # Initialized on first use
        
        # Results storage
        self.evaluation_results = {}
        
        print("   ✅ Evaluator initialized successfully")
    
    def evaluate_comprehensive(self, real_images: torch.Tensor,
                             config: Optional[EvaluationConfig] = None) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of the model.
        
        Args:
            real_images: Real images for comparison [N, C, H, W]
            config: Evaluation configuration
            
        Returns:
            Dictionary containing all evaluation results
        """
        if config is None:
            config = EvaluationConfig()
        
        print("🎯 Starting comprehensive model evaluation...")
        print(f"   📊 Real images: {len(real_images)}")
        print(f"   🎨 Generation samples: {config.fid.n_generated_samples}")
        
        # Initialize FID scorer if needed
        if config.fid.compute_fid and self.fid_scorer is None:
            self.fid_scorer = create_fid_scorer(device=self.device, cache_dir=config.fid.fid_cache_dir)
        
        # Start evaluation
        start_time = time.time()
        results = {
            'model_info': self._get_model_info(),
            'timestamp': datetime.now().isoformat(),
            # Note: config excluded to avoid JSON serialization issues
        }
        
        # 1. Generation Evaluation
        print("\n📝 1. Generation Quality Evaluation")
        generation_results = self._evaluate_generation(real_images, config)
        results['generation'] = generation_results
        results['n_generated_samples'] = config.fid.n_generated_samples
        
        # 2. Reconstruction Evaluation
        print("\n📝 2. Reconstruction Quality Evaluation")
        reconstruction_results = self._evaluate_reconstruction(real_images, config)
        results['reconstruction'] = reconstruction_results
        
        # 3. Latent Space Analysis
        if config.analyze_latent_space:
            print("\n📝 3. Latent Space Analysis")
            latent_results = self._evaluate_latent_space(real_images, config)
            results['latent_space'] = latent_results
        
        # 4. Interpolation Evaluation
        print("\n📝 4. Interpolation Quality Evaluation")
        interpolation_results = self._evaluate_interpolation(config)
        results['interpolation'] = interpolation_results
        
        # 5. Performance Evaluation
        if config.measure_timing:
            print("\n📝 5. Performance Evaluation")
            performance_results = self._evaluate_performance(config)
            results['performance'] = performance_results
        
        # Summary
        total_time = time.time() - start_time
        results['evaluation_time'] = total_time
        results['summary'] = self._create_summary(results)
        
        print(f"\n✅ Comprehensive evaluation completed in {total_time:.2f}s")
        
        # Save results if requested
        if config.save_results:
            self._save_results(results, config)
        
        # Store in instance
        self.evaluation_results = results
        # Log to wandb
        log_generation_and_inference_to_wandb(results, prefix="eval")
        return results
    
    def _evaluate_generation(self, real_images: torch.Tensor, 
                           config: EvaluationConfig) -> Dict[str, Any]:
        """Evaluate generation quality using FID and other metrics."""
        results = {}
        
        # Cache real statistics for FID if requested
        if config.fid.compute_fid:
            print("   🗄️ Caching real image statistics for FID...")
            real_sample = real_images[:config.fid.n_real_samples]
            self.fid_scorer.cache_real_statistics(real_sample, config.fid.real_data_cache_key)
        
        # Evaluate different generation methods
        for sampler_type in config.sampler_types:
            sampler_results = {}
            
            for method in config.generation_methods:
                print(f"   🎨 Evaluating {sampler_type} sampler with {method} method...")
                
                try:
                    # Generate samples
                    gen_config = GenerationConfig(
                        num_samples=config.fid.n_generated_samples,
                        sampling_method=method,
                        sampler_type=sampler_type,
                        batch_size=32,
                        sequence_length=1,  # Single images for FID
                    )
                    
                    generation_result = self.generator.generate_from_prior(gen_config)
                    generated_images = generation_result['images']
                    
                    # Remove sequence dimension for FID
                    if generated_images.dim() == 5:
                        # If shape is [B, S, C, H, W], flatten to [B*S, C, H, W]
                        if generated_images.shape[1] == 1:
                            generated_images = generated_images[:, 0]
                        else:
                            B, S, C, H, W = generated_images.shape
                            generated_images = generated_images.reshape(B * S, C, H, W)
                    
                    method_results = {
                        'n_samples': len(generated_images),
                        'n_generated_samples': len(generated_images),
                        'generation_info': generation_result['generation_info'],
                    }
                    
                    # Compute FID if requested
                    if config.fid.compute_fid:
                        fid_result = self.fid_scorer.evaluate_with_cached_real(
                            generated_images, config.fid.real_data_cache_key, 
                            batch_size=config.fid.fid_batch_size
                        )
                        
                        if fid_result:
                            method_results['fid'] = fid_result
                            print(f"      📊 FID Score: {fid_result['fid_score']:.3f}")
                        else:
                            print("      ❌ FID computation failed")
                    
                    # Additional generation metrics
                    gen_metrics = self._compute_generation_metrics(generated_images)
                    method_results['generation_metrics'] = gen_metrics
                    
                    sampler_results[method] = method_results
                    
                except Exception as e:
                    print(f"      ❌ Generation evaluation failed: {e}")
                    sampler_results[method] = {'error': str(e)}
            
            results[sampler_type] = sampler_results
        
        return results
    
    def _evaluate_reconstruction(self, real_images: torch.Tensor,
                               config: EvaluationConfig) -> Dict[str, Any]:
        """Evaluate reconstruction quality."""
        print("   🔄 Testing reconstruction quality...")
        
        # Sample subset for reconstruction evaluation
        n_samples = min(config.n_reconstruction_samples, len(real_images))
        sample_images = real_images[:n_samples]
        
        # Perform encode-reconstruct cycle
        inference_config = InferenceConfig(
            batch_size=config.reconstruction_batch_size,
            use_mean=False,  # Use sampling
            return_uncertainties=True,
        )
        
        reconstruction_result = self.inference_pipeline.encode_and_reconstruct(
            sample_images, inference_config
        )
        
        # Extract results
        metrics = reconstruction_result['reconstruction_metrics']
        uncertainties = reconstruction_result.get('uncertainties')
        
        results = {
            'n_samples': n_samples,
            'reconstruction_metrics': metrics,
            'latent_statistics': self._compute_latent_statistics(
                reconstruction_result['latents']
            ),
        }
        
        if uncertainties:
            results['uncertainty_analysis'] = self._analyze_uncertainties(uncertainties)
        
        print(f"      📊 Reconstruction MSE: {metrics['mse']:.6f}")
        print(f"      📊 Reconstruction PSNR: {metrics['psnr']:.2f} dB")
        print(f"      📊 Reconstruction SSIM: {metrics['ssim']:.4f}")
        
        return results
    
    def _evaluate_latent_space(self, real_images: torch.Tensor,
                             config: EvaluationConfig) -> Dict[str, Any]:
        """Evaluate latent space properties."""
        print("   🌌 Analyzing latent space...")
        
        # Sample subset for latent analysis
        n_samples = min(config.latent_analysis_samples, len(real_images))
        sample_images = real_images[:n_samples]
        
        # Encode to latent space
        inference_config = InferenceConfig(batch_size=64, use_mean=True)
        encoding_result = self.inference_pipeline.encode_images(sample_images, inference_config)
        
        # Analyze latent space
        latent_analysis = self.inference_pipeline.analyze_latent_space(
            encoding_result['latents']
        )
        
        # Additional latent space metrics
        latents_flat = encoding_result['latents'].view(-1, encoding_result['latents'].shape[-1])
        
        results = {
            'n_samples': n_samples,
            'latent_analysis': latent_analysis,
            'activation_statistics': self._compute_activation_statistics(latents_flat),
        }
        
        # Riemannian-specific analysis
        if hasattr(self.model, 'G_inv'):
            riemannian_metrics = self._evaluate_riemannian_properties(latents_flat)
            results['riemannian_analysis'] = riemannian_metrics
        
        effective_dim = latent_analysis['dimensionality']['effective_dim']
        intrinsic_dim = latent_analysis['dimensionality']['intrinsic_dim']
        print(f"      📊 Effective dimensionality: {effective_dim:.2f}")
        print(f"      📊 Intrinsic dimensionality: {intrinsic_dim}")
        
        return results
    
    def _evaluate_interpolation(self, config: EvaluationConfig) -> Dict[str, Any]:
        """Evaluate interpolation quality."""
        print("   🔄 Testing interpolation quality...")
        
        results = {}
        
        # Generate random latent pairs for interpolation
        latent_dim = self.model.latent_dim
        
        for method in config.interpolation_methods:
            print(f"      🎯 Testing {method} interpolation...")
            
            try:
                method_results = []
                
                for i in range(config.n_interpolations):
                    # Sample random latent points
                    latent1 = torch.randn(latent_dim, device=self.device)
                    latent2 = torch.randn(latent_dim, device=self.device)
                    
                    # Perform interpolation
                    interp_result = self.generator.interpolate(
                        latent1, latent2, 
                        num_steps=config.interpolation_steps,
                        method=method
                    )
                    
                    # Compute interpolation smoothness
                    smoothness = self._compute_interpolation_smoothness(
                        interp_result['images']
                    )
                    
                    method_results.append({
                        'smoothness': smoothness,
                        'num_steps': config.interpolation_steps,
                    })
                
                # Aggregate results
                smoothness_scores = [r['smoothness'] for r in method_results]
                results[method] = {
                    'n_interpolations': config.n_interpolations,
                    'mean_smoothness': np.mean(smoothness_scores),
                    'std_smoothness': np.std(smoothness_scores),
                    'individual_results': method_results,
                }
                
                print(f"         📊 Mean smoothness: {np.mean(smoothness_scores):.4f}")
                
            except Exception as e:
                print(f"         ❌ {method} interpolation failed: {e}")
                results[method] = {'error': str(e)}
        
        return results
    
    def _evaluate_performance(self, config: EvaluationConfig) -> Dict[str, Any]:
        """Evaluate model performance and efficiency."""
        print("   ⚡ Measuring performance...")
        
        results = {}
        
        # Generation speed
        print("      🎨 Testing generation speed...")
        gen_config = GenerationConfig(num_samples=100, batch_size=10)
        
        start_time = time.time()
        generation_result = self.generator.generate_from_prior(gen_config)
        generation_time = time.time() - start_time
        
        results['generation_speed'] = {
            'total_time': generation_time,
            'samples_per_second': 100 / generation_time,
            'time_per_sample': generation_time / 100,
        }
        
        # Inference speed
        print("      🔍 Testing inference speed...")
        test_images = torch.randn(100, 3, 64, 64, device=self.device)  # Dummy images
        inference_config = InferenceConfig(batch_size=20)
        
        start_time = time.time()
        encoding_result = self.inference_pipeline.encode_images(test_images, inference_config)
        inference_time = time.time() - start_time
        
        results['inference_speed'] = {
            'total_time': inference_time,
            'samples_per_second': 100 / inference_time,
            'time_per_sample': inference_time / 100,
        }
        
        # Memory usage
        if config.memory_profiling and torch.cuda.is_available():
            memory_stats = torch.cuda.memory_stats(self.device)
            results['memory_usage'] = {
                'peak_memory_mb': memory_stats.get('allocated_bytes.all.peak', 0) / 1024**2,
                'current_memory_mb': torch.cuda.memory_allocated(self.device) / 1024**2,
            }
        
        print(f"         📊 Generation: {results['generation_speed']['samples_per_second']:.2f} samples/s")
        print(f"         📊 Inference: {results['inference_speed']['samples_per_second']:.2f} samples/s")
        
        return results
    
    def _compute_generation_metrics(self, images: torch.Tensor) -> Dict[str, float]:
        """Compute additional generation quality metrics."""
        with torch.no_grad():
            # Image statistics
            mean_pixel = torch.mean(images).item()
            std_pixel = torch.std(images).item()
            
            # Dynamic range
            min_val = torch.min(images).item()
            max_val = torch.max(images).item()
            dynamic_range = max_val - min_val
            
            # Saturation (pixels at extremes)
            n_pixels = images.numel()
            n_saturated_low = torch.sum(images <= 0.01).item()
            n_saturated_high = torch.sum(images >= 0.99).item()
            saturation_ratio = (n_saturated_low + n_saturated_high) / n_pixels
            
        return {
            'mean_pixel_value': mean_pixel,
            'std_pixel_value': std_pixel,
            'dynamic_range': dynamic_range,
            'saturation_ratio': saturation_ratio,
            'min_value': min_val,
            'max_value': max_val,
        }
    
    def _compute_latent_statistics(self, latents: torch.Tensor) -> Dict[str, float]:
        """Compute latent code statistics."""
        latents_flat = latents.view(-1, latents.shape[-1])
        
        with torch.no_grad():
            return {
                'mean_norm': torch.norm(latents_flat, dim=1).mean().item(),
                'std_norm': torch.norm(latents_flat, dim=1).std().item(),
                'mean_activation': torch.mean(latents_flat).item(),
                'std_activation': torch.std(latents_flat).item(),
                'sparsity': (torch.abs(latents_flat) < 0.1).float().mean().item(),
            }
    
    def _analyze_uncertainties(self, uncertainties: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyze uncertainty statistics."""
        return {
            'mean_total_uncertainty': torch.mean(uncertainties['total_uncertainty']).item(),
            'std_total_uncertainty': torch.std(uncertainties['total_uncertainty']).item(),
            'mean_max_uncertainty': torch.mean(uncertainties['max_uncertainty']).item(),
            'std_max_uncertainty': torch.std(uncertainties['max_uncertainty']).item(),
        }
    
    def _compute_activation_statistics(self, latents: torch.Tensor) -> Dict[str, float]:
        """Compute activation pattern statistics."""
        with torch.no_grad():
            # Dead neurons (never activate)
            dead_threshold = 0.01
            dead_ratio = (torch.max(torch.abs(latents), dim=0)[0] < dead_threshold).float().mean().item()
            
            # Active ratio per dimension
            active_threshold = 0.1
            active_ratios = (torch.abs(latents) > active_threshold).float().mean(dim=0)
            
            return {
                'dead_neuron_ratio': dead_ratio,
                'mean_active_ratio': torch.mean(active_ratios).item(),
                'std_active_ratio': torch.std(active_ratios).item(),
                'min_active_ratio': torch.min(active_ratios).item(),
                'max_active_ratio': torch.max(active_ratios).item(),
            }
    
    def _evaluate_riemannian_properties(self, latents: torch.Tensor) -> Dict[str, float]:
        """Evaluate Riemannian geometric properties."""
        print("      🌀 Analyzing Riemannian properties...")
        
        # Sample subset for efficiency
        n_samples = min(200, len(latents))
        sample_latents = latents[:n_samples]
        
        with torch.no_grad():
            # Compute metric tensors
            G_inv = self.model.G_inv(sample_latents)
            
            # Metric properties
            eigenvals = torch.linalg.eigvals(G_inv).real
            condition_numbers = eigenvals.max(dim=1)[0] / eigenvals.min(dim=1)[0]
            determinants = torch.linalg.det(G_inv)
            
            return {
                'mean_condition_number': torch.mean(condition_numbers).item(),
                'std_condition_number': torch.std(condition_numbers).item(),
                'mean_determinant': torch.mean(determinants).item(),
                'std_determinant': torch.std(determinants).item(),
                'min_eigenvalue': torch.min(eigenvals).item(),
                'max_eigenvalue': torch.max(eigenvals).item(),
            }
    
    def _compute_interpolation_smoothness(self, interpolation_images: torch.Tensor) -> float:
        """Compute smoothness of interpolation sequence."""
        # Remove sequence dimension if present
        if interpolation_images.dim() == 5:
            interpolation_images = interpolation_images[:, 0]  # [steps, C, H, W]
        
        # Compute frame-to-frame differences
        diffs = torch.diff(interpolation_images, dim=0)
        
        # Flatten spatial dimensions and compute L2 norm
        diffs_flat = diffs.view(diffs.shape[0], -1)  # [steps-1, C*H*W]
        smoothness = torch.mean(torch.norm(diffs_flat, dim=1)).item()
        
        return smoothness
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': type(self.model).__name__,
            'latent_dim': self.model.latent_dim,
            'input_dim': self.model.input_dim,
            'n_flows': self.model.n_flows,
            'has_metric_tensor': hasattr(self.model, 'G'),
            'device': str(self.device),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create evaluation summary."""
        summary = {
            'model_info': results['model_info'],
            'evaluation_time': results['evaluation_time'],
        }
        
        # Extract key metrics
        if 'generation' in results:
            generation_fids = []
            for sampler_type, sampler_results in results['generation'].items():
                for method, method_results in sampler_results.items():
                    if 'fid' in method_results:
                        generation_fids.append(method_results['fid']['fid_score'])
            
            if generation_fids:
                summary['best_fid_score'] = min(generation_fids)
                summary['mean_fid_score'] = np.mean(generation_fids)
        
        if 'reconstruction' in results:
            recon_metrics = results['reconstruction']['reconstruction_metrics']
            summary['reconstruction_mse'] = recon_metrics['mse']
            summary['reconstruction_psnr'] = recon_metrics['psnr']
        
        if 'latent_space' in results:
            latent_dim_info = results['latent_space']['latent_analysis']['dimensionality']
            summary['effective_dimensionality'] = latent_dim_info['effective_dim']
            summary['intrinsic_dimensionality'] = latent_dim_info['intrinsic_dim']
        
        return summary
    
    def _save_results(self, results: Dict[str, Any], config: EvaluationConfig):
        """Save evaluation results."""
        results_dir = Path(config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{timestamp}.json"
        filepath = results_dir / filename
        
        # Convert tensors to lists for JSON serialization
        results_serializable = self._make_json_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"   💾 Results saved to: {filepath}")
    
    def _make_json_serializable(self, obj):
        """Convert torch tensors and numpy arrays to JSON-serializable format."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def log_generation_and_inference_to_wandb(results, prefix="eval"):  # Utility for logging
    # Log generation metrics
    if 'generation' in results:
        for sampler_type, sampler_results in results['generation'].items():
            for method, method_results in sampler_results.items():
                # FID
                if 'fid' in method_results and 'fid_score' in method_results['fid']:
                    wandb.log({f"{prefix}/fid/{sampler_type}/{method}": method_results['fid']['fid_score']})
                # Generation metrics
                if 'generation_metrics' in method_results:
                    for metric_name, value in method_results['generation_metrics'].items():
                        wandb.log({f"{prefix}/generation/{sampler_type}/{method}/{metric_name}": value})
    # Log inference/latent metrics
    if 'latent_space' in results:
        latent = results['latent_space']
        if 'latent_analysis' in latent and 'dimensionality' in latent['latent_analysis']:
            wandb.log({f"{prefix}/latent/effective_dim": latent['latent_analysis']['dimensionality']['effective_dim']})
            wandb.log({f"{prefix}/latent/intrinsic_dim": latent['latent_analysis']['dimensionality']['intrinsic_dim']})
        if 'activation_statistics' in latent:
            for k, v in latent['activation_statistics'].items():
                wandb.log({f"{prefix}/latent/{k}": v})
    if 'reconstruction' in results:
        for k, v in results['reconstruction']['reconstruction_metrics'].items():
            wandb.log({f"{prefix}/reconstruction/{k}": v})


def create_evaluator(model: ModularRiemannianFlowVAE,
                    device: Optional[torch.device] = None) -> ModelEvaluator:
    """
    Factory function to create model evaluator.
    
    Args:
        model: Trained modular RlVAE model
        device: Device for computation
        
    Returns:
        Configured evaluator instance
    """
    return ModelEvaluator(model=model, device=device) 