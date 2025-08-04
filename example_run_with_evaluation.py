#!/usr/bin/env python3
"""
Example: Running RlVAE with Integrated FID Evaluation
====================================================

This script demonstrates how to use the new integrated evaluation system
that includes FID scoring, generation analysis, and comprehensive evaluation
during your RlVAE training pipeline.

Usage Examples:
--------------

1. Run with minimal evaluation (fast for development):
   python example_run_with_evaluation.py evaluation=minimal

2. Run with comprehensive evaluation (full analysis):
   python example_run_with_evaluation.py evaluation=comprehensive

3. Run with custom evaluation settings:
   python example_run_with_evaluation.py evaluation=default evaluation.fid.n_generated_samples=2000

4. Disable evaluation entirely (maximum speed):
   python example_run_with_evaluation.py evaluation=disabled

5. Run evaluation only at the end:
   python example_run_with_evaluation.py evaluation.run_during_training=false evaluation.run_at_end_only=true

6. Use different generation methods for evaluation:
   python example_run_with_evaluation.py evaluation.generation.methods="[geodesic,enhanced]"
"""

import os
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import hydra
from omegaconf import DictConfig
from run_experiment import ExperimentRunner


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main function that runs RlVAE experiments with integrated evaluation.
    
    The evaluation system will automatically:
    1. Collect real images during training/validation for FID computation
    2. Run FID scoring using the configured generation methods
    3. Evaluate reconstruction quality and latent space properties
    4. Log all metrics to WandB
    5. Save detailed results and visualizations
    
    Evaluation Configuration Options:
    --------------------------------
    - evaluation=minimal: Fast evaluation for development
    - evaluation=default: Balanced evaluation for most use cases
    - evaluation=comprehensive: Full evaluation for research
    - evaluation=disabled: No evaluation (maximum training speed)
    
    Key Configuration Overrides:
    ----------------------------
    - evaluation.fid.enabled=true/false: Enable/disable FID scoring
    - evaluation.generation.methods=[geodesic,enhanced,basic]: Choose generation methods
    - evaluation.run_during_training=true/false: Run evaluation during training
    - evaluation.run_at_end_only=true/false: Only run at end of training
    - evaluation.fid.n_generated_samples=1000: Number of samples for FID
    """
    
    print("🚀 Running RlVAE Experiment with Integrated FID Evaluation")
    print("=" * 60)
    
    # Print evaluation configuration
    if hasattr(config, 'evaluation') and config.evaluation.enabled:
        print("📊 EVALUATION ENABLED")
        print(f"   FID Scoring: {'✅' if config.evaluation.fid.enabled else '❌'}")
        print(f"   Generation Methods: {config.evaluation.generation.methods}")
        print(f"   Run During Training: {'✅' if config.evaluation.run_during_training else '❌'}")
        print(f"   Run During Testing: {'✅' if config.evaluation.run_during_testing else '❌'}")
        print(f"   Run At End Only: {'✅' if config.evaluation.run_at_end_only else '❌'}")
        print(f"   Generated Samples for FID: {config.evaluation.fid.n_generated_samples}")
        print(f"   Real Samples Subset: {config.evaluation.fid.real_samples_subset}")
        
        # Estimation of evaluation time
        n_methods = len(config.evaluation.generation.methods)
        n_fid_samples = config.evaluation.fid.n_generated_samples
        estimated_time = n_methods * (n_fid_samples // 1000) * 30  # Rough estimate: 30s per 1000 samples per method
        print(f"   Estimated Evaluation Time: ~{estimated_time}s per evaluation")
        
    else:
        print("📊 EVALUATION DISABLED - Training will run at maximum speed")
    
    print("=" * 60)
    
    # Run the experiment
    runner = ExperimentRunner(config)
    runner.run()
    
    print("\n✅ Experiment completed!")
    
    if hasattr(config, 'evaluation') and config.evaluation.enabled:
        print("\n📊 EVALUATION RESULTS:")
        print("   Check your WandB dashboard for:")
        print("   • FID scores for different generation methods")
        print("   • Reconstruction quality metrics (PSNR, SSIM)")
        print("   • Generation performance benchmarks")
        print("   • Latent space analysis")
        print("   • Generated sample galleries")
        
        if config.evaluation.output.save_detailed_results:
            output_dir = Path(config.output_dir)
            print(f"   Detailed results saved to: {output_dir}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎯 RlVAE with Integrated FID Evaluation")
    print("   This run will include generation quality analysis!")
    print("="*80 + "\n")
    
    main() 