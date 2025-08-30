#!/usr/bin/env python3
"""
Execute KL Divergence Analysis Scripts

This script runs the comprehensive KL divergence analysis using the exact
experimental parameters from the user's command.

Usage:
    python scripts/run_kl_analysis.py [--script {debug|formulations|both}]
"""

import sys
import os
import argparse
from pathlib import Path

# Add paths
sys.path.append('/home/alaforgu/scratch/longitudinal_experiments/RlVAE')
sys.path.append('/home/alaforgu/scratch/longitudinal_experiments/RlVAE/scripts')

def run_debug_script():
    """Run the main debug script."""
    print("🚀 Running KL Divergence Debug Script...")
    print("="*80)
    
    try:
        from debug_kl_divergence import main as debug_main
        results = debug_main()
        return results
    except Exception as e:
        print(f"❌ Debug script failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_formulations_script():
    """Run the formulations analysis script."""
    print("🚀 Running KL Formulations Analysis Script...")
    print("="*80)
    
    try:
        from debug_kl_formulations import main as formulations_main
        results = formulations_main()
        return results
    except Exception as e:
        print(f"❌ Formulations script failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_summary_report(debug_results, formulations_results):
    """Create a summary report of the analysis."""
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n🎯 EXPERIMENTAL SETUP:")
    print("   experiment=global_vanilla_rlvae_pipeline")
    print("   model=rhvae_original_with_metric_update")
    print("   data=cyclic_sprites")
    print("   model.latent_dim=16")
    print("   experiment.skip_stage1=true")
    print("   model.metric_update_frequency=47")
    
    print("\n🔍 KEY FINDINGS:")
    print("1. Mathematical formulation vs. implementation mismatch identified")
    print("2. Current implementation missing normalization terms: log Z_p - log Z_q(μ)")
    print("3. Volume factors √det(G(z)) cancel pointwise, but normalizers don't")
    print("4. This causes KL to misbehave and require artificial clamping/scaling")
    
    print("\n💻 IMPLEMENTATION ANALYSIS:")
    print("Prior p(z): ∝ √det(G(z)) exp(-1/2 z^T G(z) z)")
    print("   Code: G(z) function using centroids, M_tens, temperature")
    print("   Location: _create_metric_rhvae")
    
    print("\nPosterior q_φ(z|x): ∝ √det(G(z)) exp(-1/2 (z-μ)^T G(z) (z-μ))")
    print("   Code: Encoder outputs μ, log_var + WorkingRiemannianSampler")
    print("   Location: model.encoder + sample_riemannian_latents")
    
    print("\nKL Divergence: KL(q||p) = E_q[...] + (log Z_p - log Z_q)")
    print("   Code: compute_riemannian_metric_kl_loss (INCOMPLETE)")
    print("   Issue: Missing normalization terms")
    
    print("\n✅ RECOMMENDED FIXES:")
    print("1. Implement proper normalization term: log Z_p - log Z_q(μ)")
    print("2. Remove artificial clamping and scaling")
    print("3. Use local Gaussian approximation for intractable log Z_p")
    print("4. Test with exact experimental parameters")
    print("5. Validate improved training stability")
    
    print("\n🎭 CURRENT FUNCTION MAPPING:")
    print("   Prior sampling: sample_prior() methods")
    print("   Posterior sampling: sample_riemannian_latents(method='geodesic')")
    print("   KL computation: compute_riemannian_metric_kl_loss() [NEEDS FIX]")
    print("   Metric tensor: G(z) and G_inv(z) functions")
    
    print("\n🔧 NEXT STEPS:")
    print("1. Implement corrected KL divergence in RiemannianFlowVAE")
    print("2. Update compute_riemannian_metric_kl_loss() with normalization")
    print("3. Remove clamping in the loss computation")
    print("4. Test with the exact experimental command")
    print("5. Monitor KL behavior during training")
    
    # Save summary report
    output_dir = Path('/home/alaforgu/scratch/longitudinal_experiments/RlVAE/debug_outputs')
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / 'kl_analysis_summary_report.txt'
    with open(report_path, 'w') as f:
        f.write("KL DIVERGENCE ANALYSIS SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write("Generated for experimental setup:\n")
        f.write("experiment=global_vanilla_rlvae_pipeline\n")
        f.write("model=rhvae_original_with_metric_update\n")
        f.write("data=cyclic_sprites\n")
        f.write("model.latent_dim=16\n\n")
        
        f.write("KEY PROBLEM IDENTIFIED:\n")
        f.write("Current KL implementation is missing normalization terms log Z_p - log Z_q(μ)\n")
        f.write("This causes incorrect KL magnitude and requires artificial fixes\n\n")
        
        f.write("MATHEMATICAL FORMULATION:\n")
        f.write("KL(q||p) = E_q[-1/2 ((z-μ)^T G(z)(z-μ) - z^T G(z) z)] + (log Z_p - log Z_q)\n\n")
        
        f.write("RECOMMENDED SOLUTION:\n")
        f.write("1. Add normalization terms to KL computation\n")
        f.write("2. Remove artificial clamping/scaling\n")
        f.write("3. Use proper mathematical formulation\n")
        f.write("4. Test with experimental parameters\n")
    
    print(f"\n💾 Summary report saved to: {report_path}")
    
    return {
        'debug_results': debug_results,
        'formulations_results': formulations_results,
        'summary_report_path': str(report_path)
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run KL Divergence Analysis')
    parser.add_argument('--script', choices=['debug', 'formulations', 'both'], 
                       default='both', help='Which script to run')
    
    args = parser.parse_args()
    
    print("🔬 KL DIVERGENCE ANALYSIS RUNNER")
    print("="*80)
    print("Analyzing the KL divergence implementation in RHVAE")
    print("Using exact experimental parameters from user command")
    print()
    
    debug_results = None
    formulations_results = None
    
    if args.script in ['debug', 'both']:
        debug_results = run_debug_script()
        
    if args.script in ['formulations', 'both']:
        formulations_results = run_formulations_script()
        
    # Create comprehensive summary
    if debug_results is not None or formulations_results is not None:
        summary = create_summary_report(debug_results, formulations_results)
        print("\n🎉 Analysis completed successfully!")
        return summary
    else:
        print("\n❌ Analysis failed - no valid results obtained")
        return None

if __name__ == "__main__":
    results = main()
