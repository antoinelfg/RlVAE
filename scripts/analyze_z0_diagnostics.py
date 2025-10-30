#!/usr/bin/env python3
"""
Quick parser for z0 investigation console diagnostics.

Usage:
    python scripts/analyze_z0_diagnostics.py path/to/console_output.log

Extracts key metrics from RLVAE_DEBUG=1 output:
- Candidate statistics
- Selection bias (Mahalanobis² for candidates vs selected)
- Chi-squared test results
- Correlation between h_score and Mahalanobis²
"""

import re
import sys
from pathlib import Path


def parse_candidate_diagnostics(log_content: str) -> dict:
    """Extract candidate-level diagnostics."""
    results = {
        'all_candidates_mahal_mean': [],
        'selected_mahal_mean': [],
        'delta_mahal_selection': [],
        'correlation_h_mahal': [],
        'chi_sq_deviation_pct': [],
        'ks_pvalue': [],
    }
    
    # Parse candidate diagnostics blocks
    candidate_blocks = re.findall(
        r'\[CANDIDATE DIAGNOSTICS\].*?={80}',
        log_content,
        re.DOTALL
    )
    
    for block in candidate_blocks:
        # All candidates Mahal²
        match = re.search(r'Mahal²:\s+mean=([\d.]+)', block)
        if match:
            results['all_candidates_mahal_mean'].append(float(match.group(1)))
        
        # Selected Mahal²
        match = re.search(r'Selected Mahal²:\s+mean=([\d.]+)', block)
        if match:
            results['selected_mahal_mean'].append(float(match.group(1)))
        
        # Delta (selection bias)
        match = re.search(r'Δ Mahal²\(sel - pool\):\s+([-+]?[\d.]+)', block)
        if match:
            results['delta_mahal_selection'].append(float(match.group(1)))
        
        # Correlation
        match = re.search(r'Corr\(h, Mahal²\):\s+([-+]?[\d.]+)', block)
        if match:
            results['correlation_h_mahal'].append(float(match.group(1)))
    
    # Parse chi-squared test results
    chi_blocks = re.findall(
        r'\[CHI-SQUARED TEST\].*?\[',
        log_content,
        re.DOTALL
    )
    
    for block in chi_blocks:
        # Deviation percentage
        match = re.search(r'Deviation:.*?\(([-+]?[\d.]+)%\)', block)
        if match:
            results['chi_sq_deviation_pct'].append(float(match.group(1)))
        
        # KS test p-value
        match = re.search(r'KS p-value:\s+([\d.e-]+)', block)
        if match:
            results['ks_pvalue'].append(float(match.group(1)))
    
    return results


def summarize_results(results: dict) -> None:
    """Print summary statistics."""
    print("\n" + "="*80)
    print("Z0 INVESTIGATION SUMMARY")
    print("="*80)
    
    if results['all_candidates_mahal_mean']:
        print("\n[CANDIDATE POOL STATISTICS]")
        cand_vals = results['all_candidates_mahal_mean']
        print(f"  All candidates Mahal² (across batches):")
        print(f"    Mean: {sum(cand_vals)/len(cand_vals):.4f}")
        print(f"    Min:  {min(cand_vals):.4f}")
        print(f"    Max:  {max(cand_vals):.4f}")
        print(f"  Expected for D=2: 2.0000")
    
    if results['selected_mahal_mean']:
        print("\n[SELECTED SAMPLE STATISTICS]")
        sel_vals = results['selected_mahal_mean']
        print(f"  Selected Mahal² (across batches):")
        print(f"    Mean: {sum(sel_vals)/len(sel_vals):.4f}")
        print(f"    Min:  {min(sel_vals):.4f}")
        print(f"    Max:  {max(sel_vals):.4f}")
    
    if results['delta_mahal_selection']:
        print("\n[SELECTION BIAS]")
        delta_vals = results['delta_mahal_selection']
        mean_delta = sum(delta_vals) / len(delta_vals)
        print(f"  Δ Mahal² (selected - pool):")
        print(f"    Mean: {mean_delta:+.4f}")
        if mean_delta > 0.5:
            print(f"    ⚠️  POSITIVE BIAS: Selection prefers outliers!")
        elif mean_delta < -0.5:
            print(f"    ⚠️  NEGATIVE BIAS: Selection prefers inliers!")
        else:
            print(f"    ✓ Minimal bias")
    
    if results['correlation_h_mahal']:
        print("\n[CORRELATION: h_score vs Mahalanobis²]")
        corr_vals = results['correlation_h_mahal']
        mean_corr = sum(corr_vals) / len(corr_vals)
        print(f"  Mean correlation: {mean_corr:+.4f}")
        if abs(mean_corr) > 0.5:
            direction = "positive" if mean_corr > 0 else "negative"
            print(f"    ⚠️  STRONG {direction.upper()} correlation!")
            if mean_corr > 0:
                print(f"       High-volume regions are FAR from μ → bias toward outliers")
        else:
            print(f"    ✓ Weak correlation")
    
    if results['chi_sq_deviation_pct']:
        print("\n[CHI-SQUARED FIT (FINAL z0)]")
        chi_vals = results['chi_sq_deviation_pct']
        mean_dev = sum(chi_vals) / len(chi_vals)
        print(f"  Mean deviation from χ²(2): {mean_dev:+.1f}%")
        if abs(mean_dev) > 50:
            print(f"    ⚠️  LARGE DEVIATION: z0 does NOT match N(μ, Σ)!")
        else:
            print(f"    ✓ Reasonable fit")
    
    if results['ks_pvalue']:
        print("\n[KOLMOGOROV-SMIRNOV TEST]")
        ks_vals = results['ks_pvalue']
        mean_pval = sum(ks_vals) / len(ks_vals)
        print(f"  Mean p-value: {mean_pval:.4e}")
        if mean_pval < 0.01:
            print(f"    ⚠️  REJECT null: Distribution mismatch!")
        else:
            print(f"    ✓ Cannot reject: Distribution consistent")
    
    print("\n" + "="*80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_z0_diagnostics.py <log_file>")
        print("\nAlternatively, pipe console output:")
        print("  RLVAE_DEBUG=1 python run_experiment.py ... | tee output.log")
        print("  python analyze_z0_diagnostics.py output.log")
        sys.exit(1)
    
    log_path = Path(sys.argv[1])
    
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)
    
    print(f"Parsing diagnostics from: {log_path}")
    log_content = log_path.read_text()
    
    results = parse_candidate_diagnostics(log_content)
    summarize_results(results)
    
    # Check if diagnostics were found
    if not any(results.values()):
        print("\n⚠️  WARNING: No diagnostic markers found in log.")
        print("   Make sure RLVAE_DEBUG=1 was set during the run.")


if __name__ == '__main__':
    main()

