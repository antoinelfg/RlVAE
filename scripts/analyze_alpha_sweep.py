#!/usr/bin/env python3
"""
Analyze alpha sweep results to test spatial mismatch hypothesis.

Expected pattern if hypothesis is correct:
- As α increases → Σ_μ expands → z0 reaches farther regions
- If metric field has high-volume regions at distance ~3-5 from μ:
  → log|G⁻¹(z0)| should increase with α
  → Δ(log|G⁻¹(z0)| - log|G⁻¹(μ)|) should become less negative
  → KL should eventually become positive
"""

import re
import sys
from pathlib import Path
from typing import Dict, List


def extract_metrics(log_path: Path) -> Dict:
    """Extract key metrics from a single alpha experiment log."""
    content = log_path.read_text()
    
    metrics = {
        'alpha': None,
        'log_det_mu': None,
        'log_det_z0_mean': None,
        'log_det_z0_std': None,
        'delta_logdet': None,
        'correlation': None,
        'distance_z0_mu': None,
        'mahal_sq': None,
        'chi_sq_deviation_pct': None,
        'ks_pvalue': None,
        'kl_loss': None,
    }
    
    # Extract alpha from filename
    alpha_match = re.search(r'alpha_(\d+)p(\d+)', str(log_path))
    if alpha_match:
        metrics['alpha'] = float(f"{alpha_match.group(1)}.{alpha_match.group(2)}")
    
    # Extract log|G⁻¹| at μ and z0
    metric_block = re.search(
        r'\[LOG DET G⁻¹ DISTRIBUTION\].*?At μ:\s+mean=([-\d.]+).*?'
        r'At z0:\s+mean=([-\d.]+), std=([-\d.]+).*?'
        r'Δ\(z0 - μ\):\s+([-\d.]+)',
        content,
        re.DOTALL
    )
    if metric_block:
        metrics['log_det_mu'] = float(metric_block.group(1))
        metrics['log_det_z0_mean'] = float(metric_block.group(2))
        metrics['log_det_z0_std'] = float(metric_block.group(3))
        metrics['delta_logdet'] = float(metric_block.group(4))
    
    # Extract correlation
    corr_match = re.search(r'Pearson r:\s+([-\d.]+)', content)
    if corr_match:
        metrics['correlation'] = float(corr_match.group(1))
    
    # Extract distance ||z0 - μ||
    dist_match = re.search(r'\|\|z0 - μ\|\|:\s+mean=([-\d.]+)', content)
    if dist_match:
        metrics['distance_z0_mu'] = float(dist_match.group(1))
    
    # Extract Mahalanobis²
    mahal_match = re.search(r'Actual Mahal²:\s+([-\d.]+)', content)
    if mahal_match:
        metrics['mahal_sq'] = float(mahal_match.group(1))
    
    # Extract chi-squared deviation
    chi_match = re.search(r'Deviation:.*?\(([-+\d.]+)%\)', content)
    if chi_match:
        metrics['chi_sq_deviation_pct'] = float(chi_match.group(1))
    
    # Extract KS p-value
    ks_match = re.search(r'KS p-value:\s+([\d.e-]+)', content)
    if ks_match:
        metrics['ks_pvalue'] = float(ks_match.group(1))
    
    # Extract final KL loss
    kl_match = re.search(r'FINAL KL LOSS:\s+([-\d.]+)', content)
    if kl_match:
        metrics['kl_loss'] = float(kl_match.group(1))
    
    return metrics


def print_comparison_table(results: List[Dict]):
    """Print formatted comparison table."""
    print("\n" + "="*120)
    print("ALPHA SWEEP RESULTS: Spatial Mismatch Hypothesis Test")
    print("="*120)
    
    # Header
    print(f"\n{'α':>6} | {'log|G⁻¹(μ)|':>12} | {'log|G⁻¹(z0)|':>14} | {'Δ(z0-μ)':>10} | "
          f"{'||z0-μ||':>9} | {'Corr':>7} | {'Mahal²':>7} | {'χ² dev%':>9} | {'KL':>8}")
    print("-" * 120)
    
    for r in sorted(results, key=lambda x: x['alpha'] or 0):
        if r['alpha'] is None:
            continue
        
        log_mu = f"{r['log_det_mu']:+.2f}" if r['log_det_mu'] is not None else "N/A"
        log_z0 = f"{r['log_det_z0_mean']:+.2f}" if r['log_det_z0_mean'] is not None else "N/A"
        delta = f"{r['delta_logdet']:+.2f}" if r['delta_logdet'] is not None else "N/A"
        dist = f"{r['distance_z0_mu']:.2f}" if r['distance_z0_mu'] is not None else "N/A"
        corr = f"{r['correlation']:+.3f}" if r['correlation'] is not None else "N/A"
        mahal = f"{r['mahal_sq']:.2f}" if r['mahal_sq'] is not None else "N/A"
        chi_dev = f"{r['chi_sq_deviation_pct']:+.1f}" if r['chi_sq_deviation_pct'] is not None else "N/A"
        kl = f"{r['kl_loss']:+.2f}" if r['kl_loss'] is not None else "N/A"
        
        print(f"{r['alpha']:>6.1f} | {log_mu:>12} | {log_z0:>14} | {delta:>10} | "
              f"{dist:>9} | {corr:>7} | {mahal:>7} | {chi_dev:>9} | {kl:>8}")
    
    print("\n" + "="*120)
    
    # Analysis
    print("\nKEY OBSERVATIONS:")
    
    # Check if Δ becomes less negative
    deltas = [r['delta_logdet'] for r in results if r['delta_logdet'] is not None]
    if len(deltas) >= 2:
        if deltas[-1] > deltas[0]:
            print(f"✅ Δ(z0-μ) in log|G⁻¹| IMPROVED: {deltas[0]:.2f} → {deltas[-1]:.2f}")
        else:
            print(f"❌ Δ(z0-μ) in log|G⁻¹| did NOT improve: {deltas[0]:.2f} → {deltas[-1]:.2f}")
    
    # Check correlation
    corrs = [r['correlation'] for r in results if r['correlation'] is not None]
    if len(corrs) >= 2:
        if abs(corrs[-1]) < abs(corrs[0]):
            print(f"✅ Correlation became LESS NEGATIVE: {corrs[0]:+.3f} → {corrs[-1]:+.3f}")
        else:
            print(f"❌ Correlation did NOT improve: {corrs[0]:+.3f} → {corrs[-1]:+.3f}")
    
    # Check KL
    kls = [r['kl_loss'] for r in results if r['kl_loss'] is not None]
    if len(kls) >= 2:
        positive_count = sum(1 for k in kls if k > 0)
        print(f"\n📊 KL Analysis:")
        print(f"   Negative KL count: {len(kls) - positive_count}/{len(kls)}")
        print(f"   Positive KL count: {positive_count}/{len(kls)}")
        if positive_count > 0:
            print(f"   ✅ Some configs achieved POSITIVE KL!")
        if kls[-1] > kls[0]:
            print(f"   ✅ KL IMPROVED: {kls[0]:+.2f} → {kls[-1]:+.2f}")
    
    print("\n" + "="*120)
    
    # Hypothesis verdict
    print("\nHYPOTHESIS VERDICT:")
    if len(deltas) >= 2 and deltas[-1] > deltas[0] + 2.0:
        print("✅ STRONG SUPPORT: Increasing α significantly reduces spatial mismatch")
    elif len(deltas) >= 2 and deltas[-1] > deltas[0]:
        print("⚠️  WEAK SUPPORT: Increasing α shows some improvement")
    else:
        print("❌ HYPOTHESIS REJECTED: Increasing α does not resolve spatial mismatch")
        print("   → Problem may be more fundamental (metric field structure)")


def main():
    log_dir = Path("/home/alaforgu/scratch/longitudinal_experiments/RlVAE/logs/alpha_sweep")
    
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        print("Run the alpha sweep first: bash scripts/run_alpha_sweep.sh")
        sys.exit(1)
    
    log_files = sorted(log_dir.glob("alpha_*.log"))
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        sys.exit(1)
    
    print(f"Analyzing {len(log_files)} alpha sweep logs...")
    
    results = []
    for log_file in log_files:
        print(f"  Processing {log_file.name}...")
        metrics = extract_metrics(log_file)
        results.append(metrics)
    
    print_comparison_table(results)


if __name__ == '__main__':
    main()

