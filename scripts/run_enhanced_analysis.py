#!/usr/bin/env python3
"""
Enhanced Analysis Runner for RlVAE

This script runs comprehensive analysis including:
- Enhanced generation visualization with multiple sampling methods
- Advanced inference analysis with latent space trajectories  
- Geodesic and Riemannian sampling for manifold exploration
- FID score evaluation
- Comprehensive reporting and visualization

Usage:
    python scripts/run_enhanced_analysis.py --checkpoint_path path/to/checkpoint --config_path path/to/config
"""

import os
import sys
import argparse
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.enhanced_analysis import EnhancedAnalyzer
from models.modular_rlvae import ModularRiemannianFlowVAE as RlVAE
from data.datasets import get_dataloader
from training.utils import load_checkpoint, get_config

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('enhanced_analysis.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Run enhanced RlVAE analysis")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="enhanced_analysis_outputs",
                       help="Output directory for analysis results")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples for generation analysis")
    parser.add_argument("--num_cycles", type=int, default=50,
                       help="Number of cycles for inference analysis")
    parser.add_argument("--geodesic_steps", type=int, default=20,
                       help="Number of steps for geodesic interpolation")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--log_to_wandb", action="store_true",
                       help="Log results to wandb")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint not found: {args.checkpoint_path}")
        return
    
    # Load config
    config = get_config(args.config_path)
    logger.info(f"Loaded config from {args.config_path}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model from checkpoint...")
    model, config = load_checkpoint(args.checkpoint_path, device=device)
    model.eval()
    logger.info(f"Model loaded successfully. Latent dim: {model.latent_dim}")
    
    # Setup dataloader
    logger.info("Setting up dataloader...")
    dataloader = get_dataloader(
        config.data.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    logger.info(f"Dataloader created with {len(dataloader)} batches")
    
    # Initialize enhanced analyzer
    logger.info("Initializing enhanced analyzer...")
    analyzer = EnhancedAnalyzer(
        model=model,
        device=device,
        output_dir=args.output_dir
    )
    
    # Run comprehensive analysis
    logger.info("Starting comprehensive analysis...")
    results = analyzer.run_comprehensive_analysis(
        dataloader=dataloader,
        num_samples=args.num_samples,
        num_cycles=args.num_cycles,
        geodesic_steps=args.geodesic_steps,
        log_to_wandb=args.log_to_wandb
    )
    
    # Print summary
    logger.info("Analysis completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")
    
    if 'generation' in results:
        fid_scores = results['generation']['fid_scores']
        logger.info("FID Scores:")
        for method, score in fid_scores.items():
            logger.info(f"  {method}: {score:.2f}")
    
    if 'inference' in results:
        consistency = results['inference']['mean_cycle_consistency']
        logger.info(f"Mean cycle consistency: {consistency:.4f}")
    
    logger.info("Enhanced analysis completed!")

if __name__ == "__main__":
    main() 