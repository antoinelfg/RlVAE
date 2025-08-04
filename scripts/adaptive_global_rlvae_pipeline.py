#!/usr/bin/env python3
"""
Adaptive Global RLVAE Pipeline
==============================

Enhanced pipeline that integrates your brilliant adaptive centroid idea:
1. Trains Vanilla VAE + extracts metric (Stage 1)
2. Trains RLVAE with ADAPTIVE CENTROID UPDATES (Stage 2)
3. Creates manifold evolution visualizations showing sampling on manifold 0

This implements:
- Periodic centroid recomputation every N epochs
- Living manifold evolution tracking
- Enhanced visualizations showing how sampling adapts
- Full integration with existing pipeline infrastructure

Usage:
    python scripts/adaptive_global_rlvae_pipeline.py --architecture mlp --latent-dim 2 --centroid-update-freq 2
"""

import sys
import os
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import json
import torch
from omegaconf import DictConfig, OmegaConf
import yaml
import numpy as np
import logging
import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.adaptive_centroid_trainer import AdaptiveCentroidTrainer, create_adaptive_config
from src.data.cyclic_dataset import CyclicSpritesDataModule
from src.models.modular_rlvae import ModularRiemannianFlowVAE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Adaptive Global RLVAE Pipeline")
    
    # Architecture and basic settings
    parser.add_argument("--architecture", choices=["cnn", "resnet", "mlp"], default="mlp",
                       help="Architecture type for both VAE and RLVAE (default: mlp)")
    parser.add_argument("--latent-dim", type=int, default=2,
                       help="Latent space dimension (default: 2)")
    
    # Vanilla VAE parameters
    parser.add_argument("--vae-epochs", type=int, default=50,
                       help="Vanilla VAE training epochs (default: 50)")
    parser.add_argument("--vae-preset", choices=["balanced", "max_diversity", "conservative"], 
                       default="balanced", help="VAE metric diversity preset (default: balanced)")
    
    # RLVAE parameters
    parser.add_argument("--rlvae-epochs", type=int, default=100,
                       help="RLVAE training epochs (default: 100)")
    parser.add_argument("--rlvae-batch-size", type=int, default=8,
                       help="RLVAE training batch size (default: 8)")
    parser.add_argument("--rlvae-lr", type=float, default=1e-4,
                       help="RLVAE learning rate (default: 1e-4)")
    
    # 🚀 NEW: Adaptive centroid parameters
    parser.add_argument("--centroid-update-freq", type=int, default=2,
                       help="Update centroids every N epochs (default: 2)")
    parser.add_argument("--n-samples-for-centroids", type=int, default=500,
                       help="Number of samples to use for centroid computation (default: 500)")
    parser.add_argument("--disable-adaptive", action="store_true",
                       help="Disable adaptive centroid updates (use static centroids)")
    parser.add_argument('--freeze-mode', action='store_true',
                        help='Freeze mode: analyze and visualize without actually updating centroids (for stability)')
    parser.add_argument('--kl-controlled-mode', action='store_true', default=True,
                        help='KL controlled mode: real updates with automatic KL divergence monitoring and rollback (default: True)')
    
    # Output and logging
    parser.add_argument("--output-dir", type=str, 
                       default=f"experiments/adaptive_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help="Output directory for experiment")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="adaptive-rlvae-pipeline",
                       help="Wandb project name")
    parser.add_argument("--skip-vae", action="store_true",
                       help="Skip VAE training (use existing components)")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip post-training analysis")
    
    # Training parameters
    parser.add_argument("--n-train-samples", type=int, default=1600,
                       help="Number of training samples")
    parser.add_argument("--n-val-samples", type=int, default=400,
                       help="Number of validation samples")
    parser.add_argument("--visualization-level", choices=["minimal", "standard", "full"], 
                       default="standard", help="Visualization detail level")
    
    return parser.parse_args()


def run_vae_training(args, output_dir):
    """Run Stage 1: Vanilla VAE training + metric extraction."""
    print(f"\n{'='*80}")
    print(f"🎯 STAGE 1: Training Vanilla VAE + Extracting Metric ({args.architecture.upper()})")
    print(f"{'='*80}")
    
    # Use correct VAE training script that accepts arguments
    vae_cmd = [
        sys.executable, "scripts/train_diverse_metric_vae.py", 
        "--architecture", args.architecture,
        "--latent-dim", str(args.latent_dim),
        "--epochs", str(args.vae_epochs),
        "--preset", args.vae_preset
    ]
    
    if args.wandb:
        vae_cmd.extend(["--wandb-group", args.wandb_project])
    
    print(f"[PIPELINE] Running VAE training: {' '.join(vae_cmd)}")
    subprocess.run(vae_cmd, check=True)
    
    # Find the most recently generated components in data/pretrained/
    import glob
    import time
    
    # Look for components with matching architecture and latent_dim
    pattern_prefix = f"data/pretrained/*_{args.architecture}_ld{args.latent_dim}_"
    
    # Find most recent encoder, decoder, and metric
    encoder_files = glob.glob(f"{pattern_prefix}*.pt")
    encoder_files = [f for f in encoder_files if "encoder" in f]
    decoder_files = glob.glob(f"{pattern_prefix}*.pt") 
    decoder_files = [f for f in decoder_files if "decoder" in f]
    metric_files = glob.glob(f"{pattern_prefix}*.pt")
    metric_files = [f for f in metric_files if "metric" in f]
    
    if not (encoder_files and decoder_files and metric_files):
        raise FileNotFoundError(f"Stage 1 training did not produce required components for {args.architecture}_ld{args.latent_dim}")
    
    # Get most recent files (by modification time)
    encoder_path = Path(max(encoder_files, key=lambda f: os.path.getmtime(f)))
    decoder_path = Path(max(decoder_files, key=lambda f: os.path.getmtime(f)))
    metric_path = Path(max(metric_files, key=lambda f: os.path.getmtime(f)))
    
    print(f"✅ Stage 1 completed successfully!")
    print(f"📁 Found components:")
    print(f"   Encoder: {encoder_path.name}")
    print(f"   Decoder: {decoder_path.name}")
    print(f"   Metric: {metric_path.name}")
    
    return encoder_path, decoder_path, metric_path


def create_adaptive_rlvae_config(args, encoder_path, decoder_path, metric_path):
    """Create configuration for adaptive RLVAE training."""
    print(f"\n🔧 Creating adaptive RLVAE configuration...")
    
    # Override the trainer's hardcoded paths by setting up symlinks
    # The trainer expects encoder.pt and decoder.pt in data/pretrained/
    encoder_link = Path("data/pretrained/encoder.pt")
    decoder_link = Path("data/pretrained/decoder.pt")
    
    # Create backup of existing links
    if encoder_link.exists():
        encoder_link.rename("data/pretrained/encoder_backup.pt")
    if decoder_link.exists():
        decoder_link.rename("data/pretrained/decoder_backup.pt")
    
    # Create new symlinks
    encoder_link.symlink_to(encoder_path.resolve())
    decoder_link.symlink_to(decoder_path.resolve())
    
    print(f"🔗 Created symlinks for Stage 2:")
    print(f"   encoder.pt → {encoder_path.name}")
    print(f"   decoder.pt → {decoder_path.name}")
    
    # Base configuration
    config = DictConfig({
        # Model configuration
        'input_dim': [3, 64, 64],  # Match Stage 1 dimensions (CyclicSprites are 64x64)
        'latent_dim': args.latent_dim,
        'sequence_length': 10,
        'loop_mode': 'open',
        'n_flows': 9,  # sequence_length - 1
        'posterior_type': 'riemannian_metric',
        'beta': 1.0,
        'riemannian_beta': 1.0,
        'use_riemannian': True,
        'riemannian_method': 'custom',
        'sampling_method': 'geodesic',
        
        # Architecture
        'encoder': {'architecture': args.architecture},
        'decoder': {'architecture': args.architecture},
        
        # Metric configuration
        'metric_path': str(metric_path.name),
        'temperature_fix': 3.0,
        
        # Training parameters
        'learning_rate': args.rlvae_lr,
        'n_epochs': args.rlvae_epochs,
        'batch_size': args.rlvae_batch_size,
        'n_train_samples': args.n_train_samples,
        'n_val_samples': args.n_val_samples,
        'cycle_penalty': 0.0,  # Required parameter for loop mode
        
        # Visualization
        'visualization': {
            'level': args.visualization_level,
            'enable_manifold': True,
            'enable_interactive': True,
            'manifold_frequency': 5,
            'interactive_frequency': 10
        },
        'visualization_frequency': 5,  # Required parameter for the trainer
        
        # WandB settings
        'wandb_project': args.wandb_project,
        'wandb_only': True,
        'use_vanilla_vae': False
    })
    
    # 🚀 Add adaptive centroid configuration
    if not args.disable_adaptive:
        config.adaptive_centroids = DictConfig({
            'enabled': True,
            'update_frequency': args.centroid_update_freq,
            'n_samples_for_centroids': args.n_samples_for_centroids,
            'visualize_evolution': True,
            'freeze_mode': getattr(args, 'freeze_mode', False)  # Analyze only, no actual updates
        })
        print(f"🔄 Adaptive centroids enabled: update every {args.centroid_update_freq} epochs")
    else:
        config.adaptive_centroids = DictConfig({'enabled': False})
        print("⚠️ Adaptive centroids disabled - using static centroids")
    
    return config


def train_adaptive_rlvae(args, config, output_dir):
    """Train RLVAE with adaptive centroid updates."""
    print(f"\n{'='*80}")
    print(f"🚀 STAGE 2: Training RLVAE with Adaptive Centroids ({args.architecture.upper()})")
    print(f"{'='*80}")
    
    if not args.disable_adaptive:
        print(f"🔄 Adaptive centroid updates: every {args.centroid_update_freq} epochs")
        print(f"📊 Samples per update: {args.n_samples_for_centroids}")
        print(f"🌊 Manifold evolution tracking: enabled")
    
    try:
        # Create data module (using Circle data for testing)
        data_config = DictConfig({
            'train_path': 'data/sprites/ColoredCircles_train.pt',
            'test_path': 'data/sprites/ColoredCircles_test.pt', 
            'train_meta_path': 'data/sprites/ColoredCircles_train_params.pt',
            'test_meta_path': 'data/sprites/ColoredCircles_test_params.pt',
            'sequence_length': 10,
            'image_size': [64, 64],  # Match Stage 1 dimensions
            'channels': 3,
            'batch_size': args.rlvae_batch_size,
            'num_workers': 0,
            'pin_memory': False,
            'max_train_samples': args.n_train_samples,
            'max_val_samples': args.n_val_samples,
            'verify_cyclicity': False
        })
        
        # Setup run name
        run_name = f"adaptive_stage2_rlvae_{args.architecture}_ld{args.latent_dim}"
        if not args.disable_adaptive:
            run_name += f"_freq{args.centroid_update_freq}"
        
        config.wandb_run_name = run_name
        
        # Create trainer
        if not args.disable_adaptive:
            # Use adaptive centroid trainer
            trainer = AdaptiveCentroidTrainer(
                config=config,
                project_name=args.wandb_project,
                run_name=run_name,
                centroid_update_frequency=args.centroid_update_freq,
                n_samples_for_centroids=args.n_samples_for_centroids,
                freeze_mode=args.freeze_mode,
                kl_controlled_mode=args.kl_controlled_mode and not args.freeze_mode
            )
            if args.freeze_mode:
                print("🧊 FREEZE MODE ENABLED: Will analyze but not update centroids")
            elif args.kl_controlled_mode:
                print("🎯 KL-CONTROLLED MODE ENABLED: Real updates with stability monitoring")
            else:
                print("⚠️  LEGACY MODE: Standard updates (may be unstable)")
            print("🔄 Using AdaptiveCentroidTrainer")
        else:
            # Use standard trainer (import and create here)
            from src.training.train_with_modular_visualizations import CleanCyclicLoopTrainer
            trainer = CleanCyclicLoopTrainer(
                config=config,
                project_name=args.wandb_project,
                run_name=run_name
            )
            print("📊 Using standard CleanCyclicLoopTrainer")
        
        # Train the model
        trainer.train(n_epochs=args.rlvae_epochs)
        
        # Save final model
        model_save_path = output_dir / "stage2_adaptive_rlvae" / "final_model.pt"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.model.state_dict(), model_save_path)
        
        print(f"✅ Stage 2 completed successfully!")
        print(f"💾 Model saved: {model_save_path}")
        
        if not args.disable_adaptive and hasattr(trainer, 'centroid_history'):
            print(f"🔄 Total centroid updates: {len(trainer.centroid_history)}")
            print(f"🌊 Manifold evolution tracked and visualized")
        
        return run_name, model_save_path
        
    except Exception as e:
        logger.error(f"❌ Stage 2 training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_enhanced_analysis(args, output_dir, rlvae_run_name, model_path):
    """Run enhanced post-training analysis."""
    print(f"\n{'='*80}")
    print(f"📊 STAGE 3: Enhanced Analysis")
    print(f"{'='*80}")
    
    try:
        analysis_dir = output_dir / "enhanced_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Run comprehensive analysis using our established scripts
        analysis_cmd = [
            sys.executable, "scripts/recompute_centroids_from_trained_data.py",
            "--checkpoint-path", str(model_path),
            "--output-dir", str(analysis_dir)
        ]
        
        print(f"[PIPELINE] Running enhanced analysis: {' '.join(analysis_cmd)}")
        subprocess.run(analysis_cmd, check=True)
        
        print(f"✅ Enhanced analysis completed!")
        print(f"📁 Results in: {analysis_dir}")
        
        return analysis_dir
        
    except Exception as e:
        logger.warning(f"⚠️ Enhanced analysis failed: {e}")
        return None


def create_pipeline_summary(args, output_dir, results):
    """Create comprehensive pipeline summary."""
    print(f"\n📝 Creating pipeline summary...")
    
    summary = {
        'pipeline_type': 'adaptive_global_rlvae',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'architecture': args.architecture,
            'latent_dim': args.latent_dim,
            'vae_epochs': args.vae_epochs,
            'rlvae_epochs': args.rlvae_epochs,
            'adaptive_centroids': {
                'enabled': not args.disable_adaptive,
                'update_frequency': args.centroid_update_freq if not args.disable_adaptive else None,
                'n_samples_for_centroids': args.n_samples_for_centroids if not args.disable_adaptive else None
            }
        },
        'results': results,
        'output_directory': str(output_dir)
    }
    
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Pipeline summary saved: {summary_path}")
    return summary_path


def main():
    """Main pipeline function."""
    print(f"🚀 Starting Adaptive Global RLVAE Pipeline")
    print(f"🌊 Implementing adaptive centroid updates during training")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse arguments
    args = parse_args()
    
    # Setup output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Experiment directory: {output_path}")
    print(f"🏗️  Architecture: {args.architecture.upper()}")
    print(f"🧠 Latent dimension: {args.latent_dim}")
    
    if not args.disable_adaptive:
        print(f"🔄 Adaptive centroids: every {args.centroid_update_freq} epochs")
        print(f"📊 Samples per update: {args.n_samples_for_centroids}")
    else:
        print(f"⚠️  Adaptive centroids: DISABLED (static mode)")
    
    results = {}
    
    try:
        # Step 1: Train Vanilla VAE + Extract Metric (unless skipped)
        if not args.skip_vae:
            encoder_path, decoder_path, metric_path = run_vae_training(args, output_path)
            results['stage1'] = {
                'encoder_path': str(encoder_path),
                'decoder_path': str(decoder_path),
                'metric_path': str(metric_path)
            }
        else:
            print("⏭️ Skipping VAE training (using existing components)")
            # Use existing pretrained components
            encoder_path = Path("data/pretrained/encoder.pt")
            decoder_path = Path("data/pretrained/decoder.pt")
            metric_path = Path("data/pretrained/metric_T0.7_scaled.pt")
            
            if not all(p.exists() for p in [encoder_path, decoder_path, metric_path]):
                raise FileNotFoundError("Missing pretrained components for skipped VAE training")
            
            results['stage1'] = {
                'encoder_path': str(encoder_path),
                'decoder_path': str(decoder_path),
                'metric_path': str(metric_path),
                'skipped': True
            }
        
        # Step 2: Create RLVAE config
        print(f"\n🔧 Creating adaptive RLVAE configuration...")
        rlvae_config = create_adaptive_rlvae_config(args, encoder_path, decoder_path, metric_path)
        
        # Step 3: Train Adaptive RLVAE
        rlvae_run_name, model_path = train_adaptive_rlvae(args, rlvae_config, output_path)
        results['stage2'] = {
            'run_name': rlvae_run_name,
            'model_path': str(model_path),
            'adaptive_enabled': not args.disable_adaptive
        }
        
        # Step 4: Run Enhanced Analysis (unless skipped)
        if not args.skip_analysis:
            analysis_dir = run_enhanced_analysis(args, output_path, rlvae_run_name, model_path)
            results['analysis'] = {
                'analysis_dir': str(analysis_dir) if analysis_dir else None,
                'completed': analysis_dir is not None
            }
        else:
            print("⏭️ Skipping enhanced analysis (--skip-analysis flag)")
            results['analysis'] = {'skipped': True}
        
        # Step 5: Create pipeline summary
        summary_path = create_pipeline_summary(args, output_path, results)
        
        # Success summary
        print(f"\n{'='*80}")
        print(f"🎉 ADAPTIVE GLOBAL RLVAE PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"📁 Experiment directory: {output_path}")
        print(f"📄 Summary: {summary_path}")
        
        if not args.disable_adaptive:
            print(f"🔄 Adaptive centroid updates: ✅ ENABLED")
            print(f"🌊 Manifold evolution visualizations: ✅ CREATED")
            print(f"📊 Check WandB for manifold evolution tracking!")
        else:
            print(f"⚠️  Adaptive centroid updates: ❌ DISABLED")
        
        print(f"✨ Your brilliant adaptive centroid idea has been implemented!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 