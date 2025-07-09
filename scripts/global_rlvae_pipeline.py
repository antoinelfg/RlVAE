#!/usr/bin/env python3
"""
Global RLVAE Pipeline
====================

Complete pipeline that:
1. Trains Vanilla VAE + extracts metric (using existing script)
2. Trains RLVAE with the generated components (using config system)

This gives you a fully automated vanilla → RLVAE training pipeline 
with shared encoder/decoder architectures and latent dimensions.

Usage:
    python scripts/global_rlvae_pipeline.py --architecture cnn --latent-dim 16 --vae-epochs 50 --rlvae-epochs 100

Features:
- Single command for full pipeline
- Same architecture/latent-dim for both stages
- Automatic file management and organization
- Config-based RLVAE training
- Optional wandb logging
- Comprehensive error handling
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
import matplotlib.pyplot as plt
import glob
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.modular_rlvae import ModularRiemannianFlowVAE
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from src.visualizations.manager import VisualizationManager, VisualizationConfig, VisualizationLevel


class SpritesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, for_rlvae=False):
        self.data = torch.load(data_path)
        self.for_rlvae = for_rlvae
        
        if for_rlvae:
            # Keep sequential structure for RLVAE: [batch, seq, c, h, w]
            if len(self.data.shape) == 5:
                # Already in correct format
                pass
            else:
                # If not sequential, we can't use for RLVAE
                raise ValueError("RLVAE requires sequential data but data is not in [batch, seq, c, h, w] format")
        else:
            # Flatten for traditional VAE: [batch*seq, c, h, w]
            if len(self.data.shape) == 5:
                batch_size, seq_len = self.data.shape[:2]
                self.data = self.data.reshape(batch_size * seq_len, *self.data.shape[2:])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Global RLVAE Pipeline")
    
    # Architecture and basic settings
    parser.add_argument("--architecture", choices=["cnn", "resnet", "mlp"], default="cnn",
                       help="Architecture type for both VAE and RLVAE (default: cnn)")
    parser.add_argument("--latent-dim", type=int, default=16,
                       help="Latent space dimension (default: 16)")
    
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
    
    # Output and logging
    parser.add_argument("--output-dir", type=str, 
                       default=f"experiments/global_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help="Output directory for experiment")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="global_rlvae_pipeline",
                       help="Wandb project name")
    parser.add_argument("--skip-vae", action="store_true",
                       help="Skip VAE training (use existing components)")
    
    # Advanced options
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    
    # Visualization options
    parser.add_argument("--visualization-level", type=str, choices=["minimal", "standard", "full"], default="standard",
                       help="Visualization complexity level (default: standard)")
    parser.add_argument("--include-large-files", action="store_true",
                       help="Log large files (HTML, high-res images) to wandb")
    
    # New CLI arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--n_train_samples", type=int, default=3000, help="Number of training samples (default: 3000)")
    parser.add_argument("--n_val_samples", type=int, default=800, help="Number of validation samples (default: 800)")
    
    return parser.parse_args()


def run_vae_training(args, output_dir):
    """Run vanilla VAE training + metric extraction using existing script."""
    print(f"\n{'='*80}")
    print(f"🚀 STEP 1: Training Vanilla VAE + Extracting Metric")
    print(f"{'='*80}")
    
    # (1) Run modular vanilla VAE training and visualization as a subprocess
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    vanilla_run_name = f"vanilla_vae_{args.architecture}_ld{args.latent_dim}_{timestamp}"
    print("\n[PIPELINE] Vanilla VAE run parameters:")
    print(f"  Run name: {vanilla_run_name}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Epochs: {args.vae_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Train samples: {args.n_train_samples}")
    print(f"  Val samples: {args.n_val_samples}")
    print(f"  Visualization level: {args.visualization_level}")
    print(f"  WandB project: {args.wandb_project}")
    vanilla_vae_cmd = [
        sys.executable, "src/training/train_with_modular_visualizations.py",
        "--loop_mode", "open",
        "--latent_dim", str(args.latent_dim),
        "--n_epochs", str(args.vae_epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--n_train_samples", str(args.n_train_samples),
        "--n_val_samples", str(args.n_val_samples),
        "--wandb_run_name", vanilla_run_name,
        "--wandb_project", args.wandb_project,
        "--visualization_level", args.visualization_level,
        "--wandb_only"
    ]
    # Set visualization level (only once)
    viz_level = "full" if args.include_large_files else args.visualization_level
    vanilla_vae_cmd.extend(["--visualization_level", viz_level])
    if args.wandb:
        vanilla_vae_cmd.append("--wandb_only")
    # Set PYTHONPATH to project root for subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.resolve())
    
    # Find the most recent metric file matching the architecture and latent dim
    metric_pattern = f"data/pretrained/metric_diverse_{args.architecture}_ld{args.latent_dim}_*.pt"
    metric_files = sorted(glob.glob(metric_pattern), reverse=True)
    if not metric_files:
        raise FileNotFoundError(f"No metric file found matching {metric_pattern}")
    metric_path = metric_files[0]
    vanilla_vae_cmd.extend(["--metric_path", metric_path])
    
    print(f"[PIPELINE] Running vanilla VAE training: {' '.join(vanilla_vae_cmd)}")
    subprocess.run(vanilla_vae_cmd, check=True, env=env)
    
    # After training, find the latest encoder/decoder/metric files in data/pretrained/
    pretrained_dir = Path("data/pretrained")
    encoder_files = sorted(pretrained_dir.glob(f"encoder_diverse_{args.architecture}_ld{args.latent_dim}_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    decoder_files = sorted(pretrained_dir.glob(f"decoder_diverse_{args.architecture}_ld{args.latent_dim}_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    metric_files = sorted(pretrained_dir.glob(f"metric_diverse_{args.architecture}_ld{args.latent_dim}_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not encoder_files or not decoder_files or not metric_files:
        raise FileNotFoundError("Could not find generated VAE components in data/pretrained/")
    encoder_path = encoder_files[0]
    decoder_path = decoder_files[0]
    metric_path = metric_files[0]
    # Copy to experiment directory
    vae_dir = output_dir / "vanilla_vae"
    vae_dir.mkdir(parents=True, exist_ok=True)
    exp_encoder = vae_dir / "encoder.pt"
    exp_decoder = vae_dir / "decoder.pt"
    exp_metric = vae_dir / "metric.pt"
    shutil.copy2(encoder_path, exp_encoder)
    shutil.copy2(decoder_path, exp_decoder)
    shutil.copy2(metric_path, exp_metric)
    print(f"📁 Copied components to experiment directory:")
    print(f"   Encoder: {exp_encoder}")
    print(f"   Decoder: {exp_decoder}")
    print(f"   Metric: {exp_metric}")
    
    # Load the trained vanilla VAE model using modular_vanilla_vae
    from src.models.modular_vanilla_vae import ModularVanillaVAE
    vae_model = ModularVanillaVAE(
        input_dim=(3, 64, 64),
        latent_dim=args.latent_dim,
        encoder_architecture=args.architecture,
        decoder_architecture=args.architecture,
        beta=1.0
    )
    # Load encoder and decoder weights directly into submodules
    encoder_state = torch.load(exp_encoder, map_location=vae_model.device)
    decoder_state = torch.load(exp_decoder, map_location=vae_model.device)
    vae_model.encoder.load_state_dict(encoder_state)
    vae_model.decoder.load_state_dict(decoder_state)
    vae_model.eval()
    
    # Ensure wandb run is active before running visualizations
    if args.wandb and wandb.run is None:
        wandb.init(
            project=args.wandb_project,
            name=f"vanilla_vae_{args.architecture}_ld{args.latent_dim}",
            group=experiment_group if 'experiment_group' in locals() else None,
            tags=["vanilla_vae", "pipeline", args.architecture]
        )
    # Prepare visualization config
    viz_level = VisualizationLevel[args.visualization_level.upper()]
    viz_config = VisualizationConfig.from_level(viz_level)
    # Optionally enable large files
    viz_config.enable_fancy_plots = args.include_large_files
    # Create visualization manager
    viz_manager = VisualizationManager(vae_model, device="cuda" if torch.cuda.is_available() else "cpu", config=args, viz_config=viz_config)
    # Run visualizations on a sample batch (load a small batch from train data)
    sample_data = torch.load('data/processed/Sprites_train_cyclic.pt')[:8]  # Small batch
    viz_manager.create_visualizations(sample_data, epoch=args.vae_epochs)
    # Test: log a dummy image to wandb to confirm media logging
    if args.wandb:
        dummy_img = np.random.rand(64, 64, 3)
        plt.imshow(dummy_img)
        plt.title("Dummy Test Image")
        plt.axis('off')
        plt.savefig("dummy_test_image.png")
        wandb.log({"test/dummy_image": wandb.Image("dummy_test_image.png", caption="Dummy Test Image")})
        plt.close()
    
    return exp_encoder, exp_decoder, exp_metric, vanilla_run_name


def create_rlvae_config(args, encoder_path, decoder_path, metric_path):
    """Create RLVAE configuration based on architecture and parameters."""
    
    # Set input dimension based on architecture
    if args.architecture == "mlp":
        # MLP needs flattened input dimension
        input_dim = [12288]  # 3 * 64 * 64 = 12288
    else:
        # CNN and ResNet use spatial dimensions
        input_dim = [3, 64, 64]
    
    # Base config
    config = {
        "_target_": "src.models.modular_rlvae.ModularRiemannianFlowVAE",
        "input_dim": input_dim,
        "latent_dim": args.latent_dim,
        "n_flows": 8,
        "flow_hidden_size": 256,
        "flow_n_blocks": 2,
        "flow_n_hidden": 1,
        "epsilon": 1e-6,
        "beta": 1.0,
        "riemannian_beta": 8.0,
        
        "encoder": {
            "architecture": args.architecture
        },
        "decoder": {
            "architecture": args.architecture
        },
        
        "posterior": {
            "type": "riemannian_metric"
        },
        "sampling": {
            "method": "geodesic",
            "use_riemannian": True
        },
        "loop": {
            "mode": "open",
            "penalty": 5.0
        },
        "metric": {
            "temperature_override": 3.0
        },
        "pretrained": {
            "encoder_path": str(encoder_path),
            "decoder_path": str(decoder_path),
            "metric_path": str(metric_path)
        }
    }
    
    # Architecture-specific configurations
    if args.architecture == "cnn":
        config["encoder"].update({
            "layers": [32, 64, 128, 256],
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "activation": "relu",
            "batch_norm": True
        })
        config["decoder"].update({
            "layers": [256, 128, 64, 32],
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "output_padding": 1,
            "activation": "relu",
            "batch_norm": True
        })
    elif args.architecture == "resnet":
        config["encoder"].update({
            "layers": [64, 128, 256, 512],
            "block_type": "basic",
            "num_blocks": [2, 2, 2, 2],
            "activation": "relu",
            "batch_norm": True
        })
        config["decoder"].update({
            "layers": [512, 256, 128, 64],
            "block_type": "basic",
            "num_blocks": [2, 2, 2, 2],
            "activation": "relu",
            "batch_norm": True
        })
    elif args.architecture == "mlp":
        config["encoder"].update({
            "hidden_dims": [1024, 512, 256],
            "dropout": 0.1
        })
        config["decoder"].update({
            "hidden_dims": [256, 512, 1024],
            "dropout": 0.1
        })
    
    return DictConfig(config)


def train_rlvae(args, config, output_dir, experiment_group=None):
    """Train RLVAE using the modular system."""
    print(f"\n{'='*80}")
    print(f"🚀 STEP 2: Training RLVAE ({args.architecture.upper()})")
    print(f"{'='*80}")

    # Always launch as subprocess, passing wandb args
    from datetime import datetime
    import sys, os
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    rlvae_run_name = f"pipeline_stage2_rlvae_{args.architecture}_ld{args.latent_dim}"
    print("\n[PIPELINE] RLVAE run parameters:")
    print(f"  Run name: {rlvae_run_name}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Epochs: {args.rlvae_epochs}")
    print(f"  Batch size: {getattr(args, 'rlvae_batch_size', args.batch_size)}")
    print(f"  Learning rate: {getattr(args, 'rlvae_lr', args.learning_rate)}")
    print(f"  Train samples: {args.n_train_samples}")
    print(f"  Val samples: {args.n_val_samples}")
    print(f"  Visualization level: {args.visualization_level}")
    print(f"  WandB project: {args.wandb_project}")
    rlvae_cmd = [
        sys.executable, "src/training/train_with_modular_visualizations.py",
        "--loop_mode", "open",
        "--latent_dim", str(args.latent_dim),
        "--n_epochs", str(args.rlvae_epochs),
        "--batch_size", str(getattr(args, 'rlvae_batch_size', args.batch_size)),
        "--learning_rate", str(getattr(args, 'rlvae_lr', args.learning_rate)),
        "--n_train_samples", str(args.n_train_samples),
        "--n_val_samples", str(args.n_val_samples),
        "--wandb_run_name", rlvae_run_name,
        "--wandb_project", args.wandb_project,
        "--visualization_level", args.visualization_level,
        "--wandb_only"
    ]
    # Add metric, encoder, decoder paths if needed
    if hasattr(args, 'metric_path') and args.metric_path:
        rlvae_cmd.extend(["--metric_path", args.metric_path])
    if hasattr(args, 'encoder_path') and args.encoder_path:
        rlvae_cmd.extend(["--encoder_path", args.encoder_path])
    if hasattr(args, 'decoder_path') and args.decoder_path:
        rlvae_cmd.extend(["--decoder_path", args.decoder_path])
    # Set PYTHONPATH to project root for subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.resolve())
    print(f"[PIPELINE] Running RLVAE training: {' '.join(rlvae_cmd)}")
    subprocess.run(rlvae_cmd, check=True, env=env)
    return rlvae_run_name


def main():
    """Main pipeline function."""
    print(f"🎯 Starting Global RLVAE Pipeline")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse arguments
    args = parse_args()
    
    # Setup output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Experiment directory: {output_path}")
    print(f"🏗️  Architecture: {args.architecture.upper()}")
    print(f"🧠 Latent dimension: {args.latent_dim}")
    
    try:
        # Step 1: Train Vanilla VAE + Extract Metric (unless skipped)
        if not args.skip_vae:
            encoder_path, decoder_path, metric_path, vanilla_run_name = run_vae_training(args, output_path)
        else:
            print("⏭️ Skipping VAE training (using existing components)")
            encoder_path = output_path / "vanilla_vae" / "encoder.pt"
            decoder_path = output_path / "vanilla_vae" / "decoder.pt"
            metric_path = output_path / "vanilla_vae" / "metric.pt"
            
            if not all(p.exists() for p in [encoder_path, decoder_path, metric_path]):
                raise FileNotFoundError("Missing pretrained components for skipped VAE training")
        
        # Step 2: Create RLVAE config
        print(f"\n🔧 Creating RLVAE configuration...")
        rlvae_config = create_rlvae_config(args, encoder_path, decoder_path, metric_path)
        
        # Step 3: Train RLVAE
        rlvae_run_name = train_rlvae(args, rlvae_config, output_path)
        
        # Success summary
        print(f"\n{'='*80}")
        print(f"🎉 GLOBAL RLVAE PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"📁 All outputs saved to: {output_path}")
        print(f"📊 Vanilla VAE components:")
        print(f"   - Encoder: {encoder_path}")
        print(f"   - Decoder: {decoder_path}")
        print(f"   - Metric: {metric_path}")
        
        # Create summary file
        summary = {
            "status": "completed",
            "completion_time": datetime.now().isoformat(),
            "pipeline_config": {
                "architecture": args.architecture,
                "latent_dim": args.latent_dim,
                "vae_epochs": args.vae_epochs,
                "rlvae_epochs": args.rlvae_epochs
            },
            "outputs": {
                "experiment_dir": str(output_path),
                "vanilla_vae": {
                    "encoder": str(encoder_path),
                    "decoder": str(decoder_path),
                    "metric": str(metric_path)
                }
            }
        }
        
        summary_path = output_path / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Pipeline summary: {summary_path}")
        
        print("\n================ Pipeline WandB Run Summary ================")
        print(f"Vanilla VAE run: {vanilla_run_name}")
        print(f"  Architecture: {args.architecture}, Latent dim: {args.latent_dim}, Epochs: {args.vae_epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}, Viz: {args.visualization_level}")
        print(f"RLVAE run: {rlvae_run_name}")
        print(f"  Architecture: {args.architecture}, Latent dim: {args.latent_dim}, Epochs: {args.rlvae_epochs}, Batch size: {getattr(args, 'rlvae_batch_size', args.batch_size)}, LR: {getattr(args, 'rlvae_lr', args.learning_rate)}, Viz: {args.visualization_level}")
        print(f"WandB project: {args.wandb_project}")
        print("==========================================================\n")
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED!")
        print(f"Error: {e}")
        
        # Create failure summary
        failure_summary = {
            "status": "failed",
            "failure_time": datetime.now().isoformat(),
            "error": str(e),
            "pipeline_config": {
                "architecture": args.architecture,
                "latent_dim": args.latent_dim,
                "vae_epochs": args.vae_epochs,
                "rlvae_epochs": args.rlvae_epochs
            }
        }
        
        failure_path = output_path / "pipeline_failure.json"
        with open(failure_path, 'w') as f:
            json.dump(failure_summary, f, indent=2)
        
        print(f"📄 Failure report: {failure_path}")
        raise


if __name__ == "__main__":
    main() 