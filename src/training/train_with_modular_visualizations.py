#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the src directory to Python path (for when running from training dir)
current_dir = Path(__file__).parent.absolute() if '__file__' in globals() else Path('.')
src_dir = current_dir.parent if '__file__' in globals() else Path('src')
lib_src_dir = src_dir / "lib" / "src"
project_root = src_dir.parent if '__file__' in globals() else Path('.')

# Add both src directories to path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(lib_src_dir) not in sys.path:
    sys.path.insert(0, str(lib_src_dir))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace
from tqdm import tqdm
from sklearn.decomposition import PCA
import argparse
import datetime
import warnings
from PIL import Image

# PIL decompression bomb fix
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", ".*DecompressionBombWarning.*")

# Model imports
from models.riemannian_flow_vae import RiemannianFlowVAE

# MODULAR VISUALIZATION IMPORTS
from visualizations.manager import VisualizationManager


class CyclicSpritesDataset(Dataset):
    """Cyclic Sprites dataset for loop mode comparison."""
    
    def __init__(self, data_path, subset_size=None, split='train'):
        print(f"Loading cyclic sprites data from {data_path}...")
        
        # Load cyclic data
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        
        print(f"Cyclic sprites data shape: {data.shape}")
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        
        # Data should already be in [N, T, C, H, W] format and normalized
        self.data = data.float()
        
        if subset_size is not None:
            self.data = self.data[:subset_size]
        
        print(f"✅ {split} cyclic dataset size: {len(self.data)}")
        print(f"✅ Final data shape: {self.data.shape}")
        print(f"✅ Final data range: [{self.data.min():.3f}, {self.data.max():.3f}]")
        
        # Verify cyclicity
        print(f"🔍 Verifying cyclicity of first 5 sequences:")
        for i in range(min(5, len(self.data))):
            seq = self.data[i]
            mse = torch.mean((seq[0] - seq[-1]) ** 2).item()
            print(f"   Seq {i}: MSE = {mse:.2e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]  # [T, C, H, W]


class CleanCyclicLoopTrainer:
    """Clean trainer for comparing loop modes on cyclic sequences with modular visualizations."""
    
    def __init__(self, config, project_name="cyclic-loop-mode-comparison", run_name=None):
        self.config = config
        self.project_name = project_name
        self.run_name = run_name or f"{config.loop_mode}_cyclic"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        self.model = RiemannianFlowVAE(
            input_dim=(3, 64, 64),
            latent_dim=16,
            n_flows=8,
            flow_hidden_size=256,
            flow_n_blocks=2,
            flow_n_hidden=1,
            epsilon=1e-6,
            beta=config.beta,
            loop_mode=config.loop_mode,  # This is the key parameter we're testing!
            posterior_type=getattr(config, 'posterior_type', 'gaussian'),
            riemannian_beta=getattr(config, 'riemannian_beta', None),
        ).to(self.device)
        
        # Set cycle penalty
        if hasattr(self.model, "set_loop_mode"):
            self.model.set_loop_mode(config.loop_mode, config.cycle_penalty)
            print(f"🔁 Loop mode: {config.loop_mode}, cycle penalty: {config.cycle_penalty}")
        
        # Load pretrained components
        self.model.load_pretrained_components(
            encoder_path=str(project_root / "data" / "pretrained" / "encoder.pt"),
            decoder_path=str(project_root / "data" / "pretrained" / "decoder.pt"),
            metric_path=str(project_root / "data" / "pretrained" / config.metric_path),
            temperature_override=config.temperature_fix
        )
        
        # Configure Riemannian sampling
        if config.riemannian_method in ['geodesic', 'enhanced', 'basic']:
            sampling_method = "custom"
        else:
            sampling_method = config.riemannian_method
        
        self.model.enable_pure_rhvae(enable=config.use_riemannian, method=sampling_method)
        self.model._riemannian_method = config.riemannian_method
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.8,
            patience=8,
            threshold=0.01,
            min_lr=1e-7
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.experiment_name = f"{config.loop_mode}_cyclic_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize W&B
        wandb_mode = "offline" if getattr(config, 'wandb_offline', False) else "online"
        wandb.init(
            project=self.project_name,
            name=self.run_name,
            mode=wandb_mode,
            config={
                "experiment_type": f"Cyclic Sequences - Loop Mode {config.loop_mode.upper()}",
                "loop_mode": config.loop_mode,
                "cycle_penalty": config.cycle_penalty,
                "use_riemannian": config.use_riemannian,
                "riemannian_method": config.riemannian_method,
                "posterior_type": config.posterior_type,
                "riemannian_beta": getattr(config, 'riemannian_beta', config.beta),
                "temperature_fix": config.temperature_fix,
                "input_dim": config.input_dim,
                "latent_dim": config.latent_dim,
                "n_flows": config.n_flows,
                "learning_rate": config.learning_rate,
                "beta": config.beta,
                "batch_size": config.batch_size,
                "n_epochs": config.n_epochs,
                "dataset_type": "cyclic_only",
                "train_size": config.n_train_samples,
                "val_size": config.n_val_samples,
                "visualization_level": getattr(config, 'visualization_level', 'standard'),
                "wandb_only": getattr(config, 'wandb_only', False),
                "disable_local_files": getattr(config, 'disable_local_files', False),
            }
        )
        
        # 🎨 MODULAR VISUALIZATION MANAGER
        from visualizations.manager import VisualizationLevel, VisualizationConfig
        
        # Get visualization level from config
        level_str = getattr(config, 'visualization_level', 'standard')
        level_enum = VisualizationLevel(level_str)
        viz_config = VisualizationConfig.from_level(level_enum)
        
        print(f"🎨 Visualization config: level={viz_config.level.value}, interactive={viz_config.enable_interactive}, freq={viz_config.interactive_frequency}")
        
        self.viz_manager = VisualizationManager(
            model=self.model,
            config=config,
            device=self.device,
            viz_config=viz_config
        )
        
        print(f"✅ Created {config.loop_mode} loop trainer for cyclic sequences")
        print(f"   🎯 Experiment: {self.experiment_name}")
        print(f"   🔁 Loop mode: {config.loop_mode}")
        print(f"   📊 Cycle penalty: {config.cycle_penalty}")
        print(f"   🚀 Riemannian: {config.use_riemannian} ({config.riemannian_method})")
        print(f"   🎨 Visualization level: {getattr(config, 'visualization_level', 'standard')}")
        
        # File saving options
        if getattr(config, 'wandb_only', False) or getattr(config, 'disable_local_files', False):
            print(f"   💾 Local file saving: DISABLED (WandB only)")
    
    def should_log_to_wandb(self):
        """Check if we should log to WandB."""
        return wandb.run is not None
    
    def model_forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_flow = 0
        total_reinforce = 0
        total_cycle = 0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} ({self.config.loop_mode})")
        
        for batch_idx, x in enumerate(pbar):
            x = x.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass
                result = self.model_forward(x)
                
                # Extract losses
                loss = result.loss
                recon_loss = result.recon_loss
                kl_loss = result.kld_loss
                flow_loss = result.flow_loss
                reinforce_loss = result.reinforce_loss
                
                # For closed loop, calculate cycle penalty separately if needed
                cycle_penalty = 0.0
                if self.config.loop_mode == "closed" and hasattr(result, 'z'):
                    z_seq = result.z
                    cycle_penalty = torch.mean((z_seq[:, 0] - z_seq[:, -1]) ** 2)
                
                # Check for non-finite values
                if not torch.isfinite(loss):
                    print(f"⚠️ Non-finite loss at batch {batch_idx + 1}, skipping")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Accumulate metrics
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                total_flow += flow_loss.item()
                total_reinforce += reinforce_loss.item()
                total_cycle += cycle_penalty.item() if isinstance(cycle_penalty, torch.Tensor) else cycle_penalty
                n_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Recon': f'{recon_loss.item():.4f}',
                    'KL': f'{kl_loss.item():.4f}',
                    'Flow': f'{flow_loss.item():.4f}',
                    'Cycle': f'{cycle_penalty:.4f}' if isinstance(cycle_penalty, (int, float)) else f'{cycle_penalty.item():.4f}',
                    'Mode': self.config.loop_mode
                })
                
                # Store gradient norm for epoch-level logging
                if not hasattr(self, '_current_grad_norms'):
                    self._current_grad_norms = []
                self._current_grad_norms.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                
            except Exception as e:
                print(f"⚠️ Error in batch {batch_idx + 1}: {e}")
                continue
        
        if n_batches == 0:
            print("⚠️ No valid batches processed!")
            return {'loss': float('inf'), 'recon': float('inf'), 'kl': float('inf'), 'flow': float('inf'), 'reinforce': float('inf'), 'cycle': float('inf'), 'grad_norm': float('inf')}
        
        # Calculate average gradient norm
        avg_grad_norm = sum(self._current_grad_norms) / len(self._current_grad_norms) if hasattr(self, '_current_grad_norms') and self._current_grad_norms else 0.0
        self._current_grad_norms = []  # Reset for next epoch
        
        # Return average metrics
        return {
            'loss': total_loss / n_batches,
            'recon': total_recon / n_batches,
            'kl': total_kl / n_batches,
            'flow': total_flow / n_batches,
            'reinforce': total_reinforce / n_batches,
            'cycle': total_cycle / n_batches,
            'grad_norm': avg_grad_norm
        }
    
    def validate_epoch(self, val_loader):
        """Validation epoch."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                result = self.model_forward(batch)
                val_losses.append(result.loss.item())
        
        self.model.train()
        return np.mean(val_losses)
    
    def _get_output_path(self, filename, subfolder="visualizations"):
        """Get output path for saving files."""
        output_dir = Path("wandb") / "outputs" / subfolder
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename
    
    def train(self, n_epochs=30):
        """Main training loop with modular visualizations."""
        print(f"\n🚀 Starting {self.config.loop_mode.upper()} loop training on cyclic sequences")
        print(f"🎨 Using MODULAR visualization system with level: {getattr(self.config, 'visualization_level', 'standard')}")
        
        # Create datasets
        if self.config.loop_mode == "open":
            train_path = str(project_root / "data" / "processed" / "Sprites_train_cyclic.pt")
        else:
            train_path = str(project_root / "data" / "processed" / "Sprites_train_cyclic.pt")
        
        train_dataset = CyclicSpritesDataset(train_path, subset_size=self.config.n_train_samples)
        val_dataset = CyclicSpritesDataset(train_path, subset_size=self.config.n_val_samples)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            drop_last=True,
            num_workers=2
        )
        
        self.model.train()
        
        for epoch in range(n_epochs):
            print(f"\n📈 Epoch {epoch+1}/{n_epochs} - {self.config.loop_mode.upper()} Mode")
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_loss = self.validate_epoch(val_loader)
            
            # Log training metrics
            train_log = {
                'train/total_loss': train_metrics['loss'],
                'train/recon_loss': train_metrics['recon'],
                'train/kl_loss': train_metrics['kl'],
                'train/flow_loss': train_metrics['flow'],
                'train/cycle_penalty': train_metrics['cycle'],
                'train/gradient_norm': train_metrics['grad_norm'],
                'val/total_loss': val_loss,
                'optimization/learning_rate': self.optimizer.param_groups[0]['lr'],
                'config/loop_mode': self.config.loop_mode,
                'system/memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                'epoch': epoch
            }
            wandb.log(train_log)
            
            # 🎨 MODULAR VISUALIZATIONS
            if epoch % self.config.visualization_frequency == 0:
                try:
                    # Get validation sample for visualization
                    val_sample = next(iter(val_loader))
                    val_sample = val_sample.to(self.device)
                    
                    print(f"🎨 Creating modular visualizations (level: {self.viz_manager.viz_config.level.value})...")
                    
                    # Use the modular visualization manager with its own frequency logic
                    self.viz_manager.create_visualizations(
                        x_sample=val_sample,
                        epoch=epoch
                    )
                    
                    print(f"✅ Modular visualizations completed for epoch {epoch}")
                    
                except Exception as viz_error:
                    print(f"⚠️ Modular visualization failed: {viz_error}")
                    import traceback
                    traceback.print_exc()
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            print(f"✅ Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_loss:.4f}, Cycle: {train_metrics['cycle']:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                
                model_filename = f'best_cyclic_{self.config.loop_mode}_model_epoch_{epoch}.pt'
                model_path = self._get_output_path(model_filename, "models")
                torch.save(self.model.state_dict(), model_path)
                print(f"💾 Saved best {self.config.loop_mode} model to {model_path} (val_loss: {self.best_val_loss:.4f})")
                
                wandb.log({
                    'best_model/epoch': epoch,
                    'best_model/val_loss': val_loss,
                    'best_model/train_loss': train_metrics['loss'],
                    'best_model/loop_mode': self.config.loop_mode
                })
        
        print(f"\n🎉 {self.config.loop_mode.upper()} training completed!")
        print(f"🎨 Modular visualization system successfully integrated!")
        
        wandb.finish()


def create_config(loop_mode, experiment_suffix=""):
    """Create configuration for a specific loop mode."""
    return SimpleNamespace(
        # Loop mode configuration
        loop_mode=loop_mode,  # "open" or "closed"
        cycle_penalty=5.0 if loop_mode == "closed" else 0.0,
        
        # Model parameters
        input_dim=(3, 64, 64),
        latent_dim=16,
        n_flows=8,
        flow_hidden_size=256,
        flow_n_blocks=2,
        flow_n_hidden=1,
        epsilon=1e-6,
        beta=1.0,
        
        # Separate beta for Riemannian KL
        riemannian_beta=8.0,
        
        # Training parameters
        batch_size=8,
        learning_rate=3e-4,
        n_epochs=25,
        
        # Data parameters
        n_train_samples=1000,
        n_val_samples=600,
        
        # Riemannian parameters
        use_riemannian=True,
        riemannian_method="geodesic",
        
        # Posterior type parameters
        posterior_type="riemannian_metric",
        
        # Temperature fix
        temperature_fix=3.0,
        
        # Pretrained components
        metric_path="metric_T0.7_scaled.pt",
        
        # MODULAR VISUALIZATION PARAMETERS
        visualization_level="standard",  # minimal, basic, standard, advanced, full
        visualization_frequency=5,
        
        # Experiment naming
        experiment_suffix=experiment_suffix
    )


def main():
    """Main function to run clean training with modular visualizations."""
    parser = argparse.ArgumentParser(description='Clean training with modular visualizations')
    parser.add_argument('--loop_mode', choices=['open', 'closed'], required=True,
                       help='Loop mode to train: open or closed')
    parser.add_argument('--visualization_level', choices=['minimal', 'basic', 'standard', 'advanced', 'full'],
                       default='standard', help='Visualization complexity level')
    parser.add_argument('--cycle_penalty', type=float, default=1.0,
                       help='Weight for cycle penalty when loop_mode=closed')
    parser.add_argument('--n_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n_train_samples', type=int, default=3000, help='Number of training samples')
    parser.add_argument('--n_val_samples', type=int, default=800, help='Number of validation samples')
    parser.add_argument('--visualization_frequency', type=int, default=5, help='Visualization frequency')
    parser.add_argument('--run_name', type=str, default=None, help='Custom run name')
    parser.add_argument('--riemannian_beta', type=float, default=None, 
                       help='Beta weight for Riemannian KL divergence')
    parser.add_argument('--wandb_only', action='store_true', default=False,
                       help='Only log to WandB, do not save files locally')
    parser.add_argument('--disable_local_files', action='store_true', default=False,
                       help='Disable saving visualization files locally')
    parser.add_argument('--wandb_offline', action='store_true', default=False,
                       help='Run WandB in offline mode')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting {args.loop_mode.upper()} Clean Training with Modular Visualizations")
    print(f"🎯 Mode: {args.loop_mode}")
    print(f"🎨 Visualization Level: {args.visualization_level}")
    print(f"📊 Epochs: {args.n_epochs}")
    print(f"📈 Samples: {args.n_train_samples}")
    
    # Create config for this mode
    config = create_config(args.loop_mode, experiment_suffix=f"_modular_{args.n_epochs}epochs")
    
    # Update config with command line arguments
    config.n_epochs = args.n_epochs
    config.n_train_samples = args.n_train_samples
    config.n_val_samples = args.n_val_samples
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.cycle_penalty = args.cycle_penalty
    config.visualization_frequency = args.visualization_frequency
    config.visualization_level = args.visualization_level
    config.wandb_only = args.wandb_only
    config.disable_local_files = args.disable_local_files
    config.wandb_offline = args.wandb_offline
    
    # Update Riemannian beta if provided
    if args.riemannian_beta is not None:
        config.riemannian_beta = args.riemannian_beta
    
    # Set experiment name
    project_name = "cyclic-loop-modular-visualizations"
    run_name = args.run_name or f"{args.loop_mode}_modular_{args.visualization_level}"
    
    # Create and run trainer
    try:
        trainer = CleanCyclicLoopTrainer(config, project_name=project_name, run_name=run_name)
        trainer.train(n_epochs=args.n_epochs)
        
        print(f"✅ {args.loop_mode.upper()} clean training with modular visualizations completed!")
        
    except Exception as e:
        print(f"❌ {args.loop_mode.upper()} clean training failed: {e}")
        raise
    
    print(f"🏁 Clean training experiment finished")
    print(f"🎨 Modular visualization system successfully tested!")
    print(f"📊 Check Weights & Biases for organized visualizations")
    print(f"💡 Visualization levels available:")
    print(f"   • minimal: Essential metrics only")
    print(f"   • basic: Core visualizations")
    print(f"   • standard: Balanced analysis ({args.visualization_level})")
    print(f"   • advanced: Detailed manifold analysis")
    print(f"   • full: Complete visualization suite")


if __name__ == "__main__":
    main() 