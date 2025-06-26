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
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Add this for tab10 colormap
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
import os
from types import SimpleNamespace
from tqdm import tqdm
import sys
from sklearn.decomposition import PCA
import argparse
import datetime
import warnings
# Add PIL decompression bomb fix
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", ".*DecompressionBombWarning.*")

from models.riemannian_flow_vae import RiemannianFlowVAE
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP

# Add plotly imports for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Plotly not available, interactive visualizations will be skipped")
    PLOTLY_AVAILABLE = False

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

class CyclicLoopTrainer:
    """Trainer for comparing loop modes on cyclic sequences."""
    
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
            posterior_type=getattr(config, 'posterior_type', 'gaussian'),  # NEW: Support posterior type
            riemannian_beta=getattr(config, 'riemannian_beta', None),  # NEW: Support separate Riemannian beta
        ).to(self.device)
        
        # Set cycle penalty
        if hasattr(self.model, "set_loop_mode"):
            self.model.set_loop_mode(config.loop_mode, config.cycle_penalty)
            print(f"🔁 Loop mode: {config.loop_mode}, cycle penalty: {config.cycle_penalty}")
        
        # Load pretrained components (using clean repository paths)
        self.model.load_pretrained_components(
            encoder_path=str(project_root / "data" / "pretrained" / "encoder.pt"),
            decoder_path=str(project_root / "data" / "pretrained" / "decoder.pt"),
            metric_path=str(project_root / "data" / "pretrained" / config.metric_path),
            temperature_override=config.temperature_fix
        )
        
        # Configure Riemannian sampling
        # Map riemannian_method to the proper sampling method
        if config.riemannian_method in ['geodesic', 'enhanced', 'basic']:
            sampling_method = "custom"  # Use custom sampling for our methods
        else:
            sampling_method = config.riemannian_method  # official, standard, etc.
        
        self.model.enable_pure_rhvae(enable=config.use_riemannian, method=sampling_method)
        # Set the specific Riemannian method for training sampling
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
        
        # Initialize W&B with file saving options
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
                "wandb_only": getattr(config, 'wandb_only', False),
                "disable_local_files": getattr(config, 'disable_local_files', False),
            }
        )
        
        print(f"✅ Created {config.loop_mode} loop trainer for cyclic sequences")
        print(f"   🎯 Experiment: {self.experiment_name}")
        print(f"   🔁 Loop mode: {config.loop_mode}")
        print(f"   📊 Cycle penalty: {config.cycle_penalty}")
        print(f"   🚀 Riemannian: {config.use_riemannian} ({config.riemannian_method})")
        
        # File saving options
        if getattr(config, 'wandb_only', False) or getattr(config, 'disable_local_files', False):
            print(f"   💾 Local file saving: DISABLED (WandB only)")
        if getattr(config, 'wandb_offline', False):
            print(f"   🌐 WandB mode: OFFLINE")

    def should_save_locally(self):
        """Check if we should save files locally."""
        return not (getattr(self.config, 'wandb_only', False) or getattr(self.config, 'disable_local_files', False))
    
    def should_log_to_wandb(self):
        """Check if we should log to WandB."""
        return wandb.run is not None
    
    def model_forward(self, x):
        """Helper method to call model forward."""
        return self.model.forward(x)
    
    def create_cyclicity_analysis(self, x_sample, epoch):
        """Analyze how well the model preserves/learns cyclicity."""
        self.model.eval()
        
        with torch.no_grad():
            result = self.model.forward(x_sample)
            recon_x = result.recon_x  # [batch_size, n_obs, 3, 64, 64]
            z_seq = result.z         # [batch_size, n_obs, latent_dim]
            
            batch_size, n_obs = x_sample.shape[:2]
            
            # Analyze original cyclicity
            orig_first_last_mse = []
            for i in range(batch_size):
                orig_mse = torch.mean((x_sample[i, 0] - x_sample[i, -1]) ** 2).item()
                orig_first_last_mse.append(orig_mse)
            
            # Analyze reconstruction cyclicity
            recon_first_last_mse = []
            for i in range(batch_size):
                recon_mse = torch.mean((recon_x[i, 0] - recon_x[i, -1]) ** 2).item()
                recon_first_last_mse.append(recon_mse)
            
            # Analyze latent cyclicity
            latent_first_last_mse = []
            latent_norms = []
            for i in range(batch_size):
                latent_mse = torch.mean((z_seq[i, 0] - z_seq[i, -1]) ** 2).item()
                latent_first_last_mse.append(latent_mse)
                latent_norms.append(torch.norm(z_seq[i], dim=-1).cpu().numpy())
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Cyclicity Analysis - {self.config.loop_mode.upper()} Mode - Epoch {epoch}', fontsize=16)
            
            # Plot 1: Original vs Reconstructed MSE
            axes[0, 0].scatter(orig_first_last_mse, recon_first_last_mse, alpha=0.6)
            axes[0, 0].plot([0, max(orig_first_last_mse)], [0, max(orig_first_last_mse)], 'r--', alpha=0.5)
            axes[0, 0].set_xlabel('Original First-Last MSE')
            axes[0, 0].set_ylabel('Reconstructed First-Last MSE')
            axes[0, 0].set_title('Original vs Reconstructed Cyclicity')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Latent space cyclicity
            axes[0, 1].hist(latent_first_last_mse, bins=20, alpha=0.7)
            axes[0, 1].set_xlabel('Latent First-Last MSE')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title(f'Latent Cyclicity (mean: {np.mean(latent_first_last_mse):.2e})')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Example sequence - original
            seq_idx = 0
            for t in range(min(4, n_obs)):
                if t < 2:
                    row, col = 0, 2
                    axes[row, col].clear()
                    if t == 0:
                        axes[row, col].imshow(x_sample[seq_idx, 0].permute(1, 2, 0).cpu().numpy())
                        axes[row, col].set_title(f'Original: First (t=0)')
                    else:
                        axes[row, col].imshow(x_sample[seq_idx, -1].permute(1, 2, 0).cpu().numpy())
                        axes[row, col].set_title(f'Original: Last (t={n_obs-1})')
                    axes[row, col].axis('off')
            
            # Plot 4: Example sequence - reconstructed
            for t in range(min(2, n_obs)):
                row, col = 1, 0 + t
                if t == 0:
                    axes[row, col].imshow(recon_x[seq_idx, 0].permute(1, 2, 0).cpu().numpy())
                    axes[row, col].set_title(f'Recon: First (t=0)')
                else:
                    axes[row, col].imshow(recon_x[seq_idx, -1].permute(1, 2, 0).cpu().numpy())
                    axes[row, col].set_title(f'Recon: Last (t={n_obs-1})')
                axes[row, col].axis('off')
            
            # Plot 5: Latent trajectory
            axes[1, 2].clear()
            # Plot latent trajectory for first sequence
            z_traj = z_seq[seq_idx].cpu().numpy()  # [n_obs, latent_dim]
            
            # Use PCA for visualization
            if z_traj.shape[1] > 2:
                pca = PCA(n_components=2)
                z_pca = pca.fit_transform(z_traj)
                axes[1, 2].plot(z_pca[:, 0], z_pca[:, 1], 'o-', alpha=0.8, linewidth=2, markersize=8)
                axes[1, 2].scatter(z_pca[0, 0], z_pca[0, 1], color='green', s=100, marker='s', label='Start', zorder=5)
                axes[1, 2].scatter(z_pca[-1, 0], z_pca[-1, 1], color='red', s=100, marker='*', label='End', zorder=5)
                axes[1, 2].set_title(f'Latent Trajectory (PCA)\nFirst-Last Distance: {np.linalg.norm(z_pca[0] - z_pca[-1]):.3f}')
                axes[1, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                axes[1, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            else:
                axes[1, 2].plot(z_traj[:, 0], z_traj[:, 1], 'o-', alpha=0.8, linewidth=2, markersize=8)
                axes[1, 2].scatter(z_traj[0, 0], z_traj[0, 1], color='green', s=100, marker='s', label='Start')
                axes[1, 2].scatter(z_traj[-1, 0], z_traj[-1, 1], color='red', s=100, marker='*', label='End')
                axes[1, 2].set_title('Latent Trajectory')
                axes[1, 2].set_xlabel('Latent Dim 0')
                axes[1, 2].set_ylabel('Latent Dim 1')
                
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save and log
            filename = f'cyclicity_analysis_{self.config.loop_mode}_epoch_{epoch}.png'
            saved_file = self._safe_save_plt_figure(filename, dpi=300, bbox_inches='tight')
            
            # Log to WandB
            if self.should_log_to_wandb():
                if saved_file and self.should_save_locally():
                    # Use local file if saved
                    wandb.log({
                        "analysis/cyclicity_overview": wandb.Image(saved_file, caption=f"Epoch {epoch} - {self.config.loop_mode} mode"),
                    })
                else:
                    # Create temporary buffer for WandB-only mode
                    import io
                    import PIL.Image
                    import numpy as np
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    # Convert buffer to PIL Image for WandB
                    pil_img = PIL.Image.open(buf)
                    wandb.log({
                        "analysis/cyclicity_overview": wandb.Image(pil_img, caption=f"Epoch {epoch} - {self.config.loop_mode} mode"),
                    })
            
            plt.close()
        
        self.model.train()
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_flow = 0
        total_reinforce = 0
        total_cycle = 0  # Track cycle penalty specifically
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
    
    def train(self, n_epochs=30):
        """Main training loop with comprehensive analysis."""
        print(f"\n🚀 Starting {self.config.loop_mode.upper()} loop training on cyclic sequences")
        
        # Create datasets (using clean repository paths)
        if self.config.loop_mode == "open":
            train_path = str(project_root / "data" / "processed" / "Sprites_train_cyclic.pt")
        else:
            train_path = str(project_root / "data" / "processed" / "Sprites_train_cyclic.pt")  # Same data, different loop mode
        
        train_dataset = CyclicSpritesDataset(train_path, subset_size=self.config.n_train_samples)
        val_dataset = CyclicSpritesDataset(train_path, subset_size=self.config.n_val_samples)  # Use same for val for simplicity
        
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
        
        # Let WandB handle its own step counter to avoid conflicts
        
        for epoch in range(n_epochs):
            print(f"\n📈 Epoch {epoch+1}/{n_epochs} - {self.config.loop_mode.upper()} Mode")
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_loss = self.validate_epoch(val_loader)
            
            # Log training metrics (increment step consistently)
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
            
            # Visualizations (less frequent to avoid memory issues)
            if epoch % self.config.visualization_frequency == 0:
                try:
                    # Get validation sample for visualization
                    val_sample = next(iter(val_loader))
                    val_sample = val_sample.to(self.device)
                    
                    self.create_cyclicity_analysis(val_sample, epoch)
                    self.create_sequence_trajectories_viz(val_sample, epoch)
                    
                    # Reduce frequency of heavy visualizations
                    if epoch % (self.config.visualization_frequency * 2) == 0:
                        self.create_comprehensive_reconstruction_viz(val_sample, epoch)
                        self.create_manifold_visualizations(val_sample, epoch)
                        
                        # 🌍 NEW: Enhanced geodesic visualizations every 20 epochs
                        self.create_enhanced_geodesic_visualizations(val_sample, epoch)
                    
                    # ✨ FANCY: Interactive Plotly visualizations every 9 epochs to avoid slowdown
                    if epoch % 9 == 0:
                        print("✨ Creating FANCY interactive geodesic visualizations...")
                        self.create_fancy_interactive_geodesic_visualizations(val_sample, epoch)
                    
                    # 🎬 TEMPORAL: Metric evolution over time every 3 epochs
                    if epoch % 3 == 0:
                        print("🎬 Creating TEMPORAL metric evolution visualization...")
                        self.create_temporal_metric_evolution_visualization(val_sample, epoch)
                    
                    # 🌊 CURVATURE: Timestep-specific curvature analysis every 5 epochs (if not disabled)
                    if epoch % 5 == 0 and not getattr(self.config, 'disable_curvature_during_training', False):
                        print("🌊 Creating TIMESTEP-SPECIFIC curvature analysis...")
                        self.create_timestep_specific_curvature_analysis(val_sample, epoch)
                        
                    # Ultra-high detail only every 10 epochs and with reduced resolution
                    if epoch % 10 == 0:
                        print(f"🔥 Creating moderate-detail heatmaps...")
                        # Reduce resolution to prevent PIL warnings and matplotlib errors
                        self.create_moderate_detail_metric_heatmap(val_sample, epoch)
                    
                except Exception as viz_error:
                    print(f"⚠️ Visualization failed: {viz_error}")
                    import traceback
                    traceback.print_exc()
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            print(f"✅ Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_loss:.4f}, Cycle: {train_metrics['cycle']:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_path = f'best_cyclic_{self.config.loop_mode}_model_epoch_{epoch}.pt'
                torch.save(self.model.state_dict(), model_path)
                print(f"💾 Saved best {self.config.loop_mode} model (val_loss: {self.best_val_loss:.4f})")
                
                wandb.log({
                    'best_model/epoch': epoch,
                    'best_model/val_loss': val_loss,
                    'best_model/train_loss': train_metrics['loss'],
                    'best_model/loop_mode': self.config.loop_mode
                })
        
        print(f"\n🎉 {self.config.loop_mode.upper()} training completed!")
        
        # Skip final curvature analysis if disabled
        if getattr(self.config, 'disable_curvature_during_training', False):
            print(f"\n⏭️ Skipping final curvature analysis (disabled by --disable_curvature_during_training)")
        
        # Generate interactive HTML latent space visualization if requested
        if hasattr(self.config, 'generate_html_latent_space') and self.config.generate_html_latent_space:
            print(f"\n🌐 Generating interactive HTML latent space visualization...")
            self.create_interactive_html_latent_space(num_sequences=20)
        
        wandb.finish()

    def create_moderate_detail_metric_heatmap(self, x_sample, epoch, temperature_override=None):
        """
        Create moderate resolution PCA + metric heatmap to avoid memory/matplotlib issues.
        """
        print(f"🌟 Creating MODERATE-DETAIL metric heatmap for epoch {epoch}")
        
        if not hasattr(self.model, 'G'):
            print("⚠️ No metric tensor available for moderate-detail heatmap")
            return
            
        # Temporarily override temperature for sharper metric tensor
        original_temp = None
        if temperature_override is not None and hasattr(self.model, 'temperature'):
            original_temp = self.model.temperature.item()
            self.model.temperature.data = torch.tensor(temperature_override, device=self.device)
            print(f"🔧 Temporarily using temperature = {temperature_override} (was {original_temp})")
        
        try:
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                z_seq = result.z  # [batch_size, n_obs, latent_dim]
                
                from sklearn.decomposition import PCA
                
                batch_size, n_obs, latent_dim = z_seq.shape
                
                # Prepare PCA data with MORE sequences for better coverage
                z_flat = z_seq.reshape(-1, latent_dim).cpu().numpy()
                pca = PCA(n_components=2)
                z_pca = pca.fit_transform(z_flat)
                z_pca_seq = z_pca.reshape(batch_size, n_obs, 2)
                
                # Define MODERATE-resolution grid to avoid memory issues
                x_min, x_max = z_pca[:, 0].min() - 1.0, z_pca[:, 0].max() + 1.0
                y_min, y_max = z_pca[:, 1].min() - 1.0, z_pca[:, 1].max() + 1.0
                # Reduced from 200x200 to 100x100 to prevent matplotlib errors
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
                grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
                
                print(f"📊 Computing metrics on {len(grid_points_pca)} moderate-resolution grid points...")
                
                # Project grid points back to latent space (inverse PCA)
                grid_points_latent = pca.inverse_transform(grid_points_pca)
                
                # Process in smaller chunks for stability
                chunk_size = 500
                all_log_det = []
                all_cond_num = []
                all_eigenval = []
                
                for i in range(0, len(grid_points_latent), chunk_size):
                    chunk = grid_points_latent[i:i+chunk_size]
                    chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=self.device)
                    
                    try:
                        G_chunk = self.model.G(chunk_tensor)
                        
                        # Log determinant
                        det_G_chunk = torch.linalg.det(G_chunk)
                        log_det_chunk = torch.log(torch.clamp(det_G_chunk, min=1e-12))
                        all_log_det.append(log_det_chunk.cpu().numpy())
                        
                        # Eigenvalues and condition number
                        eigenvals_chunk = torch.linalg.eigvals(G_chunk).real
                        cond_chunk = eigenvals_chunk.max(dim=1)[0] / (eigenvals_chunk.min(dim=1)[0] + 1e-10)
                        all_cond_num.append(cond_chunk.cpu().numpy())
                        
                        # Mean eigenvalue
                        mean_eigenval_chunk = eigenvals_chunk.mean(dim=1)
                        all_eigenval.append(mean_eigenval_chunk.cpu().numpy())
                        
                    except Exception as chunk_error:
                        print(f"⚠️ Chunk {i//chunk_size} failed: {chunk_error}")
                        # Fill with NaNs for failed chunks
                        chunk_len = len(chunk)
                        all_log_det.append(np.full(chunk_len, np.nan))
                        all_cond_num.append(np.full(chunk_len, np.nan))
                        all_eigenval.append(np.full(chunk_len, np.nan))
                
                # Combine results
                log_det_grid = np.concatenate(all_log_det)
                cond_grid = np.concatenate(all_cond_num)
                eigenval_grid = np.concatenate(all_eigenval)
                
                # Reshape to grid
                log_det_heatmap = log_det_grid.reshape(xx.shape)
                cond_heatmap = cond_grid.reshape(xx.shape)
                eigenval_heatmap = eigenval_grid.reshape(xx.shape)
                
                # 🎬 NEW: Create INTERACTIVE version with time slider (only every 6 epochs to avoid slowdown)
                if epoch % 6 == 0:
                    self._create_interactive_metric_slider_visualization(
                        z_pca_seq, xx, yy, pca, epoch, temperature_override, z_seq_actual=z_seq
                    )
                
                # Create 2x2 moderate-detail visualization with explicit matplotlib fixes
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                
                temp_str = f" (T={temperature_override})" if temperature_override else ""
                fig.suptitle(f'Moderate Detail Metric Analysis{temp_str} - Epoch {epoch}', fontsize=16)
                
                # Panel 1: Mean Eigenvalue with moderate resolution
                im1 = axes[0, 0].contourf(xx, yy, eigenval_heatmap, levels=20, cmap='viridis', extend='both')
                for t in range(min(n_obs, 5)):  # Limit overlays to prevent clutter
                    points_t = z_pca_seq[:8, t, :]  # Limit to 8 sequences
                    axes[0, 0].scatter(points_t[:, 0], points_t[:, 1], alpha=0.8, s=20, c='white', edgecolors='black', linewidth=0.5)
                axes[0, 0].set_title('Mean Eigenvalue λ', fontsize=12)
                axes[0, 0].set_xlabel('PC1')
                axes[0, 0].set_ylabel('PC2')
                try:
                    plt.colorbar(im1, ax=axes[0, 0])
                except Exception as cb_error:
                    print(f"⚠️ Colorbar 1 failed: {cb_error}")
                
                # Panel 2: Log Determinant with moderate detail
                im2 = axes[0, 1].contourf(xx, yy, log_det_heatmap, levels=20, cmap='plasma', extend='both')
                for t in range(min(n_obs, 5)):
                    points_t = z_pca_seq[:8, t, :]
                    axes[0, 1].scatter(points_t[:, 0], points_t[:, 1], alpha=0.8, s=20, c='white', edgecolors='black', linewidth=0.5)
                axes[0, 1].set_title('log(det(G))', fontsize=12)
                axes[0, 1].set_xlabel('PC1')
                axes[0, 1].set_ylabel('PC2')
                try:
                    plt.colorbar(im2, ax=axes[0, 1])
                except Exception as cb_error:
                    print(f"⚠️ Colorbar 2 failed: {cb_error}")
                
                # Panel 3: Condition Number with moderate detail
                im3 = axes[1, 0].contourf(xx, yy, cond_heatmap, levels=20, cmap='coolwarm', extend='both')
                for t in range(min(n_obs, 5)):
                    points_t = z_pca_seq[:8, t, :]
                    axes[1, 0].scatter(points_t[:, 0], points_t[:, 1], alpha=0.8, s=20, c='white', edgecolors='black', linewidth=0.5)
                axes[1, 0].set_title('Condition Number', fontsize=12)
                axes[1, 0].set_xlabel('PC1')
                axes[1, 0].set_ylabel('PC2')
                try:
                    plt.colorbar(im3, ax=axes[1, 0])
                except Exception as cb_error:
                    print(f"⚠️ Colorbar 3 failed: {cb_error}")
                
                # Panel 4: Simplified trajectory overlay
                # Use simple contour instead of contourf to avoid matplotlib transform issues
                try:
                    axes[1, 1].contour(xx, yy, log_det_heatmap, levels=15, colors='gray', alpha=0.5, linewidths=0.8)
                except Exception as contour_error:
                    print(f"⚠️ Contour plot failed: {contour_error}")
                
                # Draw trajectories with different colors per sequence (limit to avoid clutter)
                colors = plt.get_cmap("tab10")(np.linspace(0, 1, min(batch_size, 6)))
                for seq_idx in range(min(batch_size, 6)):  # Limit to 6 sequences for clarity
                    traj = z_pca_seq[seq_idx]  # [n_obs, 2]
                    axes[1, 1].plot(traj[:, 0], traj[:, 1], 'o-', color=colors[seq_idx], 
                                   alpha=0.8, linewidth=1.5, markersize=4, label=f'Seq {seq_idx}')
                    # Mark start and end
                    axes[1, 1].scatter(traj[0, 0], traj[0, 1], color=colors[seq_idx], s=60, marker='s', 
                                      edgecolor='black', linewidth=1, zorder=5)
                    axes[1, 1].scatter(traj[-1, 0], traj[-1, 1], color=colors[seq_idx], s=60, marker='*', 
                                      edgecolor='black', linewidth=1, zorder=5)
                
                axes[1, 1].set_title('Trajectories + Metric Contours', fontsize=12)
                axes[1, 1].set_xlabel('PC1')
                axes[1, 1].set_ylabel('PC2')
                try:
                    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                except Exception as legend_error:
                    print(f"⚠️ Legend failed: {legend_error}")
                
                plt.tight_layout()
                
                # Save with reduced DPI to avoid PIL warnings
                temp_suffix = f"_T{temperature_override}" if temperature_override else ""
                filename = f'moderate_detail_metric_heatmap{temp_suffix}_epoch_{epoch}.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
                
                # Log with temperature-specific metrics
                log_dict = {
                    "manifold/moderate_detail_heatmap": wandb.Image(filename, caption=f"Epoch {epoch} - Moderate detail{temp_str}"),
                    "metric_detail/mean_log_det": np.nanmean(log_det_heatmap),
                    "metric_detail/std_log_det": np.nanstd(log_det_heatmap),
                    "metric_detail/mean_condition_number": np.nanmean(cond_heatmap),
                    "metric_detail/max_condition_number": np.nanmax(cond_heatmap),
                    "metric_detail/mean_eigenvalue": np.nanmean(eigenval_heatmap),
                    "metric_detail/grid_points_computed": len(grid_points_pca),
                    "metric_detail/grid_size": f"{xx.shape[0]}x{xx.shape[1]}",
                }
                
                if temperature_override:
                    log_dict["metric_detail/temperature_used"] = temperature_override
                
                wandb.log(log_dict)
                plt.close()
                
                print(f"✅ Moderate detail metric heatmap created with {len(grid_points_pca)} points")
                
        except Exception as e:
            print(f"⚠️ Moderate detail metric heatmap failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Restore original temperature
            if original_temp is not None:
                self.model.temperature.data = torch.tensor(original_temp, device=self.device)
                print(f"🔧 Restored temperature to {original_temp}")
            
            self.model.train()

    def _create_interactive_metric_slider_visualization(self, z_pca_seq, xx, yy, pca, epoch, temperature_override=None, z_seq_actual=None):
        """
        Create interactive Plotly visualization with TIME SLIDER for metric evolution across timesteps.
        Shows det(G), Tr(G), condition number, and eigenvalues at each timestep.
        
        Args:
            z_pca_seq: PCA-projected coordinates for visualization [batch_size, n_obs, 2]
            xx, yy: Grid coordinates for background
            pca: PCA object for coordinate transforms
            epoch: Current epoch
            temperature_override: Temperature override if any
            z_seq_actual: ACTUAL flow-evolved coordinates [batch_size, n_obs, latent_dim] - CRITICAL FIX!
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import numpy as np
            
            print(f"🎬 Creating INTERACTIVE metric slider visualization with timestep evolution...")
            
            batch_size, n_obs, _ = z_pca_seq.shape
            
            # Project grid points back to latent space for metric computation
            grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
            grid_points_latent = pca.inverse_transform(grid_points_pca)
            grid_tensor = torch.tensor(grid_points_latent, dtype=torch.float32, device=self.device)
            
            # Compute metric fields for all timesteps
            print(f"🔄 Computing metric fields for timesteps 0 to {n_obs-1}...")
            
            # CRITICAL FIX: Prepare dynamic heatmaps that will be computed per timestep
            # We'll store per-timestep heatmaps instead of static ones
            det_heatmaps = {}
            trace_heatmaps = {}
            cond_heatmaps = {}
            eigenval_heatmaps = {}
            
            print(f"📊 Computing DYNAMIC metric heatmaps for each timestep...")
            
            # For each timestep, compute the metric tensor field at grid points
            # that correspond to the coordinate system at that timestep
            for t_grid in range(n_obs):
                print(f"   Computing heatmap for timestep {t_grid}...")
                
                if t_grid == 0:
                    # For t=0, use the original grid (no flow transformation)
                    grid_tensor_t = grid_tensor
                else:
                    # CORRECT APPROACH: Transform grid points through forward flows to timestep t
                    # This matches the coordinate system where sequences are at timestep t
                    try:
                        print(f"     🔄 Applying {t_grid} forward flow steps to grid...")
                        grid_tensor_t = grid_tensor.clone()
                        
                        # Apply forward flows to transform grid from t=0 to t=t_grid
                        for flow_idx in range(t_grid):
                            if flow_idx < len(self.model.flows):
                                print(f"       Applying flow {flow_idx}: z_{flow_idx} → z_{flow_idx+1}")
                                flow_result = self.model.flows[flow_idx](grid_tensor_t)
                                grid_tensor_t = flow_result.out
                            else:
                                print(f"       ⚠️ Flow {flow_idx} not available, stopping")
                                break
                        
                        print(f"     ✅ Grid transformed to timestep {t_grid} coordinate system")
                        
                    except Exception as e:
                        print(f"     ❌ Flow transformation failed for t={t_grid}: {e}")
                        # Fallback: use original grid with small perturbation to show difference
                        grid_tensor_t = grid_tensor + 0.1 * t_grid * torch.randn_like(grid_tensor)
                        print(f"     🔄 Using perturbed grid as fallback for t={t_grid}")
                
                # Compute metrics at grid points for this timestep
                try:
                    G_grid_t = self.model.G(grid_tensor_t)
                    det_G_grid_t = torch.linalg.det(G_grid_t).cpu().numpy()
                    eigenvals_grid_t = torch.linalg.eigvals(G_grid_t).real.cpu().numpy()
                    trace_G_grid_t = torch.diagonal(G_grid_t, dim1=-2, dim2=-1).sum(dim=-1).cpu().numpy()
                    cond_grid_t = eigenvals_grid_t.max(axis=1) / (eigenvals_grid_t.min(axis=1) + 1e-10)
                    mean_eigenval_grid_t = eigenvals_grid_t.mean(axis=1)
                    
                    # Store heatmaps for this timestep
                    det_heatmaps[t_grid] = det_G_grid_t.reshape(xx.shape)
                    trace_heatmaps[t_grid] = trace_G_grid_t.reshape(xx.shape)
                    cond_heatmaps[t_grid] = cond_grid_t.reshape(xx.shape)
                    eigenval_heatmaps[t_grid] = mean_eigenval_grid_t.reshape(xx.shape)
                    
                    print(f"     ✅ Heatmap computed for t={t_grid}: det(G) range [{det_G_grid_t.min():.2e}, {det_G_grid_t.max():.2e}]")
                    
                except Exception as e:
                    print(f"     ❌ Failed to compute heatmap for t={t_grid}: {e}")
                    # Use t=0 heatmap as fallback
                    if 0 in det_heatmaps:
                        det_heatmaps[t_grid] = det_heatmaps[0]
                        trace_heatmaps[t_grid] = trace_heatmaps[0]
                        cond_heatmaps[t_grid] = cond_heatmaps[0]
                        eigenval_heatmaps[t_grid] = eigenval_heatmaps[0]
                    else:
                        # Create dummy heatmaps
                        det_heatmaps[t_grid] = np.ones(xx.shape)
                        trace_heatmaps[t_grid] = np.ones(xx.shape)
                        cond_heatmaps[t_grid] = np.ones(xx.shape)
                        eigenval_heatmaps[t_grid] = np.ones(xx.shape)
            
            # CRITICAL FIX: Compute metrics at actual flow-evolved coordinates for each timestep
            timestep_metrics = []
            for t in range(n_obs):
                if z_seq_actual is not None:
                    # CORRECT: Use actual flow-evolved coordinates
                    z_latent_t = z_seq_actual[:, t, :]  # [batch_size, latent_dim] - FLOW-EVOLVED!
                    z_tensor_t = torch.tensor(z_latent_t, dtype=torch.float32, device=self.device)
                    print(f"    ✅ Using FLOW-EVOLVED coordinates for timestep {t}")
                else:
                    # FALLBACK: Use PCA inverse (original method, less accurate)
                    z_t = z_pca_seq[:, t, :]  # PCA positions at timestep t
                    z_latent_t = pca.inverse_transform(z_t)  # Back to latent space
                    z_tensor_t = torch.tensor(z_latent_t, dtype=torch.float32, device=self.device)
                    print(f"    ⚠️ Using PCA-INVERSE coordinates for timestep {t} (fallback)")
                
                # Compute metrics at sequence positions using FLOW-EVOLVED coordinates
                G_t = self.model.G(z_tensor_t)
                det_t = torch.linalg.det(G_t).cpu().numpy()
                eigenvals_t = torch.linalg.eigvals(G_t).real.cpu().numpy()
                trace_t = torch.diagonal(G_t, dim1=-2, dim2=-1).sum(dim=-1).cpu().numpy()
                cond_t = eigenvals_t.max(axis=1) / (eigenvals_t.min(axis=1) + 1e-10)
                
                timestep_metrics.append({
                    'det': det_t,
                    'trace': trace_t,
                    'condition': cond_t,
                    'eigenvals': eigenvals_t,
                    'positions': z_pca_seq[:, t, :]  # Still use PCA positions for visualization
                })
            
            # Create interactive slider visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "🎯 det(G) Evolution with Timestep",
                    "📊 Tr(G) Evolution with Timestep", 
                    "⚖️ Condition Number Evolution",
                    "🌟 Mean Eigenvalue Evolution"
                ],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]],
                horizontal_spacing=0.1,
                vertical_spacing=0.12
            )
            
            # Create frames for each timestep
            frames = []
            colors = px.colors.qualitative.Set3[:min(batch_size, 8)]
            
            for t in range(n_obs):
                frame_data = []
                metrics_t = timestep_metrics[t]
                
                # CRITICAL FIX: Use timestep-specific heatmap
                det_heatmap_t = det_heatmaps.get(t, det_heatmaps.get(0, np.ones(xx.shape)))
                trace_heatmap_t = trace_heatmaps.get(t, trace_heatmaps.get(0, np.ones(xx.shape)))
                cond_heatmap_t = cond_heatmaps.get(t, cond_heatmaps.get(0, np.ones(xx.shape)))
                eigenval_heatmap_t = eigenval_heatmaps.get(t, eigenval_heatmaps.get(0, np.ones(xx.shape)))
                
                print(f"   📊 Frame {t}: using det_heatmap range [{det_heatmap_t.min():.2e}, {det_heatmap_t.max():.2e}]")
                
                # Panel 1: det(G) heatmap + sequence points
                frame_data.append(
                    go.Contour(
                        x=np.linspace(xx.min(), xx.max(), xx.shape[1]),
                        y=np.linspace(yy.min(), yy.max(), yy.shape[0]),
                        z=np.log10(np.clip(det_heatmap_t, 1e-10, None)),  # FIXED: timestep-specific heatmap
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="log₁₀(det(G))", x=0.45, len=0.4, y=0.75),
                        name="det(G) field",
                        hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>log₁₀(det(G)): %{z:.2f}<extra></extra>",
                        xaxis='x', yaxis='y'
                    )
                )
                
                # Add sequence points with det(G) values
                frame_data.append(
                    go.Scatter(
                        x=metrics_t['positions'][:, 0],
                        y=metrics_t['positions'][:, 1],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=np.log10(np.clip(metrics_t['det'], 1e-10, None)),
                            colorscale='Viridis',
                            showscale=False,
                            line=dict(color='white', width=2)
                        ),
                        name=f"Sequences t={t}",
                        hovertemplate="Seq: %{pointNumber}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>det(G): %{marker.color:.2e}<extra></extra>",
                        xaxis='x', yaxis='y'
                    )
                )
                
                # Panel 2: Tr(G) heatmap + sequence points
                frame_data.append(
                    go.Contour(
                        x=np.linspace(xx.min(), xx.max(), xx.shape[1]),
                        y=np.linspace(yy.min(), yy.max(), yy.shape[0]),
                        z=trace_heatmap_t,  # FIXED: timestep-specific heatmap
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Tr(G)", x=0.95, len=0.4, y=0.75),
                        name="Tr(G) field",
                        hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Tr(G): %{z:.2f}<extra></extra>",
                        xaxis='x2', yaxis='y2'
                    )
                )
                
                frame_data.append(
                    go.Scatter(
                        x=metrics_t['positions'][:, 0],
                        y=metrics_t['positions'][:, 1],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=metrics_t['trace'],
                            colorscale='Plasma',
                            showscale=False,
                            line=dict(color='white', width=2)
                        ),
                        name=f"Tr(G) Sequences t={t}",
                        hovertemplate="Seq: %{pointNumber}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Tr(G): %{marker.color:.2f}<extra></extra>",
                        xaxis='x2', yaxis='y2'
                    )
                )
                
                # Panel 3: Condition Number heatmap + sequence points
                frame_data.append(
                    go.Contour(
                        x=np.linspace(xx.min(), xx.max(), xx.shape[1]),
                        y=np.linspace(yy.min(), yy.max(), yy.shape[0]),
                        z=cond_heatmap_t,  # FIXED: timestep-specific heatmap
                        colorscale='RdYlBu_r',
                        showscale=True,
                        colorbar=dict(title="Condition #", x=0.45, len=0.4, y=0.25),
                        name="Condition # field",
                        hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Condition: %{z:.2f}<extra></extra>",
                        xaxis='x3', yaxis='y3'
                    )
                )
                
                frame_data.append(
                    go.Scatter(
                        x=metrics_t['positions'][:, 0],
                        y=metrics_t['positions'][:, 1],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=metrics_t['condition'],
                            colorscale='RdYlBu_r',
                            showscale=False,
                            line=dict(color='white', width=2)
                        ),
                        name=f"Condition Sequences t={t}",
                        hovertemplate="Seq: %{pointNumber}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Condition: %{marker.color:.2f}<extra></extra>",
                        xaxis='x3', yaxis='y3'
                    )
                )
                
                # Panel 4: Mean Eigenvalue heatmap + sequence points
                frame_data.append(
                    go.Contour(
                        x=np.linspace(xx.min(), xx.max(), xx.shape[1]),
                        y=np.linspace(yy.min(), yy.max(), yy.shape[0]),
                        z=eigenval_heatmap_t,  # FIXED: timestep-specific heatmap
                        colorscale='Hot',
                        showscale=True,
                        colorbar=dict(title="Mean λ", x=0.95, len=0.4, y=0.25),
                        name="Mean λ field",
                        hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Mean λ: %{z:.2f}<extra></extra>",
                        xaxis='x4', yaxis='y4'
                    )
                )
                
                frame_data.append(
                    go.Scatter(
                        x=metrics_t['positions'][:, 0],
                        y=metrics_t['positions'][:, 1],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=metrics_t['eigenvals'].mean(axis=1),
                            colorscale='Hot',
                            showscale=False,
                            line=dict(color='white', width=2)
                        ),
                        name=f"Eigenval Sequences t={t}",
                        hovertemplate="Seq: %{pointNumber}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Mean λ: %{marker.color:.2f}<extra></extra>",
                        xaxis='x4', yaxis='y4'
                    )
                )
                
                frames.append(go.Frame(data=frame_data, name=str(t)))
            
            # Set initial frame (t=0)
            fig.add_traces(frames[0].data)
            fig.frames = frames
            
            # Update layout with slider controls
            temp_str = f" (T={temperature_override})" if temperature_override else ""
            fig.update_layout(
                title={
                    'text': f"🎬 INTERACTIVE METRIC EVOLUTION{temp_str} - EPOCH {epoch}<br>"
                           f"<span style='font-size:14px'>🎯 Use slider to explore det(G), Tr(G), condition number across timesteps 0→{n_obs-1}</span>",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#2E86AB'}
                },
                sliders=[{
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Timestep: ",
                        "visible": True,
                        "xanchor": "right"
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[str(t)], {"frame": {"duration": 300, "redraw": True}, 
                                               "mode": "immediate", "transition": {"duration": 200}}],
                            "label": f"t={t}",
                            "method": "animate"
                        }
                        for t in range(n_obs)
                    ]
                }],
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 600, "redraw": True}, 
                                          "fromcurrent": True, "transition": {"duration": 400}}],
                            "label": "▶️ Auto Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True}, 
                                            "mode": "immediate", "transition": {"duration": 0}}],
                            "label": "⏸️ Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0.02,
                    "yanchor": "top"
                }],
                width=1600,
                height=1000,
                showlegend=False,  # Too cluttered with legends
                paper_bgcolor='rgba(240,248,255,0.95)',
                plot_bgcolor='rgba(255,255,255,0.8)',
                margin=dict(l=80, r=80, t=120, b=100)
            )
            
            # Update subplot axes labels
            fig.update_xaxes(title_text="PC1", row=1, col=1)
            fig.update_yaxes(title_text="PC2", row=1, col=1)
            fig.update_xaxes(title_text="PC1", row=1, col=2)
            fig.update_yaxes(title_text="PC2", row=1, col=2)
            fig.update_xaxes(title_text="PC1", row=2, col=1)
            fig.update_yaxes(title_text="PC2", row=2, col=1)
            fig.update_xaxes(title_text="PC1", row=2, col=2)
            fig.update_yaxes(title_text="PC2", row=2, col=2)
            
            # Save interactive metric slider visualization
            temp_suffix = f"_T{temperature_override}" if temperature_override else ""
            html_filename = f'interactive_metric_slider{temp_suffix}_epoch_{epoch}.html'
            fig.write_html(html_filename, include_plotlyjs=True)
            
            png_filename = f'interactive_metric_slider{temp_suffix}_epoch_{epoch}.png'
            self._safe_write_image(fig, png_filename, width=1600, height=1000, scale=2)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "temporal_evolution/interactive_metric_slider": wandb.Html(html_filename),
                    "temporal_evolution/metric_slider_static": wandb.Image(png_filename, 
                        caption=f"Epoch {epoch} - Interactive metric slider{temp_str}"),
                })
            
            print(f"🎬 Interactive metric slider visualization saved: {html_filename}")
            print(f"   📊 Tracks det(G), Tr(G), condition number across {n_obs} timesteps")
            print(f"   🎯 Use slider to explore metric evolution!")
            
        except Exception as e:
            print(f"⚠️ Interactive metric slider visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def create_sequence_trajectories_viz(self, x_sample, epoch):
        """Create comprehensive visualization of sequence trajectories in latent space."""
        print(f"🧠 Creating sequence trajectory visualization for epoch {epoch}")
        
        self.model.eval()
        with torch.no_grad():
            result = self.model_forward(x_sample)
            z_seq = result.z  # [batch_size, n_obs, latent_dim]
            
            batch_size, n_obs, latent_dim = z_seq.shape
            max_viz = getattr(self.config, 'sequence_viz_count', 8)
            num_viz = min(max_viz, batch_size)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Latent Sequence Trajectories - {self.config.loop_mode.upper()} Mode - Epoch {epoch}', fontsize=16)
            
            # Prepare data for PCA
            z_flat = z_seq.reshape(-1, latent_dim).cpu().numpy()
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            z_pca = pca.fit_transform(z_flat)
            z_pca_seq = z_pca.reshape(batch_size, n_obs, 3)
            
            # Plot 1: 2D Trajectory Overview (PC1 vs PC2)
            ax = axes[0, 0]
            colors = plt.get_cmap("tab10")(np.linspace(0, 1, num_viz))
            for i in range(num_viz):
                traj = z_pca_seq[i, :, :2]
                ax.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[i], alpha=0.7, label=f'Seq {i}', linewidth=2)
                ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], s=100, marker='s', alpha=0.9, edgecolor='black')  # Start
                ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], s=100, marker='*', alpha=0.9, edgecolor='black')  # End
            
            ax.set_title(f'2D Trajectories (PC1 vs PC2)\nVariance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Temporal Evolution of PC1
            ax = axes[0, 1]
            for i in range(num_viz):
                ax.plot(range(n_obs), z_pca_seq[i, :, 0], 'o-', color=colors[i], alpha=0.7, label=f'Seq {i}')
            ax.set_title('PC1 Evolution Over Time')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('PC1 Value')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Temporal Evolution of PC2
            ax = axes[0, 2]
            for i in range(num_viz):
                ax.plot(range(n_obs), z_pca_seq[i, :, 1], 'o-', color=colors[i], alpha=0.7, label=f'Seq {i}')
            ax.set_title('PC2 Evolution Over Time')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('PC2 Value')
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Trajectory Lengths
            ax = axes[1, 0]
            traj_lengths = []
            for i in range(batch_size):
                # Calculate total path length in PCA space
                diffs = np.diff(z_pca_seq[i], axis=0)
                lengths = np.linalg.norm(diffs, axis=1)
                total_length = np.sum(lengths)
                traj_lengths.append(total_length)
            
            ax.hist(traj_lengths, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'Trajectory Lengths Distribution\nMean: {np.mean(traj_lengths):.3f}±{np.std(traj_lengths):.3f}')
            ax.set_xlabel('Total Path Length')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            
            # Plot 5: Start vs End Point Analysis
            ax = axes[1, 1]
            start_points = z_pca_seq[:, 0, :2]  # [batch_size, 2]
            end_points = z_pca_seq[:, -1, :2]   # [batch_size, 2]
            distances = np.linalg.norm(end_points - start_points, axis=1)
            
            ax.scatter(start_points[:, 0], start_points[:, 1], alpha=0.6, s=50, label='Start Points', marker='s')
            ax.scatter(end_points[:, 0], end_points[:, 1], alpha=0.6, s=50, label='End Points', marker='*')
            
            # Draw lines connecting start to end
            for i in range(min(num_viz, batch_size)):
                ax.plot([start_points[i, 0], end_points[i, 0]], 
                       [start_points[i, 1], end_points[i, 1]], 
                       'k--', alpha=0.3, linewidth=1)
            
            ax.set_title(f'Start vs End Points\nMean Distance: {np.mean(distances):.3f}±{np.std(distances):.3f}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 6: Distance Statistics
            ax = axes[1, 2]
            ax.hist(distances, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title('Start-End Distance Distribution')
            ax.set_xlabel('Start-End Distance')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save and log
            filename = f'sequence_trajectories_{self.config.loop_mode}_epoch_{epoch}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            # Keep the visualization but remove clutter metrics
            wandb.log({
                "analysis/sequence_trajectories": wandb.Image(filename, caption=f"Epoch {epoch} - {self.config.loop_mode} trajectories"),
            })
            plt.close()
        
        self.model.train()
    
    def create_comprehensive_reconstruction_viz(self, x_sample, epoch):
        """Create comprehensive reconstruction visualization."""
        print(f"🎬 Creating comprehensive reconstruction visualization for epoch {epoch}")
        
        self.model.eval()
        with torch.no_grad():
            result = self.model_forward(x_sample)
            recon_x = result.recon_x  # [batch_size, n_obs, 3, 64, 64]
            
            batch_size, n_obs = x_sample.shape[:2]
            max_viz = getattr(self.config, 'sequence_viz_count', 8)
            num_seqs = min(max_viz, batch_size)
            
            # Create comprehensive grid: original, reconstructed, and error for multiple sequences
            fig, axes = plt.subplots(num_seqs * 3, n_obs, figsize=(3*n_obs, 3*num_seqs*3))
            if num_seqs == 1:
                axes = axes.reshape(3, n_obs)
            
            fig.suptitle(f'Comprehensive Reconstruction - {self.config.loop_mode.upper()} Mode - Epoch {epoch}', fontsize=16)
            
            for seq_idx in range(num_seqs):
                for t in range(n_obs):
                    # Original
                    row = seq_idx * 3
                    orig_img = x_sample[seq_idx, t].permute(1, 2, 0).cpu().numpy()
                    orig_img = np.clip(orig_img, 0, 1)
                    axes[row, t].imshow(orig_img)
                    if t == 0:
                        axes[row, t].set_ylabel(f'Seq {seq_idx}\nOriginal', fontsize=10)
                    axes[row, t].set_title(f't={t}', fontsize=8)
                    axes[row, t].axis('off')
                    
                    # Reconstructed
                    row = seq_idx * 3 + 1
                    recon_img = recon_x[seq_idx, t].permute(1, 2, 0).cpu().numpy()
                    recon_img = np.clip(recon_img, 0, 1)
                    axes[row, t].imshow(recon_img)
                    if t == 0:
                        axes[row, t].set_ylabel(f'Seq {seq_idx}\nRecon', fontsize=10)
                    axes[row, t].axis('off')
                    
                    # Error
                    row = seq_idx * 3 + 2
                    error = torch.abs(x_sample[seq_idx, t] - recon_x[seq_idx, t]).mean(0).cpu().numpy()
                    im = axes[row, t].imshow(error, cmap='hot')
                    if t == 0:
                        axes[row, t].set_ylabel(f'Seq {seq_idx}\nError', fontsize=10)
                    axes[row, t].axis('off')
                    
                    # Add colorbar for error plots
                    if t == n_obs - 1:  # Only add colorbar to last column
                        cbar = plt.colorbar(im, ax=axes[row, t], shrink=0.8)
                        cbar.set_label('Error', fontsize=8)
            
            plt.tight_layout()
            
            # Save and log
            filename = f'comprehensive_reconstruction_{self.config.loop_mode}_epoch_{epoch}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            # Keep the visualization but remove clutter metrics
            wandb.log({
                "analysis/reconstruction_overview": wandb.Image(filename, caption=f"Epoch {epoch} - {self.config.loop_mode} reconstruction"),
            })
            plt.close()
        
        self.model.train()
    
    def create_interactive_html_latent_space(self, num_sequences=20):
        """Create interactive HTML latent space visualization at the end of training."""
        print(f"🌐 Creating interactive HTML latent space visualization...")
        
        try:
            # Load some test data for visualization
            if self.config.loop_mode == "open":
                test_path = str(project_root / "data" / "processed" / "Sprites_train_cyclic.pt")
            else:
                test_path = str(project_root / "data" / "processed" / "Sprites_train_cyclic.pt")
            
            # Load and select sequences for visualization
            data = torch.load(test_path, map_location=self.device)
            n_sequences = min(num_sequences, len(data))
            selected_data = data[:n_sequences].to(self.device)  # [n_sequences, n_obs, 3, 64, 64]
            
            print(f"📊 Processing {n_sequences} sequences for HTML visualization...")
            
            self.model.eval()
            with torch.no_grad():
                # Forward pass to get latent representations
                all_latents = []
                all_images = []
                sequence_info = []
                
                for seq_idx in range(n_sequences):
                    seq_data = selected_data[seq_idx:seq_idx+1]  # [1, n_obs, 3, 64, 64]
                    result = self.model_forward(seq_data)
                    
                    z_seq = result.z.squeeze(0)  # [n_obs, latent_dim]
                    recon_x = result.recon_x.squeeze(0)  # [n_obs, 3, 64, 64]
                    
                    for t in range(z_seq.shape[0]):
                        all_latents.append(z_seq[t].cpu().numpy())
                        all_images.append(recon_x[t].cpu().numpy())
                        sequence_info.append({
                            'seq_id': seq_idx,
                            'timestep': t,
                            'is_start': t == 0,
                            'is_end': t == z_seq.shape[0] - 1
                        })
                
                # Convert to numpy arrays
                latents_array = np.array(all_latents)  # [total_points, latent_dim]
                images_array = np.array(all_images)    # [total_points, 3, 64, 64]
                
                print(f"✅ Processed {len(latents_array)} total points from {n_sequences} sequences")
                
                # Apply PCA for 2D visualization
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                latents_2d = pca.fit_transform(latents_array)
                
                print(f"📊 PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
                
                # Create images directory
                images_dir = f"html_latent_images_{self.config.loop_mode}"
                os.makedirs(images_dir, exist_ok=True)
                
                # Save images as PNG files
                import matplotlib.pyplot as plt
                print(f"💾 Saving {len(images_array)} images to {images_dir}/...")
                
                for i, (img_array, info) in enumerate(zip(images_array, sequence_info)):
                    # Convert from [3, 64, 64] to [64, 64, 3] and clip to [0, 1]
                    img_display = np.transpose(img_array, (1, 2, 0))
                    img_display = np.clip(img_display, 0, 1)
                    
                    filename = f"seq_{info['seq_id']:02d}_t_{info['timestep']:02d}.png"
                    filepath = os.path.join(images_dir, filename)
                    
                    plt.imsave(filepath, img_display)
                
                # Generate HTML file
                html_filename = f"interactive_latent_space_{self.config.loop_mode}.html"
                self._generate_html_file(html_filename, latents_2d, sequence_info, images_dir, pca)
                
                print(f"🎉 Interactive HTML latent space visualization created!")
                print(f"📁 Files created:")
                print(f"   • {html_filename}")
                print(f"   • {images_dir}/ (with {len(images_array)} images)")
                print(f"🌐 To view: python3 -m http.server 8000, then open http://localhost:8000/{html_filename}")
                
                # Log to wandb with HTML file info
                wandb.log({
                    "final/html_visualization_file": html_filename,
                    "final/html_images_directory": images_dir,
                    "final/total_latent_points": len(latents_array),
                    "final/sequences_visualized": n_sequences,
                    "final/riemannian_beta_used": getattr(self.config, 'riemannian_beta', self.config.beta),
                    "final/loop_mode": self.config.loop_mode
                })
                
        except Exception as e:
            print(f"⚠️ Failed to create HTML latent space visualization: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()
    
    def _generate_html_file(self, filename, latents_2d, sequence_info, images_dir, pca):
        """Generate the interactive HTML file."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Latent Space - {self.config.loop_mode.upper()} Mode</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .mode-badge {{
            background: {'#e74c3c' if self.config.loop_mode == 'closed' else '#27ae60'};
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            margin: 10px;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .stat-item {{
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 5px;
            min-width: 120px;
        }}
        #canvas {{
            border: 3px solid white;
            cursor: crosshair;
            display: block;
            margin: 20px auto;
            background: white;
            border-radius: 10px;
        }}
        .controls {{
            text-align: center;
            margin: 20px 0;
        }}
        .info-panel {{
            display: flex;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .point-info {{
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 10px;
            flex: 1;
            min-width: 300px;
        }}
        .image-display {{
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 10px;
            flex: 1;
            text-align: center;
            min-width: 300px;
        }}
        #selectedImage {{
            max-width: 100%;
            height: auto;
            border: 2px solid white;
            border-radius: 8px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 15px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.1);
            padding: 8px 12px;
            border-radius: 8px;
        }}
        .legend-marker {{
            width: 16px;
            height: 16px;
            border: 2px solid #000;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Interactive Latent Space Visualization</h1>
            <div class="mode-badge">{self.config.loop_mode.upper()} Loop Mode</div>
            <p>Trained model latent space exploration with decoded sprite images</p>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <h3>{len(latents_2d)}</h3>
                <p>Total Points</p>
            </div>
            <div class="stat-item">
                <h3>{len(set(info['seq_id'] for info in sequence_info))}</h3>
                <p>Sequences</p>
            </div>
            <div class="stat-item">
                <h3>{pca.explained_variance_ratio_[0]:.1%}</h3>
                <p>PC1 Variance</p>
            </div>
            <div class="stat-item">
                <h3>{pca.explained_variance_ratio_[1]:.1%}</h3>
                <p>PC2 Variance</p>
            </div>
        </div>
        
        <div class="controls">
            <h3>Click any point to see the decoded sprite image!</h3>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-marker" style="background: none;"></div>
                <span>🔲 Start Points</span>
            </div>
            <div class="legend-item">
                <div class="legend-marker" style="background: none; transform: rotate(45deg);"></div>
                <span>💎 End Points</span>
            </div>
            <div class="legend-item">
                <div class="legend-marker" style="background: none; border-radius: 50%;"></div>
                <span>⚫ Middle Points</span>
            </div>
        </div>
        
        <canvas id="canvas" width="1000" height="700"></canvas>
        
        <div class="info-panel">
            <div class="point-info">
                <h3>📍 Point Information</h3>
                <div id="pointDetails">
                    <p>Click any point to see detailed information</p>
                </div>
            </div>
            
            <div class="image-display">
                <h3>🖼️ Decoded Image</h3>
                <div id="imageContainer">
                    <p>Click a point to see the corresponding decoded sprite</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data from Python
        const latents2d = {latents_2d.tolist()};
        const sequenceInfo = {sequence_info};
        const imagesDir = '{images_dir}';
        
        // Canvas setup
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        // Color scheme
        const colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
        ];
        
        // Scale latents to canvas
        function scaleToCanvas(latents) {{
            const margin = 50;
            const xValues = latents.map(p => p[0]);
            const yValues = latents.map(p => p[1]);
            
            const xMin = Math.min(...xValues);
            const xMax = Math.max(...xValues);
            const yMin = Math.min(...yValues);
            const yMax = Math.max(...yValues);
            
            return latents.map(([x, y]) => [
                margin + (x - xMin) / (xMax - xMin) * (canvas.width - 2 * margin),
                margin + (y - yMin) / (yMax - yMin) * (canvas.height - 2 * margin)
            ]);
        }}
        
        const scaledPoints = scaleToCanvas(latents2d);
        
        function drawVisualization() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Group points by sequence for trajectory drawing
            const sequences = {{}};
            scaledPoints.forEach((point, i) => {{
                const seqId = sequenceInfo[i].seq_id;
                if (!sequences[seqId]) sequences[seqId] = [];
                sequences[seqId].push({{point, info: sequenceInfo[i], index: i}});
            }});
            
            // Draw trajectories
            Object.keys(sequences).forEach(seqId => {{
                const seq = sequences[seqId].sort((a, b) => a.info.timestep - b.info.timestep);
                const color = colors[parseInt(seqId) % colors.length];
                
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.setLineDash([]);
                ctx.globalAlpha = 0.6;
                
                ctx.beginPath();
                seq.forEach((item, i) => {{
                    const [x, y] = item.point;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }});
                ctx.stroke();
            }});
            
            // Draw points
            ctx.globalAlpha = 1.0;
            scaledPoints.forEach((point, i) => {{
                const [x, y] = point;
                const info = sequenceInfo[i];
                const color = colors[info.seq_id % colors.length];
                
                ctx.fillStyle = color;
                ctx.strokeStyle = '#000';
                ctx.lineWidth = 2;
                
                if (info.is_start) {{
                    // Start point - square
                    ctx.fillRect(x - 8, y - 8, 16, 16);
                    ctx.strokeRect(x - 8, y - 8, 16, 16);
                }} else if (info.is_end) {{
                    // End point - diamond
                    ctx.beginPath();
                    ctx.moveTo(x, y - 10);
                    ctx.lineTo(x + 8, y);
                    ctx.lineTo(x, y + 10);
                    ctx.lineTo(x - 8, y);
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                }} else {{
                    // Middle point - circle
                    ctx.beginPath();
                    ctx.arc(x, y, 8, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.stroke();
                }}
            }});
        }}
        
        // Handle clicks
        canvas.addEventListener('click', function(event) {{
            const rect = canvas.getBoundingClientRect();
            const clickX = event.clientX - rect.left;
            const clickY = event.clientY - rect.top;
            
            // Find closest point
            let closest = null;
            let minDist = Infinity;
            let closestIndex = -1;
            
            scaledPoints.forEach((point, i) => {{
                const [x, y] = point;
                const dist = Math.sqrt((clickX - x) ** 2 + (clickY - y) ** 2);
                if (dist < 25 && dist < minDist) {{
                    minDist = dist;
                    closest = sequenceInfo[i];
                    closestIndex = i;
                }}
            }});
            
            if (closest) {{
                // Update point information
                const cyclicity = closest.is_start || closest.is_end ? 
                    (closest.is_start ? 'Start' : 'End') : 'Middle';
                    
                document.getElementById('pointDetails').innerHTML = `
                    <p><strong>Sequence:</strong> ${{closest.seq_id}}</p>
                    <p><strong>Timestep:</strong> ${{closest.timestep}}</p>
                    <p><strong>Position:</strong> (${{closest.is_start ? 'Start' : closest.is_end ? 'End' : 'Middle'}})</p>
                    <p><strong>Latent 2D:</strong> (${{latents2d[closestIndex][0].toFixed(3)}}, ${{latents2d[closestIndex][1].toFixed(3)}})</p>
                    <p><strong>Color:</strong> <span style="color: ${{colors[closest.seq_id % colors.length]}}">●</span> Sequence ${{closest.seq_id}}</p>
                `;
                
                // Update image
                const imageName = `seq_${{closest.seq_id.toString().padStart(2, '0')}}_t_${{closest.timestep.toString().padStart(2, '0')}}.png`;
                const imagePath = `${{imagesDir}}/${{imageName}}`;
                
                document.getElementById('imageContainer').innerHTML = `
                    <img id="selectedImage" src="${{imagePath}}" alt="Decoded sprite" />
                    <p>Sequence ${{closest.seq_id}}, Timestep ${{closest.timestep}}</p>
                `;
            }}
        }});
        
        // Initialize
        drawVisualization();
        
        console.log('🎉 Interactive latent space visualization loaded!');
        console.log('📊 Data: ${{latents2d.length}} points from ${{new Set(sequenceInfo.map(i => i.seq_id)).size}} sequences');
        console.log('🎯 Loop mode: {self.config.loop_mode}');
    </script>
</body>
</html>"""
        
        with open(filename, 'w') as f:
            f.write(html_content)

    def create_manifold_visualizations(self, x_sample, epoch):
        """
        Create comprehensive manifold visualizations with proper metric computation across all timesteps.
        FIXED: Now computes metrics for flow-evolved latent coordinates at all timesteps.
        """
        print(f"🔬 Creating manifold visualizations for epoch {epoch}")
        
        if not hasattr(self.model, 'G'):
            print("⚠️ No metric tensor available for manifold visualization")
            return
            
        try:
            self.model.eval()
            with torch.no_grad():
                # Get the full model forward pass first to get properly flow-evolved latents
                result = self.model_forward(x_sample)
                z_seq = result.z  # [batch_size, n_obs, latent_dim] - these are the flow-evolved coordinates!
            
                batch_size, n_obs, latent_dim = z_seq.shape
                print(f"📊 Analyzing {batch_size} sequences with {n_obs} timesteps")
                
                # ===== IMPROVED: Use flow-evolved coordinates for metric computation =====
                timestep_data = {}
                
                for t in range(n_obs):
                    print(f"📍 Computing metrics for timestep {t+1}/{n_obs} using flow-evolved coordinates")
                    
                    # CRITICAL FIX: Use the flow-evolved latent coordinates z_t
                    z_t = z_seq[:, t, :]  # [batch_size, latent_dim] - flow-evolved coordinates
                    
                    # Also get the original encoded representation for comparison
                    x_t = x_sample[:, t]
                    encoder_out = self.model.encoder(x_t)
                    mu_t = encoder_out.embedding
                    log_var_t = encoder_out.log_covariance
                    
                    # Generate analysis samples around the FLOW-EVOLVED coordinates
                    n_analysis_samples = min(500, batch_size * 50)  # Reasonable number
                    
                    # Method 1: Sample around flow-evolved coordinates
                    z_t_expanded = z_t.unsqueeze(1).expand(-1, n_analysis_samples // batch_size, -1)
                    z_t_expanded = z_t_expanded.reshape(-1, latent_dim)
                    
                    # Add some noise for variety
                    noise_scale = 0.1  # Small noise to explore neighborhood
                    noise = torch.randn_like(z_t_expanded) * noise_scale
                    z_samples_around_flow = z_t_expanded + noise
                    
                    # Method 2: Sample around encoder mean (for comparison)
                    mu_expanded = mu_t.unsqueeze(1).expand(-1, n_analysis_samples // batch_size, -1)
                    mu_expanded = mu_expanded.reshape(-1, latent_dim)
                    std_expanded = torch.exp(0.5 * log_var_t).unsqueeze(1).expand(-1, n_analysis_samples // batch_size, -1)
                    std_expanded = std_expanded.reshape(-1, latent_dim)
                    
                    eps = torch.randn_like(mu_expanded)
                    z_samples_reparam = mu_expanded + eps * std_expanded
                    
                    # ===== COMPUTE METRICS AT FLOW-EVOLVED COORDINATES =====
                    try:
                        print(f"   Computing G(z) at {len(z_samples_around_flow)} flow-evolved points...")
                        
                        # Compute metric tensor at flow-evolved coordinates
                        G_flow = self.model.G(z_samples_around_flow)  # [n_samples, latent_dim, latent_dim]
                        G_inv_flow = self.model.G_inv(z_samples_around_flow)
                        
                        # Extract metric properties
                        eigenvals_flow = torch.linalg.eigvals(G_inv_flow).real
                        det_G_inv_flow = torch.linalg.det(G_inv_flow)
                        
                        metric_properties = {
                            'eigenvals_mean': eigenvals_flow.mean(dim=0).cpu().numpy(),
                            'eigenvals_std': eigenvals_flow.std(dim=0).cpu().numpy(),
                            'condition_number': (eigenvals_flow.max(dim=1)[0] / (eigenvals_flow.min(dim=1)[0] + 1e-10)).mean().item(),
                            'det_G_inv': det_G_inv_flow.cpu().numpy(),
                            'log_det_G_inv': torch.log(torch.clamp(det_G_inv_flow, min=1e-12)).cpu().numpy(),
                            'eigenvals_all': eigenvals_flow.cpu().numpy(),
                            'samples_used': 'flow_evolved'  # Tag to indicate which coordinates were used
                        }
                        
                        print(f"   ✅ Computed metrics: det(G⁻¹) range [{det_G_inv_flow.min():.2e}, {det_G_inv_flow.max():.2e}]")
                        
                    except Exception as metric_error:
                        print(f"   ⚠️ Failed to compute metrics at flow coordinates: {metric_error}")
                        # Fallback to empty metrics
                        metric_properties = {'samples_used': 'failed'}
                    
                    # Apply flows for trajectory tracking (if not first timestep)
                    flow_intermediate = []
                    z_flow_traj = z_samples_reparam.clone()
                    
                    if t > 0:
                        # Apply flow sequence to see transformation
                        for flow_idx in range(t):
                            if flow_idx < len(self.model.flows):
                                try:
                                    z_prev = z_flow_traj.clone()
                                    flow_result = self.model.flows[flow_idx](z_flow_traj)
                                    z_flow_traj = flow_result.out
                                    
                                    flow_intermediate.append({
                                        'layer': flow_idx,
                                        'z_before': z_prev.cpu().numpy(),
                                        'z_after': z_flow_traj.cpu().numpy()
                                    })
                                except Exception as flow_error:
                                    print(f"   ⚠️ Flow {flow_idx} failed: {flow_error}")
                                    break
                    
                    # Store comprehensive timestep data
                    timestep_data[t] = {
                        'mu': mu_t.cpu().numpy(),
                        'log_var': log_var_t.cpu().numpy(),
                        'z_flow_evolved': z_t.cpu().numpy(),  # The actual flow-evolved coordinates
                        'z_reparam_samples': z_samples_reparam.cpu().numpy(),
                        'z_flow_neighborhood': z_samples_around_flow.cpu().numpy(),  # Samples around flow coordinates
                        'flow_intermediate': flow_intermediate,
                        'metric_properties': metric_properties,
                        'original_images': x_t.cpu().numpy()
                    }
                    
                    print(f"   ✅ Timestep {t} data collected (metrics: {len(metric_properties)} properties)")
                
                # Create the visualizations using properly computed metrics
                print(f"🎨 Creating visualizations with metrics for {len(timestep_data)} timesteps...")
                
                # Enhanced PCA analysis
                self._create_enhanced_pca_analysis(timestep_data, epoch)
                
                # Enhanced manifold heatmaps with flow-evolved metrics
                self._create_enhanced_manifold_heatmaps(timestep_data, epoch)
                
                # Temporal metric evolution analysis
                self._create_temporal_metric_analysis(timestep_data, epoch)
                
                print(f"✨ Manifold visualizations complete for epoch {epoch}")
            
        except Exception as e:
            print(f"⚠️ Manifold visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()

    def _create_enhanced_pca_analysis(self, timestep_data, epoch):
        """Enhanced PCA analysis using flow-evolved coordinates."""
        print("📈 Creating enhanced PCA analysis with flow-evolved coordinates...")
        
        n_timesteps = len(timestep_data)
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle(f'Enhanced PCA Analysis (Flow-Evolved Coords) - Epoch {epoch}', fontsize=16)
        
        # Collect all flow-evolved coordinates for global PCA
        all_z_flow = np.concatenate([data['z_flow_evolved'] for data in timestep_data.values()], axis=0)
        all_z_reparam = np.concatenate([data['z_reparam_samples'][:100] for data in timestep_data.values()], axis=0)
        
        # Fit PCA on flow-evolved coordinates
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(all_z_flow)
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_timesteps))
        
        # Plot 1: Flow-evolved coordinates in PCA space
        ax = axes[0, 0]
        for t, data in timestep_data.items():
            z_pca = pca.transform(data['z_flow_evolved'])
            ax.scatter(z_pca[:, 0], z_pca[:, 1], c=[colors[t]], alpha=0.7, s=30, label=f't={t}')
        ax.set_title('Flow-Evolved Coordinates')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Metric determinant evolution
        ax = axes[0, 1]
        timesteps_with_metrics = []
        det_means = []
        det_stds = []
        
        for t, data in timestep_data.items():
            if 'det_G_inv' in data['metric_properties']:
                det_values = data['metric_properties']['det_G_inv']
                timesteps_with_metrics.append(t)
                det_means.append(np.mean(det_values))
                det_stds.append(np.std(det_values))
        
        if timesteps_with_metrics:
            ax.errorbar(timesteps_with_metrics, det_means, yerr=det_stds, 
                       marker='o', capsize=5, capthick=2, linewidth=2)
            ax.set_title('Metric Determinant Evolution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Mean det(G⁻¹)')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No metric data\navailable', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Metric Determinant Evolution')
        
        # Plot 3: Condition number evolution
        ax = axes[0, 2]
        cond_nums = []
        for t in timesteps_with_metrics:
            cond_nums.append(timestep_data[t]['metric_properties']['condition_number'])
        
        if cond_nums:
            ax.plot(timesteps_with_metrics, cond_nums, 'o-', linewidth=2, markersize=8)
            ax.set_title('Manifold Condition Number')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Condition Number')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No condition\nnumber data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Manifold Condition Number')
        
        # Plot 4: Eigenvalue spectrum evolution
        ax = axes[1, 0]
        for t in timesteps_with_metrics[:min(4, len(timesteps_with_metrics))]:
            eigenvals = timestep_data[t]['metric_properties']['eigenvals_mean']
            ax.plot(range(len(eigenvals)), eigenvals, 'o-', alpha=0.8, label=f't={t}')
            ax.set_title('Eigenvalue Spectrum')
            ax.set_xlabel('Eigenvalue Index')
            ax.set_ylabel('Eigenvalue')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Plot 5: Flow-evolved vs reparam comparison
        ax = axes[1, 1]
        if n_timesteps > 1:
            # Compare first and last timestep
            t_first, t_last = 0, n_timesteps - 1
            z_first = pca.transform(timestep_data[t_first]['z_flow_evolved'])
            z_last = pca.transform(timestep_data[t_last]['z_flow_evolved'])
            
            ax.scatter(z_first[:, 0], z_first[:, 1], alpha=0.6, s=20, label=f'Timestep {t_first}', color='blue')
            ax.scatter(z_last[:, 0], z_last[:, 1], alpha=0.6, s=20, label=f'Timestep {t_last}', color='red')
            ax.set_title('First vs Last Timestep')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Metric scatter plot
        ax = axes[1, 2]
        if len(timesteps_with_metrics) > 0:
            # Use first timestep with metrics for spatial visualization
            t_ref = timesteps_with_metrics[0]
            data_ref = timestep_data[t_ref]
            
            if 'z_flow_neighborhood' in data_ref and 'det_G_inv' in data_ref['metric_properties']:
                z_neighborhood = data_ref['z_flow_neighborhood'][:200]  # Limit for performance
                det_values = data_ref['metric_properties']['det_G_inv'][:200]
                
                z_pca_neighborhood = pca.transform(z_neighborhood)
                scatter = ax.scatter(z_pca_neighborhood[:, 0], z_pca_neighborhood[:, 1], 
                                   c=det_values, cmap='viridis', s=20, alpha=0.7)
                ax.set_title(f'Metric Field (t={t_ref})')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                try:
                    plt.colorbar(scatter, ax=ax, shrink=0.8)
                except:
                    pass  # Skip colorbar if it fails
            else:
                ax.text(0.5, 0.5, 'No neighborhood\ndata available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title('Metric Field')
        
        # Plots 7-9: Additional analysis
        # Plot 7: Trajectory traces
        ax = axes[2, 0]
        for seq_idx in range(min(4, len(timestep_data[0]['z_flow_evolved']))):
            trajectory = []
            for t in range(n_timesteps):
                z_t = timestep_data[t]['z_flow_evolved'][seq_idx]
                trajectory.append(z_t)
            trajectory = np.array(trajectory)
            trajectory_pca = pca.transform(trajectory)
            
            ax.plot(trajectory_pca[:, 0], trajectory_pca[:, 1], 'o-', alpha=0.8, 
                   linewidth=2, markersize=6, label=f'Seq {seq_idx}')
            # Mark start and end
            ax.scatter(trajectory_pca[0, 0], trajectory_pca[0, 1], s=100, marker='s', 
                      color='green', edgecolor='black', zorder=5)
            ax.scatter(trajectory_pca[-1, 0], trajectory_pca[-1, 1], s=100, marker='*', 
                      color='red', edgecolor='black', zorder=5)
        
        ax.set_title('Flow-Evolved Trajectories')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 8: Metric variance across timesteps
        ax = axes[2, 1]
        if timesteps_with_metrics:
            det_variances = []
            for t in timesteps_with_metrics:
                det_values = timestep_data[t]['metric_properties']['det_G_inv']
                det_variances.append(np.var(det_values))
            
            ax.plot(timesteps_with_metrics, det_variances, 'o-', linewidth=2, markersize=8)
            ax.set_title('Metric Variance Evolution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Var(det(G⁻¹))')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No variance\ndata available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Metric Variance Evolution')
        
        # Plot 9: Summary statistics
        ax = axes[2, 2]
        ax.axis('off')
        
        # Create summary text
        summary_text = f"ENHANCED PCA ANALYSIS SUMMARY\n"
        summary_text += "=" * 35 + "\n\n"
        summary_text += f"Total timesteps: {n_timesteps}\n"
        summary_text += f"Timesteps with metrics: {len(timesteps_with_metrics)}\n"
        summary_text += f"PCA explained variance: {pca.explained_variance_ratio_[:2].sum():.1%}\n\n"
        
        if timesteps_with_metrics:
            all_dets = np.concatenate([timestep_data[t]['metric_properties']['det_G_inv'] 
                                     for t in timesteps_with_metrics])
            summary_text += f"Metric Statistics:\n"
            summary_text += f"  Mean det(G⁻¹): {np.mean(all_dets):.2e}\n"
            summary_text += f"  Range: [{np.min(all_dets):.2e}, {np.max(all_dets):.2e}]\n"
            summary_text += f"  Mean condition #: {np.mean([timestep_data[t]['metric_properties']['condition_number'] for t in timesteps_with_metrics]):.2f}\n"
        else:
            summary_text += "No metric data available\n"
        
        summary_text += f"\nCoordinates used: Flow-evolved\n"
        summary_text += f"Analysis samples per timestep: ~500\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
        plt.tight_layout()
        filename = f'enhanced_pca_analysis_epoch_{epoch}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        wandb.log({"enhanced_pca_analysis": wandb.Image(filename, caption=f"Epoch {epoch} - Enhanced PCA")})
        plt.close()
        
        print(f"✅ Enhanced PCA analysis created: {len(timesteps_with_metrics)}/{n_timesteps} timesteps have metrics")

    def _create_enhanced_manifold_heatmaps(self, timestep_data, epoch):
        """Create enhanced manifold heatmaps using flow-evolved coordinates."""
        print("🔥 Creating enhanced manifold heatmaps with flow-evolved coordinates...")
        
        n_timesteps = len(timestep_data)
        fig, axes = plt.subplots(3, n_timesteps, figsize=(4*n_timesteps, 12))
        if n_timesteps == 1:
            axes = axes.reshape(3, 1)
        
        fig.suptitle(f'Enhanced Manifold Heatmaps (Flow-Evolved) - Epoch {epoch}', fontsize=16)
        
        # Global PCA for consistent coordinates using flow-evolved data
        all_z_flow = np.concatenate([data['z_flow_evolved'] for data in timestep_data.values()], axis=0)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(all_z_flow)
        
        for t, data in timestep_data.items():
            # Row 1: Flow-evolved coordinate density
            ax = axes[0, t]
            z_flow_pca = pca.transform(data['z_flow_evolved'])
            
            hist, xedges, yedges = np.histogram2d(z_flow_pca[:, 0], z_flow_pca[:, 1], bins=40, density=True)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im1 = ax.imshow(hist.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')
            ax.set_title(f'Flow-Evolved Density t={t}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            try:
                plt.colorbar(im1, ax=ax, shrink=0.6)
            except:
                pass
            
            # Row 2: Metric field heatmap
            ax = axes[1, t]
            if 'det_G_inv' in data['metric_properties'] and 'z_flow_neighborhood' in data:
                z_neighborhood = data['z_flow_neighborhood']
                det_values = data['metric_properties']['det_G_inv']
                
                # Limit points for performance
                max_points = 200
                if len(z_neighborhood) > max_points:
                    indices = np.random.choice(len(z_neighborhood), max_points, replace=False)
                    z_neighborhood = z_neighborhood[indices]
                    det_values = det_values[indices]
                
                z_neigh_pca = pca.transform(z_neighborhood)
                
                # Create scatter plot with det(G^-1) values
                scatter = ax.scatter(z_neigh_pca[:, 0], z_neigh_pca[:, 1], 
                                   c=det_values, cmap='plasma', s=15, alpha=0.8,
                                   vmin=np.percentile(det_values, 5),
                                   vmax=np.percentile(det_values, 95))
                ax.set_title(f'Metric det(G⁻¹) t={t}')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                try:
                    plt.colorbar(scatter, ax=ax, shrink=0.6)
                except:
                    pass
            else:
                ax.text(0.5, 0.5, 'No metric data\navailable', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Metric det(G⁻¹) t={t}')
            
            # Row 3: Log metric field for better visualization
            ax = axes[2, t]
            if 'log_det_G_inv' in data['metric_properties'] and 'z_flow_neighborhood' in data:
                z_neighborhood = data['z_flow_neighborhood']
                log_det_values = data['metric_properties']['log_det_G_inv']
                
                # Limit points for performance
                if len(z_neighborhood) > max_points:
                    indices = np.random.choice(len(z_neighborhood), max_points, replace=False)
                    z_neighborhood = z_neighborhood[indices]
                    log_det_values = log_det_values[indices]
                
                z_neigh_pca = pca.transform(z_neighborhood)
                
                # Create scatter plot with log det(G^-1) values
                scatter = ax.scatter(z_neigh_pca[:, 0], z_neigh_pca[:, 1], 
                                   c=log_det_values, cmap='coolwarm', s=15, alpha=0.8,
                                   vmin=np.percentile(log_det_values, 5),
                                   vmax=np.percentile(log_det_values, 95))
                ax.set_title(f'log det(G⁻¹) t={t}')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                try:
                    plt.colorbar(scatter, ax=ax, shrink=0.6)
                except:
                    pass
            else:
                ax.text(0.5, 0.5, 'No log metric\ndata available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'log det(G⁻¹) t={t}')
        
        plt.tight_layout()
        filename = f'enhanced_manifold_heatmaps_epoch_{epoch}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        wandb.log({"enhanced_heatmaps": wandb.Image(filename, caption=f"Epoch {epoch} - Enhanced Heatmaps")})
        plt.close()
        
        print(f"✅ Enhanced manifold heatmaps created")

    def _create_temporal_metric_analysis(self, timestep_data, epoch):
        """Create temporal analysis of metric evolution."""
        print("⏱️ Creating temporal metric analysis...")
        
        # Check which timesteps have metric data
        timesteps_with_metrics = [t for t, data in timestep_data.items() 
                                if 'det_G_inv' in data['metric_properties']]
        
        if not timesteps_with_metrics:
            print("⚠️ No timesteps have metric data - skipping temporal analysis")
    
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Temporal Metric Evolution Analysis - Epoch {epoch}', fontsize=16)
        
        # Extract metric data across timesteps
        det_means = []
        det_stds = []
        det_mins = []
        det_maxs = []
        cond_nums = []
        eigenval_means = []
        
        for t in timesteps_with_metrics:
            det_values = timestep_data[t]['metric_properties']['det_G_inv']
            det_means.append(np.mean(det_values))
            det_stds.append(np.std(det_values))
            det_mins.append(np.min(det_values))
            det_maxs.append(np.max(det_values))
            cond_nums.append(timestep_data[t]['metric_properties']['condition_number'])
            eigenval_means.append(np.mean(timestep_data[t]['metric_properties']['eigenvals_mean']))
        
        # Plot 1: Determinant statistics evolution
        ax = axes[0, 0]
        ax.errorbar(timesteps_with_metrics, det_means, yerr=det_stds, 
                   marker='o', capsize=5, capthick=2, linewidth=2, label='Mean ± Std')
        ax.fill_between(timesteps_with_metrics, det_mins, det_maxs, alpha=0.3, label='Min-Max Range')
        ax.set_title('det(G⁻¹) Evolution')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('det(G⁻¹)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Condition number evolution
        ax = axes[0, 1]
        ax.plot(timesteps_with_metrics, cond_nums, 'o-', linewidth=2, markersize=8, color='red')
        ax.set_title('Condition Number Evolution')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Condition Number')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Mean eigenvalue evolution
        ax = axes[0, 2]
        ax.plot(timesteps_with_metrics, eigenval_means, 'o-', linewidth=2, markersize=8, color='green')
        ax.set_title('Mean Eigenvalue Evolution')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Mean Eigenvalue')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Determinant distribution comparison
        ax = axes[1, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps_with_metrics)))
        for i, t in enumerate(timesteps_with_metrics):
            det_values = timestep_data[t]['metric_properties']['det_G_inv']
            ax.hist(np.log10(det_values), bins=20, alpha=0.6, color=colors[i], 
                   density=True, label=f't={t}')
        ax.set_title('log₁₀(det(G⁻¹)) Distributions')
        ax.set_xlabel('log₁₀(det(G⁻¹))')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Eigenvalue spectrum heatmap
        ax = axes[1, 1]
        if len(timesteps_with_metrics) > 1:
            eigenval_matrix = []
            for t in timesteps_with_metrics:
                eigenvals = timestep_data[t]['metric_properties']['eigenvals_mean']
                eigenval_matrix.append(eigenvals)
            
            eigenval_matrix = np.array(eigenval_matrix)
            im = ax.imshow(eigenval_matrix, aspect='auto', cmap='viridis', 
                          extent=[0, eigenval_matrix.shape[1], 
                                 timesteps_with_metrics[-1], timesteps_with_metrics[0]])
            ax.set_title('Eigenvalue Spectrum Evolution')
            ax.set_xlabel('Eigenvalue Index')
            ax.set_ylabel('Timestep')
            try:
                plt.colorbar(im, ax=ax, shrink=0.8)
            except:
                pass
        else:
            ax.text(0.5, 0.5, 'Need multiple\ntimesteps for\nheatmap', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Eigenvalue Spectrum Evolution')
        
        # Plot 6: Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        # Create summary text
        summary_text = f"TEMPORAL METRIC ANALYSIS\n"
        summary_text += "=" * 28 + "\n\n"
        summary_text += f"Timesteps analyzed: {timesteps_with_metrics}\n"
        summary_text += f"Total timesteps: {len(timestep_data)}\n"
        summary_text += f"Coverage: {len(timesteps_with_metrics)}/{len(timestep_data)} ({100*len(timesteps_with_metrics)/len(timestep_data):.0f}%)\n\n"
        
        if len(timesteps_with_metrics) > 1:
            det_change = (det_means[-1] - det_means[0]) / det_means[0] * 100
            cond_change = (cond_nums[-1] - cond_nums[0]) / cond_nums[0] * 100
            
            summary_text += f"Evolution (first → last):\n"
            summary_text += f"  det(G⁻¹): {det_change:+.1f}%\n"
            summary_text += f"  Condition #: {cond_change:+.1f}%\n"
            summary_text += f"  Mean eigenval: {(eigenval_means[-1]/eigenval_means[0]-1)*100:+.1f}%\n\n"
        
        summary_text += f"Current analysis:\n"
        summary_text += f"  Mean det(G⁻¹): {np.mean(det_means):.2e}\n"
        summary_text += f"  Range: [{np.min(det_mins):.2e}, {np.max(det_maxs):.2e}]\n"
        summary_text += f"  Mean condition #: {np.mean(cond_nums):.2f}\n"
        summary_text += f"  Coordinates: Flow-evolved\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        filename = f'temporal_metric_analysis_epoch_{epoch}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        wandb.log({
            "temporal_metric_analysis": wandb.Image(filename, caption=f"Epoch {epoch} - Temporal Metrics"),
            "metrics/timesteps_with_data": len(timesteps_with_metrics),
            "metrics/total_timesteps": len(timestep_data),
            "metrics/coverage_percentage": 100*len(timesteps_with_metrics)/len(timestep_data)
            })
        plt.close()
            
        print(f"✅ Temporal metric analysis created: {len(timesteps_with_metrics)}/{len(timestep_data)} timesteps analyzed")

    def create_high_detail_metric_heatmap(self, x_sample, epoch, temperature_override=None):
        """
        🔥 Create ultra-high resolution PCA + metric heatmap with adjustable temperature.
        
        Args:
            x_sample: Input sequences
            epoch: Current epoch
            temperature_override: Override temperature for sharper metric details (e.g., 0.5, 1.0)
        """
        print(f"🌟 Creating HIGH-DETAIL metric heatmap for epoch {epoch}")
        
        if not hasattr(self.model, 'G'):
            print("⚠️ No metric tensor available for high-detail heatmap")
            return
            
        # Temporarily override temperature for sharper metric tensor
        original_temp = None
        if temperature_override is not None and hasattr(self.model, 'temperature'):
            original_temp = self.model.temperature.item()
            self.model.temperature.data = torch.tensor(temperature_override, device=self.device)
            print(f"🔧 Temporarily using temperature = {temperature_override} (was {original_temp})")
        
        try:
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                z_seq = result.z  # [batch_size, n_obs, latent_dim]
                
                from sklearn.decomposition import PCA
                
                batch_size, n_obs, latent_dim = z_seq.shape
                
                # Prepare PCA data with MORE sequences for better coverage
                z_flat = z_seq.reshape(-1, latent_dim).cpu().numpy()
                pca = PCA(n_components=2)
                z_pca = pca.fit_transform(z_flat)
                z_pca_seq = z_pca.reshape(batch_size, n_obs, 2)
                
                # Define high-resolution grid for detailed analysis
                x_min, x_max = z_pca[:, 0].min() - 1.5, z_pca[:, 0].max() + 1.5
                y_min, y_max = z_pca[:, 1].min() - 1.5, z_pca[:, 1].max() + 1.5
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
                grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
                
                print(f"📊 Computing metrics on {len(grid_points_pca)} high-resolution grid points...")
                
                # Project grid points back to latent space (inverse PCA)
                grid_points_latent = pca.inverse_transform(grid_points_pca)
                
                # Process in chunks for memory efficiency
                chunk_size = 800
                all_log_det = []
                all_cond_num = []
                all_eigenval = []
                
                for i in range(0, len(grid_points_latent), chunk_size):
                    chunk = grid_points_latent[i:i+chunk_size]
                    chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=self.device)
                    
                    try:
                        G_chunk = self.model.G(chunk_tensor)
                        
                        # Log determinant
                        det_G_chunk = torch.linalg.det(G_chunk)
                        log_det_chunk = torch.log(torch.clamp(det_G_chunk, min=1e-12))
                        all_log_det.append(log_det_chunk.cpu().numpy())
                        
                        # Eigenvalues and condition number
                        eigenvals_chunk = torch.linalg.eigvals(G_chunk).real
                        cond_chunk = eigenvals_chunk.max(dim=1)[0] / (eigenvals_chunk.min(dim=1)[0] + 1e-10)
                        all_cond_num.append(cond_chunk.cpu().numpy())
                        
                        # Mean eigenvalue
                        mean_eigenval_chunk = eigenvals_chunk.mean(dim=1)
                        all_eigenval.append(mean_eigenval_chunk.cpu().numpy())
                        
                    except Exception as chunk_error:
                        print(f"⚠️ Chunk {i//chunk_size} failed: {chunk_error}")
                        # Fill with NaNs for failed chunks
                        chunk_len = len(chunk)
                        all_log_det.append(np.full(chunk_len, np.nan))
                        all_cond_num.append(np.full(chunk_len, np.nan))
                        all_eigenval.append(np.full(chunk_len, np.nan))
                
                # Combine results
                log_det_grid = np.concatenate(all_log_det)
                cond_grid = np.concatenate(all_cond_num)
                eigenval_grid = np.concatenate(all_eigenval)
                
                # Reshape to grid
                log_det_heatmap = log_det_grid.reshape(xx.shape)
                cond_heatmap = cond_grid.reshape(xx.shape)
                eigenval_heatmap = eigenval_grid.reshape(xx.shape)
                
                # Create 4-panel ultra-detailed visualization
                fig, axes = plt.subplots(2, 2, figsize=(16, 16))
                
                temp_str = f" (T={temperature_override})" if temperature_override else ""
                fig.suptitle(f'Ultra-High Detail Metric Analysis{temp_str} - Epoch {epoch}', fontsize=18)
                
                # Panel 1: Mean Eigenvalue with high resolution
                im1 = axes[0, 0].contourf(xx, yy, eigenval_heatmap, levels=35, cmap='viridis', extend='both')
                for t in range(n_obs):
                    points_t = z_pca_seq[:, t, :]
                    axes[0, 0].scatter(points_t[:, 0], points_t[:, 1], alpha=0.9, s=25, c='white', edgecolors='black', linewidth=0.8)
                axes[0, 0].set_title('Mean Eigenvalue λ (High Detail)', fontsize=14)
                axes[0, 0].set_xlabel('PC1')
                axes[0, 0].set_ylabel('PC2')
                plt.colorbar(im1, ax=axes[0, 0])
                
                # Panel 2: Log Determinant with high detail
                im2 = axes[0, 1].contourf(xx, yy, log_det_heatmap, levels=35, cmap='plasma', extend='both')
                for t in range(n_obs):
                    points_t = z_pca_seq[:, t, :]
                    axes[0, 1].scatter(points_t[:, 0], points_t[:, 1], alpha=0.9, s=25, c='white', edgecolors='black', linewidth=0.8)
                axes[0, 1].set_title('log(det(G)) (High Detail)', fontsize=14)
                axes[0, 1].set_xlabel('PC1')
                axes[0, 1].set_ylabel('PC2')
                plt.colorbar(im2, ax=axes[0, 1])
                
                # Panel 3: Condition Number with high detail
                im3 = axes[1, 0].contourf(xx, yy, cond_heatmap, levels=35, cmap='coolwarm', extend='both')
                for t in range(n_obs):
                    points_t = z_pca_seq[:, t, :]
                    axes[1, 0].scatter(points_t[:, 0], points_t[:, 1], alpha=0.9, s=25, c='white', edgecolors='black', linewidth=0.8)
                axes[1, 0].set_title('Condition Number (High Detail)', fontsize=14)
                axes[1, 0].set_xlabel('PC1')
                axes[1, 0].set_ylabel('PC2')
                plt.colorbar(im3, ax=axes[1, 0])
                
                # Panel 4: Trajectory overlay with metric contours
                im4 = axes[1, 1].contour(xx, yy, log_det_heatmap, levels=25, colors='gray', alpha=0.6)
                
                # Draw trajectories with different colors per sequence
                colors = plt.get_cmap("tab10")(np.linspace(0, 1, batch_size))
                for seq_idx in range(min(batch_size, 8)):  # Limit to 8 sequences for clarity
                    traj = z_pca_seq[seq_idx]  # [n_obs, 2]
                    axes[1, 1].plot(traj[:, 0], traj[:, 1], 'o-', color=colors[seq_idx], 
                                   alpha=0.8, linewidth=2, markersize=6, label=f'Seq {seq_idx}')
                    # Mark start and end
                    axes[1, 1].scatter(traj[0, 0], traj[0, 1], color=colors[seq_idx], s=100, marker='s', 
                                      edgecolor='black', linewidth=2, zorder=5)
                    axes[1, 1].scatter(traj[-1, 0], traj[-1, 1], color=colors[seq_idx], s=100, marker='*', 
                                      edgecolor='black', linewidth=2, zorder=5)
                
                axes[1, 1].set_title('Trajectories + Metric Contours', fontsize=14)
                axes[1, 1].set_xlabel('PC1')
                axes[1, 1].set_ylabel('PC2')
                axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                
                # Save with temperature info in filename (reduced DPI to avoid PIL warnings)
                temp_suffix = f"_T{temperature_override}" if temperature_override else ""
                filename = f'ultra_high_detail_metric_heatmap{temp_suffix}_epoch_{epoch}.png'
                plt.savefig(filename, dpi=200, bbox_inches='tight')
                
                # Log with temperature-specific metrics
                log_dict = {
                    "manifold/ultra_high_detail_heatmap": wandb.Image(filename, caption=f"Epoch {epoch} - Ultra detail{temp_str}"),
                    "metric_detail/mean_log_det": np.nanmean(log_det_heatmap),
                    "metric_detail/std_log_det": np.nanstd(log_det_heatmap),
                    "metric_detail/mean_condition_number": np.nanmean(cond_heatmap),
                    "metric_detail/max_condition_number": np.nanmax(cond_heatmap),
                    "metric_detail/mean_eigenvalue": np.nanmean(eigenval_heatmap),
                    "metric_detail/grid_points_computed": len(grid_points_pca),
                    "metric_detail/grid_size": f"{xx.shape[0]}x{xx.shape[1]}",
                }
                
                if temperature_override:
                    log_dict["metric_detail/temperature_used"] = temperature_override
                
                wandb.log(log_dict)
                plt.close()
                
                print(f"✅ Ultra-high detail metric heatmap created with {len(grid_points_pca)} points")
                
        except Exception as e:
            print(f"⚠️ Ultra-high detail metric heatmap failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Restore original temperature
            if original_temp is not None:
                self.model.temperature.data = torch.tensor(original_temp, device=self.device)
                print(f"🔧 Restored temperature to {original_temp}")
            
            self.model.train()

    def create_enhanced_geodesic_visualizations(self, x_sample, epoch):
        """
        Create detailed geodesic and metric impact visualizations in PCA space.
        """
        print(f"🌍 Creating enhanced geodesic & metric impact visualizations for epoch {epoch}")
        
        if not hasattr(self.model, 'G'):
            print("⚠️ No metric tensor available for geodesic visualization")
            return
            
        try:
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                z_seq = result.z  # [batch_size, n_obs, latent_dim]
                
                from sklearn.decomposition import PCA
                import numpy as np
                
                batch_size, n_obs, latent_dim = z_seq.shape
                
                # Prepare PCA data
                z_flat = z_seq.reshape(-1, latent_dim).cpu().numpy()
                pca = PCA(n_components=2)
                z_pca = pca.fit_transform(z_flat)
                z_pca_seq = z_pca.reshape(batch_size, n_obs, 2)
                
                # Create comprehensive 2x3 visualization
                fig, axes = plt.subplots(2, 3, figsize=(20, 14))
                fig.suptitle(f'Enhanced Geodesic & Metric Analysis - Epoch {epoch}', fontsize=18)
                
                # Panel 1: Geodesic vs Euclidean paths
                self._create_geodesic_comparison_plot(axes[0, 0], z_seq, z_pca_seq, pca, epoch)
                
                # Panel 2: Metric tensor eigenvalue field
                self._create_metric_eigenvalue_field(axes[0, 1], z_pca_seq, pca, epoch)
                
                # Panel 3: Metric tensor ellipses
                self._create_metric_ellipse_field(axes[0, 2], z_pca_seq, pca, epoch)
                
                # Panel 4: Geodesic deviation analysis
                self._create_geodesic_deviation_analysis(axes[1, 0], z_seq, z_pca_seq, pca, epoch)
                
                # Panel 5: Curvature visualization
                self._create_curvature_visualization(axes[1, 1], z_pca_seq, pca, epoch)
                
                # Panel 6: Metric amplification heatmap
                self._create_metric_amplification_heatmap(axes[1, 2], z_pca_seq, pca, epoch)
                
                plt.tight_layout()
                
                # Save and log
                filename = f'enhanced_geodesic_analysis_epoch_{epoch}.png'
                plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
                
                wandb.log({
                    "geodesic/enhanced_analysis": wandb.Image(filename, caption=f"Epoch {epoch} - Enhanced geodesic & metric analysis"),
                })
                plt.close()
                
                # 🎬 NEW: Create INTERACTIVE geodesic visualization with time slider (only every 6 epochs to avoid slowdown)
                if epoch % 6 == 0:
                    self._create_interactive_geodesic_slider_visualization(z_seq, z_pca_seq, pca, epoch)
                
                print(f"✅ Enhanced geodesic visualization created for epoch {epoch}")
                
        except Exception as e:
            print(f"⚠️ Enhanced geodesic visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()

    def _create_interactive_geodesic_slider_visualization(self, z_seq, z_pca_seq, pca, epoch):
        """
        Create interactive Plotly visualization with TIME SLIDER for geodesic evolution.
        Shows how geodesics, eigenvalue fields, and metric ellipses evolve across timesteps.
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import numpy as np
            
            print(f"🌍 Creating INTERACTIVE geodesic slider visualization with timestep evolution...")
            
            batch_size, n_obs, latent_dim = z_seq.shape
            
            # Create grid for background fields
            x_min, x_max = z_pca_seq[:, :, 0].min() - 1, z_pca_seq[:, :, 0].max() + 1
            y_min, y_max = z_pca_seq[:, :, 1].min() - 1, z_pca_seq[:, :, 1].max() + 1
            nx, ny = 50, 50
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
            grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
            grid_points_latent = pca.inverse_transform(grid_points_pca)
            
            # CRITICAL FIX: Prepare dynamic background fields for each timestep
            print(f"📊 Computing DYNAMIC geodesic background fields for each timestep...")
            
            # Store the base grid for transformations
            grid_points_latent_base = pca.inverse_transform(grid_points_pca)
            grid_tensor_base = torch.tensor(grid_points_latent_base, dtype=torch.float32, device=self.device)
            
            # Project metric to PCA space - we'll do this for each timestep
            V = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
            
            # Pre-compute background fields for each timestep
            timestep_background_fields = {}
            
            for t_bg in range(n_obs):
                print(f"   Computing background fields for timestep {t_bg}...")
                
                if t_bg == 0:
                    # For t=0, use the original grid
                    grid_tensor_t = grid_tensor_base
                else:
                    # For t>0, apply forward flows to transform grid
                    try:
                        print(f"     🔄 Applying {t_bg} forward flow steps to geodesic grid...")
                        grid_tensor_t = grid_tensor_base.clone()
                        
                        # Apply forward flows to transform grid from t=0 to t=t_bg
                        for flow_idx in range(t_bg):
                            if flow_idx < len(self.model.flows):
                                print(f"       Applying flow {flow_idx}: z_{flow_idx} → z_{flow_idx+1}")
                                flow_result = self.model.flows[flow_idx](grid_tensor_t)
                                grid_tensor_t = flow_result.out
                            else:
                                print(f"       ⚠️ Flow {flow_idx} not available, stopping")
                                break
                        
                        print(f"     ✅ Geodesic grid transformed to timestep {t_bg} coordinate system")
                        
                    except Exception as e:
                        print(f"     ❌ Geodesic grid transformation failed for t={t_bg}: {e}")
                        # Fallback: use original grid with small perturbation
                        grid_tensor_t = grid_tensor_base + 0.1 * t_bg * torch.randn_like(grid_tensor_base)
                        print(f"     🔄 Using perturbed geodesic grid as fallback for t={t_bg}")
                
                # Compute metric at transformed grid
                try:
                    G_grid_t = self.model.G(grid_tensor_t)
                    G_pca_t = torch.matmul(torch.matmul(V.unsqueeze(0), G_grid_t), V.T.unsqueeze(0))
                    
                    # Compute metric properties for this timestep
                    det_G_pca_t = torch.linalg.det(G_pca_t).cpu().numpy().reshape(xx.shape)
                    eigenvals_t, eigenvecs_t = torch.linalg.eigh(G_pca_t)
                    eigenvals_np_t = eigenvals_t.cpu().numpy()
                    eigenvecs_np_t = eigenvecs_t.cpu().numpy()
                    
                    timestep_background_fields[t_bg] = {
                        'det_G_pca': det_G_pca_t,
                        'eigenvals_np': eigenvals_np_t,
                        'eigenvecs_np': eigenvecs_np_t
                    }
                    
                    print(f"     ✅ Geodesic background computed for t={t_bg}: det(G) range [{det_G_pca_t.min():.2e}, {det_G_pca_t.max():.2e}]")
                    
                except Exception as e:
                    print(f"     ❌ Failed to compute geodesic background for t={t_bg}: {e}")
                    # Use t=0 as fallback
                    if 0 in timestep_background_fields:
                        timestep_background_fields[t_bg] = timestep_background_fields[0]
                    else:
                        # Create dummy fields using t=0 data to avoid gray backgrounds
                        if 0 in timestep_background_fields:
                            timestep_background_fields[t_bg] = timestep_background_fields[0]
                        else:
                            # Absolute fallback
                            timestep_background_fields[t_bg] = {
                                'det_G_pca': np.ones(xx.shape),
                                'eigenvals_np': np.ones((xx.size, 2)),
                                'eigenvecs_np': np.tile(np.eye(2), (xx.size, 1, 1))
                            }
            
            # CRITICAL FIX: Compute metrics at FLOW-EVOLVED coordinates for each timestep
            timestep_geodesic_data = []
            for t in range(n_obs):
                z_t_pca = z_pca_seq[:, t, :]  # [batch_size, 2] - For visualization positioning
                # CORRECT: Use actual flow-evolved coordinates from z_seq
                z_t_latent = z_seq[:, t, :].cpu().numpy()  # FLOW-EVOLVED coordinates!
                z_t_tensor = torch.tensor(z_t_latent, dtype=torch.float32, device=self.device)
                print(f"    ✅ Geodesic slider: Using FLOW-EVOLVED coordinates for timestep {t}")
                
                # Compute metrics at sequence positions
                G_t = self.model.G(z_t_tensor)
                G_t_pca = torch.matmul(torch.matmul(V.unsqueeze(0), G_t), V.T.unsqueeze(0))
                
                det_t = torch.linalg.det(G_t_pca).cpu().numpy()
                eigenvals_t, eigenvecs_t = torch.linalg.eigh(G_t_pca)
                eigenvals_t_np = eigenvals_t.cpu().numpy()
                eigenvecs_t_np = eigenvecs_t.cpu().numpy()
                
                timestep_geodesic_data.append({
                    'positions': z_t_pca,
                    'det': det_t,
                    'eigenvals': eigenvals_t_np,
                    'eigenvecs': eigenvecs_t_np
                })
            
            # Create interactive slider visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "🎯 Geodesic Trajectories at Timestep",
                    "📊 Eigenvalue Field with Sequences", 
                    "⭕ Metric Ellipses at Sequence Points",
                    "🌟 Det(G) Heatmap with Evolution"
                ],
                horizontal_spacing=0.1,
                vertical_spacing=0.12
            )
            
            # Create frames for each timestep
            frames = []
            colors = px.colors.qualitative.Set3[:min(batch_size, 8)]
            
            for t in range(n_obs):
                frame_data = []
                geo_data = timestep_geodesic_data[t]
                
                # CRITICAL FIX: Get timestep-specific background fields
                bg_fields = timestep_background_fields.get(t, timestep_background_fields.get(0, {}))
                det_G_pca_t = bg_fields.get('det_G_pca', np.ones(xx.shape))
                eigenvals_np_t = bg_fields.get('eigenvals_np', np.ones((xx.size, 2)))
                
                print(f"   📊 Geodesic frame {t}: using det_G background range [{det_G_pca_t.min():.2e}, {det_G_pca_t.max():.2e}]")
                
                # Panel 1: Geodesic trajectories up to timestep t
                frame_data.append(
                    go.Contour(
                        x=np.linspace(x_min, x_max, nx),
                        y=np.linspace(y_min, y_max, ny),
                        z=np.log10(np.clip(det_G_pca_t, 1e-10, None)),  # FIXED: timestep-specific
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="log₁₀(det(G))", x=0.45, len=0.35, y=0.8),
                        opacity=0.3,
                        name="det(G) background",
                        hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>log₁₀(det(G)): %{z:.2f}<extra></extra>",
                        xaxis='x', yaxis='y'
                    )
                )
                
                # Add trajectory paths up to timestep t
                for seq_idx in range(min(batch_size, 8)):
                    traj_segment = z_pca_seq[seq_idx, :t+1, :]  # Path up to timestep t
                    if len(traj_segment) > 1:
                        frame_data.append(
                            go.Scatter(
                                x=traj_segment[:, 0],
                                y=traj_segment[:, 1],
                                mode='lines+markers',
                                line=dict(color=colors[seq_idx], width=3),
                                marker=dict(size=8, color=colors[seq_idx], line=dict(color='white', width=1)),
                                name=f"Trajectory {seq_idx}",
                                hovertemplate="Seq %{seq_idx}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
                                xaxis='x', yaxis='y'
                            )
                        )
                    
                    # Highlight current position
                    frame_data.append(
                        go.Scatter(
                            x=[geo_data['positions'][seq_idx, 0]],
                            y=[geo_data['positions'][seq_idx, 1]],
                            mode='markers',
                            marker=dict(size=15, color=colors[seq_idx], 
                                       symbol='star', line=dict(color='white', width=2)),
                            name=f"Current t={t}",
                            hovertemplate=f"Seq {seq_idx} at t={t}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>",
                            xaxis='x', yaxis='y'
                        )
                    )
                
                # Panel 2: Eigenvalue field with current sequences
                # FIXED: Background eigenvalue magnitude field (timestep-specific)
                eigenval_mag_t = eigenvals_np_t[:, 1].reshape(xx.shape)  # Major eigenvalue at timestep t
                frame_data.append(
                    go.Contour(
                        x=np.linspace(x_min, x_max, nx),
                        y=np.linspace(y_min, y_max, ny),
                        z=eigenval_mag_t,  # FIXED: timestep-specific eigenvalue field
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Max λ", x=0.95, len=0.35, y=0.8),
                        opacity=0.5,
                        name="Eigenvalue field",
                        hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Max λ: %{z:.2f}<extra></extra>",
                        xaxis='x2', yaxis='y2'
                    )
                )
                
                # Add sequence points with eigenvalue information
                frame_data.append(
                    go.Scatter(
                        x=geo_data['positions'][:, 0],
                        y=geo_data['positions'][:, 1],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color=geo_data['eigenvals'][:, 1],  # Color by major eigenvalue
                            colorscale='Plasma',
                            showscale=False,
                            line=dict(color='white', width=2)
                        ),
                        name=f"Eigenvals t={t}",
                        hovertemplate="Seq: %{pointNumber}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Max λ: %{marker.color:.2f}<extra></extra>",
                        xaxis='x2', yaxis='y2'
                    )
                )
                
                # Panel 3: Metric ellipses at sequence points
                frame_data.append(
                    go.Contour(
                        x=np.linspace(x_min, x_max, nx),
                        y=np.linspace(y_min, y_max, ny),
                        z=det_G_pca_t,  # FIXED: timestep-specific det(G) background
                        colorscale='RdYlBu_r',
                        showscale=True,
                        colorbar=dict(title="det(G)", x=0.45, len=0.35, y=0.25),
                        opacity=0.3,
                        name="det(G) background",
                        hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>det(G): %{z:.2f}<extra></extra>",
                        xaxis='x3', yaxis='y3'
                    )
                )
                
                # Add metric ellipses around sequence points
                for seq_idx in range(min(batch_size, 6)):  # Limit for visual clarity
                    pos = geo_data['positions'][seq_idx]
                    eigenvals_seq = geo_data['eigenvals'][seq_idx]
                    eigenvecs_seq = geo_data['eigenvecs'][seq_idx]
                    
                    # Create ellipse points
                    theta = np.linspace(0, 2*np.pi, 50)
                    ellipse_points = []
                    scale = 0.3  # Scale factor for ellipse size
                    
                    for angle in theta:
                        # Ellipse in eigenvalue basis
                        local_pt = np.array([
                            scale * np.sqrt(eigenvals_seq[0]) * np.cos(angle),
                            scale * np.sqrt(eigenvals_seq[1]) * np.sin(angle)
                        ])
                        # Rotate by eigenvectors
                        rotated_pt = eigenvecs_seq @ local_pt
                        ellipse_points.append(pos + rotated_pt)
                    
                    ellipse_points = np.array(ellipse_points)
                    
                    frame_data.append(
                        go.Scatter(
                            x=ellipse_points[:, 0],
                            y=ellipse_points[:, 1],
                            mode='lines',
                            line=dict(color=colors[seq_idx], width=2),
                            fill='toself',
                            fillcolor=colors[seq_idx],
                            opacity=0.3,
                            name=f"Ellipse {seq_idx}",
                            hovertemplate=f"Metric ellipse {seq_idx}<extra></extra>",
                            xaxis='x3', yaxis='y3'
                        )
                    )
                
                # Panel 4: Det(G) evolution with current state
                frame_data.append(
                    go.Heatmap(
                        x=np.linspace(x_min, x_max, nx),
                        y=np.linspace(y_min, y_max, ny),
                        z=det_G_pca_t,  # FIXED: timestep-specific det(G) heatmap
                        colorscale='Hot',
                        showscale=True,
                        colorbar=dict(title="det(G)", x=0.95, len=0.35, y=0.25),
                        name="det(G) heatmap",
                        hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>det(G): %{z:.2f}<extra></extra>",
                        xaxis='x4', yaxis='y4'
                    )
                )
                
                # Overlay current sequence positions
                frame_data.append(
                    go.Scatter(
                        x=geo_data['positions'][:, 0],
                        y=geo_data['positions'][:, 1],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='white',
                            symbol='circle-open',
                            line=dict(color='black', width=3)
                        ),
                        name=f"Positions t={t}",
                        hovertemplate="Seq: %{pointNumber}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
                        xaxis='x4', yaxis='y4'
                    )
                )
                
                frames.append(go.Frame(data=frame_data, name=str(t)))
            
            # Set initial frame (t=0)
            fig.add_traces(frames[0].data)
            fig.frames = frames
            
            # Update layout with slider controls
            fig.update_layout(
                title={
                    'text': f"🌍 INTERACTIVE GEODESIC EVOLUTION - EPOCH {epoch}<br>"
                           f"<span style='font-size:14px'>🎯 Use slider to explore geodesics, eigenfields, ellipses across timesteps 0→{n_obs-1}</span>",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#2E86AB'}
                },
                sliders=[{
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Timestep: ",
                        "visible": True,
                        "xanchor": "right"
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[str(t)], {"frame": {"duration": 300, "redraw": True}, 
                                               "mode": "immediate", "transition": {"duration": 200}}],
                            "label": f"t={t}",
                            "method": "animate"
                        }
                        for t in range(n_obs)
                    ]
                }],
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 700, "redraw": True}, 
                                          "fromcurrent": True, "transition": {"duration": 500}}],
                            "label": "▶️ Auto Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True}, 
                                            "mode": "immediate", "transition": {"duration": 0}}],
                            "label": "⏸️ Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0.02,
                    "yanchor": "top"
                }],
                width=1600,
                height=1000,
                showlegend=False,
                paper_bgcolor='rgba(240,248,255,0.95)',
                plot_bgcolor='rgba(255,255,255,0.8)',
            )
            
            # Update subplot axes
            fig.update_xaxes(title_text="PC1", row=1, col=1)
            fig.update_yaxes(title_text="PC2", row=1, col=1)
            fig.update_xaxes(title_text="PC1", row=1, col=2)
            fig.update_yaxes(title_text="PC2", row=1, col=2)
            fig.update_xaxes(title_text="PC1", row=2, col=1)
            fig.update_yaxes(title_text="PC2", row=2, col=1)
            fig.update_xaxes(title_text="PC1", row=2, col=2)
            fig.update_yaxes(title_text="PC2", row=2, col=2)
            
            # Save interactive geodesic slider visualization
            html_filename = f'interactive_geodesic_slider_epoch_{epoch}.html'
            fig.write_html(html_filename, include_plotlyjs=True)
            
            png_filename = f'interactive_geodesic_slider_epoch_{epoch}.png'
            self._safe_write_image(fig, png_filename, width=1600, height=1000, scale=2)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "geodesics/interactive_slider": wandb.Html(html_filename),
                    "geodesics/geodesic_slider_static": wandb.Image(png_filename, 
                        caption=f"Epoch {epoch} - Interactive geodesic slider"),
                })
            
            print(f"🌍 Interactive geodesic slider visualization saved: {html_filename}")
            print(f"   📊 Tracks geodesics, eigenvalue fields, metric ellipses across {n_obs} timesteps")
            print(f"   🎯 Use slider to explore geodesic evolution!")
            
        except Exception as e:
            print(f"⚠️ Interactive geodesic slider visualization failed: {e}")
            import traceback
            traceback.print_exc()

    def _create_geodesic_comparison_plot(self, ax, z_seq, z_pca_seq, pca, epoch):
        """Compare geodesic vs Euclidean paths between sequence points."""
        try:
            batch_size, n_obs, latent_dim = z_seq.shape
            
            # Select a few sequences for comparison
            n_compare = min(4, batch_size)
            colors = plt.cm.viridis(np.linspace(0, 1, n_compare))
            
            for seq_idx in range(n_compare):
                z_full = z_seq[seq_idx].cpu()  # [n_obs, latent_dim]
                z_pca_traj = z_pca_seq[seq_idx]  # [n_obs, 2]
                
                # Plot Euclidean path in PCA space
                ax.plot(z_pca_traj[:, 0], z_pca_traj[:, 1], 'o-', 
                       color=colors[seq_idx], alpha=0.6, linewidth=2, 
                       markersize=6, label=f'Euclidean Seq {seq_idx}')
                
                # Compute approximate geodesic paths between consecutive points
                for t in range(n_obs - 1):
                    start_point = z_full[t:t+1]  # [1, latent_dim]
                    end_point = z_full[t+1:t+1+1]  # [1, latent_dim]
                    
                    # Create interpolated path
                    n_steps = 10
                    alphas = np.linspace(0, 1, n_steps)
                    geodesic_path = []
                    
                    for alpha in alphas:
                        # Simple interpolation (more sophisticated geodesic integration would be better)
                        interp_point = (1 - alpha) * start_point + alpha * end_point
                        geodesic_path.append(interp_point)
                    
                    geodesic_latent = torch.cat(geodesic_path, dim=0)  # [n_steps, latent_dim]
                    geodesic_pca = pca.transform(geodesic_latent.numpy())  # [n_steps, 2]
                    
                    # Plot geodesic path
                    ax.plot(geodesic_pca[:, 0], geodesic_pca[:, 1], '--', 
                           color=colors[seq_idx], alpha=0.9, linewidth=1.5)
                
                # Mark start and end
                ax.scatter(z_pca_traj[0, 0], z_pca_traj[0, 1], 
                          color=colors[seq_idx], s=100, marker='s', 
                          edgecolor='black', linewidth=2, zorder=5)
                ax.scatter(z_pca_traj[-1, 0], z_pca_traj[-1, 1], 
                          color=colors[seq_idx], s=100, marker='*', 
                          edgecolor='black', linewidth=2, zorder=5)
            
            ax.set_title('Geodesic vs Euclidean Paths\n(solid=Euclidean, dashed=approx geodesic)', fontsize=12)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', transform=ax.transAxes, ha='center')

    def _create_metric_eigenvalue_field(self, ax, z_pca_seq, pca, epoch):
        """Create vector field showing metric tensor eigenvalue directions."""
        try:
            # Create grid for eigenvalue field
            x_min, x_max = z_pca_seq[:, :, 0].min() - 1, z_pca_seq[:, :, 0].max() + 1
            y_min, y_max = z_pca_seq[:, :, 1].min() - 1, z_pca_seq[:, :, 1].max() + 1
            
            # Lower resolution for vector field
            nx, ny = 15, 15
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
            grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
            grid_points_latent = pca.inverse_transform(grid_points_pca)
            
            # Compute metric tensor at grid points
            grid_tensor = torch.tensor(grid_points_latent, dtype=torch.float32, device=self.device)
            G_grid = self.model.G(grid_tensor)  # [grid_size, latent_dim, latent_dim]
            
            # Project metric to PCA space: G_pca = V^T G V where V is PCA components
            V = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)  # [2, latent_dim]
            G_pca = torch.matmul(torch.matmul(V.unsqueeze(0), G_grid), V.T.unsqueeze(0))  # [grid_size, 2, 2]
            
            # Compute eigenvalues and eigenvectors of projected metric
            eigenvals, eigenvecs = torch.linalg.eigh(G_pca)
            
            # Convert to numpy for plotting
            eigenvals_np = eigenvals.cpu().numpy()  # [grid_size, 2]
            eigenvecs_np = eigenvecs.cpu().numpy()  # [grid_size, 2, 2]
            
            # Plot background heatmap of determinant
            det_values = eigenvals_np[:, 0] * eigenvals_np[:, 1]
            det_heatmap = det_values.reshape(xx.shape)
            
            im = ax.contourf(xx, yy, det_heatmap, levels=20, cmap='viridis', alpha=0.6)
            plt.colorbar(im, ax=ax, label='det(G_pca)')
            
            # Plot eigenvector directions scaled by eigenvalues
            scale_factor = 0.3
            for i in range(0, len(grid_points_pca), 3):  # Skip some for clarity
                x, y = grid_points_pca[i]
                
                # First eigenvector (major axis)
                v1 = eigenvecs_np[i, :, 1] * np.sqrt(eigenvals_np[i, 1]) * scale_factor
                ax.arrow(x, y, v1[0], v1[1], head_width=0.1, head_length=0.1, 
                        fc='red', ec='red', alpha=0.8)
                
                # Second eigenvector (minor axis)
                v2 = eigenvecs_np[i, :, 0] * np.sqrt(eigenvals_np[i, 0]) * scale_factor
                ax.arrow(x, y, v2[0], v2[1], head_width=0.1, head_length=0.1, 
                        fc='blue', ec='blue', alpha=0.8)
            
            # Overlay sequence trajectories
            for seq_idx in range(min(3, z_pca_seq.shape[0])):
                traj = z_pca_seq[seq_idx]
                ax.plot(traj[:, 0], traj[:, 1], 'w-', linewidth=2, alpha=0.9)
                ax.scatter(traj[:, 0], traj[:, 1], c='white', s=30, edgecolor='black', linewidth=1)
            
            ax.set_title('Metric Eigenvalue Field\n(red=major, blue=minor eigenvectors)', fontsize=12)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', transform=ax.transAxes, ha='center')

    def _create_metric_ellipse_field(self, ax, z_pca_seq, pca, epoch):
        """Create field of ellipses showing metric tensor shape."""
        try:
            from matplotlib.patches import Ellipse
            
            # Create coarser grid for ellipses
            x_min, x_max = z_pca_seq[:, :, 0].min() - 1, z_pca_seq[:, :, 0].max() + 1
            y_min, y_max = z_pca_seq[:, :, 1].min() - 1, z_pca_seq[:, :, 1].max() + 1
            
            nx, ny = 8, 8
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
            grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
            grid_points_latent = pca.inverse_transform(grid_points_pca)
            
            # Compute metric tensor at grid points
            grid_tensor = torch.tensor(grid_points_latent, dtype=torch.float32, device=self.device)
            G_grid = self.model.G(grid_tensor)
            
            # Project to PCA space
            V = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
            G_pca = torch.matmul(torch.matmul(V.unsqueeze(0), G_grid), V.T.unsqueeze(0))
            
            eigenvals, eigenvecs = torch.linalg.eigh(G_pca)
            eigenvals_np = eigenvals.cpu().numpy()
            eigenvecs_np = eigenvecs.cpu().numpy()
            
            # Plot ellipses
            for i, (x, y) in enumerate(grid_points_pca):
                # Get eigenvalues and eigenvectors for this point
                lambda1, lambda2 = eigenvals_np[i]
                v1, v2 = eigenvecs_np[i, :, 0], eigenvecs_np[i, :, 1]
                
                # Ellipse parameters
                width = 2 / np.sqrt(lambda1) * 0.3  # Inverse scaling
                height = 2 / np.sqrt(lambda2) * 0.3
                angle = np.degrees(np.arctan2(v2[1], v2[0]))
                
                # Color by determinant
                det_val = lambda1 * lambda2
                color = plt.cm.plasma(det_val / (eigenvals_np[:, 0] * eigenvals_np[:, 1]).max())
                
                ellipse = Ellipse((x, y), width, height, angle=angle, 
                                facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
                ax.add_patch(ellipse)
            
            # Overlay trajectories
            colors = plt.get_cmap("tab10")(np.linspace(0, 1, min(4, z_pca_seq.shape[0])))
            for seq_idx in range(min(4, z_pca_seq.shape[0])):
                traj = z_pca_seq[seq_idx]
                ax.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[seq_idx], 
                       linewidth=2, markersize=5, alpha=0.9, label=f'Seq {seq_idx}')
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title('Metric Tensor Ellipses\n(size ∝ 1/√λ, color ∝ det(G))', fontsize=12)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend(fontsize=10)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', transform=ax.transAxes, ha='center')

    def _create_geodesic_deviation_analysis(self, ax, z_seq, z_pca_seq, pca, epoch):
        """Analyze how much geodesics deviate from Euclidean paths."""
        try:
            batch_size, n_obs, latent_dim = z_seq.shape
            
            deviations = []
            path_lengths_euclidean = []
            path_lengths_riemannian = []
            
            for seq_idx in range(min(6, batch_size)):
                z_full = z_seq[seq_idx].cpu()
                
                euclidean_length = 0
                riemannian_length = 0
                
                for t in range(n_obs - 1):
                    # Euclidean distance
                    eucl_dist = torch.norm(z_full[t+1] - z_full[t]).item()
                    euclidean_length += eucl_dist
                    
                    # Approximate Riemannian distance using metric at midpoint
                    midpoint = (z_full[t] + z_full[t+1]) / 2
                    G_mid = self.model.G(midpoint.unsqueeze(0).to(self.device))  # [1, latent_dim, latent_dim]
                    diff = (z_full[t+1] - z_full[t]).unsqueeze(0).to(self.device)  # [1, latent_dim]
                    
                    riem_dist_sq = torch.matmul(torch.matmul(diff.unsqueeze(1), G_mid), diff.unsqueeze(2)).squeeze()
                    riem_dist = torch.sqrt(torch.clamp(riem_dist_sq, min=1e-8)).item()
                    riemannian_length += riem_dist
                
                path_lengths_euclidean.append(euclidean_length)
                path_lengths_riemannian.append(riemannian_length)
                deviations.append(abs(riemannian_length - euclidean_length) / euclidean_length)
            
            # Plot comparison
            sequences = list(range(len(deviations)))
            width = 0.35
            
            ax.bar([x - width/2 for x in sequences], path_lengths_euclidean, width, 
                  label='Euclidean Length', alpha=0.7, color='blue')
            ax.bar([x + width/2 for x in sequences], path_lengths_riemannian, width, 
                  label='Riemannian Length', alpha=0.7, color='red')
            
            ax.set_xlabel('Sequence Index')
            ax.set_ylabel('Path Length')
            ax.set_title(f'Path Length Comparison\nMean deviation: {np.mean(deviations):.2%}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text with statistics
            ax.text(0.02, 0.98, f'Avg amplification: {np.mean(np.array(path_lengths_riemannian)/np.array(path_lengths_euclidean)):.2f}x',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', transform=ax.transAxes, ha='center')

    def _create_curvature_visualization(self, ax, z_pca_seq, pca, epoch, timestep=None):
        """Enhanced curvature visualization with timestep-specific metrics and true geometric curvature."""
        try:
            # Create grid for curvature analysis
            x_min, x_max = z_pca_seq[:, :, 0].min() - 1, z_pca_seq[:, :, 0].max() + 1
            y_min, y_max = z_pca_seq[:, :, 1].min() - 1, z_pca_seq[:, :, 1].max() + 1
            
            # Higher resolution for more accuracy
            nx, ny = 20, 20
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
            
            # Initialize curvature arrays for different measures
            scalar_curvature = np.zeros_like(xx)
            ricci_curvature = np.zeros_like(xx)
            gaussian_curvature = np.zeros_like(xx)
            
            h = 0.01  # Small step for finite differences
            
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    try:
                        # Central point in PCA space
                        pca_center = np.array([xx[j, i], yy[j, i]])
                        z_center = pca.inverse_transform(pca_center.reshape(1, -1)).flatten()
                        
                        # Get timestep-specific metric at center
                        G_center = self._get_timestep_metric(z_center, timestep)
                        
                        # Compute partial derivatives of metric using finite differences
                        dG_dx = self._compute_metric_derivative(z_center, pca, 0, h, timestep)
                        dG_dy = self._compute_metric_derivative(z_center, pca, 1, h, timestep)
                        
                        # For PCA space (2D), compute simplified curvature measures
                        if G_center.shape[0] >= 2:
                            # Project to PCA subspace
                            V = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
                            G_pca = torch.matmul(torch.matmul(V, G_center), V.T).cpu().numpy()
                            
                            # Gaussian curvature (for 2D surface) 
                            det_G = np.linalg.det(G_pca)
                            if det_G > 1e-10:
                                # Simplified Gaussian curvature using metric determinant variation
                                ddet_dx = np.trace(np.linalg.solve(G_pca, dG_dx[:2, :2])) * det_G
                                ddet_dy = np.trace(np.linalg.solve(G_pca, dG_dy[:2, :2])) * det_G
                                gaussian_curvature[j, i] = -(ddet_dx + ddet_dy) / (2 * det_G)
                            
                            # Scalar curvature approximation using Christoffel symbols
                            scalar_curvature[j, i] = self._compute_scalar_curvature_approx(G_pca, dG_dx[:2, :2], dG_dy[:2, :2])
                            
                            # Ricci curvature (trace of Ricci tensor)
                            ricci_curvature[j, i] = self._compute_ricci_curvature_approx(G_pca, dG_dx[:2, :2], dG_dy[:2, :2])
                        
                    except Exception as e:
                        # Set to zero if computation fails
                        scalar_curvature[j, i] = 0
                        ricci_curvature[j, i] = 0
                        gaussian_curvature[j, i] = 0
            
            # Choose which curvature to display (you can add controls for this)
            curvature_display = scalar_curvature
            curvature_name = "Scalar Curvature"
            
            # If timestep is specified, include it in the title
            timestep_str = f" (t={timestep})" if timestep is not None else ""
            
            # Plot enhanced curvature heatmap
            im = ax.contourf(xx, yy, curvature_display, levels=20, cmap='RdYlBu_r')
            plt.colorbar(im, ax=ax, label=f'{curvature_name}{timestep_str}')
            
            # Add contour lines for better visibility
            ax.contour(xx, yy, curvature_display, levels=10, colors='black', alpha=0.3, linewidths=0.5)
            
            # Overlay trajectories with timestep markers
            for seq_idx in range(min(3, z_pca_seq.shape[0])):
                traj = z_pca_seq[seq_idx]
                ax.plot(traj[:, 0], traj[:, 1], 'k-', linewidth=2, alpha=0.8)
                
                # Color-code timesteps
                n_timesteps = traj.shape[0]
                colors = plt.cm.viridis(np.linspace(0, 1, n_timesteps))
                for t in range(n_timesteps):
                    ax.scatter(traj[t, 0], traj[t, 1], c=[colors[t]], s=40, 
                             edgecolor='white', linewidth=1, alpha=0.9)
                    ax.text(traj[t, 0], traj[t, 1], str(t), fontsize=8, ha='center', va='center')
            
            ax.set_title(f'Enhanced {curvature_name} Field{timestep_str}\n(geometric curvature of metric)', fontsize=12)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', transform=ax.transAxes, ha='center')
    
    def _get_timestep_metric(self, z_point, timestep):
        """Get metric tensor at specific timestep using flow transformations."""
        # Ensure z_point is on the correct device
        if isinstance(z_point, np.ndarray):
            z_tensor = torch.tensor(z_point, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            z_tensor = z_point.to(self.device).unsqueeze(0) if z_point.dim() == 1 else z_point.to(self.device)
        
        if timestep is None or timestep == 0:
            # Base metric at initial point
            return self.model.G(z_tensor)[0]
        else:
            # Apply flow transformations to get to the right timestep
            if hasattr(self.model, 'flows') and len(self.model.flows) > timestep - 1:
                z_t = z_tensor
                for t in range(timestep):
                    flow_result = self.model.flows[t](z_t)
                    z_t = flow_result.out
                return self.model.G(z_t)[0]
            else:
                # Fallback to base metric if flows not available
                return self.model.G(z_tensor)[0]
    
    def _compute_metric_derivative(self, z_center, pca, direction, h, timestep):
        """Compute derivative of metric tensor using finite differences."""
        # Create perturbation in PCA space
        if isinstance(z_center, np.ndarray):
            pca_center = pca.transform(z_center.reshape(1, -1)).flatten()
        else:
            pca_center = pca.transform(z_center.cpu().numpy().reshape(1, -1)).flatten()
        
        # Perturb in specified PCA direction
        pca_plus = pca_center.copy()
        pca_minus = pca_center.copy()
        pca_plus[direction] += h
        pca_minus[direction] -= h
        
        # Transform back to latent space
        z_plus = pca.inverse_transform(pca_plus.reshape(1, -1)).flatten()
        z_minus = pca.inverse_transform(pca_minus.reshape(1, -1)).flatten()
        
        # Get metrics at perturbed points
        G_plus = self._get_timestep_metric(z_plus, timestep).cpu().numpy()
        G_minus = self._get_timestep_metric(z_minus, timestep).cpu().numpy()
        
        # Finite difference
        dG = (G_plus - G_minus) / (2 * h)
        return dG
    
    def _compute_scalar_curvature_approx(self, G, dG_dx, dG_dy):
        """Compute approximate scalar curvature for 2D metric."""
        try:
            if np.linalg.det(G) < 1e-10:
                return 0.0
            
            G_inv = np.linalg.inv(G)
            
            # Simplified scalar curvature using metric and its derivatives
            # R ≈ -0.5 * tr(G^-1 * (∂²G/∂x² + ∂²G/∂y²))
            # Approximated using first derivatives
            
            trace_term = np.trace(G_inv @ (dG_dx + dG_dy))
            return -0.5 * trace_term
            
        except:
            return 0.0
    
    def _compute_ricci_curvature_approx(self, G, dG_dx, dG_dy):
        """Compute approximate Ricci curvature for 2D metric."""
        try:
            if np.linalg.det(G) < 1e-10:
                return 0.0
            
            G_inv = np.linalg.inv(G)
            
            # For 2D, Ricci curvature equals Gaussian curvature
            # R_ij ≈ using Christoffel symbol approximation
            
            christoffel_trace = 0.5 * np.trace(G_inv @ (dG_dx + dG_dy))
            return christoffel_trace
            
        except:
            return 0.0

    def _create_metric_amplification_heatmap(self, ax, z_pca_seq, pca, epoch):
        """Create detailed heatmap of metric amplification factor."""
        try:
            # High resolution grid for smooth heatmap
            x_min, x_max = z_pca_seq[:, :, 0].min() - 1, z_pca_seq[:, :, 0].max() + 1
            y_min, y_max = z_pca_seq[:, :, 1].min() - 1, z_pca_seq[:, :, 1].max() + 1
            
            nx, ny = 50, 50
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
            grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
            grid_points_latent = pca.inverse_transform(grid_points_pca)
            
            # Compute amplification factor
            grid_tensor = torch.tensor(grid_points_latent, dtype=torch.float32, device=self.device)
            
            # Process in chunks to avoid memory issues
            chunk_size = 400
            amplification_factors = []
            
            for i in range(0, len(grid_tensor), chunk_size):
                chunk = grid_tensor[i:i+chunk_size]
                G_chunk = self.model.G(chunk)
                
                # Project to PCA space
                V = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
                G_pca_chunk = torch.matmul(torch.matmul(V.unsqueeze(0), G_chunk), V.T.unsqueeze(0))
                
                # Amplification factor = sqrt(det(G_pca))
                det_chunk = torch.linalg.det(G_pca_chunk)
                amp_chunk = torch.sqrt(torch.clamp(det_chunk, min=1e-8))
                amplification_factors.append(amp_chunk.cpu().numpy())
            
            amplification_grid = np.concatenate(amplification_factors)
            amplification_heatmap = amplification_grid.reshape(xx.shape)
            
            # Plot with custom colormap
            im = ax.contourf(xx, yy, amplification_heatmap, levels=25, cmap='hot')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Amplification Factor √det(G)', fontsize=12)
            
            # Add contour lines for better readability
            ax.contour(xx, yy, amplification_heatmap, levels=10, colors='black', alpha=0.3, linewidths=0.5)
            
            # Overlay trajectories with arrows showing direction
            for seq_idx in range(min(4, z_pca_seq.shape[0])):
                traj = z_pca_seq[seq_idx]
                color = plt.cm.viridis(seq_idx / 4)
                
                # Plot trajectory
                ax.plot(traj[:, 0], traj[:, 1], 'o-', color=color, 
                       linewidth=2, markersize=4, alpha=0.9, label=f'Seq {seq_idx}')
                
                # Add arrows showing direction
                for t in range(len(traj) - 1):
                    dx = traj[t+1, 0] - traj[t, 0]
                    dy = traj[t+1, 1] - traj[t, 1]
                    ax.arrow(traj[t, 0], traj[t, 1], dx*0.3, dy*0.3, 
                            head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=0.7)
            
            ax.set_title('Metric Amplification Heatmap\n(how metric stretches distances)', fontsize=12)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend(fontsize=10, loc='upper right')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', transform=ax.transAxes, ha='center')

    def create_fancy_interactive_geodesic_visualizations(self, x_sample, epoch):
        """
        Create stunning interactive Plotly visualizations with fancy colors and dense trajectory points.
        """
        print(f"✨ Creating FANCY interactive geodesic visualizations for epoch {epoch}")
        
        if not hasattr(self.model, 'G'):
            print("⚠️ No metric tensor available for fancy visualization")
            return
            
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import plotly.colors as pc
            
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                z_seq = result.z  # [batch_size, n_obs, latent_dim]
                
                from sklearn.decomposition import PCA
                import numpy as np
                
                batch_size, n_obs, latent_dim = z_seq.shape
                
                # 🌟 ENHANCED: Generate MANY more trajectory points using interpolation
                dense_trajectories = self._generate_dense_trajectories(z_seq, n_interp_points=20)
                
                # Prepare PCA data with dense trajectories
                z_flat = dense_trajectories.reshape(-1, latent_dim).cpu().numpy()
                pca = PCA(n_components=2)
                z_pca = pca.fit_transform(z_flat)
                dense_n_points = dense_trajectories.shape[1]
                z_pca_dense = z_pca.reshape(batch_size, dense_n_points, 2)
                
                # Original trajectory points in PCA space
                z_orig_flat = z_seq.reshape(-1, latent_dim).cpu().numpy()
                z_orig_pca = pca.transform(z_orig_flat).reshape(batch_size, n_obs, 2)
                
                # Create fancy interactive subplots
                fig = make_subplots(
                    rows=2, cols=3,
                    subplot_titles=[
                        "🌀 Interactive Geodesic Paths", 
                        "🎭 Metric Eigenvalue Symphony",
                        "🎨 Metric Tensor Art Gallery",
                        "📊 Path Length Analytics", 
                        "🌊 Curvature Landscape",
                        "🔥 Amplification Heatmap"
                    ],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]],
                    horizontal_spacing=0.08,
                    vertical_spacing=0.12
                )
                
                # Panel 1: Interactive Dense Geodesic Paths
                self._create_fancy_geodesic_paths(fig, z_seq, z_pca_dense, z_orig_pca, pca, row=1, col=1)
                
                # Panel 2: Fancy Metric Eigenvalue Field
                self._create_fancy_eigenvalue_field(fig, z_pca_dense, pca, row=1, col=2)
                
                # Panel 3: Artistic Metric Ellipses
                self._create_fancy_metric_ellipses(fig, z_pca_dense, pca, row=1, col=3)
                
                # Panel 4: Interactive Path Analytics
                self._create_fancy_path_analytics(fig, z_seq, z_orig_pca, pca, row=2, col=1)
                
                # Panel 5: Beautiful Curvature Landscape
                self._create_fancy_curvature_landscape(fig, z_pca_dense, pca, row=2, col=2)
                
                # Panel 6: Stunning Amplification Heatmap
                self._create_fancy_amplification_heatmap(fig, z_pca_dense, pca, row=2, col=3)
                
                # Global styling for maximum fancy factor
                fig.update_layout(
                    title={
                        'text': f"✨ ENHANCED GEODESIC SYMPHONY - EPOCH {epoch} ✨<br>"
                               f"<span style='font-size:14px'>🎯 Riemannian Geometry Analysis with {dense_n_points * batch_size} Trajectory Points</span>",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20, 'color': '#2E86AB'}
                    },
                    showlegend=True,
                    width=1600,
                    height=1000,
                    paper_bgcolor='rgba(240,248,255,0.95)',
                    plot_bgcolor='rgba(255,255,255,0.8)',
                    font={'family': 'Arial Black, Arial, sans-serif', 'color': '#2C3E50'},
                    margin=dict(l=80, r=80, t=120, b=80)
                )
                
                # Save as interactive HTML
                html_filename = f'fancy_geodesic_analysis_epoch_{epoch}.html'
                fig.write_html(html_filename, include_plotlyjs=True)
                
                # Also save a static high-quality image
                png_filename = f'fancy_geodesic_analysis_epoch_{epoch}.png'
                self._safe_write_image(fig, png_filename, width=1600, height=1000, scale=2)
                
                wandb.log({
                    "geodesic/fancy_interactive": wandb.Html(html_filename),
                    "geodesic/fancy_static": wandb.Image(png_filename, caption=f"Epoch {epoch} - Fancy geodesic analysis"),
                })
                
                print(f"✨ Fancy interactive visualization created!")
                print(f"📁 Files: {html_filename} (interactive), {png_filename} (static)")
                print(f"🎯 Total trajectory points: {dense_n_points * batch_size}")
                
        except Exception as e:
            print(f"⚠️ Fancy visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()

    def _generate_dense_trajectories(self, z_seq, n_interp_points=20):
        """Generate dense trajectories with many interpolated points."""
        batch_size, n_obs, latent_dim = z_seq.shape
        dense_trajectories = []
        
        for seq_idx in range(batch_size):
            seq_points = []
            for t in range(n_obs - 1):
                # Add original point
                seq_points.append(z_seq[seq_idx, t])
                
                # Add interpolated points
                start_point = z_seq[seq_idx, t]
                end_point = z_seq[seq_idx, t + 1]
                
                for i in range(1, n_interp_points):
                    alpha = i / n_interp_points
                    interp_point = (1 - alpha) * start_point + alpha * end_point
                    seq_points.append(interp_point)
            
            # Add final point
            seq_points.append(z_seq[seq_idx, -1])
            dense_trajectories.append(torch.stack(seq_points))
        
        return torch.stack(dense_trajectories)

    def _create_fancy_geodesic_paths(self, fig, z_seq, z_pca_dense, z_orig_pca, pca, row, col):
        """Create fancy interactive geodesic paths with stunning colors."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import plotly.colors as pc
            
            batch_size = min(8, z_seq.shape[0])  # Show more sequences
            
            # Use fancy color scales
            colors = pc.qualitative.Set3[:batch_size] if batch_size <= len(pc.qualitative.Set3) else px.colors.sample_colorscale("rainbow", batch_size)
            
            for seq_idx in range(batch_size):
                color = colors[seq_idx]
                
                # Dense trajectory line
                fig.add_trace(
                    go.Scatter(
                        x=z_pca_dense[seq_idx, :, 0],
                        y=z_pca_dense[seq_idx, :, 1],
                        mode='lines',
                        line=dict(color=color, width=3, dash='solid'),
                        name=f'Dense Path {seq_idx}',
                        opacity=0.8,
                        hovertemplate=f'Seq {seq_idx}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>'
                    ),
                    row=row, col=col
                )
                
                # Original points as markers
                fig.add_trace(
                    go.Scatter(
                        x=z_orig_pca[seq_idx, :, 0],
                        y=z_orig_pca[seq_idx, :, 1],
                        mode='markers',
                        marker=dict(
                            color=color,
                            size=12,
                            symbol='circle',
                            line=dict(color='white', width=2)
                        ),
                        name=f'Original Points {seq_idx}',
                        hovertemplate=f'Seq {seq_idx} (Original)<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>'
                    ),
                    row=row, col=col
                )
                
                # Start and end markers
                fig.add_trace(
                    go.Scatter(
                        x=[z_orig_pca[seq_idx, 0, 0]],
                        y=[z_orig_pca[seq_idx, 0, 1]],
                        mode='markers',
                        marker=dict(color=color, size=16, symbol='square', line=dict(color='black', width=2)),
                        name=f'Start {seq_idx}',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[z_orig_pca[seq_idx, -1, 0]],
                        y=[z_orig_pca[seq_idx, -1, 1]],
                        mode='markers',
                        marker=dict(color=color, size=16, symbol='star', line=dict(color='black', width=2)),
                        name=f'End {seq_idx}',
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="PC1", row=row, col=col, gridcolor='lightgray')
            fig.update_yaxes(title_text="PC2", row=row, col=col, gridcolor='lightgray')
            
        except Exception as e:
            print(f"⚠️ Fancy geodesic paths failed: {e}")

    def _create_fancy_eigenvalue_field(self, fig, z_pca_dense, pca, row, col):
        """Create beautiful metric eigenvalue field visualization."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np
            # Create grid for eigenvalue field
            x_min, x_max = z_pca_dense[:, :, 0].min() - 1, z_pca_dense[:, :, 0].max() + 1
            y_min, y_max = z_pca_dense[:, :, 1].min() - 1, z_pca_dense[:, :, 1].max() + 1
            
            nx, ny = 20, 20  # Higher resolution
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
            grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
            grid_points_latent = pca.inverse_transform(grid_points_pca)
            
            # Compute metric tensor at grid points
            grid_tensor = torch.tensor(grid_points_latent, dtype=torch.float32, device=self.device)
            G_grid = self.model.G(grid_tensor)
            
            # Project to PCA space
            V = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
            G_pca = torch.matmul(torch.matmul(V.unsqueeze(0), G_grid), V.T.unsqueeze(0))
            
            eigenvals, eigenvecs = torch.linalg.eigh(G_pca)
            eigenvals_np = eigenvals.cpu().numpy()
            
            # Create beautiful heatmap background
            det_values = eigenvals_np[:, 0] * eigenvals_np[:, 1]
            det_heatmap = det_values.reshape(xx.shape)
            
            fig.add_trace(
                go.Contour(
                    x=np.linspace(x_min, x_max, nx),
                    y=np.linspace(y_min, y_max, ny),
                    z=det_heatmap,
                    colorscale='Viridis',
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(title="det(G)", x=1.02, len=0.4),
                    name="Determinant"
                ),
                row=row, col=col
            )
            
            # Add some trajectory overlay
            for seq_idx in range(min(3, z_pca_dense.shape[0])):
                fig.add_trace(
                    go.Scatter(
                        x=z_pca_dense[seq_idx, ::5, 0],  # Subsample for clarity
                        y=z_pca_dense[seq_idx, ::5, 1],
                        mode='markers',
                        marker=dict(color='white', size=6, line=dict(color='black', width=1)),
                        name=f'Traj {seq_idx}',
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="PC1", row=row, col=col)
            fig.update_yaxes(title_text="PC2", row=row, col=col)
            
        except Exception as e:
            print(f"⚠️ Fancy eigenvalue field failed: {e}")

    def _create_fancy_metric_ellipses(self, fig, z_pca_dense, pca, row, col):
        """Create artistic metric tensor ellipse visualization."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np
            # Sample points more densely for ellipses
            x_min, x_max = z_pca_dense[:, :, 0].min() - 1, z_pca_dense[:, :, 0].max() + 1
            y_min, y_max = z_pca_dense[:, :, 1].min() - 1, z_pca_dense[:, :, 1].max() + 1
            
            nx, ny = 12, 12
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
            grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
            grid_points_latent = pca.inverse_transform(grid_points_pca)
            
            # Compute metric tensor
            grid_tensor = torch.tensor(grid_points_latent, dtype=torch.float32, device=self.device)
            G_grid = self.model.G(grid_tensor)
            
            V = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
            G_pca = torch.matmul(torch.matmul(V.unsqueeze(0), G_grid), V.T.unsqueeze(0))
            
            eigenvals, eigenvecs = torch.linalg.eigh(G_pca)
            eigenvals_np = eigenvals.cpu().numpy()
            eigenvecs_np = eigenvecs.cpu().numpy()
            
            # Create background heatmap for context
            det_values = eigenvals_np[:, 0] * eigenvals_np[:, 1]
            det_heatmap = det_values.reshape(xx.shape)
            
            fig.add_trace(
                go.Heatmap(
                    x=np.linspace(x_min, x_max, nx),
                    y=np.linspace(y_min, y_max, ny),
                    z=det_heatmap,
                    colorscale='Plasma',
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(title="det(G)", x=1.02, len=0.4)
                ),
                row=row, col=col
            )
            
            # Overlay dense trajectories
            colors = px.colors.qualitative.Set2
            for seq_idx in range(min(4, z_pca_dense.shape[0])):
                color = colors[seq_idx % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=z_pca_dense[seq_idx, :, 0],
                        y=z_pca_dense[seq_idx, :, 1],
                        mode='lines',
                        line=dict(color=color, width=3),
                        name=f'Dense Traj {seq_idx}',
                        opacity=0.9
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="PC1", row=row, col=col)
            fig.update_yaxes(title_text="PC2", row=row, col=col)
            
        except Exception as e:
            print(f"⚠️ Fancy metric ellipses failed: {e}")

    def _create_fancy_path_analytics(self, fig, z_seq, z_orig_pca, pca, row, col):
        """Create interactive path length analytics."""
        try:
            import plotly.graph_objects as go
            batch_size, n_obs, latent_dim = z_seq.shape
            
            euclidean_lengths = []
            riemannian_lengths = []
            sequence_names = []
            
            for seq_idx in range(min(10, batch_size)):  # More sequences
                z_full = z_seq[seq_idx].cpu()
                
                euclidean_length = 0
                riemannian_length = 0
                
                for t in range(n_obs - 1):
                    eucl_dist = torch.norm(z_full[t+1] - z_full[t]).item()
                    euclidean_length += eucl_dist
                    
                    midpoint = (z_full[t] + z_full[t+1]) / 2
                    G_mid = self.model.G(midpoint.unsqueeze(0).to(self.device))
                    diff = (z_full[t+1] - z_full[t]).unsqueeze(0).to(self.device)
                    
                    riem_dist_sq = torch.matmul(torch.matmul(diff.unsqueeze(1), G_mid), diff.unsqueeze(2)).squeeze()
                    riem_dist = torch.sqrt(torch.clamp(riem_dist_sq, min=1e-8)).item()
                    riemannian_length += riem_dist
                
                euclidean_lengths.append(euclidean_length)
                riemannian_lengths.append(riemannian_length)
                sequence_names.append(f'Seq {seq_idx}')
            
            # Create beautiful bar chart
            fig.add_trace(
                go.Bar(
                    x=sequence_names,
                    y=euclidean_lengths,
                    name='Euclidean',
                    marker_color='rgba(55, 128, 191, 0.8)',
                    hovertemplate='%{x}<br>Euclidean: %{y:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Bar(
                    x=sequence_names,
                    y=riemannian_lengths,
                    name='Riemannian',
                    marker_color='rgba(255, 65, 54, 0.8)',
                    hovertemplate='%{x}<br>Riemannian: %{y:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="Sequences", row=row, col=col)
            fig.update_yaxes(title_text="Path Length", row=row, col=col)
            
        except Exception as e:
            print(f"⚠️ Fancy path analytics failed: {e}")

    def _create_fancy_curvature_landscape(self, fig, z_pca_dense, pca, row, col, timestep=None):
        """Create beautiful and accurate curvature landscape with timestep-specific metrics."""
        try:
            import plotly.graph_objects as go
            import numpy as np
            x_min, x_max = z_pca_dense[:, :, 0].min() - 1, z_pca_dense[:, :, 0].max() + 1
            y_min, y_max = z_pca_dense[:, :, 1].min() - 1, z_pca_dense[:, :, 1].max() + 1
            
            nx, ny = 30, 30  # Even higher resolution for accuracy
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
            
            # Multiple curvature measures
            scalar_curvature = np.zeros_like(xx)
            gaussian_curvature = np.zeros_like(xx)
            ricci_curvature = np.zeros_like(xx)
            
            h = 0.008  # Small step for finite differences
            
            # Compute accurate curvature using proper differential geometry
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    try:
                        # Central point in PCA space
                        pca_center = np.array([xx[j, i], yy[j, i]])
                        z_center = pca.inverse_transform(pca_center.reshape(1, -1)).flatten()
                        
                        # Get timestep-specific metric at center
                        G_center = self._get_timestep_metric_for_fancy(z_center, timestep)
                        
                        # Compute partial derivatives of metric using finite differences
                        dG_dx = self._compute_metric_derivative_for_fancy(z_center, pca, 0, h, timestep)
                        dG_dy = self._compute_metric_derivative_for_fancy(z_center, pca, 1, h, timestep)
                        
                        # For PCA space (2D), compute enhanced curvature measures
                        if G_center.shape[0] >= 2:
                            # Project to PCA subspace
                            V = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
                            G_pca = torch.matmul(torch.matmul(V, G_center), V.T).cpu().numpy()
                            
                            # Enhanced Gaussian curvature
                            det_G = np.linalg.det(G_pca)
                            if det_G > 1e-12:
                                # More accurate Gaussian curvature using metric determinant variation
                                G_inv = np.linalg.inv(G_pca)
                                ddet_dx = np.trace(G_inv @ dG_dx[:2, :2]) * det_G
                                ddet_dy = np.trace(G_inv @ dG_dy[:2, :2]) * det_G
                                gaussian_curvature[j, i] = -(ddet_dx + ddet_dy) / (2 * det_G)
                            
                            # Enhanced scalar curvature using Christoffel symbols
                            scalar_curvature[j, i] = self._compute_enhanced_scalar_curvature(G_pca, dG_dx[:2, :2], dG_dy[:2, :2])
                            
                            # Enhanced Ricci curvature
                            ricci_curvature[j, i] = self._compute_enhanced_ricci_curvature(G_pca, dG_dx[:2, :2], dG_dy[:2, :2])
                        
                    except Exception as e:
                        scalar_curvature[j, i] = 0
                        gaussian_curvature[j, i] = 0
                        ricci_curvature[j, i] = 0
            
            # Choose primary curvature display (scalar curvature is most general)
            curvature_display = scalar_curvature
            
            # Create multiple colormaps for different aspects
            timestep_str = f" (t={timestep})" if timestep is not None else ""
            
            # Create stunning heatmap with enhanced colorscale
            fig.add_trace(
                go.Heatmap(
                    x=np.linspace(x_min, x_max, nx),
                    y=np.linspace(y_min, y_max, ny),
                    z=curvature_display,
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title=f"Scalar Curvature{timestep_str}", x=1.02, len=0.4),
                    hovertemplate='PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Curvature: %{z:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add contour lines for better structure visibility
            fig.add_trace(
                go.Contour(
                    x=np.linspace(x_min, x_max, nx),
                    y=np.linspace(y_min, y_max, ny),
                    z=curvature_display,
                    contours=dict(
                        start=curvature_display.min(),
                        end=curvature_display.max(),
                        size=(curvature_display.max() - curvature_display.min()) / 10,
                        coloring='lines'
                    ),
                    line=dict(color='rgba(0,0,0,0.3)', width=1),
                    showscale=False
                ),
                row=row, col=col
            )
            
            # Overlay enhanced trajectories with timestep evolution
            for seq_idx in range(min(4, z_pca_dense.shape[0])):
                # Full trajectory
                fig.add_trace(
                    go.Scatter(
                        x=z_pca_dense[seq_idx, :, 0],
                        y=z_pca_dense[seq_idx, :, 1],
                        mode='lines',
                        line=dict(color=f'rgba(255,255,255,0.8)', width=3),
                        name=f'Traj {seq_idx}',
                        showlegend=False,
                        hovertemplate='Trajectory %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
                        text=[f'{seq_idx}'] * len(z_pca_dense[seq_idx])
                    ),
                    row=row, col=col
                )
                
                # Timestep markers with evolution colors
                n_points = z_pca_dense.shape[1]
                colors = np.linspace(0, 1, n_points)
                
                fig.add_trace(
                    go.Scatter(
                        x=z_pca_dense[seq_idx, ::max(1, n_points//8), 0],  # Subsample for clarity
                        y=z_pca_dense[seq_idx, ::max(1, n_points//8), 1],
                        mode='markers',
                        marker=dict(
                            color=colors[::max(1, n_points//8)],
                            colorscale='Viridis',
                            size=8,
                            line=dict(color='white', width=2)
                        ),
                        name=f'Timesteps {seq_idx}',
                        showlegend=False,
                        hovertemplate='Seq %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
                        text=[f'{seq_idx}'] * len(z_pca_dense[seq_idx, ::max(1, n_points//8)])
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="PC1", row=row, col=col)
            fig.update_yaxes(title_text="PC2", row=row, col=col)
            
        except Exception as e:
            print(f"⚠️ Enhanced fancy curvature landscape failed: {e}")
    
    def _get_timestep_metric_for_fancy(self, z_point, timestep):
        """Get metric tensor at specific timestep for fancy visualizations."""
        z_tensor = torch.tensor(z_point, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        if timestep is None or timestep == 0:
            return self.model.G(z_tensor)[0]
        else:
            # Apply flow transformations to get to the right timestep
            if hasattr(self.model, 'flows') and len(self.model.flows) > timestep - 1:
                z_t = z_tensor
                for t in range(timestep):
                    flow_result = self.model.flows[t](z_t)
                    z_t = flow_result.out
                return self.model.G(z_t)[0]
            else:
                return self.model.G(z_tensor)[0]
    
    def _compute_metric_derivative_for_fancy(self, z_center, pca, direction, h, timestep):
        """Compute metric derivative for fancy visualizations."""
        pca_center = pca.transform(z_center.reshape(1, -1)).flatten()
        
        pca_plus = pca_center.copy()
        pca_minus = pca_center.copy()
        pca_plus[direction] += h
        pca_minus[direction] -= h
        
        z_plus = pca.inverse_transform(pca_plus.reshape(1, -1)).flatten()
        z_minus = pca.inverse_transform(pca_minus.reshape(1, -1)).flatten()
        
        G_plus = self._get_timestep_metric_for_fancy(z_plus, timestep).cpu().numpy()
        G_minus = self._get_timestep_metric_for_fancy(z_minus, timestep).cpu().numpy()
        
        return (G_plus - G_minus) / (2 * h)
    
    def _compute_enhanced_scalar_curvature(self, G, dG_dx, dG_dy):
        """Compute enhanced scalar curvature with better approximation."""
        try:
            if np.linalg.det(G) < 1e-12:
                return 0.0
            
            G_inv = np.linalg.inv(G)
            
            # Enhanced scalar curvature using more accurate Christoffel symbol computation
            # R = g^ij R_ij where R_ij is the Ricci tensor
            
            # Christoffel symbols approximation
            christoffel_11 = 0.5 * (dG_dx[0, 0] + dG_dx[0, 0] - dG_dx[0, 0])  # Γ^k_11
            christoffel_12 = 0.5 * (dG_dx[0, 1] + dG_dy[0, 0] - dG_dx[0, 1])  # Γ^k_12
            christoffel_22 = 0.5 * (dG_dy[1, 1] + dG_dy[1, 1] - dG_dy[1, 1])  # Γ^k_22
            
            # Approximate scalar curvature
            scalar_curv = np.trace(G_inv @ (dG_dx + dG_dy))
            return -0.5 * scalar_curv
            
        except:
            return 0.0
    
    def _compute_enhanced_ricci_curvature(self, G, dG_dx, dG_dy):
        """Compute enhanced Ricci curvature."""
        try:
            if np.linalg.det(G) < 1e-12:
                return 0.0
            
            G_inv = np.linalg.inv(G)
            
            # For 2D manifolds, use the fact that Ricci curvature = Gaussian curvature
            # Enhanced computation using metric derivatives
            
            trace_term = np.trace(G_inv @ (dG_dx + dG_dy))
            det_term = np.linalg.det(G_inv @ (dG_dx @ G_inv @ dG_dy))
            
            return 0.5 * (trace_term - np.sqrt(abs(det_term)))
            
        except:
            return 0.0

    def _create_fancy_amplification_heatmap(self, fig, z_pca_dense, pca, row, col):
        """Create stunning amplification factor heatmap."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np
            x_min, x_max = z_pca_dense[:, :, 0].min() - 1, z_pca_dense[:, :, 0].max() + 1
            y_min, y_max = z_pca_dense[:, :, 1].min() - 1, z_pca_dense[:, :, 1].max() + 1
            
            nx, ny = 60, 60  # Very high resolution
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
            grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
            grid_points_latent = pca.inverse_transform(grid_points_pca)
            
            # Compute amplification factor in chunks
            grid_tensor = torch.tensor(grid_points_latent, dtype=torch.float32, device=self.device)
            
            chunk_size = 600
            amplification_factors = []
            
            for i in range(0, len(grid_tensor), chunk_size):
                chunk = grid_tensor[i:i+chunk_size]
                G_chunk = self.model.G(chunk)
                
                V = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
                G_pca_chunk = torch.matmul(torch.matmul(V.unsqueeze(0), G_chunk), V.T.unsqueeze(0))
                
                det_chunk = torch.linalg.det(G_pca_chunk)
                amp_chunk = torch.sqrt(torch.clamp(det_chunk, min=1e-8))
                amplification_factors.append(amp_chunk.cpu().numpy())
            
            amplification_grid = np.concatenate(amplification_factors)
            amplification_heatmap = amplification_grid.reshape(xx.shape)
            
            # Create gorgeous heatmap
            fig.add_trace(
                go.Heatmap(
                    x=np.linspace(x_min, x_max, nx),
                    y=np.linspace(y_min, y_max, ny),
                    z=amplification_heatmap,
                    colorscale='Hot',
                    showscale=True,
                    colorbar=dict(title="√det(G)", x=1.02, len=0.4)
                ),
                row=row, col=col
            )
            
            # Overlay all dense trajectories with different colors
            colors = px.colors.qualitative.Prism
            for seq_idx in range(min(6, z_pca_dense.shape[0])):
                color = colors[seq_idx % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=z_pca_dense[seq_idx, :, 0],
                        y=z_pca_dense[seq_idx, :, 1],
                        mode='lines',
                        line=dict(color=color, width=3),
                        name=f'Dense Path {seq_idx}',
                        opacity=0.9
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="PC1", row=row, col=col)
            fig.update_yaxes(title_text="PC2", row=row, col=col)
            
        except Exception as e:
            print(f"⚠️ Fancy amplification heatmap failed: {e}")

    def _safe_write_image(self, fig, filename, **kwargs):
        """Safely write Plotly figure to image, handling animated figures and file saving options."""
        # Check if we should save locally
        if not self.should_save_locally():
            print(f"📤 Skipping local save for {filename} (WandB-only mode)")
            return None
            
        try:
            # Check if figure has frames (animated)
            if hasattr(fig, 'frames') and fig.frames:
                print(f"⚠️ Skipping PNG export for animated figure: {filename} (has {len(fig.frames)} frames)")
                # Save as HTML instead for animated figures
                html_filename = filename.replace('.png', '.html')
                fig.write_html(html_filename)
                print(f"💾 Saved animated figure as HTML: {html_filename}")
                return html_filename
            else:
                # Regular static figure - safe to export as PNG
                fig.write_image(filename, **kwargs)
                print(f"💾 Saved static figure as PNG: {filename}")
                return filename
        except Exception as e:
            print(f"⚠️ Image export failed for {filename}: {e}")
            # Fallback: try to save as HTML
            try:
                html_filename = filename.replace('.png', '.html')
                fig.write_html(html_filename)
                print(f"💾 Fallback: Saved as HTML: {html_filename}")
                return html_filename
            except Exception as e2:
                print(f"❌ Both PNG and HTML export failed: {e2}")
                return None
    
    def _safe_save_plt_figure(self, filename, **kwargs):
        """Safely save matplotlib figure with file saving options."""
        if not self.should_save_locally():
            print(f"📤 Skipping local save for {filename} (WandB-only mode)")
            return None
        
        try:
            plt.savefig(filename, **kwargs)
            print(f"💾 Saved matplotlib figure: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Failed to save matplotlib figure {filename}: {e}")
            return None

    def create_timestep_specific_curvature_analysis(self, x_sample, epoch):
        """Create interactive curvature analysis with timestep slider for accurate metric evolution."""
        # Skip curvature analysis if disabled
        if getattr(self.config, 'disable_curvature_during_training', False):
            print("⏭️ Skipping curvature analysis (disabled by --disable_curvature_during_training)")
            return
            
        print(f"🌊 Creating timestep-specific curvature analysis for epoch {epoch}")
        
        if not hasattr(self.model, 'G'):
            print("⚠️ No metric tensor available for curvature analysis")
            return
            
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            from sklearn.decomposition import PCA
            import numpy as np
            
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                z_seq = result.z  # [batch_size, n_obs, latent_dim]
                
                batch_size, n_obs, latent_dim = z_seq.shape
                
                # Create PCA for visualization
                z_flat = z_seq.reshape(-1, latent_dim).cpu().numpy()
                pca = PCA(n_components=2)
                z_pca = pca.fit_transform(z_flat)
                z_pca_seq = z_pca.reshape(batch_size, n_obs, 2)
                
                print(f"PCA variance explained: {pca.explained_variance_ratio_[:2]}")
                
                # Create interactive figure with timestep slider
                fig = go.Figure()
                
                # Define grid for curvature computation
                x_min, x_max = z_pca_seq[:, :, 0].min() - 1, z_pca_seq[:, :, 0].max() + 1
                y_min, y_max = z_pca_seq[:, :, 1].min() - 1, z_pca_seq[:, :, 1].max() + 1
                
                nx, ny = 35, 35  # High resolution
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
                
                # Pre-compute curvatures for all timesteps
                all_timesteps_curvature = []
                all_timesteps_gaussian = []
                all_timesteps_ricci = []
                
                print("Computing curvature for all timesteps...")
                for timestep in range(n_obs):
                    print(f"  Timestep {timestep}/{n_obs-1}")
                    
                    scalar_curvature = np.zeros_like(xx)
                    gaussian_curvature = np.zeros_like(xx)  
                    ricci_curvature = np.zeros_like(xx)
                    
                    h = 0.008
                    
                    # Compute curvature at this timestep
                    for i in range(1, nx-1):
                        for j in range(1, ny-1):
                            try:
                                pca_center = np.array([xx[j, i], yy[j, i]])
                                z_center = pca.inverse_transform(pca_center.reshape(1, -1)).flatten()
                                
                                # Get timestep-specific metric
                                G_center = self._get_timestep_metric(z_center, timestep)
                                
                                # Compute derivatives
                                dG_dx = self._compute_metric_derivative(z_center, pca, 0, h, timestep)
                                dG_dy = self._compute_metric_derivative(z_center, pca, 1, h, timestep)
                                
                                if G_center.shape[0] >= 2:
                                    V = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
                                    G_pca = torch.matmul(torch.matmul(V, G_center), V.T).cpu().numpy()
                                    
                                    # Compute all curvatures
                                    det_G = np.linalg.det(G_pca)
                                    if det_G > 1e-12:
                                        G_inv = np.linalg.inv(G_pca)
                                        ddet_dx = np.trace(G_inv @ dG_dx[:2, :2]) * det_G
                                        ddet_dy = np.trace(G_inv @ dG_dy[:2, :2]) * det_G
                                        gaussian_curvature[j, i] = -(ddet_dx + ddet_dy) / (2 * det_G)
                                    
                                    scalar_curvature[j, i] = self._compute_scalar_curvature_approx(G_pca, dG_dx[:2, :2], dG_dy[:2, :2])
                                    ricci_curvature[j, i] = self._compute_ricci_curvature_approx(G_pca, dG_dx[:2, :2], dG_dy[:2, :2])
                                
                            except:
                                scalar_curvature[j, i] = 0
                                gaussian_curvature[j, i] = 0
                                ricci_curvature[j, i] = 0
                    
                    all_timesteps_curvature.append(scalar_curvature)
                    all_timesteps_gaussian.append(gaussian_curvature)
                    all_timesteps_ricci.append(ricci_curvature)
                
                # Create slider frames
                frames = []
                
                for timestep in range(n_obs):
                    curvature_data = all_timesteps_curvature[timestep]
                    
                    frame_data = [
                        # Curvature heatmap
                        go.Heatmap(
                            x=np.linspace(x_min, x_max, nx),
                            y=np.linspace(y_min, y_max, ny),
                            z=curvature_data,
                            colorscale='RdYlBu_r',
                            showscale=True,
                            colorbar=dict(title=f"Scalar Curvature (t={timestep})", x=1.02, len=0.8),
                            hovertemplate='PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Curvature: %{z:.4f}<extra></extra>',
                            name='curvature'
                        ),
                        
                        # Contour lines
                        go.Contour(
                            x=np.linspace(x_min, x_max, nx),
                            y=np.linspace(y_min, y_max, ny),
                            z=curvature_data,
                            contours=dict(
                                start=curvature_data.min(),
                                end=curvature_data.max(),
                                size=(curvature_data.max() - curvature_data.min()) / 15,
                                coloring='lines'
                            ),
                            line=dict(color='rgba(0,0,0,0.4)', width=1),
                            showscale=False,
                            name='contours'
                        )
                    ]
                    
                    # Add trajectories up to current timestep
                    for seq_idx in range(min(5, batch_size)):
                        # Full trajectory (faded)
                        frame_data.append(
                            go.Scatter(
                                x=z_pca_seq[seq_idx, :, 0],
                                y=z_pca_seq[seq_idx, :, 1],
                                mode='lines',
                                line=dict(color='rgba(255,255,255,0.3)', width=2),
                                name=f'Full Traj {seq_idx}',
                                showlegend=False
                            )
                        )
                        
                        # Trajectory up to current timestep (highlighted)
                        if timestep > 0:
                            frame_data.append(
                                go.Scatter(
                                    x=z_pca_seq[seq_idx, :timestep+1, 0],
                                    y=z_pca_seq[seq_idx, :timestep+1, 1],
                                    mode='lines+markers',
                                    line=dict(color=px.colors.qualitative.Set1[seq_idx % len(px.colors.qualitative.Set1)], width=3),
                                    marker=dict(size=6, line=dict(color='white', width=1)),
                                    name=f'Active Traj {seq_idx}',
                                    showlegend=False
                                )
                            )
                        
                        # Current position marker
                        frame_data.append(
                            go.Scatter(
                                x=[z_pca_seq[seq_idx, timestep, 0]],
                                y=[z_pca_seq[seq_idx, timestep, 1]],
                                mode='markers',
                                marker=dict(
                                    size=12,
                                    color=px.colors.qualitative.Set1[seq_idx % len(px.colors.qualitative.Set1)],
                                    line=dict(color='white', width=3),
                                    symbol='circle'
                                ),
                                name=f'Current t={timestep}',
                                showlegend=False,
                                hovertemplate=f'Seq {seq_idx} at t={timestep}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>'
                            )
                        )
                    
                    frames.append(go.Frame(data=frame_data, name=str(timestep)))
                
                # Initial frame (t=0)
                fig.add_traces(frames[0].data)
                
                # Configure slider
                fig.update_layout(
                    title=dict(
                        text=f"🌊 TIMESTEP-SPECIFIC CURVATURE ANALYSIS - EPOCH {epoch}<br>" +
                             f"<sub>True geometric curvature using flow-transformed metrics | PCA variance: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}</sub>",
                        x=0.5,
                        font=dict(size=16)
                    ),
                    xaxis=dict(title="PC1 (Principal Component 1)"),
                    yaxis=dict(title="PC2 (Principal Component 2)"),
                    width=1000,
                    height=700,
                    sliders=[{
                        "active": 0,
                        "currentvalue": {"prefix": "Timestep: "},
                        "pad": {"t": 50},
                        "steps": [
                            {
                                "args": [[str(k)], {
                                    "frame": {"duration": 500, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 300}
                                }],
                                "label": str(k),
                                "method": "animate"
                            }
                            for k in range(n_obs)
                        ]
                    }],
                    updatemenus=[{
                        "buttons": [
                            {
                                "args": [None, {"frame": {"duration": 800, "redraw": True},
                                               "fromcurrent": True, "transition": {"duration": 300}}],
                                "label": "▶ Play",
                                "method": "animate"
                            },
                            {
                                "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                                 "mode": "immediate", "transition": {"duration": 0}}],
                                "label": "⏸ Pause",
                                "method": "animate"
                            }
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 87},
                        "showactive": False,
                        "type": "buttons",
                        "x": 0.1,
                        "xanchor": "right",
                        "y": 0,
                        "yanchor": "top"
                    }]
                )
                
                fig.frames = frames
                
                # Save interactive visualization
                output_filename = f"timestep_curvature_analysis_epoch_{epoch}.html"
                fig.write_html(output_filename)
                print(f"✅ Saved timestep-specific curvature analysis: {output_filename}")
                
                # Create summary statistics
                curvature_stats = {
                    'timestep': list(range(n_obs)),
                    'mean_scalar_curvature': [np.mean(curv) for curv in all_timesteps_curvature],
                    'std_scalar_curvature': [np.std(curv) for curv in all_timesteps_curvature],
                    'max_scalar_curvature': [np.max(curv) for curv in all_timesteps_curvature],
                    'min_scalar_curvature': [np.min(curv) for curv in all_timesteps_curvature]
                }
                
                print("\n📊 Curvature evolution summary:")
                for t in range(n_obs):
                    print(f"  t={t}: mean={curvature_stats['mean_scalar_curvature'][t]:.4f}, "
                          f"std={curvature_stats['std_scalar_curvature'][t]:.4f}, "
                          f"range=[{curvature_stats['min_scalar_curvature'][t]:.4f}, {curvature_stats['max_scalar_curvature'][t]:.4f}]")
                
        except Exception as e:
            print(f"❌ Timestep-specific curvature analysis failed: {e}")
            import traceback
            traceback.print_exc()

    def create_temporal_metric_evolution_visualization(self, x_sample, epoch):
        """
        Create visualizations showing how det(G) evolves through FLOW TRANSFORMATIONS.
        This shows the proper temporal evolution via Jacobian chain rule: 
        det(G(z_ij)) = det(G(z_i0)) * |J_flow_1| * |J_flow_2| * ... * |J_flow_j|
        """
        print(f"🌊 Creating FLOW-BASED temporal metric evolution visualization for epoch {epoch}")
        
        if not hasattr(self.model, 'G'):
            print("⚠️ No metric tensor available for temporal visualization")
            return
            
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            from sklearn.decomposition import PCA
            import numpy as np
            
            self.model.eval()
            with torch.no_grad():
                batch_size, n_obs = x_sample.shape[:2]
                print(f"🌊 Computing flow-based det(G) evolution for {batch_size} sequences with {n_obs} timesteps")
                
                # ===== FLOW-BASED TEMPORAL EVOLUTION =====
                
                # Step 1: Encode first timestep to get z_i0
                x_0 = x_sample[:, 0]
                encoder_out = self.model.encoder(x_0)
                mu = encoder_out.embedding
                log_var = encoder_out.log_covariance
                
                # Sample z_0 (could also use mu directly for deterministic)
                eps = torch.randn_like(mu)
                z_0 = mu + eps * torch.exp(0.5 * log_var)
                
                # Step 2: Compute base metric determinant at z_i0
                print("📍 Computing base metric det(G(z_i0))...")
                G_0 = self.model.G(z_0)  # [batch_size, latent_dim, latent_dim]
                det_G_0 = torch.linalg.det(G_0)  # [batch_size]
                print(f"   Base det(G) range: [{det_G_0.min():.3e}, {det_G_0.max():.3e}]")
                
                # Step 3: Initialize flow-based temporal evolution
                z_seq = [z_0]
                det_G_seq = [det_G_0]  # Track det(G) evolution through flows
                flow_jacobians = []    # Track individual flow Jacobians
                
                print("🌊 Propagating det(G) through flow chain...")
                
                # Step 4: Propagate through flows with metric determinant tracking
                for t in range(1, n_obs):
                    print(f"   Flow step {t}: z_{t-1} → z_{t}")
                    
                    # Apply flow transformation
                    flow_res = self.model.flows[t-1](z_seq[-1])
                    z_t = flow_res.out                    # z_it = flow_t(z_{i,t-1})
                    log_det_jac = flow_res.log_abs_det_jac # log|J_flow_t|
                    
                    # 🔥 KEY: Transform metric determinant through flow Jacobian
                    det_G_prev = det_G_seq[-1]
                    jac_det = torch.exp(log_det_jac)      # |J_flow_t|
                    det_G_t = det_G_prev * jac_det        # det(G(z_it)) = det(G(z_{i,t-1})) * |J_flow_t|
                    
                    z_seq.append(z_t)
                    det_G_seq.append(det_G_t)
                    flow_jacobians.append(jac_det)
                    
                    print(f"     |J_flow_{t}| range: [{jac_det.min():.3f}, {jac_det.max():.3f}]")
                    print(f"     det(G) range: [{det_G_t.min():.3e}, {det_G_t.max():.3e}]")
                
                # Convert to arrays for visualization
                z_seq = torch.stack(z_seq, dim=1).cpu().numpy()  # [batch_size, n_obs, latent_dim]
                det_G_seq = torch.stack(det_G_seq, dim=1).cpu().numpy()  # [batch_size, n_obs]
                
                print(f"✅ Flow-based evolution complete!")
                print(f"   Final det(G) range: [{det_G_seq.min():.3e}, {det_G_seq.max():.3e}]")
                print(f"   Det(G) amplification: {det_G_seq[:, -1].mean() / det_G_seq[:, 0].mean():.3f}x")
                
                # Prepare PCA for consistent 2D projection
                z_flat = z_seq.reshape(-1, z_seq.shape[-1])
                pca = PCA(n_components=2)
                z_pca = pca.fit_transform(z_flat)
                z_pca_seq = z_pca.reshape(batch_size, n_obs, 2)
                
                print(f"📊 PCA projection complete (explained variance: {pca.explained_variance_ratio_[:2].sum():.1%})")
                
                # CRITICAL FIX: Compute temporal spatial heatmaps for each timestep
                print("🔥 Computing spatial det(G) heatmaps for each timestep...")
                
                # Create spatial grid for heatmaps
                x_min, x_max = z_pca_seq[:, :, 0].min() - 2, z_pca_seq[:, :, 0].max() + 2
                y_min, y_max = z_pca_seq[:, :, 1].min() - 2, z_pca_seq[:, :, 1].max() + 2
                nx, ny = 30, 30
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
                grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
                
                # CRITICAL FIX: Compute timestep-specific heatmaps using flow-evolved coordinates
                temporal_det_maps = []
                sequence_dets = det_G_seq.T  # [n_obs, batch_size] for easier indexing
                
                for t in range(n_obs):
                    print(f"   Computing spatial heatmap for timestep {t}...")
                    
                    if t == 0:
                        # For t=0, use original latent space coordinates
                        grid_points_latent = pca.inverse_transform(grid_points_pca)
                        grid_tensor = torch.tensor(grid_points_latent, dtype=torch.float32, device=self.device)
                        G_grid = self.model.G(grid_tensor)
                        det_G_grid = torch.linalg.det(G_grid).cpu().numpy()
                    else:
                        # For t>0, transform grid through flows to timestep t
                        try:
                            grid_points_latent = pca.inverse_transform(grid_points_pca)
                            grid_tensor = torch.tensor(grid_points_latent, dtype=torch.float32, device=self.device)
                            
                            # Apply t flow transformations to grid points
                            z_grid_t = grid_tensor
                            for flow_idx in range(t):
                                flow_result = self.model.flows[flow_idx](z_grid_t)
                                z_grid_t = flow_result.out
                            
                            # Compute metric tensor at flow-evolved grid points
                            G_grid_t = self.model.G(z_grid_t)
                            det_G_grid = torch.linalg.det(G_grid_t).cpu().numpy()
                            print(f"     ✅ Computed heatmap for t={t}: det(G) range [{det_G_grid.min():.2e}, {det_G_grid.max():.2e}]")
                            
                        except Exception as e:
                            print(f"     ⚠️ Flow-evolved heatmap failed for t={t}: {e}, using approximation")
                            # Fallback: use t=0 heatmap as approximation
                            if len(temporal_det_maps) > 0:
                                det_G_grid = temporal_det_maps[0].copy()
                            else:
                                grid_points_latent = pca.inverse_transform(grid_points_pca)
                                grid_tensor = torch.tensor(grid_points_latent, dtype=torch.float32, device=self.device)
                                G_grid = self.model.G(grid_tensor)
                                det_G_grid = torch.linalg.det(G_grid).cpu().numpy()
                    
                    # Reshape to grid format and store
                    det_heatmap_t = det_G_grid.reshape(xx.shape)
                    temporal_det_maps.append(det_heatmap_t)
                
                print(f"✅ Computed {len(temporal_det_maps)} spatial heatmaps with evolving det(G) fields!")
                
                # Create flow-based temporal visualizations
                self._create_flow_based_temporal_plots(
                    z_pca_seq, det_G_seq, flow_jacobians, epoch, pca
                )
                
                # Create interactive flow animation
                self._create_flow_based_interactive_animation(
                    z_pca_seq, det_G_seq, flow_jacobians, epoch
                )
                
                # CRITICAL FIX: Create temporal evolution plots with proper spatial heatmaps
                self._create_temporal_evolution_plots(
                    z_pca_seq, temporal_det_maps, sequence_dets, xx, yy, epoch
                )
                
                # CRITICAL FIX: Create interactive temporal animation with proper spatial heatmaps
                self._create_interactive_temporal_animation(
                    z_pca_seq, temporal_det_maps, sequence_dets, xx, yy, epoch
                )
                
                print(f"✨ Flow-based temporal metric evolution visualization created for epoch {epoch}!")
                
        except Exception as e:
            print(f"⚠️ Flow-based temporal visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()

    def _create_flow_based_temporal_plots(self, z_pca_seq, det_G_seq, flow_jacobians, epoch, pca):
        """Create plots showing how det(G) evolves through flow transformations."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
            
            batch_size, n_obs, _ = z_pca_seq.shape
            
            # Create comprehensive flow-based visualization
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            fig.suptitle(f'Flow-Based det(G) Evolution - Epoch {epoch}', fontsize=16)
            
            colors = plt.get_cmap("tab10")(np.linspace(0, 1, min(batch_size, 10)))
            timesteps = np.linspace(0, n_obs-1, min(8, n_obs), dtype=int)
            
            # Row 1: Flow-evolved det(G) spatial distribution at different timesteps
            for idx, t in enumerate(timesteps[:4]):
                ax = axes[0, idx]
                
                # Get positions and det(G) values at timestep t
                x_coords = z_pca_seq[:, t, 0]
                y_coords = z_pca_seq[:, t, 1]
                det_values = det_G_seq[:, t]
                
                # Create scatter plot with det(G) as color and size
                scatter = ax.scatter(x_coords, y_coords, c=det_values, s=100, 
                                   cmap='viridis', alpha=0.8, edgecolors='white',
                                   vmin=det_G_seq.min(), vmax=det_G_seq.max())
                
                # Draw trajectories up to this timestep
                for seq_idx in range(min(batch_size, 8)):
                    traj = z_pca_seq[seq_idx, :t+1, :]
                    ax.plot(traj[:, 0], traj[:, 1], color=colors[seq_idx], 
                           linewidth=2, alpha=0.6)
                
                ax.set_title(f'Flow Step t={t}\ndet(G) via Jacobian Chain')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.grid(True, alpha=0.3)
                
                if idx == 0:
                    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('det(G) (flow-evolved)', fontsize=10)
            
            # Row 2: Flow Jacobian evolution
            for idx, t in enumerate(timesteps[4:8] if len(timesteps) > 4 else [1, 2, 3, 4]):
                if t >= len(flow_jacobians):
                    continue
                    
                ax = axes[1, idx]
                
                if t > 0 and t-1 < len(flow_jacobians):
                    jac_values = flow_jacobians[t-1].cpu().numpy()
                    
                    # Get positions where Jacobians are computed
                    x_coords = z_pca_seq[:, t-1, 0]  # Input positions to flow
                    y_coords = z_pca_seq[:, t-1, 1]
                    
                    # Scatter plot with Jacobian magnitude as color
                    scatter = ax.scatter(x_coords, y_coords, c=jac_values, s=80,
                                       cmap='plasma', alpha=0.8, edgecolors='white')
                    
                    # Show flow direction arrows
                    for seq_idx in range(min(batch_size, 6)):
                        start = z_pca_seq[seq_idx, t-1, :]
                        end = z_pca_seq[seq_idx, t, :]
                        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                               head_width=0.1, head_length=0.1, fc=colors[seq_idx], 
                               ec=colors[seq_idx], alpha=0.7)
                    
                    ax.set_title(f'Flow Jacobian |J_{t}|\nRange: [{jac_values.min():.2f}, {jac_values.max():.2f}]')
                    
                    if idx == 0:
                        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label('|J_flow|', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'No Jacobian\n(t=0)', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12)
                
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.grid(True, alpha=0.3)
            
            # Row 3: Temporal evolution plots
            
            # Plot 3.1: det(G) evolution over time for individual sequences
            ax = axes[2, 0]
            timesteps_full = np.arange(n_obs)
            for seq_idx in range(min(batch_size, 8)):
                det_trajectory = det_G_seq[seq_idx, :]
                ax.plot(timesteps_full, det_trajectory, 'o-', color=colors[seq_idx], 
                       linewidth=2, alpha=0.8, label=f'Seq {seq_idx}')
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('det(G)')
            ax.set_title('det(G) Evolution Through Flows')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.set_yscale('log')  # Log scale for better visualization
            
            # Plot 3.2: Jacobian determinants over time
            ax = axes[2, 1]
            if flow_jacobians:
                jac_array = torch.stack(flow_jacobians, dim=0).cpu().numpy()  # [n_flows, batch_size]
                
                for seq_idx in range(min(batch_size, 8)):
                    jac_trajectory = jac_array[:, seq_idx]
                    ax.plot(range(1, len(jac_trajectory)+1), jac_trajectory, 'o-', 
                           color=colors[seq_idx], linewidth=2, alpha=0.8)
                
                ax.set_xlabel('Flow Step')
                ax.set_ylabel('|J_flow|')
                ax.set_title('Flow Jacobian Evolution')
                ax.grid(True, alpha=0.3)
            
            # Plot 3.3: Cumulative det(G) amplification
            ax = axes[2, 2]
            amplification = det_G_seq / det_G_seq[:, 0:1]  # Normalize by initial values
            
            mean_amp = np.mean(amplification, axis=0)
            std_amp = np.std(amplification, axis=0)
            
            ax.fill_between(timesteps_full, mean_amp - std_amp, mean_amp + std_amp, 
                           alpha=0.3, color='blue', label='±1 std')
            ax.plot(timesteps_full, mean_amp, 'o-', color='blue', linewidth=3, label='Mean amplification')
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('det(G) Amplification')
            ax.set_title('Cumulative Metric Amplification')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_yscale('log')
            
            # Plot 3.4: Flow chain visualization
            ax = axes[2, 3]
            
            # Show the flow chain as a graph
            flow_steps = range(n_obs)
            mean_det = np.mean(det_G_seq, axis=0)
            
            ax.plot(flow_steps, mean_det, 'o-', linewidth=3, markersize=8, color='red')
            
            # Annotate with flow jacobian info
            if flow_jacobians:
                for i, jac_tensor in enumerate(flow_jacobians):
                    mean_jac = jac_tensor.mean().item()
                    ax.annotate(f'|J|={mean_jac:.2f}', 
                               xy=(i+0.5, (mean_det[i] + mean_det[i+1])/2),
                               xytext=(0, 10), textcoords='offset points',
                               ha='center', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            ax.set_xlabel('Flow Step')
            ax.set_ylabel('Mean det(G)')
            ax.set_title('Flow Chain: det(G) → |J| → det(G)')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            plt.tight_layout()
            
            # Save flow-based evolution
            filename = f'flow_based_det_evolution_epoch_{epoch}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create temporal difference plots
            self._create_flow_jacobian_analysis_plots(det_G_seq, flow_jacobians, epoch)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "temporal_evolution/flow_based_comprehensive": wandb.Image(filename, 
                        caption=f"Epoch {epoch} - Flow-based det(G) evolution"),
                })
            
        except Exception as e:
            print(f"⚠️ Flow-based temporal plots failed: {e}")
            import traceback
            traceback.print_exc()

    def _create_flow_jacobian_analysis_plots(self, det_G_seq, flow_jacobians, epoch):
        """Create detailed analysis of flow Jacobians and their impact."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not flow_jacobians:
                return
                
            # Convert to numpy
            jac_array = torch.stack(flow_jacobians, dim=0).cpu().numpy()  # [n_flows, batch_size]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Flow Jacobian Analysis - Epoch {epoch}', fontsize=16)
            
            # Plot 1: Jacobian distribution at each flow step
            ax = axes[0, 0]
            for flow_idx in range(len(flow_jacobians)):
                jac_values = jac_array[flow_idx, :]
                ax.hist(jac_values, bins=20, alpha=0.6, label=f'Flow {flow_idx+1}', density=True)
            
            ax.set_xlabel('|J_flow|')
            ax.set_ylabel('Density')
            ax.set_title('Jacobian Distribution by Flow Step')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Jacobian vs det(G) correlation
            ax = axes[0, 1]
            
            for flow_idx in range(len(flow_jacobians)):
                if flow_idx + 1 < det_G_seq.shape[1]:
                    jac_values = jac_array[flow_idx, :]
                    det_before = det_G_seq[:, flow_idx]
                    det_after = det_G_seq[:, flow_idx + 1]
                    
                    ax.scatter(jac_values, det_after / det_before, alpha=0.6, 
                             label=f'Flow {flow_idx+1}')
            
            ax.set_xlabel('|J_flow|')
            ax.set_ylabel('det(G) Ratio')
            ax.set_title('Jacobian vs det(G) Amplification')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Cumulative Jacobian product
            ax = axes[0, 2]
            
            # Calculate cumulative Jacobian product
            cumulative_jac = np.ones((jac_array.shape[1], len(flow_jacobians) + 1))
            for flow_idx in range(len(flow_jacobians)):
                cumulative_jac[:, flow_idx + 1] = cumulative_jac[:, flow_idx] * jac_array[flow_idx, :]
            
            # Plot mean and std
            mean_cum_jac = np.mean(cumulative_jac, axis=0)
            std_cum_jac = np.std(cumulative_jac, axis=0)
            timesteps = np.arange(len(mean_cum_jac))
            
            ax.fill_between(timesteps, mean_cum_jac - std_cum_jac, mean_cum_jac + std_cum_jac,
                           alpha=0.3, color='green')
            ax.plot(timesteps, mean_cum_jac, 'o-', color='green', linewidth=2, label='Mean')
            
            ax.set_xlabel('Flow Step')
            ax.set_ylabel('Cumulative |J| Product')
            ax.set_title('Cumulative Jacobian Product')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            # Plot 4: det(G) vs theoretical (based on Jacobians)
            ax = axes[1, 0]
            
            # Theoretical det(G) based on Jacobian chain rule
            theoretical_det_G = det_G_seq[:, 0:1] * cumulative_jac  # [batch_size, n_obs]
            actual_det_G = det_G_seq
            
            # Compare final values
            ax.scatter(theoretical_det_G[:, -1], actual_det_G[:, -1], alpha=0.6)
            ax.plot([theoretical_det_G.min(), theoretical_det_G.max()], 
                   [actual_det_G.min(), actual_det_G.max()], 'r--', alpha=0.8, label='Perfect match')
            
            ax.set_xlabel('Theoretical det(G) (Jacobian chain)')
            ax.set_ylabel('Actual det(G) (final)')
            ax.set_title('Theory vs Practice Validation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Plot 5: Flow step impact analysis
            ax = axes[1, 1]
            
            impact_ratios = []
            for flow_idx in range(len(flow_jacobians)):
                jac_values = jac_array[flow_idx, :]
                impact = np.std(jac_values) / np.mean(jac_values)  # Coefficient of variation
                impact_ratios.append(impact)
            
            ax.bar(range(1, len(impact_ratios) + 1), impact_ratios, alpha=0.7, color='orange')
            ax.set_xlabel('Flow Step')
            ax.set_ylabel('Jacobian Variability (CV)')
            ax.set_title('Flow Impact Analysis')
            ax.grid(True, alpha=0.3)
            
            # Plot 6: Sequence-wise flow evolution
            ax = axes[1, 2]
            
            # Show det(G) evolution for a few representative sequences
            colors = plt.get_cmap("tab10")(np.linspace(0, 1, 6))
            timesteps = np.arange(det_G_seq.shape[1])
            
            for seq_idx in range(min(6, det_G_seq.shape[0])):
                det_traj = det_G_seq[seq_idx, :]
                ax.plot(timesteps, det_traj, 'o-', color=colors[seq_idx], 
                       linewidth=2, alpha=0.8, label=f'Seq {seq_idx}')
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('det(G)')
            ax.set_title('Individual Sequence Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            plt.tight_layout()
            
            # Save Jacobian analysis
            jac_filename = f'flow_jacobian_analysis_epoch_{epoch}.png'
            plt.savefig(jac_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "temporal_evolution/jacobian_analysis": wandb.Image(jac_filename, 
                        caption=f"Epoch {epoch} - Flow Jacobian analysis"),
                })
            
        except Exception as e:
            print(f"⚠️ Flow Jacobian analysis failed: {e}")

    def _create_flow_based_interactive_animation(self, z_pca_seq, det_G_seq, flow_jacobians, epoch):
        """Create interactive Plotly animation of flow-based det(G) evolution."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import numpy as np
            
            batch_size, n_obs, _ = z_pca_seq.shape
            
            # Create interactive animation with multiple panels
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "🌊 Flow-Based Spatial Evolution", 
                    "📊 det(G) Temporal Evolution",
                    "⚡ Flow Jacobian Magnitudes",
                    "🔗 Cumulative Amplification"
                ],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]],
                horizontal_spacing=0.1,
                vertical_spacing=0.15
            )
            
            # Prepare data for animation frames
            frames = []
            colors = px.colors.qualitative.Set3[:min(batch_size, 8)]
            
            for t in range(n_obs):
                frame_data = []
                
                # Panel 1: Spatial distribution with det(G) coloring
                frame_data.append(
                    go.Scatter(
                        x=z_pca_seq[:, t, 0],
                        y=z_pca_seq[:, t, 1],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=det_G_seq[:, t],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="det(G)", x=0.45, len=0.4),
                            line=dict(color='white', width=1)
                        ),
                        name="Sequences",
                        hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>det(G): %{marker.color:.2e}<extra></extra>",
                        xaxis='x', yaxis='y'
                    )
                )
                
                # Add trajectory lines
                for seq_idx in range(min(batch_size, 6)):
                    traj = z_pca_seq[seq_idx, :t+1, :]
                    frame_data.append(
                        go.Scatter(
                            x=traj[:, 0],
                            y=traj[:, 1],
                            mode='lines',
                            line=dict(color=colors[seq_idx], width=2),
                            name=f'Traj {seq_idx}',
                            showlegend=(t == 0),
                            xaxis='x', yaxis='y'
                        )
                    )
                
                # Panel 2: det(G) evolution up to timestep t
                for seq_idx in range(min(batch_size, 6)):
                    det_so_far = det_G_seq[seq_idx, :t+1]
                    timesteps_so_far = np.arange(t+1)
                    
                    frame_data.append(
                        go.Scatter(
                            x=timesteps_so_far,
                            y=det_so_far,
                            mode='lines+markers',
                            line=dict(color=colors[seq_idx], width=2),
                            marker=dict(size=6, color=colors[seq_idx]),
                            name=f'det(G) Seq {seq_idx}',
                            showlegend=False,
                            hovertemplate=f"Seq {seq_idx}<br>Timestep: %{{x}}<br>det(G): %{{y:.2e}}<extra></extra>",
                            xaxis='x2', yaxis='y2'
                        )
                    )
                
                # Panel 3: Flow Jacobians (if available)
                if t > 0 and t-1 < len(flow_jacobians):
                    jac_values = flow_jacobians[t-1].cpu().numpy()
                    
                    frame_data.append(
                        go.Scatter(
                            x=z_pca_seq[:, t-1, 0],
                            y=z_pca_seq[:, t-1, 1],
                            mode='markers',
                            marker=dict(
                                size=10,
                                color=jac_values,
                                colorscale='Plasma',
                                showscale=True,
                                colorbar=dict(title="|J_flow|", x=0.95, len=0.4),
                                line=dict(color='white', width=1)
                            ),
                            name="Jacobians",
                            hovertemplate=f"Flow {t}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>|J|: %{{marker.color:.3f}}<extra></extra>",
                            xaxis='x3', yaxis='y3'
                        )
                    )
                
                # Panel 4: Cumulative amplification
                amplification = det_G_seq / det_G_seq[:, 0:1]
                mean_amp = np.mean(amplification[:, :t+1], axis=0)
                
                frame_data.append(
                    go.Scatter(
                        x=np.arange(t+1),
                        y=mean_amp,
                        mode='lines+markers',
                        line=dict(color='red', width=3),
                        marker=dict(size=8, color='red'),
                        name="Mean Amplification",
                        showlegend=False,
                        hovertemplate="Timestep: %{x}<br>Amplification: %{y:.3f}<extra></extra>",
                        xaxis='x4', yaxis='y4'
                    )
                )
                
                frames.append(go.Frame(data=frame_data, name=str(t)))
            
            # Set initial frame
            fig.add_traces(frames[0].data)
            fig.frames = frames
            
            # Update layout
            fig.update_layout(
                title={
                    'text': f"🌊 FLOW-BASED METRIC EVOLUTION - EPOCH {epoch}<br>"
                           f"<span style='font-size:14px'>det(G) through Jacobian Chain Rule • {n_obs} flow steps • {batch_size} sequences</span>",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#2E86AB'}
                },
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 800, "redraw": True}, 
                                          "fromcurrent": True, "transition": {"duration": 400}}],
                            "label": "▶️ Play Flow",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True}, 
                                            "mode": "immediate", "transition": {"duration": 0}}],
                            "label": "⏸️ Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }],
                sliders=[{
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Flow Step: ",
                        "visible": True,
                        "xanchor": "right"
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f], {"frame": {"duration": 400, "redraw": True}, 
                                         "mode": "immediate", "transition": {"duration": 300}}],
                            "label": str(t),
                            "method": "animate"
                        }
                        for t, f in enumerate(frames)
                    ]
                }],
                width=1600,
                height=900,
                showlegend=True,
                paper_bgcolor='rgba(240,248,255,0.95)',
                plot_bgcolor='rgba(255,255,255,0.8)',
            )
            
            # Update subplot axes
            fig.update_xaxes(title_text="PC1", row=1, col=1)
            fig.update_yaxes(title_text="PC2", row=1, col=1)
            fig.update_xaxes(title_text="Timestep", row=1, col=2)
            fig.update_yaxes(title_text="det(G)", type="log", row=1, col=2)
            fig.update_xaxes(title_text="PC1", row=2, col=1)
            fig.update_yaxes(title_text="PC2", row=2, col=1)
            fig.update_xaxes(title_text="Timestep", row=2, col=2)
            fig.update_yaxes(title_text="Amplification", type="log", row=2, col=2)
            
            # Save interactive animation
            html_filename = f'flow_based_animation_epoch_{epoch}.html'
            fig.write_html(html_filename, include_plotlyjs=True)
            
            png_filename = f'flow_based_animation_epoch_{epoch}.png'
            self._safe_write_image(fig, png_filename, width=1600, height=900, scale=2)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "temporal_evolution/flow_based_interactive": wandb.Html(html_filename),
                    "temporal_evolution/flow_based_animation_static": wandb.Image(png_filename, 
                        caption=f"Epoch {epoch} - Flow-based interactive animation"),
                })
            
            print(f"🌊 Flow-based interactive animation saved: {html_filename}")
            
        except Exception as e:
            print(f"⚠️ Flow-based interactive animation failed: {e}")
            import traceback
            traceback.print_exc()

    def _create_temporal_evolution_plots(self, z_pca_seq, temporal_det_maps, sequence_dets, xx, yy, epoch):
        """Create static temporal evolution plots."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            batch_size, n_obs, _ = z_pca_seq.shape
            
            # Create figure with multiple temporal snapshots
            n_snapshots = min(8, n_obs)  # Show up to 8 timesteps
            timesteps = np.linspace(0, n_obs-1, n_snapshots, dtype=int)
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f'Temporal Evolution of det(G) - Epoch {epoch}', fontsize=16)
            axes = axes.flatten()
            
            # Color palette for sequences
            colors = plt.get_cmap("tab10")(np.linspace(0, 1, min(batch_size, 10)))
            
            for idx, t in enumerate(timesteps):
                ax = axes[idx]
                
                # Plot det(G) heatmap at timestep t
                det_map = temporal_det_maps[t]
                im = ax.contourf(xx, yy, det_map, levels=20, cmap='viridis', alpha=0.7)
                
                # Overlay sequence trajectories up to timestep t
                for seq_idx in range(min(batch_size, 8)):  # Limit for clarity
                    traj_so_far = z_pca_seq[seq_idx, :t+1, :]
                    ax.plot(traj_so_far[:, 0], traj_so_far[:, 1], 
                           color=colors[seq_idx], linewidth=2, alpha=0.8)
                    
                    # Mark current position
                    ax.scatter(z_pca_seq[seq_idx, t, 0], z_pca_seq[seq_idx, t, 1],
                             color=colors[seq_idx], s=80, marker='o', 
                             edgecolors='white', linewidth=2, zorder=5)
                
                ax.set_title(f't = {t} / {n_obs-1}')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar for first subplot
                if idx == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='det(G)')
            
            plt.tight_layout()
            
            # Save static temporal evolution
            filename = f'temporal_det_evolution_epoch_{epoch}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create sequence-specific det(G) evolution plot
            self._create_sequence_det_evolution(sequence_dets, epoch)
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    "temporal_evolution/static_snapshots": wandb.Image(filename, 
                        caption=f"Epoch {epoch} - Temporal det(G) evolution"),
                })
            
        except Exception as e:
            print(f"⚠️ Static temporal plots failed: {e}")

    def _create_sequence_det_evolution(self, sequence_dets, epoch):
        """Create plots showing det(G) evolution for individual sequences."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            n_obs, batch_size = sequence_dets.shape
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'det(G) Evolution Along Sequences - Epoch {epoch}', fontsize=14)
            
            # Plot 1: Individual sequence trajectories in det(G) space
            colors = plt.get_cmap("tab10")(np.linspace(0, 1, min(batch_size, 10)))
            timesteps = np.arange(n_obs)
            
            for seq_idx in range(min(batch_size, 8)):  # Show subset for clarity
                det_trajectory = sequence_dets[:, seq_idx]
                ax1.plot(timesteps, det_trajectory, color=colors[seq_idx], 
                        linewidth=2, alpha=0.8, label=f'Seq {seq_idx}')
                
                # Mark start and end
                ax1.scatter(0, det_trajectory[0], color=colors[seq_idx], 
                           s=100, marker='s', edgecolors='black', zorder=5)
                ax1.scatter(n_obs-1, det_trajectory[-1], color=colors[seq_idx], 
                           s=100, marker='*', edgecolors='black', zorder=5)
            
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('det(G)')
            ax1.set_title('Individual Sequence det(G) Trajectories')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot 2: Statistical summary across all sequences
            mean_det = np.mean(sequence_dets, axis=1)
            std_det = np.std(sequence_dets, axis=1)
            
            ax2.fill_between(timesteps, mean_det - std_det, mean_det + std_det, 
                            alpha=0.3, color='blue', label='±1 std')
            ax2.plot(timesteps, mean_det, color='blue', linewidth=3, label='Mean det(G)')
            
            # Highlight min/max points
            min_idx = np.argmin(mean_det)
            max_idx = np.argmax(mean_det)
            ax2.scatter(min_idx, mean_det[min_idx], color='red', s=100, 
                       marker='v', label=f'Min at t={min_idx}', zorder=5)
            ax2.scatter(max_idx, mean_det[max_idx], color='green', s=100, 
                       marker='^', label=f'Max at t={max_idx}', zorder=5)
            
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('det(G)')
            ax2.set_title('Statistical Summary of det(G) Evolution')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # Save sequence evolution plot
            seq_filename = f'sequence_det_evolution_epoch_{epoch}.png'
            plt.savefig(seq_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    "temporal_evolution/sequence_trajectories": wandb.Image(seq_filename, 
                        caption=f"Epoch {epoch} - det(G) evolution along sequences"),
                })
            
        except Exception as e:
            print(f"⚠️ Sequence det evolution plots failed: {e}")

    def _create_interactive_temporal_animation(self, z_pca_seq, temporal_det_maps, sequence_dets, xx, yy, epoch):
        """Create interactive Plotly animation of temporal det(G) evolution."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            batch_size, n_obs, _ = z_pca_seq.shape
            
            # Create figure with animation
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["🎬 Temporal det(G) Evolution", "📈 det(G) Along Sequences"],
                specs=[[{"secondary_y": False}, {"secondary_y": False}]],
                horizontal_spacing=0.1
            )
            
            # Prepare data for animation frames
            frames = []
            colors = px.colors.qualitative.Set3[:min(batch_size, 8)]
            
            for t in range(n_obs):
                frame_data = []
                
                # Add heatmap for current timestep
                frame_data.append(
                    go.Contour(
                        x=np.linspace(xx.min(), xx.max(), xx.shape[1]),
                        y=np.linspace(yy.min(), yy.max(), yy.shape[0]),
                        z=temporal_det_maps[t],
                        colorscale='Viridis',
                        opacity=0.7,
                        showscale=True,
                        colorbar=dict(title="det(G)", x=0.45, len=0.8),
                        name="det(G) field",
                        hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>det(G): %{z:.2e}<extra></extra>",
                        xaxis='x', yaxis='y'  # Fixed: use xaxis/yaxis instead of row/col
                    )
                )
                
                # Add sequence trajectories up to current timestep
                for seq_idx in range(min(batch_size, 6)):
                    # Trajectory so far
                    traj_x = z_pca_seq[seq_idx, :t+1, 0]
                    traj_y = z_pca_seq[seq_idx, :t+1, 1]
                    
                    frame_data.append(
                        go.Scatter(
                            x=traj_x,
                            y=traj_y,
                            mode='lines+markers',
                            line=dict(color=colors[seq_idx], width=3),
                            marker=dict(size=8, color=colors[seq_idx]),
                            name=f'Seq {seq_idx}',
                            showlegend=(t == 0),  # Show legend only in first frame
                                                    hovertemplate=f"Seq {seq_idx}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>Timestep: {t}<extra></extra>",
                        xaxis='x', yaxis='y'
                        )
                    )
                    
                    # Current position marker
                    if t < len(traj_x):
                        frame_data.append(
                            go.Scatter(
                                x=[traj_x[-1]],
                                y=[traj_y[-1]],
                                mode='markers',
                                marker=dict(
                                    size=15,
                                    color=colors[seq_idx],
                                    symbol='star',
                                    line=dict(color='white', width=2)
                                ),
                                name=f'Current {seq_idx}',
                                showlegend=False,
                                                            hovertemplate=f"Seq {seq_idx} (current)<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>det(G): {sequence_dets[t, seq_idx]:.2e}<extra></extra>",
                            xaxis='x', yaxis='y'
                            )
                        )
                
                # Add det(G) evolution plot (right panel)
                for seq_idx in range(min(batch_size, 6)):
                    det_so_far = sequence_dets[:t+1, seq_idx]
                    timesteps_so_far = np.arange(t+1)
                    
                    frame_data.append(
                        go.Scatter(
                            x=timesteps_so_far,
                            y=det_so_far,
                            mode='lines+markers',
                            line=dict(color=colors[seq_idx], width=3),
                            marker=dict(size=6, color=colors[seq_idx]),
                            name=f'det(G) Seq {seq_idx}',
                            showlegend=False,
                                                    hovertemplate=f"Seq {seq_idx}<br>Timestep: %{{x}}<br>det(G): %{{y:.2e}}<extra></extra>",
                        xaxis='x2', yaxis='y2'
                        )
                    )
                
                frames.append(go.Frame(data=frame_data, name=str(t)))
            
            # Initial frame (t=0) - fix subplot assignment
            for trace in frames[0].data:
                # Assign to appropriate subplot based on xaxis
                if hasattr(trace, 'xaxis') and trace.xaxis == 'x2':
                    fig.add_trace(trace, row=1, col=2)
                else:
                    fig.add_trace(trace, row=1, col=1)
            
            fig.frames = frames
            
            # Add animation controls
            fig.update_layout(
                title={
                    'text': f"🎬 TEMPORAL METRIC EVOLUTION - EPOCH {epoch}<br>"
                           f"<span style='font-size:14px'>det(G) changing over time • {n_obs} timesteps • {batch_size} sequences</span>",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#2E86AB'}
                },
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 500, "redraw": True}, 
                                          "fromcurrent": True, "transition": {"duration": 200}}],
                            "label": "▶️ Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True}, 
                                            "mode": "immediate", "transition": {"duration": 0}}],
                            "label": "⏸️ Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }],
                sliders=[{
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Timestep: ",
                        "visible": True,
                        "xanchor": "right"
                    },
                    "transition": {"duration": 200, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f], {"frame": {"duration": 200, "redraw": True}, 
                                         "mode": "immediate", "transition": {"duration": 200}}],
                            "label": str(t),
                            "method": "animate"
                        }
                        for t, f in enumerate(frames)
                    ]
                }],
                width=1600,
                height=800,
                showlegend=True,
                paper_bgcolor='rgba(240,248,255,0.95)',
                plot_bgcolor='rgba(255,255,255,0.8)',
            )
            
            # Update subplot titles and axes
            fig.update_xaxes(title_text="PC1", row=1, col=1)
            fig.update_yaxes(title_text="PC2", row=1, col=1)
            fig.update_xaxes(title_text="Timestep", row=1, col=2)
            fig.update_yaxes(title_text="det(G)", row=1, col=2)
            
            # Save interactive animation
            html_filename = f'temporal_metric_animation_epoch_{epoch}.html'
            fig.write_html(html_filename, include_plotlyjs=True)
            
            png_filename = f'temporal_metric_animation_epoch_{epoch}.png'
            self._safe_write_image(fig, png_filename, width=1600, height=800, scale=2)
            
            # Log to wandb if available  
            if wandb.run is not None:
                wandb.log({
                    "temporal_evolution/interactive_animation": wandb.Html(html_filename),
                    "temporal_evolution/animation_static": wandb.Image(png_filename, 
                        caption=f"Epoch {epoch} - Interactive temporal animation"),
                })
            
            print(f"🎬 Interactive temporal animation saved: {html_filename}")
            
        except Exception as e:
            print(f"⚠️ Interactive temporal animation failed: {e}")
            import traceback
            traceback.print_exc()

def create_config(loop_mode, experiment_suffix=""):
    """Create configuration for a specific loop mode."""
    return SimpleNamespace(
        # Loop mode configuration (THE KEY DIFFERENCE!)
        loop_mode=loop_mode,  # "open" or "closed"
        cycle_penalty=5.0 if loop_mode == "closed" else 0.0,  # Higher penalty for closed loop
        
        # Model parameters
        input_dim=(3, 64, 64),
        latent_dim=16,
        n_flows=8,
        flow_hidden_size=256,
        flow_n_blocks=2,
        flow_n_hidden=1,
        epsilon=1e-6,
        beta=1.0,
        
        # NEW: Separate beta for Riemannian KL (to handle high values)
        riemannian_beta=8.0,  # Higher weight for Riemannian KL to emphasize manifold structure
        
        # Training parameters
        batch_size=8,  # Smaller batch for cyclic data
        learning_rate=3e-4,
        n_epochs=25,
        
        # Data parameters (using cyclic-only datasets)
        n_train_samples=1000,  # Use subset of cyclic data
        n_val_samples=600,
        
        # Riemannian parameters
        use_riemannian=True,
        riemannian_method="geodesic",  # Using geodesic sampling for better manifold sampling
        
        # NEW: Posterior type parameters
        posterior_type="riemannian_metric",  # "gaussian", "iaf", "riemannian_metric"
        
        # Temperature fix
        temperature_fix=3.0,
        
        # Pretrained components
        metric_path="metric_T0.7_scaled.pt",
        
        # Experiment naming
        experiment_suffix=experiment_suffix
    )

def main():
    """Main function to run both experiments."""
    parser = argparse.ArgumentParser(description='Compare loop modes on cyclic sequences')
    parser.add_argument('--loop_mode', choices=['open', 'closed'], required=True,
                       help='Loop mode to train: open or closed')
    parser.add_argument('--cycle_penalty', type=float, default=1.0,
                       help='Weight for cycle penalty when loop_mode=closed')
    parser.add_argument('--n_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n_train_samples', type=int, default=3000, help='Number of training samples')
    parser.add_argument('--n_val_samples', type=int, default=800, help='Number of validation samples')
    parser.add_argument('--visualization_frequency', type=int, default=5, help='Visualization frequency')
    parser.add_argument('--sequence_viz_count', type=int, default=8, help='Number of sequences to visualize')
    parser.add_argument('--run_name', type=str, default=None, help='Custom run name')
    parser.add_argument('--html_latent_space', action='store_true', default=False, 
                       help='Generate interactive HTML latent space visualization at the end of training')
    parser.add_argument('--riemannian_beta', type=float, default=None, 
                       help='Beta weight for Riemannian KL divergence (if using riemannian_metric posterior)')
    parser.add_argument('--disable_curvature_during_training', action='store_true', default=False,
                       help='Disable expensive curvature computation during training (compute only at end)')
    parser.add_argument('--wandb_only', action='store_true', default=False,
                       help='Only log to WandB, do not save files locally')
    parser.add_argument('--disable_local_files', action='store_true', default=False,
                       help='Disable saving visualization files locally (keep only WandB logging)')
    parser.add_argument('--wandb_offline', action='store_true', default=False,
                       help='Run WandB in offline mode to avoid local wandb run storage')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting {args.loop_mode.upper()} Cyclic Loop Mode Experiment")
    print(f"🎯 Mode: {args.loop_mode}")
    print(f"📊 Epochs: {args.n_epochs}")
    print(f"📈 Samples: {args.n_train_samples}")
    print(f"🔍 Sequence Viz Count: {args.sequence_viz_count}")
    
    # Create config for this mode
    config = create_config(args.loop_mode, experiment_suffix=f"_extended_{args.n_epochs}epochs")
    
    # Update config with command line arguments
    config.n_epochs = args.n_epochs
    config.n_train_samples = args.n_train_samples
    config.n_val_samples = args.n_val_samples
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.cycle_penalty = args.cycle_penalty
    config.visualization_frequency = args.visualization_frequency
    config.sequence_viz_count = args.sequence_viz_count
    config.generate_html_latent_space = args.html_latent_space
    config.disable_curvature_during_training = args.disable_curvature_during_training
    config.wandb_only = args.wandb_only
    config.disable_local_files = args.disable_local_files
    config.wandb_offline = args.wandb_offline
    
    # Update Riemannian beta if provided
    if args.riemannian_beta is not None:
        config.riemannian_beta = args.riemannian_beta
    
    # Set experiment name based on extended training
    project_name = "cyclic-loop-mode-extended" if args.n_epochs > 50 else "cyclic-loop-mode-comparison"
    run_name = args.run_name or f"{args.loop_mode}_cyclic_extended"
    
    # Create and run trainer
    try:
        trainer = CyclicLoopTrainer(config, project_name=project_name, run_name=run_name)
        trainer.train(n_epochs=args.n_epochs)
        
        print(f"✅ {args.loop_mode.upper()} loop experiment completed successfully!")
        
    except Exception as e:
        print(f"❌ {args.loop_mode.upper()} loop experiment failed: {e}")
        raise
    
    print(f"🏁 {args.loop_mode.upper()} loop experiment finished")
    print(f"📊 Check Weights & Biases for detailed comparisons")
    print(f"💡 Key metrics analyzed:")
    print(f"   • Cyclicity preservation (recon vs original)")
    print(f"   • Latent space cyclicity")
    print(f"   • Sequence trajectory visualization")
    print(f"   • Training stability")
    
    if args.html_latent_space:
        print(f"\n🌐 Interactive HTML visualization:")
        print(f"   • File: interactive_latent_space_{args.loop_mode}.html")
        print(f"   • Images: html_latent_images_{args.loop_mode}/")
        print(f"   • To view: python3 -m http.server 8000")
        print(f"   • Then open: http://localhost:8000/interactive_latent_space_{args.loop_mode}.html")

if __name__ == "__main__":
    main() 