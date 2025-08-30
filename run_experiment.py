#!/usr/bin/env python3
"""
RlVAE Experiment Runner
======================

Hydra-powered experiment runner for systematic comparison of Riemannian VAE variants.

Usage Examples:
--------------

1. Single experiment:
   python run_experiment.py

2. Quick development test:
   python run_experiment.py training=quick visualization=minimal

3. Compare models:
   python run_experiment.py experiment=comparison_study

4. Hyperparameter sweep:
   python run_experiment.py experiment=hyperparameter_sweep -m

5. Custom configuration:
   python run_experiment.py model=vanilla_vae training.n_epochs=50 wandb.mode=offline

6. Override specific parameters:
   python run_experiment.py model.riemannian_beta=10.0 training.learning_rate=1e-3
"""

import os
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"
lib_src_dir = src_dir / "lib" / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(lib_src_dir) not in sys.path:
    sys.path.insert(0, str(lib_src_dir))

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime
from typing import Dict, Any, List, Optional
import yaml

# Local imports
from utils.reproducibility import configure_for_experiment
try:
    from rlvae.models.modular_rlvae import ModularRiemannianFlowVAE, ModelFactory, MetricsCollector
except Exception:
    from models.modular_rlvae import ModularRiemannianFlowVAE, ModelFactory, MetricsCollector
from data.cyclic_dataset import CyclicSpritesDataModule
try:
    from rlvae.training.lightning_trainer import LightningRlVAETrainer
except Exception:
    from training.lightning_trainer import LightningRlVAETrainer
from visualizations.manager import VisualizationManager


class ExperimentRunner:
    """Main experiment runner with Hydra configuration."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.results = {}
        
        # --- AUTOMATION: Ensure model.input_dim always matches data shape ---
        if (
            hasattr(self.config, 'data') and
            hasattr(self.config.data, 'channels') and
            hasattr(self.config.data, 'image_size') and
            isinstance(self.config.data.image_size, (list, tuple)) and
            len(self.config.data.image_size) == 2
        ):
            self.config.model.input_dim = [
                self.config.data.channels,
                self.config.data.image_size[0],
                self.config.data.image_size[1]
            ]
        # ---------------------------------------------------------------
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        print(f"🚀 RlVAE Experiment Runner")
        print(f"📅 Experiment: {config.experiment_name}")
        print(f"💻 Device: {self.device}")
        print(f"📁 Output: {self.output_dir}")
        
        # Set comprehensive random seed for reproducibility
        seed = config.get('seed', 42)
        experiment_type = config.get('experiment_type', 'research')
        configure_for_experiment(seed, experiment_type, self.device)
        print(f"🎲 Comprehensive seeding configured (seed: {seed})")
    
    def run(self):
        """Run the experiment based on configuration."""
        experiment_type = self.config.experiment.type
        print(f"\n🧪 Running {experiment_type} experiment: {self.config.experiment.name}")
        if experiment_type == "single":
            self.run_single_experiment()
            # Stage C: add recon vs real panel using validation batch
            try:
                import torchvision.utils as vutils
                data_module = CyclicSpritesDataModule(self.config.data)
                data_module.setup("fit", self.config.training)
                loader_c = data_module.val_dataloader()
                batch_c = next(iter(loader_c))[:8].to(self.device)
                model_wrapper = LightningRlVAETrainer(self.config, data_module=data_module)
                model_wrapper.model.eval()
                with torch.no_grad():
                    outc = model_wrapper.model(batch_c)
                recon_c = None
                if isinstance(outc, dict):
                    recon_c = outc.get('reconstruction', None) or outc.get('recon_x', None)
                elif hasattr(outc, 'reconstruction'):
                    recon_c = outc.reconstruction
                elif hasattr(outc, 'recon_x'):
                    recon_c = outc.recon_x
                if recon_c is not None:
                    recon_c = recon_c.clamp(0, 1)
                    grid_c = vutils.make_grid(torch.cat([batch_c, recon_c], dim=0), nrow=8, normalize=False)
                    wandb.log({"stageC/recon_vs_real": wandb.Image(grid_c)})
            except Exception as e:
                print(f"[Stage C] ⚠️ recon_vs_real logging failed: {e}")
        elif experiment_type == "comparison":
            self.run_comparison_study()
        elif experiment_type == "sweep":
            self.run_hyperparameter_sweep()
        elif experiment_type == "pipeline":
            self.run_pipeline_experiment()
        elif experiment_type == "three_stage":
            self.run_three_stage_experiment()
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    def run_single_experiment(self):
        """Run a single experiment with current configuration."""
        
        # Initialize wandb
        wandb_logger = self._setup_wandb("single_run")
        
        # Create data module
        data_module = CyclicSpritesDataModule(self.config.data)
        data_module.setup("fit", self.config.training)
        
        # Create model
        model_wrapper = LightningRlVAETrainer(
            self.config,
            data_module=data_module
        )
        # === Config-driven generator, inference, evaluator ===
        self.generator = model_wrapper.model.create_generator(self.config.get('generation', None))
        self.inference_pipeline = model_wrapper.model.create_inference_pipeline(self.config.get('inference', None))
        self.evaluator = model_wrapper.model.create_evaluator() if hasattr(model_wrapper.model, 'create_evaluator') else None
        # === End config-driven instantiation ===
        
        # Setup trainer
        trainer = self._create_trainer(wandb_logger)
        
        # Train
        print(f"🚀 Starting training...")
        trainer.fit(model_wrapper, data_module)
        
        # Test
        print(f"🧪 Running test...")
        test_results = trainer.test(model_wrapper, data_module)
        
        # Save results
        self.results = {
            'test_results': test_results[0] if test_results else {},
            'model_summary': model_wrapper.model.get_model_summary()
        }
        
        print(f"✅ Single experiment completed!")
        self._save_results()
    
    def run_comparison_study(self):
        """Run comparison between multiple model variants."""
        
        print(f"🔬 Running comparison study...")
        models_to_compare = self.config.experiment.models
        comparison_metrics = self.config.experiment.comparison_metrics
        
        metrics_collector = MetricsCollector()
        all_results = {}
        
        for model_name in models_to_compare:
            print(f"\n🎯 Training model: {model_name}")
            
            # Create model-specific config
            model_config = self._create_model_config(model_name)
            
            # Setup wandb for this model
            wandb_logger = self._setup_wandb(f"comparison_{model_name}")
            
            # Create data module
            data_module = CyclicSpritesDataModule(model_config.data)
            data_module.setup("fit", model_config.training)
            
            # Create model wrapper
            model_wrapper = LightningRlVAETrainer(
                model_config,
                data_module=data_module
            )
            
            # Setup trainer
            trainer = self._create_trainer(wandb_logger)
            
            # Train
            trainer.fit(model_wrapper, data_module)
            
            # Test
            test_results = trainer.test(model_wrapper, data_module)
            
            # Collect metrics
            if test_results:
                metrics = self._extract_comparison_metrics(test_results[0], comparison_metrics)
                metrics_collector.add_model_metrics(model_name, metrics)
            
            # Store results
            all_results[model_name] = {
                'test_results': test_results[0] if test_results else {},
                'model_summary': model_wrapper.model.get_model_summary()
            }
            
            # Finish this wandb run
            wandb.finish()
        
        # Create comparison analysis
        print(f"\n📊 Analyzing comparison results...")
        comparison_summary = metrics_collector.get_comparison_summary()
        
        # Start final wandb run for comparison
        wandb_logger = self._setup_wandb("comparison_analysis")
        metrics_collector.log_comparison_to_wandb()
        
        self.results = {
            'comparison_summary': comparison_summary,
            'individual_results': all_results,
            'comparison_metrics': comparison_metrics
        }
        
        print(f"✅ Comparison study completed!")
        self._save_results()
        self._print_comparison_summary(comparison_summary)
    
    def run_hyperparameter_sweep(self):
        """Run hyperparameter sweep (placeholder for Hydra sweep)."""
        print(f"🌊 Hyperparameter sweep mode")
        print(f"⚠️ This should be run with Hydra multirun (-m flag)")
        print(f"Example: python run_experiment.py experiment=hyperparameter_sweep -m")
        
        # For individual sweep runs, just run single experiment
        self.run_single_experiment()
    
    def run_pipeline_experiment(self):
        """
        Run a two-stage pipeline:
        1. Vanilla VAE training + diverse metric extraction (with wandb logging and results file)
        2. RLVAE training with pretrained components from stage 1
        """
        import torch
        import wandb
        from pathlib import Path
        from datetime import datetime
        from scripts.train_diverse_metric_vae import create_model, SpritesDataset, extract_diverse_metric, save_model_components
        from torch.utils.data import DataLoader, ConcatDataset
        import torch.nn.functional as F
        import torchvision.utils as vutils
        import numpy as np
        import yaml
        # --- Stage 1: Vanilla VAE + Diverse Metric ---
        if hasattr(self.config.experiment, 'skip_stage1') and self.config.experiment.skip_stage1:
            print("⏭️ Skipping Stage 1 (vanilla VAE) as requested by experiment.skip_stage1=true")
            # Expect user to provide pretrained paths in config.pretrained
            encoder_path = getattr(self.config.pretrained, 'encoder_path', None)
            decoder_path = getattr(self.config.pretrained, 'decoder_path', None)
            metric_path = getattr(self.config.pretrained, 'metric_path', None)
            if not all([encoder_path, decoder_path, metric_path]):
                raise FileNotFoundError("Missing pretrained components in config.pretrained for skipped Stage 1")
            component_paths = {'encoder': encoder_path, 'decoder': decoder_path, 'metric': metric_path}
        else:
            stage1_cfg = self.config.experiment.stage1
            print("\n=== [Pipeline] Stage 1: Vanilla VAE + Diverse Metric ===")
            architecture = getattr(stage1_cfg, 'architecture', 'mlp') if hasattr(stage1_cfg, 'architecture') else 'mlp'
            latent_dim = getattr(stage1_cfg, 'latent_dim', 16) if hasattr(stage1_cfg, 'latent_dim') else 16
            epochs = getattr(stage1_cfg, 'epochs', 50) if hasattr(stage1_cfg, 'epochs') else 50
            temperature = getattr(stage1_cfg, 'temperature', 0.5) if hasattr(stage1_cfg, 'temperature') else 0.5
            regularization = getattr(stage1_cfg, 'regularization', 0.01) if hasattr(stage1_cfg, 'regularization') else 0.01
            preset = getattr(stage1_cfg, 'preset', 'balanced') if hasattr(stage1_cfg, 'preset') else 'balanced'
            n_heatmaps = getattr(stage1_cfg, 'n_heatmaps', 6) if hasattr(stage1_cfg, 'n_heatmaps') else 6
            data_config_name = getattr(stage1_cfg, 'data', 'cyclic_sprites')
            data_cfg = self.config.data if data_config_name == self.config.data.get('name', data_config_name) else hydra.compose(config_name=data_config_name, overrides=[]).data
            train_data_path = data_cfg.get('train_path', 'data/processed/Sprites_train_cyclic.pt')
            test_data_path = data_cfg.get('test_path', 'data/processed/Sprites_test_cyclic.pt')
            print(f"[Stage 1] Using train data: {train_data_path}")
            print(f"[Stage 1] Using test data: {test_data_path}")
            train_dataset = SpritesDataset(train_data_path, normalize=False)
            test_dataset = SpritesDataset(test_data_path, normalize=False)
            full_dataset = ConcatDataset([train_dataset, test_dataset])
            train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # --- MANUAL PATCH: Set model.sequence_length and n_flows from dataset (ALWAYS RUN) ---
            from omegaconf import OmegaConf
            OmegaConf.set_struct(self.config.model, False)
            import torch
            raw_data = torch.load(self.config.data.train_path, map_location='cpu')
            if raw_data.ndim >= 2:
                detected_seq_len = raw_data.shape[1]
            else:
                detected_seq_len = 1
            self.config.model.sequence_length = detected_seq_len
            self.config.model.n_flows = detected_seq_len - 1
            print(f"[PATCH] Set self.config.model.sequence_length = {detected_seq_len}")
            print(f"[PATCH] Set self.config.model.n_flows = {detected_seq_len - 1}")
            assert self.config.model.n_flows == self.config.model.sequence_length - 1, (
                f"[CONFIG ERROR] n_flows ({self.config.model.n_flows}) != sequence_length - 1 ({self.config.model.sequence_length - 1})! "
                "Check your config overrides."
            )
            # Ensure input_dim is set from data config for Stage 1 (robust automation)
            input_dim = [self.config.data.channels, self.config.data.image_size[0], self.config.data.image_size[1]]
            model = create_model(architecture, input_dim=tuple(input_dim), latent_dim=latent_dim).to(self.device)
            if architecture.lower() in ['cnn', 'resnet']:
                lr = 5e-5
                print(f"[Stage 1] Using lower learning rate {lr} for {architecture} architecture")
            else:
                lr = 1e-4
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            wandb_run = wandb.init(
                project=self.config.wandb.project, 
                name=f"pipeline_stage1_vanilla_vae_{architecture}_ld{latent_dim}",
                settings=wandb.Settings(init_timeout=300),
                config={
                "architecture": architecture,
                "latent_dim": latent_dim,
                "epochs": epochs,
                "temperature": temperature,
                "regularization": regularization,
                "preset": preset,
                "n_heatmaps": n_heatmaps,
                "stage": "pipeline_stage1_vanilla_vae"
            })
            print(f"[Stage 1] Training for {epochs} epochs...")
            batch_counter = 0
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                total_recon_loss = 0
                total_kld_loss = 0
                for batch in train_loader:
                    batch = batch.to(self.device)
                    if architecture.lower() in ["mlp", "pythae"]:
                        output = model({"data": batch})
                        loss = output.loss
                        recon_loss = output.recon_loss.item()
                        kld_loss = output.reg_loss.item()
                        recon_batch = output.recon_x
                    else:
                        output = model(batch)
                        loss = output.loss
                        recon_loss = output.reconstruction_loss.item()
                        kld_loss = output.reg_loss.item()
                        recon_batch = output.recon_x
                    optimizer.zero_grad()
                    loss.backward()
                    if architecture.lower() in ['cnn', 'resnet']:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    total_recon_loss += recon_loss
                    total_kld_loss += kld_loss
                    if batch_counter % 100 == 0:
                        with torch.no_grad():
                            recon_display = recon_batch.clamp(0, 1)
                            comparison = torch.cat([batch[:8], recon_display[:8]], dim=0)
                            grid = vutils.make_grid(comparison, nrow=8, normalize=False)
                            wandb.log({
                                "reconstructions": wandb.Image(grid),
                                "batch": batch_counter,
                                "epoch": epoch + 1
                            })
                    batch_counter += 1
                avg_loss = total_loss / len(train_loader)
                avg_recon_loss = total_recon_loss / len(train_loader)
                avg_kld_loss = total_kld_loss / len(train_loader)
                model.eval()
                val_loss = 0
                val_recon_loss = 0
                val_kld_loss = 0
                val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        if architecture.lower() in ["mlp", "pythae"]:
                            output = model({"data": batch})
                            recon_loss_val = output.recon_loss.item()
                            kld_loss_val = output.reg_loss.item()
                        else:
                            output = model(batch)
                            recon_loss_val = output.reconstruction_loss.item()
                            kld_loss_val = output.reg_loss.item()
                        if not (torch.isnan(output.loss) or torch.isinf(output.loss)):
                            val_loss += output.loss.item()
                            val_recon_loss += recon_loss_val
                            val_kld_loss += kld_loss_val
                            val_batches += 1
                if val_batches > 0:
                    avg_val_loss = val_loss / val_batches
                    avg_val_recon_loss = val_recon_loss / val_batches
                    avg_val_kld_loss = val_kld_loss / val_batches
                else:
                    avg_val_loss = avg_val_recon_loss = avg_val_kld_loss = 0
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/reconstruction_loss": avg_recon_loss,
                    "train/kld_loss": avg_kld_loss,
                    "val/loss": avg_val_loss,
                    "val/reconstruction_loss": avg_val_recon_loss,
                    "val/kld_loss": avg_val_kld_loss
                })
                print(f"[Stage 1] Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} (Recon: {avg_recon_loss:.4f} + KL: {avg_kld_loss:.4f}) | Val Loss: {avg_val_loss:.4f}")
            print("[Stage 1] Training complete. Saving components...")
            component_paths = save_model_components(model, architecture, latent_dim)
            print("[Stage 1] Extracting diverse metric...")
            metric_path = extract_diverse_metric(model, architecture, latent_dim, temperature=temperature, regularization=regularization, input_dim=tuple(input_dim), data_path=train_data_path)
            
            # --- METRIC ANALYSIS & VISUALIZATION (from train_diverse_metric_vae.py) ---
            print("[Stage 1] Creating metric analysis and visualizations...")
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('default')
            sns.set_palette("husl")
            
            try:
                metric_data = torch.load(metric_path, map_location='cpu', weights_only=False)
                M_matrices = metric_data['M_matrices']
                centroids = metric_data['centroids']
                eigenvals = torch.linalg.eigvals(M_matrices).real
                min_eigenvals = eigenvals.min(dim=-1)[0]
                max_eigenvals = eigenvals.max(dim=-1)[0]
                mean_eigenvals = eigenvals.mean(dim=-1)
                cond_nums = max_eigenvals / (min_eigenvals + 1e-12)
                determinants = torch.linalg.det(M_matrices)
                eigenval_spread = max_eigenvals - min_eigenvals
                
                # 1. Eigenvalue distributions
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle('Metric Eigenvalue & Condition Analysis', fontsize=16, fontweight='bold')
                axes[0,0].hist(min_eigenvals.numpy(), bins=40, color='red', edgecolor='black')
                axes[0,0].set_title('Min Eigenvalue')
                axes[0,1].hist(max_eigenvals.numpy(), bins=40, color='blue', edgecolor='black')
                axes[0,1].set_title('Max Eigenvalue')
                axes[0,2].hist(mean_eigenvals.numpy(), bins=40, color='green', edgecolor='black')
                axes[0,2].set_title('Mean Eigenvalue')
                axes[1,0].hist(cond_nums.numpy(), bins=40, color='orange', edgecolor='black')
                axes[1,0].set_title('Condition Number')
                log_dets = torch.log10(torch.abs(determinants) + 1e-50)
                axes[1,1].hist(log_dets.numpy(), bins=40, color='purple', edgecolor='black')
                axes[1,1].set_title('Log₁₀|Determinant|')
                axes[1,2].hist(eigenval_spread.numpy(), bins=40, color='cyan', edgecolor='black')
                axes[1,2].set_title('Eigenvalue Spread')
                plt.tight_layout()
                wandb.log({"metric_analysis/eigenvalue_distributions": wandb.Image(fig)})
                plt.close(fig)
                
                # 2. Heatmaps of metric matrices (configurable number, or fewer if less matrices available)
                n_heatmaps = min(n_heatmaps, len(M_matrices))
                fig, axes = plt.subplots(1, n_heatmaps, figsize=(4*n_heatmaps, 4))
                for i in range(n_heatmaps):
                    im = axes[i].imshow(M_matrices[i].numpy(), cmap='RdYlBu_r', aspect='auto')
                    axes[i].set_title(f'Matrix {i}')
                    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                plt.tight_layout()
                wandb.log({"metric_analysis/metric_matrix_heatmaps": wandb.Image(fig)})
                plt.close(fig)
                
                # 3. Centroid statistics
                centroid_norms = torch.norm(centroids, dim=1)
                pairwise_dists = torch.cdist(centroids, centroids)
                triu_mask = torch.triu(torch.ones_like(pairwise_dists, dtype=bool), diagonal=1)
                pairwise_vals = pairwise_dists[triu_mask]
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].hist(centroid_norms.numpy(), bins=30, color='red', edgecolor='black')
                axes[0].set_title('Centroid Norms')
                axes[1].hist(pairwise_vals.numpy(), bins=40, color='blue', edgecolor='black')
                axes[1].set_title('Pairwise Centroid Distances')
                plt.tight_layout()
                wandb.log({"metric_analysis/centroid_stats": wandb.Image(fig)})
                plt.close(fig)
                
                print(f"[Stage 1] ✅ Metric analysis complete: {n_heatmaps} matrix heatmaps created")
                
            except Exception as e:
                print(f"[Stage 1] ⚠️ Could not analyze metric for graphs: {e}")
            
            # Log metric file as wandb artifact
            if Path(metric_path).exists():
                artifact = wandb.Artifact(f"diverse_metric_{architecture}_ld{latent_dim}", type="metric")
                artifact.add_file(metric_path)
                wandb.log_artifact(artifact)
            
            # Save results YAML for Stage 1
            vanilla_results = {
                'architecture': architecture,
                'latent_dim': latent_dim,
                'epochs': epochs,
                'temperature': temperature,
                'regularization': regularization,
                'preset': preset,
                'encoder_path': component_paths['encoder'],
                'decoder_path': component_paths['decoder'],
                'metric_path': metric_path
            }
            results_path = Path(self.output_dir) / "vanilla_vae_results.yaml"
            with open(results_path, 'w') as f:
                yaml.dump(vanilla_results, f)
            print(f"[Stage 1] Results saved to: {results_path}")
            wandb.finish()
        # --- Stage 2: RLVAE Training ---
        print("\n=== [Pipeline] Stage 2: RLVAE Training ===")
        # Update config for pretrained paths
        self.config.model.pretrained.encoder_path = component_paths['encoder']
        self.config.model.pretrained.decoder_path = component_paths['decoder']
        self.config.model.pretrained.metric_path = metric_path
        # Ensure latent_dim is set from stage2 config if present
        # Resolve all interpolations (e.g., oc.decode expressions) before accessing Stage 2 values
        try:
            OmegaConf.resolve(self.config)
        except Exception as e:
            print(f"[PATCH] OmegaConf.resolve failed: {e}")
        stage2_cfg = self.config.experiment.stage2
        latent_dim2 = getattr(stage2_cfg, 'latent_dim', None)
        if latent_dim2 is not None:
            self.config.model.latent_dim = latent_dim2
        # --- PATCH: Ensure model.riemannian_beta is set from experiment.stage2 ---
        if hasattr(self.config, 'experiment') and hasattr(self.config.experiment, 'stage2') and hasattr(self.config.experiment.stage2, 'riemannian_beta'):
            self.config.model.riemannian_beta = self.config.experiment.stage2.riemannian_beta
            print(f"[PATCH] Set self.config.model.riemannian_beta = {self.config.model.riemannian_beta}")
        # --- PATCH: Ensure model.n_flows is set from experiment.stage2 ---
        if hasattr(self.config, 'experiment') and hasattr(self.config.experiment, 'stage2') and hasattr(self.config.experiment.stage2, 'n_flows'):
            n_flows_val = self.config.experiment.stage2.n_flows
            # Coerce string expressions like "8 - 1" to integers if present
            if isinstance(n_flows_val, str):
                import re
                m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", n_flows_val)
                if m:
                    try:
                        n_flows_val = int(m.group(1)) - int(m.group(2))
                        print(f"[PATCH] Coerced stage2.n_flows expression to int: {n_flows_val}")
                    except Exception:
                        pass
                else:
                    try:
                        n_flows_val = int(n_flows_val)
                        print(f"[PATCH] Coerced stage2.n_flows string to int: {n_flows_val}")
                    except Exception:
                        pass
            self.config.model.n_flows = n_flows_val
            print(f"[PATCH] Set self.config.model.n_flows = {self.config.model.n_flows}")
        # Ensure input_dim is set from data config for Stage 2 (robust automation)
        self.config.model.input_dim = [
            self.config.data.channels,
            self.config.data.image_size[0],
            self.config.data.image_size[1]
        ]
        # Ensure visualization level respects CLI overrides; only apply stage2 default if no CLI override
        visualization_level2 = getattr(stage2_cfg, 'visualization', None)
        cli_overrides_visualization = False
        try:
            from hydra.core.hydra_config import HydraConfig
            task_overrides = getattr(HydraConfig.get(), 'overrides', None)
            if task_overrides and hasattr(task_overrides, 'task'):
                cli_overrides_visualization = any(
                    o.startswith('visualization=') or o.startswith('visualization.level=')
                    for o in task_overrides.task
                )
        except Exception:
            pass
        if visualization_level2 is not None and not cli_overrides_visualization:
            if hasattr(self.config, 'visualization') and hasattr(self.config.visualization, 'level'):
                self.config.visualization.level = visualization_level2
            elif hasattr(self.config, 'visualization') and isinstance(self.config.visualization, dict):
                self.config.visualization['level'] = visualization_level2
            else:
                self.config.visualization = {'level': visualization_level2}
        elif cli_overrides_visualization:
            print("[PATCH] Respecting CLI visualization override; not applying stage2.visualization")
        # === PATCH: Ensure evaluation config is present for Stage 2 ===
        from omegaconf import OmegaConf
        import yaml
        if not hasattr(self.config, "evaluation") or self.config.evaluation is None:
            with open("conf/evaluation/default.yaml", "r") as f:
                eval_cfg = OmegaConf.create(yaml.safe_load(f))
            self.config.evaluation = eval_cfg
        # === END PATCH ===
        # --- FINAL PATCH: Set model.sequence_length and n_flows right before Stage 2 model creation ---
        from omegaconf import OmegaConf
        OmegaConf.set_struct(self.config.model, False)  # <-- must be first!
        import torch
        raw_data = torch.load(self.config.data.train_path, map_location='cpu')
        if raw_data.ndim >= 2:
            detected_seq_len = raw_data.shape[1]
        else:
            detected_seq_len = 1
        self.config.model.sequence_length = detected_seq_len
        
        # CRITICAL FIX: Only override n_flows if not explicitly set in stage2 config
        if not (hasattr(self.config, 'experiment') and hasattr(self.config.experiment, 'stage2') and hasattr(self.config.experiment.stage2, 'n_flows')):
            self.config.model.n_flows = detected_seq_len - 1
            print(f"[FINAL PATCH] Auto-set self.config.model.n_flows = {detected_seq_len - 1}")
        else:
            print(f"[FINAL PATCH] Keeping user-specified self.config.model.n_flows = {self.config.model.n_flows}")
        print(f"[FINAL PATCH] Set self.config.model.sequence_length = {detected_seq_len}")
        
        # Use modular model as configured
        print(f"[FINAL PATCH] Using modular model as configured: {self.config.model._target_}")
        
        print("[FINAL PATCH] Model config before creation:")
        print(OmegaConf.to_yaml(self.config.model))
        # Ensure input_dim is set from data config for Stage 2 (robust automation)
        self.config.model.input_dim = [
            self.config.data.channels,
            self.config.data.image_size[0],
            self.config.data.image_size[1]
        ]
        # Ensure visualization level respects CLI overrides; only apply stage2 default if no CLI override
        visualization_level2 = getattr(stage2_cfg, 'visualization', None)
        cli_overrides_visualization = False
        try:
            from hydra.core.hydra_config import HydraConfig
            task_overrides = getattr(HydraConfig.get(), 'overrides', None)
            if task_overrides and hasattr(task_overrides, 'task'):
                cli_overrides_visualization = any(
                    o.startswith('visualization=') or o.startswith('visualization.level=')
                    for o in task_overrides.task
                )
        except Exception:
            pass
        if visualization_level2 is not None and not cli_overrides_visualization:
            if hasattr(self.config, 'visualization') and hasattr(self.config.visualization, 'level'):
                self.config.visualization.level = visualization_level2
            elif hasattr(self.config, 'visualization') and isinstance(self.config.visualization, dict):
                self.config.visualization['level'] = visualization_level2
            else:
                self.config.visualization = {'level': visualization_level2}
        elif cli_overrides_visualization:
            print("[PATCH] Respecting CLI visualization override; not applying stage2.visualization")
        # CRITICAL FIX: Override training epochs for Stage 2 
        stage2_epochs = getattr(self.config.experiment.stage2, 'epochs', None)
        if stage2_epochs is not None:
            self.config.training.trainer.max_epochs = stage2_epochs
            print(f"[PATCH] Stage 2: Set training.trainer.max_epochs = {stage2_epochs}")
        
        n_flows = getattr(self.config.model, 'n_flows', None)
        print(f"[Stage 2] DEBUG: Training for {self.config.training.trainer.max_epochs} epochs")
        print(f"[Stage 2] DEBUG: Model n_flows = {n_flows} (config.model.n_flows)")
        # Run RLVAE training as in single experiment
        self.run_single_experiment()

        # === FINAL VISUALIZATIONS (after Stage 2) ===
        try:
            print("\n=== [Pipeline] Final Visualizations ===")
            from visualizations.manager import VisualizationManager
            # Assume model_wrapper is available from run_single_experiment (or reload model if needed)
            # For demonstration, we use the last trained model
            model = None
            if hasattr(self, 'evaluator') and hasattr(self.evaluator, 'model'):
                model = self.evaluator.model
            elif hasattr(self, 'generator') and hasattr(self.generator, 'model'):
                model = self.generator.model
            elif hasattr(self, 'inference_pipeline') and hasattr(self.inference_pipeline, 'model'):
                model = self.inference_pipeline.model
            if model is None:
                print("[Pipeline] Could not find trained model for visualization.")
                return
            viz_manager = VisualizationManager(model, self.device, self.config)
            basic_viz = viz_manager.modules['basic']
            # 1. Generation grid (prior samples)
            basic_viz.create_generation_grid(num_samples=16, epoch=self.config.training.trainer.max_epochs)
            # 2. Interpolation grid (random pairs)
            basic_viz.create_interpolation_grid(num_interpolations=5, steps=8, epoch=self.config.training.trainer.max_epochs)
            # 3. Comprehensive generation comparison (if multiple methods available)
            # Example: try to use generator to get images for several methods
            generation_methods = ["geodesic", "enhanced", "basic", "standard"]
            generation_results = {}
            fid_scores = {}
            for method in generation_methods:
                try:
                    if hasattr(model, 'generate_samples'):
                        result = model.generate_samples(num_samples=4, method=method)
                        images = result['images']
                        if images.dim() == 5:
                            images = images[:, 0]
                        generation_results[method] = images
                        # Optionally compute FID if available
                        if hasattr(model, 'compute_fid_score'):
                            fid = model.compute_fid_score(real_images=images, num_generated=min(16, len(images)), cache_key=f"pipeline_{method}")
                            if fid and 'fid_score' in fid:
                                fid_scores[method] = fid['fid_score']
                    else:
                        generation_results[method] = None
                except Exception as e:
                    print(f"[Pipeline] Generation for method {method} failed: {e}")
                    generation_results[method] = None
            # Only plot if at least one method succeeded
            if any(v is not None for v in generation_results.values()):
                basic_viz.create_comprehensive_generation_visualization(
                    generation_results=generation_results,
                    fid_scores=fid_scores,
                    num_samples_per_method=4,
                    epoch=self.config.training.trainer.max_epochs
                )
            # Log all final visualizations to wandb
            viz_manager.log_final_visualizations_to_wandb(epoch=self.config.training.trainer.max_epochs)
        except Exception as e:
            print(f"[Pipeline] ⚠️ Final visualizations failed: {e}")

    def run_three_stage_experiment(self):
        """
        Multi-stage pipeline:
          - Stage A: Warm VAE on full sequences (flattened)
          - Stage B: Metric learning at t=0 with selectable implementation (RHVAE-style or precision)
          - Optional RHMC sampling using the learned metric
          - Stage C: RLVAE with Riemannian prior/posterior, optionally updating metric
        """
        from torch.utils.data import DataLoader
        from scripts.train_diverse_metric_vae import (
            create_model as create_vanilla,
            SpritesDataset,
            extract_diverse_metric,
            save_model_components,
        )
        from models.components.metric_tensor import MetricTensor
        from models.components.native_inverse_metric import NativeInverseMetricTensor, NativeInverseRHMC
        import torch
        import os
        import yaml
        from pathlib import Path
        import wandb

        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create a single WandB run for the whole three-stage pipeline
        if cfg.wandb.mode != "disabled" and wandb.run is None:
            try:
                wandb.init(
                    project=cfg.wandb.project,
                    name=f"three_stage_pipeline_{cfg.experiment.name}",
                    mode=cfg.wandb.mode,
                    settings=wandb.Settings(init_timeout=300),
                    group=getattr(cfg.wandb, 'group', None),
                    config=OmegaConf.to_container(cfg, resolve=True)
                )
                print(f"✅ WandB initialized successfully for three-stage pipeline")
            except Exception as e:
                print(f"⚠️ WandB initialization failed: {e}")
                print("Continuing without WandB logging...")
                cfg.wandb.mode = "disabled"

        # Stage A: Train base model
        if getattr(cfg.experiment, 'run_stage_a', True):
            model_choice = str(getattr(cfg.experiment.stage_a, 'model', 'vanilla_vae')).lower()
            # Set defaults if not specified
            arch = getattr(cfg.model.encoder, 'architecture', 'mlp')
            latent_dim = getattr(cfg.model, 'latent_dim', 16)
            print(f"[Stage A] Using architecture: {arch}, latent_dim: {latent_dim}")
            input_dim = (cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1])
            train_path = cfg.data.train_path
            test_path = cfg.data.test_path

            if model_choice == 'rhvae':
                print("\n=== [Stage A] RHVAE training ===")
                from models.rhvae_experiment import RHVAEExperiment as _RHVAEExp
                # Initialize RHVAEExperiment
                rh_exp = _RHVAEExp(
                    input_dim=list(input_dim),
                    latent_dim=int(getattr(cfg.model, 'latent_dim', latent_dim)),
                    # Core HMC / RHVAE params (fallbacks preserve previous behavior)
                    n_lf=int(getattr(cfg.model, 'n_lf', 6)),
                    eps_lf=float(getattr(cfg.model, 'eps_lf', 2e-5)),
                    beta_zero=float(getattr(cfg.model, 'beta_zero', 1.0)),
                    temperature=float(getattr(cfg.model, 'temperature', getattr(cfg.metric, 'temperature', 0.7))),
                    regularization=float(getattr(cfg.model, 'regularization', getattr(cfg.metric, 'regularization', 1e-3))),
                    # Enc/Dec (optional hydra-style dicts)
                    encoder=getattr(cfg.model, 'encoder', None),
                    decoder=getattr(cfg.model, 'decoder', None),
                    # Advanced metric/weighting controls to match Hydra script behavior
                    align_with_knn_cov=bool(getattr(cfg.model, 'align_with_knn_cov', False)),
                    knn_k=int(getattr(cfg.model, 'knn_k', 300)),
                    alpha_align=float(getattr(cfg.model, 'alpha_align', 0.5)),
                    metric_normalization=str(getattr(cfg.model, 'metric_normalization', 'none')),
                    weight_kernel=str(getattr(cfg.model, 'weight_kernel', 'isotropic')),
                    weight_metric_normalization=str(getattr(cfg.model, 'weight_metric_normalization', 'trace')),
                    normalize_weight_sum=bool(getattr(cfg.model, 'normalize_weight_sum', False)),
                    topk_weights=getattr(cfg.model, 'topk_weights', None),
                    reestimate_metric_from_decoder_jacobian=bool(getattr(cfg.model, 'reestimate_metric_from_decoder_jacobian', False)),
                    jacobian_alpha=float(getattr(cfg.model, 'jacobian_alpha', 0.5)),
                    jacobian_h=float(getattr(cfg.model, 'jacobian_h', 1e-3)),
                    jacobian_stride=int(getattr(cfg.model, 'jacobian_stride', 4)),
                    metric_scale=float(getattr(cfg.model, 'metric_scale', 1.0)),
                    realign_centroids=bool(getattr(cfg.model, 'realign_centroids', False)),
                    centroid_realign_method=str(getattr(cfg.model, 'centroid_realign_method', 'kmeans')),
                    max_centroids=getattr(cfg.model, 'max_centroids', None),
                    centroid_subsample_method=str(getattr(cfg.model, 'centroid_subsample_method', 'fps')),
                    # System
                    device=str(self.device),
                    seed=int(getattr(cfg, 'seed', 42)),
                )
                # Use configured batch size for RHVAE Stage A
                rh_exp.load_data(train_path=train_path, test_path=test_path, batch_size=int(cfg.experiment.stage_a.batch_size))
                rh_exp.train(
                    epochs=int(cfg.experiment.stage_a.epochs),
                    learning_rate=1e-4,
                    weight_decay=1e-5,
                    log_every=50,
                    save_every=10,
                    output_dir=str(Path(cfg.checkpoint.stageA_dir)),
                    use_wandb=(cfg.wandb.mode != 'disabled'),
                    wandb_config={
                        "project": cfg.wandb.project,
                        "name": f"pipeline_stage1_rhvae_ld{latent_dim}",
                        "group": getattr(cfg.wandb, 'group', None),
                        "tags": ["pipeline", "stageA", "rhvae"],
                    },
                )
                # Stage A: Reconstruction vs Original (RHVAE path)
                try:
                    import torchvision.utils as vutils
                    # Load a small batch from test set
                    raw_test = torch.load(test_path, map_location='cpu')
                    if raw_test.ndim == 5:
                        # Use first timestep frames for clarity
                        imgs = raw_test[:8, 0].to(self.device)
                    elif raw_test.ndim == 4:
                        imgs = raw_test[:8].to(self.device)
                    else:
                        raise ValueError(f"Unsupported data shape for recon logging: {tuple(raw_test.shape)}")
                    rh_exp.model.eval()
                    with torch.no_grad():
                        enc = rh_exp.model.encoder(imgs)
                        z = enc.get('embedding', None) if isinstance(enc, dict) else enc
                        if z is None and isinstance(enc, dict):
                            z = enc.get('z', None)
                        dec = rh_exp.model.decoder(z)
                        recon = dec.get('reconstruction', None) if isinstance(dec, dict) else dec
                        if recon is None and isinstance(dec, dict):
                            recon = dec.get('recon_x', None)
                        recon = torch.sigmoid(recon) if recon.dtype.is_floating_point else recon
                        recon = recon.clamp(0, 1)
                        # Normalize for visualization only to avoid dim-looking outputs
                        grid = vutils.make_grid(
                            torch.cat([imgs, recon], dim=0),
                            nrow=8,
                            normalize=True,
                            value_range=(0.0, 1.0)
                        )
                        if wandb.run is not None:
                            wandb.log({"stageA/recon_vs_real": wandb.Image(grid)})
                except Exception as e:
                    print(f"[Stage A] ⚠️ RHVAE recon_vs_real logging failed: {e}")
                # Compute t=0 latents with the trained RHVAE encoder (to subset centroids later)
                try:
                    # Load raw training tensor and extract timestep 0 frames
                    from torch.utils.data import DataLoader as TorchLoader, TensorDataset
                    raw = torch.load(train_path, map_location='cpu')
                    if raw.ndim == 5:  # [N, S, C, H, W]
                        imgs0 = raw[:, 0]
                    elif raw.ndim == 4:  # [N, C, H, W]
                        imgs0 = raw
                    else:
                        raise ValueError(f"Unsupported data shape for t=0 extraction: {tuple(raw.shape)}")
                    ds0 = TensorDataset(imgs0)
                    dl0 = TorchLoader(ds0, batch_size=256, shuffle=False)
                    mu_list = []
                    rh_exp.model.eval()
                    with torch.no_grad():
                        for (xb,) in dl0:
                            xb = xb.to(self.device)
                            enc = rh_exp.model.encoder(xb)
                            if isinstance(enc, dict) and 'embedding' in enc:
                                mu_list.append(enc['embedding'].detach().cpu())
                            elif isinstance(enc, dict) and 'z' in enc:
                                mu_list.append(enc['z'].detach().cpu())
                            elif isinstance(enc, torch.Tensor):
                                mu_list.append(enc.detach().cpu())
                    t0_latents = torch.cat(mu_list, dim=0) if mu_list else None
                except Exception as e:
                    print(f"[Stage A] ⚠️ Could not compute t=0 latents: {e}")
                    t0_latents = None

                # Export metric to organized Stage A folder
                stageA_paths = get_stage_paths(cfg, 'A', 'RHVAE', arch, latent_dim)
                try:
                    centroids = rh_exp.metric_adapter.centroids_tens.detach().cpu()
                    M = rh_exp.metric_adapter.M_tens.detach().cpu()
                    # Save encoder/decoder and full model for reuse
                    enc_path = stageA_paths['encoder_path']
                    dec_path = stageA_paths['decoder_path']
                    model_path = stageA_paths['model_path']
                    try:
                        torch.save(rh_exp.model.encoder.state_dict(), enc_path)
                    except Exception:
                        enc_path = None
                    try:
                        torch.save(rh_exp.model.decoder.state_dict(), dec_path)
                    except Exception:
                        dec_path = None
                    try:
                        # Save a portable checkpoint via experiment helper
                        if hasattr(rh_exp, 'save_model'):
                            rh_exp.save_model(str(model_path))
                        else:
                            torch.save({'model_state_dict': rh_exp.model.state_dict()}, model_path)
                    except Exception:
                        model_path = None
                    payload = {
                        'centroids': centroids,
                        'metric_matrices': M,
                        'temperature': float(rh_exp.temperature),
                        'regularization': float(rh_exp.regularization),
                        'latent_dim': int(latent_dim),
                        't0_latents': t0_latents,
                        'encoder_path': str(enc_path) if enc_path is not None else None,
                        'decoder_path': str(dec_path) if dec_path is not None else None,
                        'model_path': str(model_path) if model_path is not None else None,
                    }
                    torch.save(payload, stageA_paths['metric_path'])
                    print(f"[Stage A] ✅ Saved RHVAE metric to {stageA_paths['metric_path']}")
                    
                    # Save Stage A configuration
                    stageA_config = {
                        'stage': 'A',
                        'model_type': 'RHVAE',
                        'architecture': arch,
                        'latent_dim': latent_dim,
                        'epochs': cfg.experiment.stage_a.epochs,
                        'temperature': float(rh_exp.temperature),
                        'regularization': float(rh_exp.regularization),
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(stageA_paths['config_path'], 'w') as f:
                        yaml.dump(stageA_config, f)
                    print(f"[Stage A] ✅ Saved Stage A config to {stageA_paths['config_path']}")
                except Exception as e:
                    print(f"[Stage A] ⚠️ Could not export RHVAE metric: {e}")
                # Propagate component paths for Stage B/C
                comp_paths = {}
                if enc_path is not None:
                    comp_paths['encoder'] = str(enc_path)
                if dec_path is not None:
                    comp_paths['decoder'] = str(dec_path)
                # Also expose the live model for Stage B recon logging
                stageB_model = getattr(rh_exp, 'model', None)
                # keep single run open
            else:
                print("\n=== [Stage A] Warm VAE training ===")
                print(f"[Stage A] Configuration:")
                print(f"  - Architecture: {arch}")
                print(f"  - Latent dim: {latent_dim}")
                print(f"  - Epochs: {cfg.experiment.stage_a.epochs}")
                print(f"  - Batch size: {cfg.experiment.stage_a.batch_size}")
                print(f"  - Learning rate: {cfg.experiment.stage_a.lr}")
                
                # Build dataset: flatten sequences
                ds_train = SpritesDataset(train_path, normalize=False, timestep_only=None)
                ds_test = SpritesDataset(test_path, normalize=False, timestep_only=None)
                loader = DataLoader(torch.utils.data.ConcatDataset([ds_train, ds_test]), batch_size=cfg.experiment.stage_a.batch_size, shuffle=True)
                vanilla = create_vanilla(arch, input_dim=input_dim, latent_dim=latent_dim).to(self.device)
                optim = torch.optim.Adam(vanilla.parameters(), lr=cfg.experiment.stage_a.lr, weight_decay=1e-5)
                # use single run; no extra wandb.init here
                for epoch in range(cfg.experiment.stage_a.epochs):
                    vanilla.train()
                    total = 0.0
                    for batch in loader:
                        batch = batch.to(self.device)
                        if arch.lower() in ["mlp", "pythae"]:
                            out = vanilla({"data": batch}); loss = out.loss
                        else:
                            out = vanilla(batch); loss = out.loss
                        optim.zero_grad(); loss.backward(); optim.step(); total += loss.item()
                    if wandb.run is not None:
                        wandb.log({"stageA/train_loss": total/len(loader), "stageA/epoch": epoch+1})

                    # Every 10 epochs, log PCA of latent means at t=0 to monitor clustering
                    try:
                        if (epoch + 1) % 10 == 0 or (epoch + 1) == int(cfg.experiment.stage_a.epochs):
                            from scripts.train_diverse_metric_vae import SpritesDataset as _SD
                            from torch.utils.data import DataLoader as _DL
                            import matplotlib.pyplot as plt
                            # Build t=0 dataset (train only to save time)
                            ds0 = _SD(train_path, normalize=False, timestep_only=0)
                            dl0 = _DL(ds0, batch_size=256, shuffle=False)
                            vanilla.eval()
                            mus = []
                            with torch.no_grad():
                                for xb in dl0:
                                    xb = xb.to(self.device)
                                    if arch.lower() in ["mlp", "pythae"]:
                                        enc = vanilla.encoder(xb); mu = enc.embedding
                                    else:
                                        mu, _ = vanilla.encode(xb)
                                    mus.append(mu.detach().cpu())
                                    if sum(t.shape[0] for t in mus) >= 6000:
                                        break
                            if mus:
                                Z = torch.cat(mus, dim=0)
                                Zc = Z - Z.mean(dim=0, keepdim=True)
                                U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
                                comp = Vh[:2].T
                                proj = (Zc @ comp).numpy()
                                plt.figure(figsize=(6,5))
                                plt.scatter(proj[:,0], proj[:,1], s=4, alpha=0.35, c='tab:blue')
                                plt.title(f'Stage A: Latent μ PCA(2) — epoch {epoch+1}')
                                plt.xlabel('PC1'); plt.ylabel('PC2'); plt.tight_layout()
                                if wandb.run is not None:
                                    wandb.log({"stageA/latent_pca_t0": wandb.Image(plt.gcf()), "stageA/epoch": epoch+1})
                                plt.close()
                    except Exception as e:
                        print(f"[Stage A] ⚠️ Periodic PCA failed (epoch {epoch+1}): {e}")
                # Reconstructions vs real for Stage A (vanilla path)
                try:
                    import torchvision.utils as vutils
                    vanilla.eval()
                    # Sample random, independent sprites (not a temporal sequence)
                    sample_loader = DataLoader(
                        torch.utils.data.ConcatDataset([ds_train, ds_test]),
                        batch_size=8,
                        shuffle=True
                    )
                    batch = next(iter(sample_loader)).to(self.device)
                    with torch.no_grad():
                        if arch.lower() in ["mlp", "pythae"]:
                            out = vanilla({"data": batch}); recon = out.recon_x.clamp(0, 1)
                        else:
                            out = vanilla(batch); recon = out.recon_x.clamp(0, 1)
                    grid = vutils.make_grid(
                        torch.cat([batch, recon], dim=0),
                        nrow=8,
                        normalize=False
                    )
                    if wandb.run is not None:
                        wandb.log({"stageA/recon_vs_real": wandb.Image(grid)})
                except Exception as e:
                    print(f"[Stage A] ⚠️ Recon vs real logging failed: {e}")

                # Stage A latent space PCA visualization (μ embeddings)
                try:
                    import matplotlib.pyplot as plt
                    from torch.utils.data import DataLoader as TorchLoader
                    from torch.utils.data import ConcatDataset as TorchConcat
                    vanilla.eval()
                    full_loader = TorchLoader(TorchConcat([ds_train, ds_test]), batch_size=256, shuffle=False)
                    mus = []
                    with torch.no_grad():
                        for xb in full_loader:
                            xb = xb.to(self.device)
                            if arch.lower() in ["mlp", "pythae"]:
                                enc = vanilla.encoder(xb)
                                mu = enc.embedding
                            else:
                                mu, _ = vanilla.encode(xb)
                            mus.append(mu.detach().cpu())
                            if sum(t.shape[0] for t in mus) >= 5000:
                                break
                    if mus:
                        Z = torch.cat(mus, dim=0)
                        Zc = Z - Z.mean(dim=0, keepdim=True)
                        # Torch PCA via SVD
                        U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
                        comp = Vh[:2].T  # [D,2]
                        proj = (Zc @ comp).numpy()
                        plt.figure(figsize=(6,5))
                        plt.scatter(proj[:,0], proj[:,1], s=4, alpha=0.4, c='tab:blue')
                        plt.title('Stage A: Latent μ PCA(2)')
                        plt.xlabel('PC1'); plt.ylabel('PC2'); plt.tight_layout()
                        if wandb.run is not None:
                            wandb.log({"stageA/latent_pca": wandb.Image(plt.gcf())})
                        plt.close()
                except Exception as e:
                    print(f"[Stage A] ⚠️ Latent PCA visualization failed: {e}")
                # Save vanilla VAE to organized Stage A folder
                stageA_paths = get_stage_paths(cfg, 'A', 'VANILLA', arch, latent_dim)
                comp_paths = save_model_components(vanilla, arch, latent_dim, save_dir=str(stageA_paths['base_dir']))
                torch.save(vanilla.state_dict(), stageA_paths['model_path'])
                
                # Save Stage A configuration
                stageA_config = {
                    'stage': 'A',
                    'model_type': 'VANILLA',
                    'architecture': arch,
                    'latent_dim': latent_dim,
                    'epochs': cfg.experiment.stage_a.epochs,
                    'lr': cfg.experiment.stage_a.lr,
                    'beta': cfg.experiment.stage_a.beta,
                    'timestamp': datetime.now().isoformat()
                }
                with open(stageA_paths['config_path'], 'w') as f:
                    yaml.dump(stageA_config, f)
                print(f"[Stage A] ✅ Saved Stage A config to {stageA_paths['config_path']}")
                # Log Stage A artifacts (encoder/decoder/model) to WandB for pipeline chaining
                if cfg.wandb.mode != "disabled" and wandb.run is not None and getattr(cfg.wandb, 'artifacts', {}).get('enabled', False):
                    try:
                        art = wandb.Artifact(
                            name=f"stageA_vae_{arch}_ld{latent_dim}",
                            type="model",
                            metadata={"stage": "A", "architecture": arch, "latent_dim": latent_dim}
                        )
                        if 'encoder' in comp_paths: art.add_file(comp_paths['encoder'])
                        if 'decoder' in comp_paths: art.add_file(comp_paths['decoder'])
                        if 'model' in comp_paths:   art.add_file(comp_paths['model'])
                        aliases = [getattr(cfg.wandb.artifacts.aliases, 'stage_a_latest', 'stageA_latest')]
                        wandb.log_artifact(art, aliases=aliases)
                    except Exception as e:
                        print(f"[Stage A] ⚠️ Artifact logging failed: {e}")
                    # keep single run open
        else:
            comp_paths = {
                'encoder': cfg.pretrained.encoder_path,
                'decoder': cfg.pretrained.decoder_path
            }

        # Stage B: Metric learning at t=0
        if getattr(cfg.experiment, 'run_stage_b', True):
            print("\n=== [Stage B] Metric learning at t=0 ===")
            arch = cfg.model.encoder.architecture
            latent_dim = cfg.model.latent_dim
            metric_impl = cfg.experiment.stage_b.implementation
            
            # Get organized Stage B paths
            stageB_paths = get_stage_paths(cfg, 'B', metric_impl.upper(), arch, latent_dim)
            
            # Try to find Stage A data automatically
            stage_a_data = find_stage_a_data(cfg, arch, latent_dim)
            
            # Fix the metric_file reference for WandB logging
            metric_file = stageB_paths['metric_path']
            reused_stageA_metric = False
            
            print(f"\n=== [Stage B] COMPONENT LOADING DEBUG ===")
            print(f"[Stage B] Architecture: {arch}, Latent dim: {latent_dim}")
            print(f"[Stage B] Metric implementation: {metric_impl}")
            print(f"[Stage B] Stage A data found: {stage_a_data is not None}")
            
            if stage_a_data:
                print(f"[Stage B] Stage A paths:")
                print(f"  - Base dir: {stage_a_data['base_dir']}")
                print(f"  - Encoder: {stage_a_data['encoder_path']}")
                print(f"  - Decoder: {stage_a_data['decoder_path']}")
                print(f"  - Config: {stage_a_data['config_path']}")
                if stage_a_data.get('metric_path'):
                    print(f"  - Metric: {stage_a_data['metric_path']}")
                
                # Verify architecture and latent_dim consistency with Stage A
                print(f"[Stage B] Architecture/Latent Dim Verification:")
                try:
                    with open(stage_a_data['config_path'], 'r') as f:
                        stage_a_config = yaml.safe_load(f)
                    stage_a_arch = stage_a_config.get('architecture', 'unknown')
                    stage_a_ld = stage_a_config.get('latent_dim', 'unknown')
                    print(f"  - Stage A: arch={stage_a_arch}, latent_dim={stage_a_ld}")
                    print(f"  - Stage B: arch={arch}, latent_dim={latent_dim}")
                    if stage_a_arch != arch or stage_a_ld != latent_dim:
                        print(f"  - ⚠️ MISMATCH DETECTED! Stage A and Stage B have different architecture/latent_dim!")
                        print(f"  - Forcing Stage B to use Stage A parameters...")
                        # Force Stage B to use Stage A's architecture and latent_dim
                        arch = stage_a_arch
                        latent_dim = stage_a_ld
                        print(f"  - ✅ Updated Stage B to use arch={arch}, latent_dim={latent_dim}")
                    else:
                        print(f"  - ✅ Architecture and latent_dim match between Stage A and Stage B")
                except Exception as e:
                    print(f"  - ⚠️ Could not verify Stage A config: {e}")
            print(f"=== [Stage B] END COMPONENT DEBUG ===\n")
            
            if stage_a_data is not None and metric_impl == 'rhvae':
                try:
                    _st = torch.load(stage_a_data['metric_path'], map_location='cpu', weights_only=False)
                    if _st is not None and 'centroids' in _st and ('metric_matrices' in _st or 'M_matrices' in _st or 'inverse_metrics' in _st):
                        print(f"[Stage B] Reusing RHVAE metric from Stage A: {stage_a_data['metric_path']}")
                        metric_path = str(stage_a_data['metric_path'])
                        reused_stageA_metric = True
                    else:
                        print("[Stage B] Existing metric file is incompatible; re-extracting.")
                except Exception as _e:
                    print(f"[Stage B] Could not reuse Stage A metric ({_e}); re-extracting.")
            if metric_impl == 'rhvae' and not reused_stageA_metric:
                # Load model from Stage A if available
                if stage_a_data is not None:
                    print(f"[Stage B] ✅ Loading model from Stage A:")
                    print(f"  - Encoder path: {stage_a_data['encoder_path']}")
                    print(f"  - Decoder path: {stage_a_data['decoder_path']}")
                    stageB_model = create_vanilla(arch, input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]), latent_dim=latent_dim).to(self.device)
                    stageB_model.encoder.load_state_dict(torch.load(stage_a_data['encoder_path'], map_location=self.device, weights_only=False))
                    stageB_model.decoder.load_state_dict(torch.load(stage_a_data['decoder_path'], map_location=self.device, weights_only=False))
                    print(f"[Stage B] ✅ Successfully loaded Stage A encoder/decoder")
                else:
                    print(f"[Stage B] ⚠️ No Stage A data found, using fallback model")
                    # Fallback to existing model
                    stageB_model = (
                        rh_exp.model if 'rh_exp' in locals() and hasattr(rh_exp, 'model') else
                        (vanilla if 'vanilla' in locals() else create_vanilla(arch, input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]), latent_dim=latent_dim).to(self.device))
                    )
                
                metric_path = extract_diverse_metric(
                    model=stageB_model,
                    architecture=arch,
                    latent_dim=latent_dim,
                    temperature=cfg.experiment.stage_b.temperature,
                    regularization=cfg.experiment.stage_b.regularization,
                    num_centroids=cfg.experiment.stage_b.n_centroids,
                    save_dir=str(stageB_paths['base_dir']),
                    input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]),
                    data_path=cfg.data.train_path,
                    timestep_only=cfg.experiment.stage_b.use_timestep,
                    standardize_latents=cfg.experiment.stage_b.standardize_latents,
                    centroid_method=cfg.experiment.stage_b.centroid_method,
                    neighbor_mode=cfg.experiment.stage_b.neighbor_mode,
                    knn_k=cfg.experiment.stage_b.knn_k,
                )
            elif metric_impl == 'rhvae' and reused_stageA_metric:
                # When reusing Stage A metric, we need to copy it to Stage B location
                metric_path = str(stage_a_data['metric_path'])
                print(f"[Stage B] ✅ Reusing Stage A metric: {metric_path}")
            elif metric_impl == 'precision':
                # Load model from Stage A if available
                if stage_a_data is not None:
                    print(f"[Stage B] Loading model from Stage A for precision metric: {stage_a_data['encoder_path']}")
                    stageB_model = create_vanilla(arch, input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]), latent_dim=latent_dim).to(self.device)
                    stageB_model.encoder.load_state_dict(torch.load(stage_a_data['encoder_path'], map_location=self.device, weights_only=False))
                    stageB_model.decoder.load_state_dict(torch.load(stage_a_data['decoder_path'], map_location=self.device, weights_only=False))
                else:
                    stageB_model = vanilla if 'vanilla' in locals() else create_vanilla(arch, input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]), latent_dim=latent_dim).to(self.device)
                
                # Precision metric from posterior: reuse extraction with local KNN covariance -> invert
                metric_path = extract_diverse_metric(
                    model=stageB_model,
                    architecture=arch,
                    latent_dim=latent_dim,
                    temperature=cfg.experiment.stage_b.temperature,
                    regularization=cfg.experiment.stage_b.regularization,
                    num_centroids=cfg.experiment.stage_b.n_centroids,
                    save_dir=str(stageB_paths['base_dir']),
                    input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]),
                    data_path=cfg.data.train_path,
                    timestep_only=cfg.experiment.stage_b.use_timestep,
                    standardize_latents=cfg.experiment.stage_b.standardize_latents,
                    centroid_method='kmeans',
                    neighbor_mode='knn',
                    knn_k=max(100, cfg.experiment.stage_b.knn_k),
                    normalize_M='none',
                )
                # Convert saved covariance matrices to precision (G^{-1}) for Stage C compatibility
                try:
                    state = torch.load(metric_path, map_location=self.device, weights_only=False)
                    if 'M_matrices' in state and 'inverse_metrics' not in state:
                        cov = state['M_matrices']
                        precision = torch.linalg.inv(cov)
                        state['M_matrices'] = precision
                        state['extraction_method'] = str(state.get('extraction_method', '')) + '_precision'
                        torch.save(state, metric_path)
                        print("[Stage B] Converted covariance matrices to precision (G^{-1}) for precision metric.")
                except Exception as e:
                    print(f"[Stage B] ⚠️ Precision conversion failed: {e}")
            else:
                raise ValueError(f"Unknown metric implementation: {metric_impl}")
            # Copy metric to organized Stage B location
            if Path(metric_path) != stageB_paths['metric_path']:
                try:
                    import shutil
                    # Load the metric data to ensure we preserve all fields including t0_latents
                    metric_data = torch.load(metric_path, map_location='cpu', weights_only=False)
                    # Save to Stage B location with all data preserved
                    torch.save(metric_data, stageB_paths['metric_path'])
                    metric_path = str(stageB_paths['metric_path'])
                    print(f"[Stage B] ✅ Copied metric with all data to: {metric_path}")
                except Exception as e:
                    print(f"[Stage B] ⚠️ Could not copy metric to organized location: {e}")
                    # Fallback to direct file copy
                    try:
                        shutil.copyfile(metric_path, stageB_paths['metric_path'])
                        metric_path = str(stageB_paths['metric_path'])
                    except Exception as e2:
                        print(f"[Stage B] ⚠️ Direct file copy also failed: {e2}")
                except Exception as e:
                    print(f"[Stage B] ⚠️ Could not copy metric to organized location: {e}")
            
            # Save Stage B configuration
            stageB_config = {
                'stage': 'B',
                'model_type': metric_impl.upper(),
                'architecture': arch,
                'latent_dim': latent_dim,
                'temperature': cfg.experiment.stage_b.temperature,
                'regularization': cfg.experiment.stage_b.regularization,
                'n_centroids': cfg.experiment.stage_b.n_centroids,
                'centroid_method': cfg.experiment.stage_b.centroid_method,
                'neighbor_mode': cfg.experiment.stage_b.neighbor_mode,
                'knn_k': cfg.experiment.stage_b.knn_k,
                'use_timestep': cfg.experiment.stage_b.use_timestep,
                'standardize_latents': cfg.experiment.stage_b.standardize_latents,
                'stage_a_source': str(stage_a_data['base_dir']) if stage_a_data else None,
                'timestamp': datetime.now().isoformat()
            }
            with open(stageB_paths['config_path'], 'w') as f:
                yaml.dump(stageB_config, f)
            print(f"[Stage B] ✅ Saved Stage B config to {stageB_paths['config_path']}")
            print(f"[Stage B] ✅ Saved metric checkpoint: {stageB_paths['metric_path']}")
            # Stage B basic visuals (eigenvalue/condition/heatmaps) for quick verification
            if cfg.wandb.mode != "disabled":
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    plt.style.use('default')
                    sns.set_palette("husl")
                    state = torch.load(metric_path, map_location='cpu', weights_only=False)
                    M_matrices = state.get('M_matrices')
                    if M_matrices is None:
                        M_matrices = state.get('metric_matrices')
                    centroids = state.get('centroids')
                    if M_matrices is not None and centroids is not None:
                        eigenvals = torch.linalg.eigvals(M_matrices).real
                        min_eigenvals = eigenvals.min(dim=-1)[0]
                        max_eigenvals = eigenvals.max(dim=-1)[0]
                        mean_eigenvals = eigenvals.mean(dim=-1)
                        cond_nums = max_eigenvals / (min_eigenvals + 1e-12)
                        determinants = torch.linalg.det(M_matrices)
                        # 1) Eigenvalue distributions
                        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                        fig.suptitle('Stage B Metric — Eigenvalue & Condition Analysis', fontsize=16, fontweight='bold')
                        axes[0,0].hist(min_eigenvals.numpy(), bins=40, color='red', edgecolor='black'); axes[0,0].set_title('Min Eigenvalue')
                        axes[0,1].hist(max_eigenvals.numpy(), bins=40, color='blue', edgecolor='black'); axes[0,1].set_title('Max Eigenvalue')
                        axes[0,2].hist(mean_eigenvals.numpy(), bins=40, color='green', edgecolor='black'); axes[0,2].set_title('Mean Eigenvalue')
                        axes[1,0].hist(cond_nums.numpy(), bins=40, color='orange', edgecolor='black'); axes[1,0].set_title('Condition Number')
                        log_dets = torch.log10(torch.abs(determinants) + 1e-50)
                        axes[1,1].hist(log_dets.numpy(), bins=40, color='purple', edgecolor='black'); axes[1,1].set_title('Log₁₀|Determinant|')
                        eig_spread = (max_eigenvals - min_eigenvals).numpy()
                        axes[1,2].hist(eig_spread, bins=40, color='cyan', edgecolor='black'); axes[1,2].set_title('Eigenvalue Spread')
                        plt.tight_layout()
                        wandb.log({"stageB/metric_eigen_diagnostics": wandb.Image(fig)})
                        plt.close(fig)
                        # 2) Heatmaps
                        n_heatmaps = min(6, len(M_matrices))
                        fig, axes = plt.subplots(1, n_heatmaps, figsize=(4*n_heatmaps, 4))
                        for i in range(n_heatmaps):
                            im = axes[i].imshow(M_matrices[i].numpy(), cmap='RdYlBu_r', aspect='auto')
                            axes[i].set_title(f'Matrix {i}')
                            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                        plt.tight_layout()
                        wandb.log({"stageB/metric_heatmaps": wandb.Image(fig)})
                        plt.close(fig)
                        # 3) Centroid stats
                        centroid_norms = torch.norm(centroids, dim=1)
                        pairwise_dists = torch.cdist(centroids, centroids)
                        triu_mask = torch.triu(torch.ones_like(pairwise_dists, dtype=bool), diagonal=1)
                        pairwise_vals = pairwise_dists[triu_mask]
                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                        axes[0].hist(centroid_norms.numpy(), bins=30, color='red', edgecolor='black'); axes[0].set_title('Centroid Norms')
                        axes[1].hist(pairwise_vals.numpy(), bins=40, color='blue', edgecolor='black'); axes[1].set_title('Pairwise Distances')
                        plt.tight_layout()
                        wandb.log({"stageB/centroid_stats": wandb.Image(fig)})
                        plt.close(fig)
                except Exception as e:
                    print(f"[Stage B] ⚠️ Visual diagnostics failed: {e}")

            # Stage B PCA(2) heatmap of det(G^{-1}) in latent subspace with overlays
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                from rlvae.models.components.metric_loader import MetricLoader
                loader = MetricLoader(device=self.device)
                blob = loader.load_from_file(metric_path, cfg.experiment.stage_b.temperature, cfg.experiment.stage_b.regularization)
                C = blob['centroids'].to(self.device)
                M = blob['metric_matrices'].to(self.device)
                # Use cached t0 latents if available
                state = torch.load(metric_path, map_location='cpu', weights_only=False)
                Z0 = state.get('t0_latents', None)
                if Z0 is None:
                    print("[Stage B] No t0_latents cached; sampling 4000 points around centroids for PCA basis")
                    Z0 = C.detach().cpu()
                else:
                    Z0 = Z0.detach().cpu()
                # PCA basis
                Zc = Z0 - Z0.mean(dim=0, keepdim=True)
                U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
                comp = Vh[:2].T  # [D,2]
                mean = Z0.mean(dim=0)
                # Grid in PCA space
                Proj = (Zc @ comp).numpy()
                x_min, x_max = np.percentile(Proj[:,0], [1, 99])
                y_min, y_max = np.percentile(Proj[:,1], [1, 99])
                xs = np.linspace(x_min, x_max, 100)
                ys = np.linspace(y_min, y_max, 100)
                XX, YY = np.meshgrid(xs, ys)
                pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
                # Back to latent space: z = mean + comp @ p
                comp_t = comp
                Zgrid = mean.unsqueeze(0) + torch.from_numpy(pts).to(mean.dtype) @ comp_t.T
                Zgrid = Zgrid.to(self.device)
                # Compute det(G^{-1}) using MetricTensor rebuilt from loaded data
                from rlvae.models.components.metric_tensor import MetricTensor as _MetricTensor
                mt = _MetricTensor(latent_dim=Zgrid.shape[1], device=self.device)
                mt.load_pretrained(C, M, blob['temperature'], blob['regularization'])
                Ginvg = mt.compute_inverse_metric(Zgrid)
                _, logdet = torch.linalg.slogdet(Ginvg)
                det_vals = torch.exp(logdet).detach().cpu().numpy().reshape(XX.shape)
                # RHMC samples overlay if available
                rhmc_pts = None
                try:
                    from src.models.samplers.hmc_sampler import RHVAEVolumeElementHMCSampler
                    sampler = RHVAEVolumeElementHMCSampler(mt, device=self.device)
                    rhmc = sampler.sample(num_samples=1000)
                    rhmc_centered = (rhmc.detach().cpu() - mean)
                    rhmc_proj = (rhmc_centered @ comp).numpy()
                    rhmc_pts = rhmc_proj
                except Exception:
                    pass
                # Build figure
                plt.figure(figsize=(7,6))
                plt.imshow(det_vals, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='viridis', aspect='auto')
                # overlay t0 latents and centroids
                P0 = ( (Z0 - mean) @ comp ).numpy()
                plt.scatter(P0[:,0], P0[:,1], s=3, c='white', alpha=0.15, label='t=0 latents')
                Pc = ( (C.detach().cpu() - mean) @ comp ).numpy()
                plt.scatter(Pc[:,0], Pc[:,1], s=40, c='red', marker='*', label='centroids')
                if rhmc_pts is not None:
                    plt.scatter(rhmc_pts[:,0], rhmc_pts[:,1], s=3, c='deepskyblue', alpha=0.3, label='RHMC samples')
                plt.colorbar(label='det(G^{-1}) (PCA subspace)')
                plt.title('Stage B: det(G^{-1}) in PCA(2) subspace (t=0)')
                plt.legend(loc='upper right', fontsize=8)
                plt.tight_layout()
                if wandb.run is not None:
                    wandb.log({"stageB/pca_det_heatmap": wandb.Image(plt.gcf())})
                plt.close()
            except Exception as e:
                print(f"[Stage B] ⚠️ PCA det(G^-1) heatmap failed: {e}")

            # Stage B WandB logging (artifact)
            if cfg.wandb.mode != "disabled":
                try:
                    state = torch.load(metric_path, map_location='cpu', weights_only=False)
                    centroids = state.get('centroids')
                    # Log metric artifact for reuse in Stage C or future runs
                    try:
                        art = wandb.Artifact(
                            name=f"stageB_metric_{arch}_ld{latent_dim}",
                            type="metric",
                            metadata={"stage": "B", "architecture": arch, "latent_dim": latent_dim, "implementation": metric_impl}
                        )
                        art.add_file(str(metric_path))
                        aliases = [getattr(cfg.wandb.artifacts.aliases, 'stage_b_latest', 'stageB_latest')]
                        wandb.log_artifact(art, aliases=aliases)
                    except Exception as eart:
                        print(f"[Stage B] ⚠️ Artifact logging failed: {eart}")
                    # keep run open
                except Exception as e:
                    print(f"[Stage B] ⚠️ WandB logging failed: {e}")
        else:
            # Only check metric_file if it's defined (i.e., if we're in Stage B context)
            if 'metric_file' in locals():
                assert metric_file.exists(), f"Metric file not found: {metric_file}"

        # RHMC sampling at t=0 (use volume-element sampler)
        if getattr(cfg.experiment, 'run_sampling', True) and cfg.sampling.enabled:
            print("\n=== RHMC sampling at t=0 ===")
            # Define arch and latent_dim for sampling if not already defined
            if 'arch' not in locals():
                arch = getattr(cfg.model.encoder, 'architecture', 'mlp')
            if 'latent_dim' not in locals():
                latent_dim = getattr(cfg.model, 'latent_dim', 16)
            if 'metric_impl' not in locals():
                metric_impl = getattr(cfg.experiment.stage_b, 'implementation', 'rhvae')
            
            # Reuse the unified sampler from scripts tester for reliability
            from src.models.components.metric_loader import MetricLoader
            from src.models.samplers.hmc_sampler import RHVAEVolumeElementHMCSampler
            loader = MetricLoader(device=self.device)
            # Use Stage B metric if available, otherwise fallback
            stage_b_data_for_sampling = find_stage_b_data(cfg, arch, latent_dim, metric_impl.upper())
            # Define metric_file for sampling if not already defined
            if 'metric_file' not in locals():
                metric_file = stage_b_data_for_sampling['metric_path'] if stage_b_data_for_sampling else None
            metric_path_for_sampling = stage_b_data_for_sampling['metric_path'] if stage_b_data_for_sampling else metric_file
            blob = loader.load_from_file(str(metric_path_for_sampling), cfg.experiment.stage_b.temperature, cfg.experiment.stage_b.regularization)
            # Filter centroids to those that are actually used at timestep 0 (as in RHVAE, but restricted to t=0)
            C_all = blob['centroids'].to(self.device)
            M_all = blob['metric_matrices'].to(self.device)
            C_use, M_use = C_all, M_all
            # Prefer precomputed t=0 latents from Stage A payload for deterministic filtering
            try:
                state = torch.load(metric_path_for_sampling, map_location='cpu', weights_only=False)
                mu0 = state.get('t0_latents', None)
                if mu0 is not None:
                    mu0 = mu0.to(self.device)
                    d2 = torch.cdist(mu0, C_all)  # [N0, K]
                    winners = torch.argmin(d2, dim=1)
                    used_ids = torch.unique(winners)
                    C_use = C_all[used_ids]
                    M_use = M_all[used_ids]
                    print(f"[Stage B] Using {C_use.shape[0]} centroids relevant to t=0 (from {C_all.shape[0]} total).")
                else:
                    print("[Stage B] No t=0 latents in metric payload; using all centroids.")
            except Exception as e:
                print(f"[Stage B] ⚠️ t=0 centroid filtering skipped: {e}")
            class _MetricStub:
                def __init__(
                    self,
                    centroids,
                    M,
                    T,
                    lbd,
                    device,
                    normalize_weight_sum=False,
                    weight_kernel: str = 'mahalanobis_normed',
                    weight_metric_normalization: str = 'trace',
                    topk_weights: int | None = None,
                    metric_scale: float = 1.0,
                ):
                    self.centroids_tens = centroids.to(device)
                    self.M_raw = M.to(device)
                    self.temperature = float(T)
                    self.regularization = float(lbd)
                    self.device = device
                    self.normalize_weight_sum = bool(normalize_weight_sum)
                    self.weight_kernel = (weight_kernel or 'mahalanobis_normed').lower()
                    self.weight_metric_normalization = (weight_metric_normalization or 'trace').lower()
                    self.topk_weights = int(topk_weights) if topk_weights is not None else None
                    self.metric_scale = float(metric_scale)

                    # Precompute normalized metric matrices per chosen normalization
                    if self.weight_metric_normalization == 'trace':
                        traces = torch.einsum('kii->k', self.M_raw).unsqueeze(-1).unsqueeze(-1) + 1e-12
                        self.M_normed = self.M_raw / traces
                    else:
                        self.M_normed = self.M_raw
                    self.M_tens = self.metric_scale * self.M_normed
                # Provide a torch-like parameters iterator for sampler device detection
                def parameters(self):
                    return iter([torch.empty(0, device=self.device)])
                def _weights_and_mask(self, z):
                    z = z.to(self.device)
                    diff = z.unsqueeze(1) - self.centroids_tens.unsqueeze(0)  # [B, K, D]
                    if self.weight_kernel == 'isotropic':
                        d2 = torch.sum(diff * diff, dim=-1) / (self.temperature ** 2)
                    else:
                        # Mahalanobis variants
                        # Using normalized matrices if requested
                        tmp = torch.einsum('bkd,kde->bke', diff, self.M_normed)
                        d2 = torch.sum(tmp * diff, dim=-1) / (self.temperature ** 2)
                    # Top-k selection on distances (keep nearest centroids)
                    if self.topk_weights is not None and self.topk_weights > 0 and self.topk_weights < d2.shape[1]:
                        vals, idx = torch.topk(d2, k=self.topk_weights, dim=1, largest=False)
                        mask = torch.full_like(d2, fill_value=float('inf'))
                        mask.scatter_(1, idx, vals)
                        d2 = mask
                    w = torch.exp(-d2)
                    if self.normalize_weight_sum:
                        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
                    return w
                def G(self, z):
                    # Regularized inverse and matrix inverse to get G
                    w = self._weights_and_mask(z)
                    Ginv = torch.einsum('bk,kij->bij', w, self.M_tens)
                    eye = torch.eye(Ginv.shape[-1], device=self.device).unsqueeze(0).expand(Ginv.shape[0], -1, -1)
                    Ginv = Ginv + self.regularization * eye
                    return torch.linalg.inv(Ginv)
                def G_inv(self, z):
                    w = self._weights_and_mask(z)
                    Ginv = torch.einsum('bk,kij->bij', w, self.M_tens)
                    eye = torch.eye(Ginv.shape[-1], device=self.device).unsqueeze(0).expand(Ginv.shape[0], -1, -1)
                    return Ginv + self.regularization * eye
            # Match Stage A behavior if available (fallback False)
            model_cfg = getattr(cfg, 'model', None)
            normalize_w = bool(getattr(model_cfg, 'normalize_weight_sum', False)) if model_cfg else False
            weight_kernel = str(getattr(model_cfg, 'weight_kernel', 'mahalanobis_normed')) if model_cfg else 'mahalanobis_normed'
            weight_metric_norm = str(getattr(model_cfg, 'weight_metric_normalization', 'trace')) if model_cfg else 'trace'
            topk_weights = getattr(model_cfg, 'topk_weights', None) if model_cfg else None
            metric_scale = float(getattr(model_cfg, 'metric_scale', 1.0)) if model_cfg else 1.0
            metric_model = _MetricStub(
                C_use,
                M_use,
                blob['temperature'],
                blob['regularization'],
                self.device,
                normalize_weight_sum=normalize_w,
                weight_kernel=weight_kernel,
                weight_metric_normalization=weight_metric_norm,
                topk_weights=topk_weights,
                metric_scale=metric_scale,
            )
            # Tune RHMC for better acceptance on Stage B diagnostic overlay
            step_size = float(getattr(cfg.sampling, 'step_size', 0.001)) if hasattr(cfg, 'sampling') else 0.001
            tuned_eps = min(0.005, max(1e-5, step_size))
            rhmc = RHVAEVolumeElementHMCSampler(
                metric_model,
                mcmc_steps_nbr=max(100, cfg.sampling.n_steps),
                n_lf=max(20, cfg.sampling.n_leapfrog),
                eps_lf=tuned_eps,
                beta_zero=1.0
            )
            samples = rhmc.sample(n_samples=1024)
            torch.save({'samples': samples.detach().cpu()}, Path(cfg.checkpoint.metric_dir) / cfg.checkpoint.rhmc_samples)
            print(f"[Sampling] Saved RHMC samples to {Path(cfg.checkpoint.metric_dir) / cfg.checkpoint.rhmc_samples}")
            # (Disabled raw z overlay by request; PCA overlays logged below)

            # PCA(2) visualization on manifold t=0 with RHMC overlay and reconstructions
            try:
                import numpy as np
                import matplotlib.pyplot as plt
                from sklearn.decomposition import PCA
                import torchvision.utils as vutils

                # Load latent references from metric file if available
                state = torch.load(metric_file, map_location='cpu', weights_only=False)
                Z_ref = state.get('t0_latents', None)
                pca_fit_basis = 'mu0' if Z_ref is not None else None
                if Z_ref is None:
                    Z_ref = state.get('z_sample', None)
                    pca_fit_basis = 'z_sample' if Z_ref is not None else None
                if Z_ref is None:
                    # Fallback to centroids for PCA fit
                    Z_ref = C_use.detach().cpu()
                    pca_fit_basis = 'centroids'
                else:
                    Z_ref = Z_ref.detach().cpu()
                Z_ref_np = Z_ref.numpy()
                # Fit PCA(2) on t=0 latent references
                pca = PCA(n_components=2, random_state=42)
                Z_ref_pca = pca.fit_transform(Z_ref_np)
                U = pca.components_  # [2, D]
                mean_np = pca.mean_   # [D]
                U_d2 = torch.tensor(U.T, device=self.device, dtype=blob['centroids'].dtype)
                mean_t = torch.tensor(mean_np, device=self.device, dtype=blob['centroids'].dtype)

                # Build grid in PCA coordinates
                xmin, xmax = np.percentile(Z_ref_pca[:,0], [1, 99])
                ymin, ymax = np.percentile(Z_ref_pca[:,1], [1, 99])
                pad_x = 0.1 * (xmax - xmin + 1e-6); pad_y = 0.1 * (ymax - ymin + 1e-6)
                xmin -= pad_x; xmax += pad_x; ymin -= pad_y; ymax += pad_y
                gx, gy = np.meshgrid(np.linspace(xmin, xmax, 180), np.linspace(ymin, ymax, 180))
                grid2 = np.stack([gx.ravel(), gy.ravel()], axis=1)
                grid2_t = torch.tensor(grid2, device=self.device, dtype=blob['centroids'].dtype)
                # Compute det(G^{-1}) strictly in the PCA(2) subspace, matching Stage A logic
                # Project centroids and per-centroid matrices onto PCA(2)
                C2 = torch.matmul((C_use - mean_t), U_d2)                               # [K, 2]
                MU = torch.einsum('kde,ej->kdj', M_use, U_d2)                           # [K, D, 2]
                M2 = torch.einsum('id,kdj->kij', U_d2.T, MU)                            # [K, 2, 2]

                # Select S2 for Mahalanobis weights in 2D (trace/det normalization) — used only for distance, not aggregation
                if weight_kernel == 'mahalanobis_normed':
                    if weight_metric_norm == 'trace':
                        traces2 = torch.einsum('kii->k', M2).clamp_min(1e-12)
                        S2 = M2 / (traces2.view(-1, 1, 1) / 2.0)
                    elif weight_metric_norm == 'det':
                        dets2 = torch.linalg.det(M2).abs().clamp_min(1e-24)
                        scales2 = dets2.pow(1.0 / 2.0).clamp_min(1e-12)
                        S2 = M2 / scales2.view(-1, 1, 1)
                    else:
                        S2 = M2
                elif weight_kernel == 'mahalanobis':
                    S2 = M2
                else:  # isotropic
                    S2 = None

                # Evaluate det(G^{-1}_subspace) on the PCA grid using RAW (non-normalized) weights
                z2_all = grid2_t  # [B, 2] coordinates in PCA plane
                det_map_sub = torch.empty(z2_all.shape[0], device=self.device, dtype=blob['centroids'].dtype)
                chunk = 2048
                eye2 = torch.eye(2, device=self.device, dtype=blob['centroids'].dtype).unsqueeze(0)
                T2 = (float(blob['temperature']) ** 2) + 1e-12
                for s in range(0, z2_all.shape[0], chunk):
                    z2 = z2_all[s:s+chunk]                                                    # [b, 2]
                    with torch.no_grad():
                        diff2 = C2.unsqueeze(0) - z2.unsqueeze(1)                              # [b, K, 2]
                        if weight_kernel == 'isotropic':
                            d2 = torch.sum(diff2 * diff2, dim=-1)                              # [b, K]
                        else:
                            Sd = torch.einsum('bkd,kde->bke', diff2, S2)                      # [b, K, 2]
                            d2 = torch.sum(Sd * diff2, dim=-1)                                 # [b, K]
                        w = torch.exp(-d2 / T2)                                                # RAW weights
                        if topk_weights is not None and topk_weights > 0 and topk_weights < w.shape[1]:
                            ksel = min(int(topk_weights), w.shape[1])
                            topv, topi = torch.topk(w, k=ksel, dim=1, largest=True, sorted=False)
                            mask = torch.zeros_like(w)
                            mask.scatter_(1, topi, 1.0)
                            w = w * mask
                        if normalize_w:
                            w = w / (w.sum(dim=1, keepdim=True).clamp_min(1e-12))
                        Ginv2 = torch.einsum('bk,kij->bij', w, M2) + blob['regularization'] * eye2  # [b, 2, 2]
                        det2 = torch.linalg.det(Ginv2).clamp_min(1e-20)
                        det_map_sub[s:s+z2.shape[0]] = det2
                det_img_sub = det_map_sub.detach().cpu().numpy().reshape(gx.shape)

                # Project centroids and samples to PCA(2)
                C_np = C_use.detach().cpu().numpy()
                C_pca = (C_np - mean_np) @ U.T
                S_np = samples.detach().cpu().numpy()
                S_pca = (S_np - mean_np) @ U.T

                # Plot heatmap + overlays (subspace determinant to match Stage A visuals)
                fig, ax = plt.subplots(1,1,figsize=(7,6))
                # Improve contrast using percentile clipping
                lo, hi = np.nanpercentile(det_img_sub, [5, 95])
                im = ax.imshow(det_img_sub, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto', vmin=lo, vmax=hi)
                ax.scatter(Z_ref_pca[:,0][::max(1, len(Z_ref_pca)//3000)], Z_ref_pca[:,1][::max(1, len(Z_ref_pca)//3000)], s=4, c='white', alpha=0.15, label='t=0 latents')
                ax.scatter(C_pca[:,0], C_pca[:,1], c='red', marker='*', s=80, label='centroids')
                ax.scatter(S_pca[:,0], S_pca[:,1], s=6, c='xkcd:blue', alpha=0.35, label='RHMC samples')
                ax.set_title('Stage B: det(G^{-1}) in PCA(2) subspace (t=0) + RHMC')
                ax.legend(frameon=False, loc='upper right')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='det(G^{-1}) (PCA subspace)')
                # Single run already open
                if wandb.run is not None:
                    wandb.log({"stageB/pca/rhmc_overlay_t0": wandb.Image(fig)})
                plt.close(fig)

                # Also log standalone heatmap and centroids-only scatter
                fig_h, ax_h = plt.subplots(1,1,figsize=(6,5))
                ax_h.set_title('Stage B: det(G^{-1}) heatmap (PCA subspace, t=0)')
                imh = ax_h.imshow(det_img_sub, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto', vmin=lo, vmax=hi)
                plt.colorbar(imh, ax=ax_h, fraction=0.046, pad=0.04)
                if wandb.run is not None:
                    wandb.log({"stageB/pca/det_heatmap_t0": wandb.Image(fig_h)})
                plt.close(fig_h)

                fig_c, ax_c = plt.subplots(1,1,figsize=(6,5))
                ax_c.scatter(C_pca[:,0], C_pca[:,1], c='red', marker='*', s=80, label='centroids')
                ax_c.set_title('Stage B: Centroids on PCA(2) (t=0)')
                ax_c.legend(frameon=False)
                if wandb.run is not None:
                    wandb.log({"stageB/pca/centroids_t0": wandb.Image(fig_c)})
                plt.close(fig_c)

                # Concise Stage B summary as a table (instead of extra plots)
                if wandb.run is not None:
                    try:
                        acceptance = getattr(rhmc, 'last_acceptance_rate', None)
                        summary_cols = [
                            'timestep', 'centroids_source', 'pca_fit_basis', 'n_centroids_total', 'n_centroids_t0',
                            'latent_dim', 'temperature', 'regularization', 'weight_kernel', 'weight_metric_normalization',
                            'normalize_weight_sum', 'topk_weights', 'metric_scale', 'n_steps', 'n_leapfrog', 'step_size', 'acceptance_rate'
                        ]
                        summary_row = [
                            0,
                            't0-filtered' if state.get('t0_latents', None) is not None else 'all',
                            pca_fit_basis,
                            int(C_all.shape[0]),
                            int(C_use.shape[0]),
                            int(C_all.shape[1]) if C_all.ndim == 2 else int(M_all.shape[-1]),
                            float(blob['temperature']),
                            float(blob['regularization']),
                            weight_kernel,
                            weight_metric_norm,
                            bool(normalize_w),
                            int(topk_weights) if topk_weights is not None else None,
                            float(metric_scale),
                            int(getattr(cfg.sampling, 'n_steps', 100)),
                            int(getattr(cfg.sampling, 'n_leapfrog', 50)),
                            float(getattr(cfg.sampling, 'step_size', 0.001)),
                            float(acceptance) if acceptance is not None else None,
                        ]
                        table = wandb.Table(data=[summary_row], columns=summary_cols)
                        wandb.log({"stageB/summary_table": table})
                    except Exception as e:
                        print(f"[Stage B] ⚠️ Summary table logging failed: {e}")

                # Stage B: Reconstruction vs Original using available encoder/decoder (RHVAE or Vanilla)
                try:
                    import torchvision.utils as vutils
                    model_enc = None
                    model_dec = None
                    if 'rh_exp' in locals() and hasattr(rh_exp, 'model'):
                        model_enc = getattr(rh_exp.model, 'encoder', None)
                        model_dec = getattr(rh_exp.model, 'decoder', None)
                    elif 'vanilla' in locals():
                        model_enc = getattr(vanilla, 'encoder', None)
                        model_dec = getattr(vanilla, 'decoder', None)
                    elif 'stageB_model' in locals():
                        model_enc = getattr(stageB_model, 'encoder', None)
                        model_dec = getattr(stageB_model, 'decoder', None)
                    if model_enc is not None and model_dec is not None:
                        model_enc = model_enc.to(self.device).eval()
                        model_dec = model_dec.to(self.device).eval()
                        raw_train = torch.load(cfg.data.train_path, map_location='cpu')
                        if raw_train.ndim == 5:
                            imgs0 = raw_train[:8, 0].to(self.device)
                        elif raw_train.ndim == 4:
                            imgs0 = raw_train[:8].to(self.device)
                        else:
                            imgs0 = None
                        if imgs0 is not None:
                            with torch.no_grad():
                                enc_out = model_enc(imgs0)
                                z = enc_out.get('embedding', None) if isinstance(enc_out, dict) else enc_out
                                if z is None and isinstance(enc_out, dict):
                                    z = enc_out.get('z', None)
                                dec_out = model_dec(z)
                                recon = dec_out.get('reconstruction', None) if isinstance(dec_out, dict) else dec_out
                                if recon is None and isinstance(dec_out, dict):
                                    recon = dec_out.get('recon_x', None)
                                recon = torch.sigmoid(recon) if recon.dtype.is_floating_point else recon
                                recon = recon.clamp(0, 1)
                            grid = vutils.make_grid(
                                torch.cat([imgs0, recon], dim=0),
                                nrow=8,
                                normalize=True,
                                value_range=(0.0, 1.0)
                            )
                            if wandb.run is not None:
                                wandb.log({"stageB/recon_vs_real": wandb.Image(grid)})
                    else:
                        print("[Stage B] Note: No encoder/decoder available for recon_vs_real.")
                except Exception as e:
                    print(f"[Stage B] ⚠️ stageB/recon_vs_real logging failed: {e}")
            except Exception as e:
                print(f"[Stage B] ⚠️ PCA RHMC overlay failed: {e}")

        # Stage C: RLVAE
        if getattr(cfg.experiment, 'run_stage_c', True):
            print("\n=== [Stage C] RLVAE training ===")
            
            # Define architecture and latent_dim for Stage C
            arch = cfg.model.encoder.architecture if hasattr(cfg.model, 'encoder') and cfg.model.encoder is not None else 'mlp'
            latent_dim = cfg.model.latent_dim
            metric_impl = cfg.experiment.stage_b.implementation
            
            # Get organized Stage C paths
            stageC_paths = get_stage_paths(cfg, 'C', 'RLVAE', arch, latent_dim)
            
            # Try to find Stage A and B data automatically
            # For Stage C, we specifically want RHVAE Stage A data, not vanilla VAE
            stage_a_data = find_stage_a_data(cfg, arch, latent_dim)
            stage_b_data = find_stage_b_data(cfg, arch, latent_dim, metric_impl.upper())
            
            # Accept whatever Stage A data exists (RHVAE or VANILLA)
            # For ModRLVAE, vanilla enc/dec are perfectly valid.
            pass
            
            # Define metric_file for fallback cases
            metric_file = stage_b_data['metric_path'] if stage_b_data else None
            
            print(f"\n=== [Stage C] COMPONENT LOADING DEBUG ===")
            print(f"[Stage C] Architecture: {arch}, Latent dim: {latent_dim}")
            print(f"[Stage C] Stage A data found: {stage_a_data is not None}")
            print(f"[Stage C] Stage B data found: {stage_b_data is not None}")
            
            if stage_a_data:
                print(f"[Stage C] Stage A paths:")
                print(f"  - Base dir: {stage_a_data['base_dir']}")
                print(f"  - Encoder: {stage_a_data['encoder_path']}")
                print(f"  - Decoder: {stage_a_data['decoder_path']}")
                print(f"  - Config: {stage_a_data['config_path']}")
                if stage_a_data.get('metric_path'):
                    print(f"  - Metric: {stage_a_data['metric_path']}")
            
            if stage_b_data:
                print(f"[Stage C] Stage B paths:")
                print(f"  - Base dir: {stage_b_data['base_dir']}")
                print(f"  - Metric: {stage_b_data['metric_path']}")
                print(f"  - Config: {stage_b_data['config_path']}")
            
            # Wire metric path and pretrained components into model config
            if stage_b_data is not None:
                # Always set the pretrained metric path (works across model variants)
                try:
                    self.config.model.pretrained.metric_path = str(stage_b_data['metric_path'])
                except Exception as e:
                    print(f"[Stage C] ⚠️ Could not set pretrained.metric_path: {e}")
                # Try to set fixed metric path on metric block when schema supports it
                try:
                    if hasattr(self.config.model, 'metric') and self.config.model.metric is not None:
                        # Only set if key exists to avoid struct errors
                        try:
                            if 'fixed_metric_path' in self.config.model.metric:
                                self.config.model.metric.fixed_metric_path = str(stage_b_data['metric_path'])
                        except Exception:
                            pass
                        try:
                            # For Phase‑2 adaptability: initialize trainable net from fixed metric
                            if 'init_from_fixed' in self.config.model.metric:
                                self.config.model.metric.init_from_fixed = True
                            if 'trainable' in self.config.model.metric:
                                self.config.model.metric.trainable = True
                            if 'architecture' in self.config.model.metric and not self.config.model.metric.architecture:
                                self.config.model.metric.architecture = 'mlp'
                        except Exception:
                            pass
                except Exception as e:
                    # Non-fatal; different model schemas may not expose these keys
                    print(f"[Stage C] ℹ️ Skipping metric.fixed_metric_path/init_from_fixed wiring: {e}")
                print(f"[Stage C] ✅ Using Stage B metric: {stage_b_data['metric_path']}")
            else:
                # Fallback to old method
                if metric_file is not None:
                    self.config.model.pretrained.metric_path = str(metric_file)
                    self.config.model.metric.fixed_metric_path = str(metric_file)
                    # Allow metric to update during training for better temporal dynamics
                    self.config.model.metric.init_from_fixed = False
                    print(f"[Stage C] ⚠️ Using fallback metric: {metric_file}")
                else:
                    print(f"[Stage C] ❌ No metric file found!")
                    raise ValueError("No metric file available for Stage C")
            
            if stage_a_data is not None:
                self.config.model.pretrained.encoder_path = str(stage_a_data['encoder_path'])
                self.config.model.pretrained.decoder_path = str(stage_a_data['decoder_path'])
                print(f"[Stage C] ✅ Using Stage A encoder/decoder:")
                print(f"  - Encoder: {stage_a_data['encoder_path']}")
                print(f"  - Decoder: {stage_a_data['decoder_path']}")
            elif 'encoder' in comp_paths:
                self.config.model.pretrained.encoder_path = comp_paths['encoder']
                self.config.model.pretrained.decoder_path = comp_paths['decoder']
                print(f"[Stage C] ⚠️ Using fallback encoder/decoder: {comp_paths['encoder']}")
            else:
                print(f"[Stage C] ❌ No encoder/decoder found!")
            
            print(f"[Stage C] Final config paths:")
            print(f"  - Encoder: {self.config.model.pretrained.encoder_path}")
            print(f"  - Decoder: {self.config.model.pretrained.decoder_path}")
            print(f"  - Metric: {self.config.model.pretrained.metric_path}")
            print(f"[Stage C] Final metric config:")
            try:
                metric_cfg = getattr(self.config.model, 'metric', None)
                if metric_cfg is not None:
                    fixed_path_val = None
                    init_from_fixed_val = None
                    try:
                        if 'fixed_metric_path' in metric_cfg:
                            fixed_path_val = metric_cfg.fixed_metric_path
                    except Exception:
                        pass
                    try:
                        if 'init_from_fixed' in metric_cfg:
                            init_from_fixed_val = metric_cfg.init_from_fixed
                    except Exception:
                        pass
                    print(f"  - fixed_metric_path: {fixed_path_val if fixed_path_val is not None else 'n/a'}")
                    print(f"  - init_from_fixed: {init_from_fixed_val if init_from_fixed_val is not None else 'n/a'}")
                else:
                    print("  - metric config: n/a")
            except Exception as e:
                print(f"  - metric config unavailable ({e})")
            
            # Verify architecture and latent_dim consistency with Stage A
            if stage_a_data:
                print(f"[Stage C] Architecture/Latent Dim Verification:")
                print(f"  - Stage A config: {stage_a_data['config_path']}")
                try:
                    with open(stage_a_data['config_path'], 'r') as f:
                        stage_a_config = yaml.safe_load(f)
                    stage_a_arch = stage_a_config.get('architecture', 'unknown')
                    stage_a_ld = stage_a_config.get('latent_dim', 'unknown')
                    print(f"  - Stage A: arch={stage_a_arch}, latent_dim={stage_a_ld}")
                    print(f"  - Stage C: arch={arch}, latent_dim={latent_dim}")
                    if stage_a_arch != arch or stage_a_ld != latent_dim:
                        print(f"  - ⚠️ MISMATCH DETECTED! Stage A and Stage C have different architecture/latent_dim!")
                        print(f"  - Forcing Stage C to use Stage A parameters...")
                        # Force Stage C to use Stage A's architecture and latent_dim
                        self.config.model.encoder.architecture = stage_a_arch
                        self.config.model.decoder.architecture = stage_a_arch
                        self.config.model.latent_dim = stage_a_ld
                        print(f"  - ✅ Updated Stage C config to match Stage A")
                    else:
                        print(f"  - ✅ Architecture and latent_dim match between Stage A and Stage C")
                except Exception as e:
                    print(f"  - ⚠️ Could not verify Stage A config: {e}")
            
            print(f"=== [Stage C] END COMPONENT DEBUG ===\n")
            # Ensure latent_dim matches the metric from Stage B
            try:
                metric_file_to_check = stage_b_data['metric_path'] if stage_b_data else (metric_file if metric_file is not None else None)
                state_metric = torch.load(metric_file_to_check, map_location='cpu', weights_only=False)
                ld = state_metric.get('latent_dim', None)
                if ld is None:
                    C_ = state_metric.get('centroids', None)
                    M_ = state_metric.get('metric_matrices', None)
                    if C_ is not None and C_.ndim == 2:
                        ld = int(C_.shape[1])
                    elif M_ is not None and M_.ndim == 3:
                        ld = int(M_.shape[-1])
                if ld is not None:
                    self.config.model.latent_dim = int(ld)
                    print(f"[Stage C] Set model.latent_dim = {self.config.model.latent_dim} from Stage A metric")
                # Fallback: read encoder/decoder paths from metric payload if not already set
                try:
                    if not getattr(self.config.model.pretrained, 'encoder_path', None):
                        enc_p = state_metric.get('encoder_path', None)
                        if enc_p:
                            self.config.model.pretrained.encoder_path = enc_p
                    if not getattr(self.config.model.pretrained, 'decoder_path', None):
                        dec_p = state_metric.get('decoder_path', None)
                        if dec_p:
                            self.config.model.pretrained.decoder_path = dec_p
                except Exception:
                    pass
            except Exception as e:
                print(f"[Stage C] ⚠️ Could not set latent_dim from metric: {e}")
            # Epoch override
            self.config.training.trainer.max_epochs = cfg.experiment.stage_c.epochs

            # Apply Stage C model overrides (posterior/KL parameters etc.)
            try:
                sc = cfg.experiment.stage_c
                # Propagate riemannian_beta if provided at stage level
                if hasattr(sc, 'riemannian_beta') and sc.riemannian_beta is not None:
                    self.config.model.riemannian_beta = float(sc.riemannian_beta)
                # Propagate sampling method if present
                if hasattr(sc, 'sampling') and sc.sampling is not None and hasattr(sc.sampling, 'method'):
                    if not hasattr(self.config.model, 'sampling'):
                        self.config.model.sampling = {}
                    self.config.model.sampling.method = sc.sampling.method
                # Generic overrides map (applied to model fields)
                if hasattr(sc, 'overrides') and sc.overrides is not None:
                    for k, v in sc.overrides.items():
                        try:
                            setattr(self.config.model, k, v)
                        except Exception:
                            # Non‑fatal for structured configs
                            pass
                # Force n_flows if provided at stage C level
                try:
                    if hasattr(sc, 'n_flows') and sc.n_flows is not None:
                        self.config.model.n_flows = int(sc.n_flows)
                except Exception:
                    pass
                # Merge any training overrides declared inside experiment.stage_c (e.g., metric alternation)
                try:
                    if hasattr(sc, 'training') and sc.training is not None:
                        # Shallow merge metric_alternation
                        tr = getattr(self.config, 'training', None)
                        if tr is not None and hasattr(sc.training, 'metric_alternation'):
                            if not hasattr(tr, 'metric_alternation'):
                                tr.metric_alternation = {}
                            for k, v in sc.training.metric_alternation.items():
                                try:
                                    tr.metric_alternation[k] = v
                                except Exception:
                                    pass
                except Exception as e:
                    print(f"[Stage C] ⚠️ Failed to merge training overrides: {e}")
            except Exception as e:
                print(f"[Stage C] ⚠️ Failed to apply Stage C overrides: {e}")

            # Use fully modular Stage C (ModRLVAE)
            try:
                self.config.model._target_ = 'rlvae.models.modrlvae.ModRLVAE'
            except Exception:
                pass

            # Set specific model parameters for Stage C (guarded for struct configs)
            try:
                # Ensure posterior is Riemannian for ModRLVAE
                if hasattr(self.config.model, 'posterior') and self.config.model.posterior is not None:
                    try:
                        if 'type' in self.config.model.posterior:
                            self.config.model.posterior.type = 'riemannian_metric'
                    except Exception:
                        pass
                # Also set top-level posterior_type when present
                try:
                    if 'posterior_type' in self.config.model:
                        self.config.model.posterior_type = 'riemannian_metric'
                except Exception:
                    pass
                # Sequence length and flows for modular temporal dynamics
                try:
                    seq_len = int(getattr(self.config.data, 'sequence_length', 0))
                    if seq_len > 0:
                        # Prefer sequence_length; ModRLVAE auto-derives n_flows = seq_len - 1 when n_flows missing
                        self.config.model.sequence_length = seq_len
                        self.config.model.n_flows = max(0, seq_len - 1)
                except Exception:
                    pass
                try:
                    if 'riemannian_kl_mode' in self.config.model and (self.config.model.riemannian_kl_mode is None or self.config.model.riemannian_kl_mode == ''):
                        self.config.model.riemannian_kl_mode = 'sample_logq_logp'
                except Exception:
                    pass
                try:
                    if 'riemannian_beta' in self.config.model and (self.config.model.riemannian_beta is None):
                        self.config.model.riemannian_beta = 1.0
                except Exception:
                    pass
                # Ensure temporal reconstruction is enabled for RLVAE
                try:
                    if 'reconstruction_mode' in self.config.model:
                        self.config.model.reconstruction_mode = 'all'
                except Exception:
                    pass
                # Ensure temporal dynamics are properly configured
                try:
                    if 'encode_all_timesteps_if_no_flows' in self.config.model:
                        self.config.model.encode_all_timesteps_if_no_flows = True
                except Exception:
                    pass
                try:
                    if 'kl_over_all_timesteps_if_flows' in self.config.model:
                        self.config.model.kl_over_all_timesteps_if_flows = False  # KL only on z₀ (t=0)
                except Exception:
                    pass
            except Exception:
                pass
            print(f"[Stage C] Set model parameters:")
            try:
                post_type = getattr(self.config.model.posterior, 'type', 'n/a') if hasattr(self.config.model, 'posterior') else 'n/a'
            except Exception:
                post_type = 'n/a'
            try:
                n_flows_val = self.config.model.n_flows if hasattr(self.config.model, 'n_flows') else 'n/a'
            except Exception:
                n_flows_val = 'n/a'
            try:
                rkm_val = self.config.model.riemannian_kl_mode if ('riemannian_kl_mode' in self.config.model) else 'n/a'
            except Exception:
                rkm_val = 'n/a'
            try:
                rbeta_val = self.config.model.riemannian_beta if ('riemannian_beta' in self.config.model) else 'n/a'
            except Exception:
                rbeta_val = 'n/a'
            print(f"  - posterior.type: {post_type}")
            print(f"  - n_flows: {n_flows_val}")
            print(f"  - riemannian_kl_mode: {rkm_val}")
            print(f"  - riemannian_beta: {rbeta_val}")
            print(f"  - reconstruction_mode: all (temporal reconstruction enabled)")
            print(f"  - encode_all_timesteps_if_no_flows: True")
            print(f"  - kl_over_all_timesteps_if_flows: False (KL only on z₀)")
            # Respect CLI overrides: only set defaults if not provided
            # Posterior type default (do not override if already set)
            try:
                current_post = getattr(self.config.model.posterior, 'type', None) if hasattr(self.config.model, 'posterior') else None
                if current_post in (None, ""):
                    self.config.model.posterior.type = 'riem_hmc'
            except Exception:
                pass
            # Visualization defaults: do not override when level is 'none'
            try:
                vis = getattr(self.config, 'visualization', None)
                if vis is None:
                    self.config.visualization.level = 'standard'
                    self.config.visualization.enable_basic = True
                    self.config.visualization.enable_manifold = True
                else:
                    level = str(getattr(self.config.visualization, 'level', 'standard')).lower()
                    if level != 'none':
                        # Keep user's level, only ensure sane frequencies
                        self.config.visualization.enable_basic = True
                        self.config.visualization.enable_manifold = True
                # Ensure positive frequencies if present
                self.config.visualization.basic_frequency = max(1, int(getattr(self.config.visualization, 'basic_frequency', 5)))
                self.config.visualization.manifold_frequency = max(1, int(getattr(self.config.visualization, 'manifold_frequency', 10)))
            except Exception:
                pass
            # Disable heavy evaluation during quick Stage C to keep pipeline fast and clean
            try:
                if hasattr(self.config, 'evaluation') and self.config.evaluation is not None:
                    self.config.evaluation.enabled = False
                    # Also avoid generator init via evaluation
                    if hasattr(self.config.evaluation, 'generation'):
                        self.config.evaluation.generation.enabled = False
                    if hasattr(self.config.evaluation, 'fid'):
                        self.config.evaluation.fid.enabled = False
                    if hasattr(self.config.evaluation, 'inference'):
                        self.config.evaluation.inference.enabled = False
                    if hasattr(self.config.evaluation, 'reconstruction'):
                        self.config.evaluation.reconstruction.enabled = False
                    print("[Stage C] Evaluation disabled for modular Stage C (clean, fast run)")
            except Exception:
                pass

            # Link Stage B metric as an input artifact for provenance
            if cfg.wandb.mode != "disabled" and getattr(cfg.wandb, 'artifacts', {}).get('enabled', False):
                try:
                    # Use previously logged stageB artifact if present
                    art_name = f"stageB_metric_{arch}_ld{latent_dim}:{getattr(cfg.wandb.artifacts.aliases, 'stage_b_latest', 'stageB_latest')}"
                    wandb.use_artifact(art_name)
                except Exception as e:
                    print(f"[Stage C] ⚠️ Could not reference Stage B artifact: {e}")
            self.run_single_experiment()

            # Stage C latent PCA + det(G^{-1}) heatmap with centroids overlay
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                # Load Stage B metric for centroids overlay
                if stage_b_data is not None:
                    stateB = torch.load(stage_b_data['metric_path'], map_location='cpu', weights_only=False)
                    Cb = stateB.get('centroids')
                else:
                    Cb = None
                # Build small dataloader for val set
                from data.cyclic_dataset import CyclicSpritesDataModule
                dm = CyclicSpritesDataModule(cfg.data)
                dm.setup('fit', cfg.training)
                vl = dm.val_dataloader()
                model_wrapper = LightningRlVAETrainer(cfg, data_module=dm)
                model = model_wrapper.model.to(self.device)
                model.eval()
                mus = []
                with torch.no_grad():
                    for xb in vl:
                        xb = xb.to(self.device)
                        x0 = xb[:,0]
                        enc = model.encoder(x0)
                        mu = enc.embedding
                        mus.append(mu.detach().cpu())
                        if sum(t.shape[0] for t in mus) >= 4000:
                            break
                if mus:
                    Z = torch.cat(mus, dim=0)
                    Zc = Z - Z.mean(dim=0, keepdim=True)
                    U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
                    comp = Vh[:2].T
                    mean = Z.mean(dim=0)
                    Proj = (Zc @ comp).numpy()
                    x_min, x_max = np.percentile(Proj[:,0], [1,99])
                    y_min, y_max = np.percentile(Proj[:,1], [1,99])
                    xs = np.linspace(x_min, x_max, 100)
                    ys = np.linspace(y_min, y_max, 100)
                    XX, YY = np.meshgrid(xs, ys)
                    pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
                    Zgrid = mean.unsqueeze(0) + torch.from_numpy(pts).to(mean.dtype) @ comp.T
                    Zgrid = Zgrid.to(self.device)
                    Ginv = model.modular_metric.compute_inverse_metric(Zgrid)
                    _, logdet = torch.linalg.slogdet(Ginv)
                    det_vals = torch.exp(logdet).detach().cpu().numpy().reshape(XX.shape)
                    plt.figure(figsize=(7,6))
                    plt.imshow(det_vals, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='viridis', aspect='auto')
                    P = ((Z - mean) @ comp).numpy()
                    plt.scatter(P[:,0], P[:,1], s=3, c='white', alpha=0.15, label='μ (val t=0)')
                    if Cb is not None:
                        Pc = ((Cb - mean).detach().cpu() @ comp).numpy()
                        plt.scatter(Pc[:,0], Pc[:,1], s=40, c='red', marker='*', label='centroids (B)')
                    # Prior samples from Stage C
                    try:
                        zs = model.sample_prior(1000, method='geodesic').detach().cpu()
                        Ps = ((zs - mean) @ comp).numpy()
                        plt.scatter(Ps[:,0], Ps[:,1], s=3, c='deepskyblue', alpha=0.3, label='prior samples')
                    except Exception:
                        pass
                    plt.colorbar(label='det(G^{-1}) (PCA subspace)')
                    plt.title('Stage C: det(G^{-1}) in PCA(2) subspace')
                    plt.legend(loc='upper right', fontsize=8)
                    plt.tight_layout()
                    if wandb.run is not None:
                        wandb.log({"stageC/pca_det_heatmap": wandb.Image(plt.gcf())})
                    plt.close()
            except Exception as e:
                print(f"[Stage C] ⚠️ PCA det(G^-1) heatmap failed: {e}")
            
            # Save Stage C configuration
            stageC_config = {
                'stage': 'C',
                'model_type': 'RLVAE',
                'architecture': arch,
                'latent_dim': latent_dim,
                'epochs': cfg.experiment.stage_c.epochs,
                'riemannian_beta': cfg.experiment.stage_c.riemannian_beta,
                'stage_a_source': str(stage_a_data['base_dir']) if stage_a_data else None,
                'stage_b_source': str(stage_b_data['base_dir']) if stage_b_data else None,
                'timestamp': datetime.now().isoformat()
            }
            with open(stageC_paths['config_path'], 'w') as f:
                yaml.dump(stageC_config, f)
            print(f"[Stage C] ✅ Saved Stage C config to {stageC_paths['config_path']}")
            # After Stage C, snapshot and log the trained metric (if trainable) as an artifact
            if cfg.wandb.mode != "disabled" and getattr(cfg.wandb, 'artifacts', {}).get('enabled', False):
                try:
                    # Try to locate last saved snapshot
                    last_snap = None
                    snap_dir = Path("metric_snapshots")
                    if snap_dir.exists():
                        snaps = sorted(snap_dir.glob("metric_epoch_*.pt"))
                        if snaps:
                            last_snap = snaps[-1]
                    if last_snap is not None and last_snap.exists():
                        art = wandb.Artifact(
                            name=f"stageC_metric_{arch}_ld{latent_dim}",
                            type="metric",
                            metadata={"stage": "C", "architecture": arch, "latent_dim": latent_dim}
                        )
                        art.add_file(str(last_snap))
                        aliases = [getattr(cfg.wandb.artifacts.aliases, 'stage_c_latest', 'stageC_latest')]
                        wandb.log_artifact(art, aliases=aliases)
                except Exception as e:
                    print(f"[Stage C] ⚠️ Artifact logging failed: {e}")
            # Optional summary run aggregating key graphs
            if cfg.wandb.mode != "disabled":
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    plt.style.use('default'); sns.set_palette("husl")
                    state = torch.load(metric_file, map_location='cpu', weights_only=False)
                    M_matrices = state.get('M_matrices')
                    if M_matrices is None:
                        M_matrices = state.get('metric_matrices')
                    if M_matrices is None:
                        M_matrices = state.get('inverse_metrics')
                    centroids = state.get('centroids')
                    if M_matrices is not None and centroids is not None:
                        eigenvals = torch.linalg.eigvals(M_matrices).real
                        min_eigenvals = eigenvals.min(dim=-1)[0]
                        max_eigenvals = eigenvals.max(dim=-1)[0]
                        determinants = torch.linalg.det(M_matrices)
                        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
                        axes[0].hist(min_eigenvals.numpy(), bins=40, color='red', edgecolor='black'); axes[0].set_title('Min Eig')
                        axes[1].hist(max_eigenvals.numpy(), bins=40, color='blue', edgecolor='black'); axes[1].set_title('Max Eig')
                        axes[2].hist((torch.log10(torch.abs(determinants)+1e-50)).numpy(), bins=40, color='purple', edgecolor='black'); axes[2].set_title('Log10|Det|')
                        plt.tight_layout(); wandb.log({"summary/stageB/metric_overview": wandb.Image(fig)}); plt.close(fig)
                        art = wandb.Artifact(f"stageB_metric_{arch}_ld{latent_dim}", type="metric"); art.add_file(str(metric_file)); wandb.log_artifact(art)
                    # Stage A quick recon grid (if Stage A model was saved)
                    try:
                        if 'model' in comp_paths:
                            from scripts.train_diverse_metric_vae import create_model as create_stage1
                            import torchvision.utils as vutils
                            stage1 = create_stage1(arch, input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]), latent_dim=latent_dim)
                            stage1.load_state_dict(torch.load(comp_paths['model'], map_location='cpu', weights_only=False))
                            stage1.eval()
                            raw = torch.load(cfg.data.train_path, map_location='cpu')
                            if raw.ndim == 5:
                                # [B, S, C, H, W] -> flatten
                                b, s = raw.shape[:2]
                                raw = raw.reshape(b*s, *raw.shape[2:])
                            batch = raw[:8]
                            with torch.no_grad():
                                if arch.lower() in ["mlp", "pythae"]:
                                    out = stage1({"data": batch})
                                    recon = out.recon_x.clamp(0, 1)
                                else:
                                    out = stage1(batch)
                                    recon = out.recon_x.clamp(0, 1)
                            grid = vutils.make_grid(torch.cat([batch[:8], recon[:8]], dim=0), nrow=8, normalize=False)
                            wandb.log({"summary/stageA/final_recon_grid": wandb.Image(grid)})
                    except Exception as e:
                        print(f"[SUMMARY] ⚠️ Could not log Stage A recon grid: {e}")

                    # Stage C test metrics
                    if isinstance(self.results, dict) and 'test_results' in self.results and isinstance(self.results['test_results'], dict) and len(self.results['test_results']):
                        payload = {}
                        for k, v in self.results['test_results'].items():
                            if isinstance(v, torch.Tensor):
                                try:
                                    v = v.item()
                                except Exception:
                                    continue
                            payload[f"summary/stageC/{k}"] = v
                        if payload:
                            wandb.log(payload)
                    # keep run open
                except Exception as e:
                    print(f"[SUMMARY] ⚠️ WandB summary logging failed: {e}")
    
    def _create_model_config(self, model_name: str) -> DictConfig:
        """Create configuration for a specific model variant."""
        config = OmegaConf.structured(self.config)
        
        # Apply model-specific overrides
        if model_name == 'vanilla_vae':
            config.model.n_flows = 0
            config.model.riemannian_beta = 0.0
            config.model.posterior.type = 'gaussian'
            config.model.sampling.use_riemannian = False
            config.model.sampling.method = 'standard'
            config.model.loop.mode = 'open'
            config.model.loop.penalty = 0.0
        elif model_name == 'riemannian_flow_vae':
            # Use default Riemannian configuration
            pass
        
        # Apply experiment overrides
        if hasattr(self.config.experiment, 'training_override'):
            config.training.update(self.config.experiment.training_override)
        
        if hasattr(self.config.experiment, 'visualization_override'):
            config.visualization.update(self.config.experiment.visualization_override)
        
        return config
    
    def _setup_wandb(self, run_name: str) -> Optional[WandbLogger]:
        """Setup Weights & Biases logging."""
        if self.config.wandb.mode == "disabled":
            return None
        
        # For pipeline experiments, include stage information in the run name
        if hasattr(self.config.experiment, 'type') and self.config.experiment.type == "pipeline":
            # Extract architecture and latent_dim for consistent naming
            architecture = getattr(self.config.experiment.stage1, 'architecture', 'mlp')
            latent_dim = getattr(self.config.experiment.stage1, 'latent_dim', 16)
            full_run_name = f"pipeline_stage2_rlvae_{architecture}_ld{latent_dim}"
        else:
            # Create unique run name for other experiments
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_run_name = f"{run_name}_{timestamp}"
        
        wandb_logger = WandbLogger(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=full_run_name,
            mode=self.config.wandb.mode,
            tags=self.config.wandb.get('tags', []),
            config=OmegaConf.to_container(self.config, resolve=True)
        )
        
        return wandb_logger
    
    def _create_trainer(self, wandb_logger) -> L.Trainer:
        """Create Lightning trainer."""
        callbacks = []
        
        # Early stopping
        if hasattr(self.config.training, 'early_stopping'):
            early_stop = EarlyStopping(
                monitor=self.config.training.early_stopping.monitor,
                patience=self.config.training.early_stopping.patience,
                mode=self.config.training.early_stopping.mode
            )
            callbacks.append(early_stop)
        
        # Model checkpointing
        checkpoint = ModelCheckpoint(
            monitor=self.config.training.logging.monitor,
            save_top_k=self.config.training.logging.save_top_k,
            mode=self.config.training.logging.mode,
            dirpath=self.output_dir / "checkpoints",
            filename="{epoch:02d}-{val_loss:.3f}"
        )
        callbacks.append(checkpoint)
        
        # Create trainer with new config structure
        trainer_kwargs = {
            'max_epochs': self.config.training.trainer.max_epochs,
            'accelerator': self.config.training.trainer.accelerator,
            'devices': self.config.training.trainer.devices,
            'strategy': self.config.training.trainer.strategy,
            'precision': self.config.training.trainer.precision,
            'log_every_n_steps': self.config.training.trainer.log_every_n_steps,
            'val_check_interval': self.config.training.trainer.val_check_interval,
            'num_sanity_val_steps': self.config.training.trainer.num_sanity_val_steps,
            'enable_progress_bar': self.config.training.trainer.enable_progress_bar,
            'enable_model_summary': self.config.training.trainer.enable_model_summary,
            'deterministic': self.config.training.trainer.deterministic,
            'logger': wandb_logger,
            'callbacks': callbacks,
        }
        
        trainer = L.Trainer(**trainer_kwargs)
        
        return trainer
    
    def _extract_comparison_metrics(self, test_results: Dict, metric_names: List[str]) -> Dict[str, float]:
        """Extract specific metrics for comparison."""
        extracted = {}
        
        for metric_name in metric_names:
            if metric_name in test_results:
                value = test_results[metric_name]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                extracted[metric_name] = value
        
        return extracted
    
    def _save_results(self):
        """Save experiment results."""
        results_path = self.output_dir / "results.yaml"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            OmegaConf.save(OmegaConf.create(self.results), f)
        
        print(f"💾 Results saved to: {results_path}")
    
    def _print_comparison_summary(self, summary: Dict[str, Any]):
        """Print comparison summary to console."""
        print(f"\n📊 COMPARISON SUMMARY")
        print("=" * 60)
        
        for model_name, metrics in summary.items():
            print(f"\n🎯 {model_name.upper()}:")
            for metric_name, value in metrics.items():
                if metric_name.endswith('_final'):
                    clean_name = metric_name.replace('_final', '')
                    print(f"   {clean_name}: {value:.4f}")

def get_stage_folder_name(stage, model_type, architecture, latent_dim, dataset_name="SPRITES"):
    """
    Create organized folder names for the three-stage pipeline.
    
    Args:
        stage: 'A', 'B', or 'C'
        model_type: 'RHVAE', 'VANILLA', 'PRECISION', etc.
        architecture: 'MLP', 'CNN', etc.
        latent_dim: latent dimension as int
        dataset_name: dataset name (default: "SPRITES")
    
    Returns:
        Folder name like 'A_RHVAE_MLP_16_SPRITES' or 'B_VANILLA_MLP_16_SPRITES'
    """
    return f"{stage}_{model_type}_{architecture.upper()}_{latent_dim}_{dataset_name}"

def get_stage_paths(cfg, stage, model_type, architecture, latent_dim):
    """
    Get organized paths for a specific stage.
    
    Args:
        cfg: configuration object
        stage: 'A', 'B', or 'C'
        model_type: model type for the stage
        architecture: architecture type
        latent_dim: latent dimension
    
    Returns:
        dict with organized paths
    """
    stage_folder = get_stage_folder_name(stage, model_type, architecture, latent_dim)
    base_dir = Path(cfg.output_dir) / "stages" / stage_folder
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if stage == 'A':
        return {
            'base_dir': base_dir,
            'encoder_path': base_dir / 'encoder.pt',
            'decoder_path': base_dir / 'decoder.pt',
            'model_path': base_dir / 'model.pt',
            'metric_path': base_dir / 'metric.pt',
            'config_path': base_dir / 'config.yaml'
        }
    elif stage == 'B':
        return {
            'base_dir': base_dir,
            'metric_path': base_dir / 'metric.pt',
            'samples_path': base_dir / 'rhmc_samples.pt',
            'config_path': base_dir / 'config.yaml'
        }
    elif stage == 'C':
        return {
            'base_dir': base_dir,
            'checkpoint_path': base_dir / 'checkpoint.pt',
            'config_path': base_dir / 'config.yaml'
        }
    else:
        raise ValueError(f"Unknown stage: {stage}")

def find_stage_a_data(cfg, architecture, latent_dim):
    """
    Automatically find Stage A data from organized folders.
    
    Args:
        cfg: configuration object
        architecture: architecture type
        latent_dim: latent dimension
    
    Returns:
        dict with Stage A paths or None if not found
    """
    stages_dir = Path(cfg.output_dir) / "stages"
    if not stages_dir.exists():
        return None
    
    # Decide preference order based on configured Stage A model
    prefer_vanilla = False
    try:
        model_choice = str(getattr(cfg.experiment.stage_a, 'model', '')).lower()
        prefer_vanilla = (model_choice.startswith('vanilla'))
    except Exception:
        prefer_vanilla = False
    if prefer_vanilla:
        possible_folders = [
            get_stage_folder_name('A', 'VANILLA', architecture, latent_dim),
            get_stage_folder_name('A', 'RHVAE', architecture, latent_dim)
        ]
    else:
        possible_folders = [
            get_stage_folder_name('A', 'RHVAE', architecture, latent_dim),
            get_stage_folder_name('A', 'VANILLA', architecture, latent_dim)
        ]
    
    # Add debug info
    print(f"[Stage B] Looking for Stage A data in folders: {possible_folders}")
    
    for folder_name in possible_folders:
        stage_a_dir = stages_dir / folder_name
        if stage_a_dir.exists():
            # Check if it has the required files
            config_path = stage_a_dir / 'config.yaml'
            
            # Look for encoder and decoder files (handle both timestamped and simple naming, and both .pt and .pkl)
            encoder_files = (list(stage_a_dir.glob('encoder_*.pt')) + list(stage_a_dir.glob('encoder.pt')) + 
                           list(stage_a_dir.glob('encoder_*.pkl')) + list(stage_a_dir.glob('encoder.pkl')))
            decoder_files = (list(stage_a_dir.glob('decoder_*.pt')) + list(stage_a_dir.glob('decoder.pt')) + 
                           list(stage_a_dir.glob('decoder_*.pkl')) + list(stage_a_dir.glob('decoder.pkl')))
            metric_files = (list(stage_a_dir.glob('metric*.pt')) + list(stage_a_dir.glob('metric.pt')) + 
                          list(stage_a_dir.glob('metric*.pkl')) + list(stage_a_dir.glob('metric.pkl')))
            
            if encoder_files and decoder_files and config_path.exists():
                encoder_path = encoder_files[0]  # Take the first one
                decoder_path = decoder_files[0]  # Take the first one
                metric_path = metric_files[0] if metric_files else None
                print(f"[Stage B] Found Stage A data in: {stage_a_dir}")
                return {
                    'base_dir': stage_a_dir,
                    'metric_path': metric_path,
                    'encoder_path': encoder_path,
                    'decoder_path': decoder_path,
                    'config_path': config_path
                }
    
    print(f"[Stage B] No Stage A data found for {architecture}_{latent_dim}")
    return None

def find_stage_b_data(cfg, architecture, latent_dim, metric_type):
    """
    Automatically find Stage B data from organized folders.
    
    Args:
        cfg: configuration object
        architecture: architecture type
        latent_dim: latent dimension
        metric_type: metric type (RHVAE, PRECISION, etc.)
    
    Returns:
        dict with Stage B paths or None if not found
    """
    stages_dir = Path(cfg.output_dir) / "stages"
    if not stages_dir.exists():
        return None
    
    # Look for Stage B folder
    folder_name = get_stage_folder_name('B', metric_type, architecture, latent_dim)
    stage_b_dir = stages_dir / folder_name
    
    if stage_b_dir.exists():
        # Check if it has the required files
        metric_path = stage_b_dir / 'metric.pt'
        config_path = stage_b_dir / 'config.yaml'
        
        if metric_path.exists() and config_path.exists():
            print(f"[Stage C] Found Stage B data in: {stage_b_dir}")
            return {
                'base_dir': stage_b_dir,
                'metric_path': metric_path,
                'config_path': config_path
            }
    
    print(f"[Stage C] No Stage B data found for {metric_type}_{architecture}_{latent_dim}")
    return None

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    
    print("🧠 RlVAE Experiment Runner with Hydra")
    print("=" * 50)
    
    # Print configuration
    print("📋 Configuration:")
    from omegaconf import OmegaConf
    print(OmegaConf.to_yaml(config))
    
    # Print model.metric config for extra debugging
    print("[HYDRA DEBUG] config.model.metric:", OmegaConf.to_yaml(config.model.metric) if hasattr(config, 'model') and hasattr(config.model, 'metric') else 'N/A')
    
    # Run experiment
    runner = ExperimentRunner(config)
    runner.run()
    
    print("\n🏁 Experiment completed successfully!")


if __name__ == "__main__":
    main() 
