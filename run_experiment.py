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
import datetime
from typing import Dict, Any, List, Optional
import yaml

# Local imports
from models.modular_rlvae import ModularRiemannianFlowVAE, ModelFactory, MetricsCollector
from data.cyclic_dataset import CyclicSpritesDataModule
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
        
        # Set random seed for reproducibility
        if config.get('seed'):
            L.seed_everything(config.seed)
            print(f"🎲 Seed: {config.seed}")
    
    def run(self):
        """Run the experiment based on configuration."""
        experiment_type = self.config.experiment.type
        print(f"\n🧪 Running {experiment_type} experiment: {self.config.experiment.name}")
        if experiment_type == "single":
            self.run_single_experiment()
        elif experiment_type == "comparison":
            self.run_comparison_study()
        elif experiment_type == "sweep":
            self.run_hyperparameter_sweep()
        elif experiment_type == "pipeline":
            self.run_pipeline_experiment()
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
            self.config.model.n_flows = self.config.experiment.stage2.n_flows
            print(f"[PATCH] Set self.config.model.n_flows = {self.config.model.n_flows}")
        # Ensure input_dim is set from data config for Stage 2 (robust automation)
        self.config.model.input_dim = [
            self.config.data.channels,
            self.config.data.image_size[0],
            self.config.data.image_size[1]
        ]
        # Ensure visualization level is set from stage2 config if present
        visualization_level2 = getattr(stage2_cfg, 'visualization', None)
        if visualization_level2 is not None:
            if hasattr(self.config, 'visualization') and hasattr(self.config.visualization, 'level'):
                self.config.visualization.level = visualization_level2
            elif hasattr(self.config, 'visualization') and isinstance(self.config.visualization, dict):
                self.config.visualization['level'] = visualization_level2
            else:
                self.config.visualization = {'level': visualization_level2}
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
        print("[FINAL PATCH] Model config before creation:")
        print(OmegaConf.to_yaml(self.config.model))
        # Ensure input_dim is set from data config for Stage 2 (robust automation)
        self.config.model.input_dim = [
            self.config.data.channels,
            self.config.data.image_size[0],
            self.config.data.image_size[1]
        ]
        # Ensure visualization level is set from stage2 config if present
        visualization_level2 = getattr(stage2_cfg, 'visualization', None)
        if visualization_level2 is not None:
            if hasattr(self.config, 'visualization') and hasattr(self.config.visualization, 'level'):
                self.config.visualization.level = visualization_level2
            elif hasattr(self.config, 'visualization') and isinstance(self.config.visualization, dict):
                self.config.visualization['level'] = visualization_level2
            else:
                self.config.visualization = {'level': visualization_level2}
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
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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