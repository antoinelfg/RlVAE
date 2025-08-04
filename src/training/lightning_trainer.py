"""
Lightning Trainer Module for RlVAE
==================================

PyTorch Lightning wrapper for ModularRiemannianFlowVAE with integrated visualizations.
"""

import os
import torch
import torch.nn as nn
import lightning as L
from typing import Dict, Any, Optional
from omegaconf import DictConfig
import wandb
import sys
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

from models.modular_rlvae import ModularRiemannianFlowVAE
from visualizations.manager import VisualizationManager, VisualizationLevel, VisualizationConfig
from generation.generator import RlVAEGenerator
from evaluation.fid_scorer import FIDScorer
from evaluation.evaluator import ModelEvaluator
from inference.inference_pipeline import RlVAEInferencePipeline


class LightningRlVAETrainer(L.LightningModule):
    """Lightning module for RiemannianFlowVAE training."""
    
    def __init__(self, config: DictConfig, data_module=None):
        super().__init__()
        
        self.config = config
        self.data_module = data_module
        
        # Create model
        self.model = ModularRiemannianFlowVAE(config.model)
        # --- FLOW DIAGNOSTICS: Log initial flow weights ---
        if hasattr(self.model, 'flow_manager'):
            print("[FLOW DIAGNOSTICS] Initial flow weights:")
            for i, flow in enumerate(self.model.flow_manager.flows):
                for name, param in flow.named_parameters():
                    print(f"  Flow {i} param {name}: mean={param.data.mean().item():.4e}, std={param.data.std().item():.4e}")
        # Setup visualizations
        self._setup_visualizations()
        
        # Setup evaluation components
        self._setup_evaluation()
        
        # Save hyperparameters
        self.save_hyperparameters(config)
        
        print(f"⚡ Lightning trainer initialized")
        print(f"   Model: {self.model.model_name}")
        print(f"   Visualization level: {self.config.visualization.level}")
        print(f"   Evaluation enabled: {getattr(self.config.evaluation, 'enabled', False)}")
        # --- FLOW DIAGNOSTICS: Log regularization and learning rate settings ---
        if hasattr(self.config, 'training') and hasattr(self.config.training, 'optimizer'):
            print(f"[FLOW DIAGNOSTICS] Optimizer LR: {self.config.training.optimizer.lr}")
            print(f"[FLOW DIAGNOSTICS] Optimizer weight_decay: {self.config.training.optimizer.weight_decay}")
        if hasattr(self.config.model, 'metric'):
            print(f"[FLOW DIAGNOSTICS] Metric config: {self.config.model.metric}")
        if hasattr(self.config.model, 'flow_hidden_size'):
            print(f"[FLOW DIAGNOSTICS] Flow hidden size: {self.config.model.flow_hidden_size}")
        if hasattr(self.config.model, 'n_flows'):
            print(f"[FLOW DIAGNOSTICS] n_flows: {self.config.model.n_flows}")
    
    def setup(self, stage=None):
        """Setup method called by Lightning."""
        super().setup(stage)
        # Ensure the entire model is properly on device
        self._ensure_model_on_device()
        # Update visualization manager device
        if self.viz_manager is not None:
            self.viz_manager.device = self.device
            # Update device for all visualization modules
            for viz_name, viz_module in self.viz_manager.modules.items():
                viz_module.device = self.device

    def _setup_visualizations(self):
        """Setup visualization manager."""
        try:
            # Create visualization config - handle level as string
            level_str = self.config.visualization.level
            if isinstance(level_str, str):
                level_enum = VisualizationLevel(level_str)
            else:
                level_enum = level_str
                
            viz_config = VisualizationConfig.from_level(level_enum)
            
            # Override with specific config values
            for key, value in self.config.visualization.items():
                if hasattr(viz_config, key) and key != 'level':
                    setattr(viz_config, key, value)
            
            self.viz_manager = VisualizationManager(
                model=self.model,
                config=self.config,
                device=self.device,
                viz_config=viz_config
            )
            
            self.enable_visualizations = True
            print(f"🎨 Visualizations enabled: {viz_config.level.value}")
            
        except Exception as e:
            print(f"⚠️ Visualization setup failed: {e}")
            self.viz_manager = None
            self.enable_visualizations = False
    
    def _setup_evaluation(self):
        """Setup evaluation components for FID scoring and generation analysis."""
        try:
            if not hasattr(self.config, 'evaluation') or not self.config.evaluation.enabled:
                self.enable_evaluation = False
                self.generator = None
                self.fid_scorer = None
                self.evaluator = None
                self.inference_pipeline = None
                print("📊 Evaluation disabled")
                return
            # Initialize evaluation components
            self.generator = self.model.create_generator(self.config.get('generation', None))
            self.inference_pipeline = self.model.create_inference_pipeline(self.config.get('inference', None))
            self.fid_scorer = None  # Lazy initialization (if needed)
            self.evaluator = None  # Lazy initialization (if needed)
            self.enable_evaluation = True
            self.evaluation_config = self.config.evaluation
            # Track evaluation state
            self.real_images_collected = False
            self.real_image_batch = None
            print(f"📊 Evaluation enabled")
            print(f"   FID enabled: {self.evaluation_config.fid.enabled}")
            print(f"   Generation enabled: {self.evaluation_config.generation.enabled}")
            print(f"   Run during training: {self.evaluation_config.run_during_training}")
            print(f"   Run during testing: {self.evaluation_config.run_during_testing}")
        except Exception as e:
            print(f"⚠️ Evaluation setup failed: {e}")
            self.enable_evaluation = False
            self.generator = None
            self.fid_scorer = None
            self.evaluator = None
            self.inference_pipeline = None
    
    def _lazy_init_evaluation_components(self):
        """Lazy initialization of evaluation components when first needed."""
        if not self.enable_evaluation:
            return False
        try:
            if self.generator is None:
                self.generator = self.model.create_generator(self.config.get('generation', None))
            if self.fid_scorer is None and self.evaluation_config.fid.enabled:
                self.fid_scorer = FIDScorer(device=self.device)
            if self.inference_pipeline is None and self.evaluation_config.inference.enabled:
                self.inference_pipeline = self.model.create_inference_pipeline(self.config.get('inference', None))
            if self.evaluator is None:
                self.evaluator = self.model.create_evaluator()
            return True
        except Exception as e:
            print(f"⚠️ Failed to initialize evaluation components: {e}")
            return False
    
    def _collect_real_images_for_fid(self, batch):
        """Collect real images for FID computation."""
        if (self.enable_evaluation and 
            not self.real_images_collected and 
            self.evaluation_config.fid.enabled):
            
            try:
                # Handle sequence data: [B, T, C, H, W] -> [B*T, C, H, W] for FID
                if batch.dim() == 5:
                    # Reshape sequence data: [B, T, C, H, W] -> [B*T, C, H, W]
                    B, T, C, H, W = batch.shape
                    batch_for_fid = batch.view(B*T, C, H, W)
                elif batch.dim() == 4:
                    # Already in correct format: [B, C, H, W]
                    batch_for_fid = batch
                else:
                    print(f"⚠️ Unexpected batch shape for FID: {batch.shape}")
                    return
                
                # Collect a subset of real images for FID
                if self.real_image_batch is None:
                    self.real_image_batch = batch_for_fid.detach().cpu()
                else:
                    self.real_image_batch = torch.cat([self.real_image_batch, batch_for_fid.detach().cpu()], dim=0)
                
                # Stop collecting when we have enough
                if self.real_image_batch.shape[0] >= self.evaluation_config.fid.real_samples_subset:
                    self.real_image_batch = self.real_image_batch[:self.evaluation_config.fid.real_samples_subset]
                    self.real_images_collected = True
                    print(f"📊 Collected {self.real_image_batch.shape[0]} real images for FID computation")
                    print(f"   Real image batch shape: {self.real_image_batch.shape}")
                    # Ensure FID scorer is initialized before caching
                    if self.fid_scorer is None and self.evaluation_config.fid.enabled:
                        from src.evaluation.fid_scorer import FIDScorer
                        self.fid_scorer = FIDScorer(device=self.device)
                    if self.fid_scorer is not None:
                        self.fid_scorer.cache_real_statistics(
                            self.real_image_batch.to(self.device),
                            cache_key="evaluation",
                            batch_size=self.evaluation_config.fid.inception_batch_size
                        )

            except Exception as e:
                print(f"⚠️ Failed to collect real images for FID: {e}")
                import traceback
                print(traceback.format_exc())
    
    def _should_run_evaluation(self):
        """Check if evaluation should run at current epoch."""
        if not self.enable_evaluation:
            return False
        
        # Always run at end if enabled
        if self.evaluation_config.run_at_end_only:
            return False  # Will be handled in on_test_end
        
        # Check if we should run during training
        if not self.evaluation_config.run_during_training:
            return False
        
        # Check epoch constraints
        freq_config = self.evaluation_config.frequency
        
        if self.current_epoch < freq_config.min_epoch:
            return False
        
        # Check specific epochs
        if freq_config.at_epochs and self.current_epoch in freq_config.at_epochs:
            return True
        
        # Check frequency
        if freq_config.every_n_epochs > 0 and self.current_epoch % freq_config.every_n_epochs == 0:
            return True
        
        return False
    
    def _run_generation_fid_evaluation(self, prefix="eval"):
        """Run generation and FID evaluation."""
        if not self._lazy_init_evaluation_components():
            return {}
        
        try:
            results = {}
            
            # Run FID evaluation
            if (self.fid_scorer is not None and 
                self.real_images_collected and 
                self.evaluation_config.fid.enabled):
                
                print(f"🔍 Computing FID score...")
                
                # Generate samples for FID
                n_samples = self.evaluation_config.fid.n_generated_samples
                batch_size = self.evaluation_config.fid.batch_size
                
                generated_samples = self.generator.generate_samples(
                    n_samples=n_samples,
                    batch_size=batch_size,
                    method=self.evaluation_config.generation.methods[0]  # Use first method
                )
                
                # Reshape generated samples if needed for FID
                if generated_samples.dim() == 5:
                    B, S, C, H, W = generated_samples.shape
                    generated_samples = generated_samples.view(B * S, C, H, W)

                # Compute FID
                fid_score = self.fid_scorer.evaluate_with_cached_real(
                    generated_images=generated_samples,
                    real_cache_key="evaluation",  # Use the same cache key as above
                    batch_size=self.evaluation_config.fid.inception_batch_size
                )
                
                results[f'{prefix}_fid_score'] = fid_score
                if isinstance(fid_score, dict) and 'fid_score' in fid_score:
                    print(f"📊 FID Score: {fid_score['fid_score']:.2f}")
                elif isinstance(fid_score, (float, int)):
                    print(f"📊 FID Score: {fid_score:.2f}")
                else:
                    print(f"📊 FID Score: {fid_score}")
            
            # Run generation evaluation for multiple methods
            if self.evaluator is not None and self.evaluation_config.generation.enabled:
                print(f"🎲 Evaluating generation methods...")
                
                for method in self.evaluation_config.generation.methods:
                    try:
                        # Generate samples
                        n_samples = self.evaluation_config.generation.n_samples_per_method
                        samples = self.generator.generate_samples(
                            n_samples=n_samples,
                            batch_size=self.evaluation_config.generation.batch_size,
                            method=method
                        )
                        
                        # Basic quality metrics (could extend this)
                        results[f'{prefix}_generation_{method}_samples'] = n_samples
                        results[f'{prefix}_generation_{method}_mean_pixel'] = torch.mean(samples).item()
                        results[f'{prefix}_generation_{method}_std_pixel'] = torch.std(samples).item()
                        
                    except Exception as e:
                        print(f"⚠️ Failed to evaluate generation method {method}: {e}")
            
            return results
            
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
            return {}
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x, compute_metrics=True)
    
    def training_step(self, batch, batch_idx):
        # Flow diagnostics (only on first step for cleaner output)
        if hasattr(self.model, 'flow_manager') and batch_idx == 0 and self.current_epoch % 10 == 0:
            total_flow_grad_norm = 0
            for i, flow in enumerate(self.model.flow_manager.flows):
                for param in flow.parameters():
                    if param.grad is not None:
                        total_flow_grad_norm += param.grad.norm().item() ** 2
            total_flow_grad_norm = total_flow_grad_norm ** 0.5
            print(f"[Flow] Epoch {self.current_epoch}, Total gradient norm: {total_flow_grad_norm:.2e}")
        x = batch  # [B, T, C, H, W]
        result = self.model(x)
        # Robust loss extraction
        if 'loss' in result:
            main_loss = result['loss']
        elif 'total_loss' in result:
            main_loss = result['total_loss']
        else:
            print("[WARNING] No 'loss' or 'total_loss' in model output during training_step. Skipping loss logging.")
            main_loss = None
        # Diversity regularization (only log occasionally)
        diversity_weight = 0.05
        if hasattr(result, 'z'):
            z_seq_tensor = result.z  # [batch, timesteps, latent_dim]
            latent_var = z_seq_tensor.var(dim=0).mean()
            diversity_loss = -latent_var
            if batch_idx == 0 and self.current_epoch % 10 == 0:
                print(f"[Diversity] Epoch {self.current_epoch}, Latent variance: {latent_var.item():.4e}")
            total_loss = main_loss + diversity_weight * diversity_loss
        else:
            total_loss = main_loss
        # Log losses
        self.log('train_loss', total_loss, prog_bar=True)
        recon_loss = result.get('reconstruction_loss', None)
        kl_loss = result.get('kl_divergence', None)
        # Only log if present
        if recon_loss is not None:
            self.log('train_recon_loss', recon_loss)
        else:
            print("[WARNING] 'reconstruction_loss' not in model output during training_step.")
        if kl_loss is not None:
            self.log('train_kl_loss', kl_loss)
        else:
            print("[WARNING] 'kl_divergence' not in model output during training_step.")
        
        # Log additional metrics if available
        if 'riemannian_kl' in result:
            self.log('train_riemannian_kl', result['riemannian_kl'])
        if 'cyclicity_error' in result:
            self.log('train_cyclicity_error', result['cyclicity_error'])
        if 'metric_conditioning' in result:
            self.log('train_metric_conditioning', result['metric_conditioning'])
        if 'manifold_regularity' in result:
            self.log('train_manifold_regularity', result['manifold_regularity'])
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x = batch
        # Debug: Print batch stats
        print(f"[DEBUG] Validation batch {batch_idx} stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[DEBUG] Validation batch {batch_idx} contains NaN or Inf!")
        # Collect real images for FID computation
        self._collect_real_images_for_fid(x)
        # Forward pass
        result = self.model(x)
        # Robust loss extraction
        if 'loss' in result:
            main_loss = result['loss']
        elif 'total_loss' in result:
            main_loss = result['total_loss']
        else:
            print("[WARNING] No 'loss' or 'total_loss' in model output during validation_step. Skipping loss logging.")
            main_loss = None
        # Debug: Print output keys and stats
        print(f"[DEBUG] Model output keys: {list(result.keys())}")
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any() or torch.isinf(v).any():
                    print(f"[DEBUG] Output {k} contains NaN or Inf! Value: {v}")
                else:
                    print(f"[DEBUG] Output {k}: min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}, std={v.std().item():.4f}")
        # Extract losses
        total_loss = result['total_loss']
        recon_loss = result['reconstruction_loss']
        kl_loss = result['kl_divergence']
        # Debug: Print loss values
        print(f"[DEBUG] Losses: total={total_loss.item():.4f}, recon={recon_loss.item():.4f}, kl={kl_loss.item():.4f}")
        # Log losses
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_recon_loss', recon_loss)
        self.log('val_kl_loss', kl_loss)
        # Log additional metrics
        if 'riemannian_kl' in result:
            self.log('val_riemannian_kl', result['riemannian_kl'])
        if 'cyclicity_error' in result:
            self.log('val_cyclicity_error', result['cyclicity_error'])
        # Store for visualization
        if batch_idx == 0:  # Only store first batch for efficiency
            self.validation_batch = x.detach().cpu()
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x = batch
        
        # Collect real images for FID computation if not already done
        self._collect_real_images_for_fid(x)
        
        # Forward pass
        result = self.model(x)
        # Robust loss extraction
        if 'loss' in result:
            main_loss = result['loss']
        elif 'total_loss' in result:
            main_loss = result['total_loss']
        else:
            print("[WARNING] No 'loss' or 'total_loss' in model output during test_step. Skipping loss logging.")
            main_loss = None
        
        # Create comprehensive test metrics
        total_loss = result.get('total_loss', None)
        recon_loss = result.get('reconstruction_loss', None) 
        kl_loss = result.get('kl_divergence', None)
        
        if total_loss is None:
            print("[WARNING] 'total_loss' not in model output during test_step.")
        metrics = {
            'test_loss': total_loss,
            'test_recon_loss': recon_loss,
            'test_kl_loss': kl_loss,
        }
        
        # Add additional metrics
        if 'riemannian_kl' in result:
            metrics['test_riemannian_kl'] = result['riemannian_kl']
        
        if 'cyclicity_error' in result:
            metrics['test_cyclicity_error'] = result['cyclicity_error']
        
        # Add Riemannian-specific metrics
        if 'metric_conditioning' in result:
            metrics['test_metric_conditioning'] = result['metric_conditioning']
        
        if 'manifold_regularity' in result:
            metrics['test_manifold_regularity'] = result['manifold_regularity']
        
        # Run evaluation during testing if enabled
        if (self.enable_evaluation and 
            self.evaluation_config.run_during_testing and 
            batch_idx == 0):  # Only run once per test epoch
            
            try:
                eval_results = self._run_generation_fid_evaluation(prefix="test")
                metrics.update(eval_results)
            except Exception as e:
                print(f"⚠️ Test evaluation failed: {e}")
        
        # Log all metrics
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    # Only log numeric values
                    if isinstance(subval, (int, float, torch.Tensor)):
                        self.log(f"{key}_{subkey}", subval)
            else:
                if isinstance(value, (int, float, torch.Tensor)):
                    self.log(key, value)
        
        return metrics
    
    def on_validation_epoch_end(self):
        """Create visualizations and run evaluation at end of validation epoch."""
        # Run evaluation if scheduled
        print("[DEBUG] Entering on_validation_epoch_end")
        sys.stdout.flush()
        if self._should_run_evaluation():
            try:
                print(f"🔍 Running evaluation at epoch {self.current_epoch}")
                sys.stdout.flush()
                # Use config-driven evaluator if available
                print("[DEBUG] Checking evaluator existence and method...")
                sys.stdout.flush()
                if self.evaluator is not None and hasattr(self.evaluator, 'evaluate_comprehensive'):
                    print("[DEBUG] Evaluator and method found, about to call evaluate_comprehensive")
                    sys.stdout.flush()
                    try:
                        eval_results = {'evaluation': self.evaluator.evaluate_comprehensive(
                            real_images=self.real_image_batch if hasattr(self, 'real_image_batch') else None,
                            config=self.config.evaluation
                        )}
                        print("[DEBUG] eval_results structure:", eval_results)
                        print("[DEBUG] eval_results['evaluation'] keys:", list(eval_results['evaluation'].keys()))
                        sys.stdout.flush()
                    except Exception as e:
                        print("[DEBUG] Exception in evaluate_comprehensive:", e)
                        import traceback
                        print(traceback.format_exc())
                        sys.stdout.flush()
                    # Log evaluation results to wandb
                    if wandb.run is not None:
                        for key, value in eval_results.items():
                            if isinstance(value, (int, float)):
                                wandb.log({f"val/{key}": value}, step=self.current_epoch)
                            elif isinstance(value, dict):
                                # Flatten nested dicts
                                for subkey, subval in value.items():
                                    if isinstance(subval, (int, float)):
                                        wandb.log({f"val/{key}/{subkey}": subval}, step=self.current_epoch)
                    print(f"📊 Logged evaluation metrics to WandB")
                else:
                    # Fallback to legacy FID/generation if no evaluator
                    eval_results = self._run_generation_fid_evaluation(prefix="val")
                    for key, value in eval_results.items():
                        if wandb.run is not None:
                            wandb.log({f"val/{key}": value}, step=self.current_epoch)
            except Exception as e:
                print(f"⚠️ Validation evaluation failed: {e}")
        # Run visualizations
        if not self.enable_visualizations or self.viz_manager is None:
            return
        # Only create visualizations at specified frequency
        if self.current_epoch % self.config.visualization.frequency != 0:
            return
        try:
            # Get desired number of sequences for visualization
            seq_count = getattr(self.config.visualization, 'sequence_viz_count', 8)
            if isinstance(seq_count, str) and seq_count == 'all':
                seq_count = 128  # Hard cap for safety
            # Aggregate enough sequences from val dataloader
            if hasattr(self, 'data_module') and hasattr(self.data_module, 'val_dataloader'):
                val_loader = self.data_module.val_dataloader()
                batches = []
                total = 0
                for batch in val_loader:
                    # Extract the input tensor (handle tuple, dict, or tensor)
                    if isinstance(batch, (tuple, list)):
                        x = batch[0]
                    elif isinstance(batch, dict):
                        x = batch.get('x', next(iter(batch.values())))
                    else:
                        x = batch
                    print(f"[DEBUG] Batch shape: {getattr(batch, 'shape', 'N/A')}, x shape: {x.shape}")
                    batches.append(x)
                    total += x.shape[0]
                    if total >= seq_count:
                        break
                x_sample = torch.cat(batches, dim=0)
                seq_count = x_sample.shape[0]  # Use all available sequences in the batch
                x_sample = x_sample[:seq_count].to(self.device)
            elif hasattr(self, 'validation_batch'):
                x_sample = self.validation_batch.to(self.device)
            else:
                print("⚠️ No sample data available for visualization")
                return
            print(f"[DEBUG] x_sample shape before visualization: {x_sample.shape}")
            print(f"🎨 Creating visualizations for epoch {self.current_epoch} (n_seq={x_sample.shape[0]})")
            self.viz_manager.create_visualizations(
                x_sample=x_sample,
                epoch=self.current_epoch
            )
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")

    def on_test_end(self):
        """Run evaluation and log results at the end of testing."""
        print("[DEBUG] Entering on_test_end")
        sys.stdout.flush()
        if self.enable_evaluation:
            try:
                print(f"🔍 Running final evaluation at test end")
                sys.stdout.flush()
                # Use config-driven evaluator if available
                print("[DEBUG] Checking evaluator existence and method...")
                sys.stdout.flush()
                if self.evaluator is not None and hasattr(self.evaluator, 'evaluate_comprehensive'):
                    print("[DEBUG] Evaluator and method found, about to call evaluate_comprehensive")
                    sys.stdout.flush()
                    try:
                        eval_results = {'evaluation': self.evaluator.evaluate_comprehensive(
                            real_images=self.real_image_batch if hasattr(self, 'real_image_batch') else None,
                            config=self.config.evaluation
                        )}
                        print("[DEBUG] eval_results structure:", eval_results)
                        print("[DEBUG] eval_results['evaluation'] keys:", list(eval_results['evaluation'].keys()))
                        sys.stdout.flush()
                    except Exception as e:
                        print("[DEBUG] Exception in evaluate_comprehensive:", e)
                        import traceback
                        print(traceback.format_exc())
                        sys.stdout.flush()
                    # Log evaluation results to wandb
                    if wandb.run is not None:
                        for key, value in eval_results.items():
                            if isinstance(value, (int, float)):
                                wandb.log({f"test/{key}": value})
                            elif isinstance(value, dict):
                                for subkey, subval in value.items():
                                    if isinstance(subval, (int, float)):
                                        wandb.log({f"test/{key}/{subkey}": subval})
                    print(f"📊 Logged test evaluation metrics to WandB")
                else:
                    # Fallback to legacy FID/generation if no evaluator
                    eval_results = self._run_generation_fid_evaluation(prefix="test")
                    for key, value in eval_results.items():
                        if wandb.run is not None:
                            wandb.log({f"test/{key}": value})
            except Exception as e:
                print(f"⚠️ Test evaluation failed: {e}")
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer_config = self.config.training.optimizer
        param_groups = []
        # --- FLOW DIAGNOSTICS: Custom learning rate for all flows ---
        if hasattr(self.model, 'flow_manager'):
            all_flow_params = []
            for flow in self.model.flow_manager.flows:
                all_flow_params += list(flow.parameters())
            param_groups.append({'params': all_flow_params, 'lr': optimizer_config.lr * 0.1, 'weight_decay': optimizer_config.weight_decay})
            # Add all other model params (encoder, decoder, etc.)
            flow_param_ids = set([id(p) for p in all_flow_params])
            main_params = [p for n, p in self.model.named_parameters() if id(p) not in flow_param_ids]
            if main_params:
                param_groups.append({'params': main_params, 'lr': optimizer_config.lr, 'weight_decay': optimizer_config.weight_decay})
        else:
            param_groups.append({'params': self.model.parameters(), 'lr': optimizer_config.lr, 'weight_decay': optimizer_config.weight_decay})
        optimizer = torch.optim.Adam(param_groups)
        # --- FLOW DIAGNOSTICS: Print optimizer parameter groups ---
        print("[FLOW DIAGNOSTICS] Optimizer parameter groups:")
        for i, group in enumerate(optimizer.param_groups):
            print(f"  Group {i}: lr={group['lr']}, weight_decay={group['weight_decay']}, num_params={len(group['params'])}")
        # Create scheduler if specified
        if hasattr(self.config.training, 'scheduler'):
            scheduler_config = self.config.training.scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.mode,
                factor=scheduler_config.factor,
                patience=scheduler_config.patience,
                min_lr=scheduler_config.min_lr
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.config.training.logging.monitor,
                    "frequency": 1
                }
            }
        return optimizer
    
    def on_train_start(self):
        """Log model summary at start of training."""
        # Ensure model is on device again at train start
        self._ensure_model_on_device()
        
        if wandb.run is not None:
            try:
                summary = self.model.get_model_summary()
                
                # Convert ListConfig objects to regular lists for JSON serialization
                def convert_config_to_dict(obj):
                    if hasattr(obj, '_content'):
                        # This is a DictConfig or ListConfig
                        if isinstance(obj._content, dict):
                            return {k: convert_config_to_dict(v) for k, v in obj._content.items()}
                        elif isinstance(obj._content, list):
                            return [convert_config_to_dict(v) for v in obj._content]
                        else:
                            return obj._content
                    elif isinstance(obj, dict):
                        return {k: convert_config_to_dict(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_config_to_dict(v) for v in obj]
                    elif hasattr(obj, '__dict__'):
                        # Handle objects with __dict__
                        return str(obj)
                    else:
                        return obj
                
                summary_serializable = convert_config_to_dict(summary)
                wandb.log({"model_summary": summary_serializable})
                
                # Log data statistics if available
                if self.data_module and hasattr(self.data_module, 'get_data_stats'):
                    data_stats = self.data_module.get_data_stats()
                    data_stats_serializable = convert_config_to_dict(data_stats)
                    wandb.log({"data_stats": data_stats_serializable})
                    
            except Exception as e:
                print(f"⚠️ Failed to log model summary to wandb: {e}")
        
        print(f"🚀 Starting training for {self.config.training.trainer.max_epochs} epochs")
        print(f"   Model parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   Visualization frequency: every {self.config.visualization.frequency} epochs") 

    def on_train_epoch_end(self):
        # Option 2: Save metric_net weights every epoch
        metric_net = getattr(self.model, 'modular_metric', None)
        if metric_net is not None and getattr(metric_net, 'trainable', False):
            save_dir = "metric_snapshots"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(metric_net.metric_net.state_dict(), f"{save_dir}/metric_epoch_{self.current_epoch}.pt")

            # Option 3: Plot metric tensor for a fixed latent vector
            latent_dim = metric_net.latent_dim
            arch_kwargs = metric_net.arch_kwargs if hasattr(metric_net, 'arch_kwargs') else {}
            device = next(metric_net.metric_net.parameters()).device
            z = torch.zeros(1, latent_dim, device=device)  # Example: fixed z
            G = metric_net.metric_net(z).detach().cpu().squeeze().numpy()

            # Compute statistics
            eigvals = np.linalg.eigvalsh(G)
            min_eig = np.min(eigvals)
            max_eig = np.max(eigvals)
            cond_number = max_eig / (min_eig + 1e-12)
            det = np.linalg.det(G)

            # Plot
            fig, ax = plt.subplots()
            im = ax.imshow(G, cmap='viridis')
            fig.colorbar(im, ax=ax)
            ax.set_title(f"Metric tensor at epoch {self.current_epoch}")
            plt.tight_layout()

            # Save locally
            plot_dir = os.path.join(save_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            fig_path = os.path.join(plot_dir, f"metric_tensor_epoch_{self.current_epoch}.png")
            fig.savefig(fig_path)
            plt.close(fig)

            # Log to wandb every 5 epochs
            if wandb.run is not None and self.current_epoch % 5 == 0:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                # Use tostring_argb and convert ARGB to RGB
                argb = np.frombuffer(renderer.tostring_argb(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                argb = argb.reshape((h, w, 4))
                # Convert ARGB to RGB
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                rgb[:, :, 0] = argb[:, :, 1]  # R
                rgb[:, :, 1] = argb[:, :, 2]  # G
                rgb[:, :, 2] = argb[:, :, 3]  # B
                wandb.log({
                    "metric_tensor": wandb.Image(rgb, caption=f"Metric tensor at epoch {self.current_epoch}"),
                    "metric_tensor_mean": np.mean(G),
                    "metric_tensor_std": np.std(G),
                    "metric_tensor_min_eig": float(min_eig),
                    "metric_tensor_max_eig": float(max_eig),
                    "metric_tensor_condition": float(cond_number),
                    "metric_tensor_det": float(det),
                    "metric_tensor_eig_hist": wandb.Histogram(eigvals)
                }, step=self.current_epoch)

    def _ensure_model_on_device(self):
        """Ensure all model components are on the correct device."""
        device = self.device
        # Move main model
        self.model = self.model.to(device)
        # Ensure encoder/decoder are on device
        if hasattr(self.model, 'encoder') and self.model.encoder is not None:
            self.model.encoder = self.model.encoder.to(device)
        if hasattr(self.model, 'decoder') and self.model.decoder is not None:
            self.model.decoder = self.model.decoder.to(device)
        # Ensure metric components are on device
        for attr_name in ['G', 'G_inv', 'centroids', 'flows']:
            if hasattr(self.model, attr_name):
                attr_value = getattr(self.model, attr_name)
                if attr_value is not None:
                    if hasattr(attr_value, 'to'):
                        setattr(self.model, attr_name, attr_value.to(device))
                    elif isinstance(attr_value, (list, nn.ModuleList)):
                        for i, item in enumerate(attr_value):
                            if hasattr(item, 'to'):
                                attr_value[i] = item.to(device)
        print(f"✅ Ensured model is on device: {device}") 