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
import random
import torchvision.utils as vutils

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
        
        # Create model based on config._target_
        if hasattr(config.model, '_target_'):
            # Check if it's the original RiemannianFlowVAE
            if 'original_rlvae.src.models.riemannian_flow_vae.RiemannianFlowVAE' in config.model._target_:
                # Create the original model directly with proper parameter conversion
                from original_rlvae.src.models.riemannian_flow_vae import RiemannianFlowVAE
                
                # Convert config parameters to proper types
                model_config = config.model
                input_dim = tuple(model_config.input_dim) if hasattr(model_config.input_dim, '_content') else tuple(model_config.input_dim)
                latent_dim = int(model_config.latent_dim)
                n_flows = int(model_config.n_flows)
                # Handle both flow_hidden_size and flow_hidden_dims
        if hasattr(model_config, 'flow_hidden_size'):
            flow_hidden_size = int(model_config.flow_hidden_size)
        elif hasattr(model_config, 'flow_hidden_dims'):
            flow_hidden_size = int(model_config.flow_hidden_dims[0])
        else:
            flow_hidden_size = 64  # Default fallback
            flow_n_blocks = int(model_config.flow_n_blocks)
            flow_n_hidden = int(model_config.flow_n_hidden)
            epsilon = float(model_config.epsilon)
            beta = float(model_config.beta)
            riemannian_beta = float(model_config.riemannian_beta) if hasattr(model_config, 'riemannian_beta') else beta
            posterior_type = str(model_config.posterior.type) if hasattr(model_config.posterior, 'type') else 'gaussian'
            loop_mode = str(model_config.loop.mode) if hasattr(model_config.loop, 'mode') else 'open'
            
            # Extract pretrained and metric configs
            pretrained_config = getattr(model_config, 'pretrained', {})
            metric_config = getattr(model_config, 'metric', {})
            
            # Create model with all parameters
            self.model = RiemannianFlowVAE(
                input_dim=input_dim,
                latent_dim=latent_dim,
                n_flows=n_flows,
                flow_hidden_size=flow_hidden_size,
                    flow_n_blocks=flow_n_blocks,
                    flow_n_hidden=flow_n_hidden,
                    epsilon=epsilon,
                    beta=beta,
                    riemannian_beta=riemannian_beta,
                    posterior_type=posterior_type,
                    loop_mode=loop_mode,
                    update_metric_during_training=getattr(model_config, 'update_metric_during_training', False),
                    metric_update_frequency=int(getattr(model_config, 'metric_update_frequency', 100)),
                    metric_update_alpha=float(getattr(model_config, 'metric_update_alpha', 0.01)),
                    metric_update_temperature=float(getattr(model_config, 'metric_update_temperature', 0.1)),
                    metric_update_regularization=float(getattr(model_config, 'metric_update_regularization', 0.01)),
                    pretrained=pretrained_config,
                    metric=metric_config
                )
            else:
                # Use hydra's instantiate for other models
                from hydra.utils import instantiate
                self.model = instantiate(config.model)
        else:
            # Fallback to ModularRiemannianFlowVAE
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

        # === Metric alternation schedule (Stage C) ===
        self.metric_alt_cfg = getattr(self.config.training, 'metric_alternation', None)
        self.metric_alt_enabled = bool(self.metric_alt_cfg and getattr(self.metric_alt_cfg, 'enabled', False))
        if self.metric_alt_enabled:
            # Defaults
            self.metric_warmup_epochs = int(getattr(self.metric_alt_cfg, 'warmup_epochs', 5))
            self.metric_k_rlvae_epochs = int(getattr(self.metric_alt_cfg, 'k_rlvae_epochs', 3))
            self.metric_step_epochs = int(getattr(self.metric_alt_cfg, 'metric_step_epochs', 1))
            self.metric_logdet_clip = float(getattr(self.metric_alt_cfg, 'logdet_clip', 50.0))
            self.metric_anchor_size = int(getattr(self.metric_alt_cfg, 'anchor_size', 20000))
            self.metric_anchor_refresh_frac = float(getattr(self.metric_alt_cfg, 'anchor_refresh_frac', 0.1))
            self.metric_consistency_weight = float(getattr(self.metric_alt_cfg, 'consistency_weight', 0.0))
            self.metric_only_epoch = False
            self._anchors_z = None  # Tensor [A, D]
            print(f"[METRIC ALT] Enabled | warmup={self.metric_warmup_epochs}, schedule={self.metric_k_rlvae_epochs}:{self.metric_step_epochs}, anchors={self.metric_anchor_size}")
        else:
            self.metric_only_epoch = False
    
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

    # === Alternating schedule utilities ===
    def _set_requires_grad(self, module: nn.Module, requires: bool) -> None:
        for param in module.parameters(recurse=True):
            param.requires_grad = requires

    def _freeze_for_metric_step(self) -> None:
        # Freeze everything except the metric network
        self._set_requires_grad(self.model, False)
        metric_net = getattr(self.model, 'modular_metric', None)
        if metric_net is not None:
            self._set_requires_grad(metric_net, True)
        print("[METRIC ALT] Freeze encoder/decoder/flows; training metric only")

    def _freeze_for_rlvae_step(self) -> None:
        # Freeze metric, train others
        metric_net = getattr(self.model, 'modular_metric', None)
        if metric_net is not None:
            self._set_requires_grad(metric_net, False)
        # Ensure others are trainable
        # Turn everything on, then metric stays off
        for name, module in self.model.named_children():
            self._set_requires_grad(module, True)
        if metric_net is not None:
            self._set_requires_grad(metric_net, False)
        print("[METRIC ALT] Freeze metric; training encoder/decoder/flows")

    @torch.no_grad()
    def _collect_anchor_samples(self, max_needed: int) -> torch.Tensor:
        """Collect z0 samples from encoder over training data for anchor pool.
        Uses encoder μ as deterministic representative for stability.
        """
        assert self.data_module is not None, "Data module required to build anchor set"
        loader = self.data_module.train_dataloader()
        collected: list[torch.Tensor] = []
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            elif isinstance(batch, dict):
                x = batch.get('x', next(iter(batch.values())))
            else:
                x = batch
            x = x.to(self.device)
            # Encode first timestep
            x0 = x[:, 0]
            enc_out = self.model.encoder(x0)
            mu = enc_out.embedding.detach()
            collected.append(mu.cpu())
            total = sum(t.shape[0] for t in collected)
            if total >= max_needed:
                break
        if not collected:
            raise RuntimeError("Failed to collect anchors: empty dataset?")
        anchors = torch.cat(collected, dim=0)[:max_needed]
        return anchors

    def _ensure_anchor_set(self) -> None:
        if self._anchors_z is None or self._anchors_z.shape[0] < self.metric_anchor_size:
            print("[METRIC ALT] Building anchor set…")
            anchors = self._collect_anchor_samples(self.metric_anchor_size)
            self._anchors_z = anchors  # on CPU to save GPU mem
            print(f"[METRIC ALT] Anchors ready: {self._anchors_z.shape}")

    def _refresh_anchor_subset(self) -> None:
        if self._anchors_z is None:
            return
        refresh_n = max(1, int(self.metric_anchor_size * self.metric_anchor_refresh_frac))
        if refresh_n <= 0:
            return
        print(f"[METRIC ALT] Refreshing {refresh_n} anchors…")
        # Guard: if anchor pool is smaller than desired size (e.g. small dataset), cap refresh_n
        pool_size = self._anchors_z.shape[0]
        refresh_n = min(refresh_n, pool_size)
        # Collect exactly refresh_n new samples
        new_samples = self._collect_anchor_samples(refresh_n)
        if new_samples.shape[0] > refresh_n:
            new_samples = new_samples[:refresh_n]
        elif new_samples.shape[0] < refresh_n:
            refresh_n = new_samples.shape[0]
        # Replace random indices of current pool
        idxs = random.sample(range(pool_size), k=refresh_n)
        self._anchors_z[idxs] = new_samples

    @torch.no_grad()
    def _prepare_batch_z0(self, x: torch.Tensor) -> torch.Tensor:
        # Compute z0 using current encoder; detach to stop gradients to encoder
        x0 = x[:, 0]
        enc_out = self.model.encoder(x0)
        mu = enc_out.embedding
        log_var = enc_out.log_covariance if hasattr(enc_out, 'log_covariance') else torch.zeros_like(mu)
        # Use metric-aware sampling if available but detach
        try:
            z0 = self.model.sample_metric_aware_posterior(mu, log_var)
        except Exception:
            eps = torch.randn_like(mu)
            z0 = mu + eps * torch.exp(0.5 * log_var)
        return z0.detach()

    def _metric_only_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute metric-only objective: maximize E_z[0.5 log det G^{-1}(z)] - log Z(psi).
        Implemented as minimizing: L = -(mean d(z)) + logsumexp(d(anchors)) - log(|A|).
        """
        metric_net = getattr(self.model, 'modular_metric', None)
        # Must be present, loaded, and trainable with parameters
        if (
            metric_net is None or
            not getattr(metric_net, '_is_loaded', True) or
            not getattr(metric_net, 'trainable', False) or
            len(list(metric_net.parameters())) == 0
        ):
            print("[METRIC ALT] Skipping metric-only epoch: metric not loaded")
            return torch.zeros((), device=self.device, requires_grad=True)
        # Prepare z batch without grad to encoder/decoder
        with torch.no_grad():
            z_batch = self._prepare_batch_z0(x)
        # Move anchors to device for compute
        self._ensure_anchor_set()
        anchors = self._anchors_z.to(self.device)

        # Compute 0.5 log det G^{-1} using modular metric
        Ginv_batch = self.model.modular_metric.compute_inverse_metric(z_batch)
        sign_b, logdet_b = torch.linalg.slogdet(Ginv_batch)
        logdet_b = torch.clamp(logdet_b, min=-self.metric_logdet_clip, max=self.metric_logdet_clip)
        d_batch = 0.5 * logdet_b  # [B]

        Ginv_anch = self.model.modular_metric.compute_inverse_metric(anchors)
        sign_a, logdet_a = torch.linalg.slogdet(Ginv_anch)
        logdet_a = torch.clamp(logdet_a, min=-self.metric_logdet_clip, max=self.metric_logdet_clip)
        d_anch = 0.5 * logdet_a  # [A]

        # log Z ≈ log( (1/A) sum_k exp(d_k) ) = logsumexp(d_k) - log(A)
        logZ = torch.logsumexp(d_anch, dim=0) - torch.log(torch.tensor(float(d_anch.shape[0]), device=self.device))
        # Final loss (minimize): -E[d(z)] + logZ
        loss = -(d_batch.mean()) + logZ

        # Optional consistency term (placeholder)
        if self.metric_consistency_weight > 0.0:
            loss = loss + 0.0 * self.metric_consistency_weight  # hook for future pushforward consistency
        return loss

    def on_train_epoch_start(self) -> None:
        """Decide whether this epoch is metric-only or RLVAE step and freeze params accordingly."""
        if not self.metric_alt_enabled:
            self.metric_only_epoch = False
            return
        e = self.current_epoch
        if e < self.metric_warmup_epochs:
            self.metric_only_epoch = False
            self._freeze_for_rlvae_step()
            wandb.log({"phase": "warmup_rlvae", "epoch": e}) if wandb.run else None
            return
        # Alternating: k epochs of RLVAE then 1 epoch of metric-only
        period = self.metric_k_rlvae_epochs + self.metric_step_epochs
        in_metric_block = ((e - self.metric_warmup_epochs) % period) >= self.metric_k_rlvae_epochs
        self.metric_only_epoch = in_metric_block
        # If metric not ready or not trainable, force RLVAE step
        metric_net = getattr(self.model, 'modular_metric', None)
        if self.metric_only_epoch and (
            metric_net is None or
            not getattr(metric_net, '_is_loaded', True) or
            not getattr(metric_net, 'trainable', False) or
            len(list(metric_net.parameters())) == 0
        ):
            self.metric_only_epoch = False
            print("[METRIC ALT] Forcing RLVAE step (metric unavailable or not trainable)")
        if self.metric_only_epoch:
            self._freeze_for_metric_step()
            # Ensure anchors and refresh a fraction
            self._ensure_anchor_set()
            self._refresh_anchor_subset()
            wandb.log({"phase": "metric", "epoch": e}) if wandb.run else None
        else:
            self._freeze_for_rlvae_step()
            wandb.log({"phase": "rlvae", "epoch": e}) if wandb.run else None

    def _setup_visualizations(self):
        """Setup visualization manager."""
        try:
            # Optionally disable visualizations via config
            vis_cfg = getattr(self.config, 'visualization', {})
            if isinstance(vis_cfg, dict) and (vis_cfg.get('enabled') is False or vis_cfg.get('level') == 'none'):
                self.viz_manager = None
                self.enable_visualizations = False
                print("🎨 Visualizations disabled by config")
                return
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
        # Metric-only epoch handling
        if self.metric_alt_enabled and self.metric_only_epoch:
            x = batch
            loss = self._metric_only_loss(x)
            self.log('train_metric_only_loss', loss, prog_bar=True)
            return loss

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
        z_seq_tensor = result.get('latent_samples', None)
        if isinstance(z_seq_tensor, torch.Tensor):
            try:
                latent_var = z_seq_tensor.var(dim=0).mean()
                diversity_loss = -latent_var
                if batch_idx == 0 and self.current_epoch % 10 == 0:
                    print(f"[Diversity] Epoch {self.current_epoch}, Latent variance: {latent_var.item():.4e}")
                total_loss = main_loss + diversity_weight * diversity_loss
            except Exception:
                total_loss = main_loss
        else:
            total_loss = main_loss
        # Log losses
        self.log('train_loss', total_loss, prog_bar=True)
        # Support both key conventions
        recon_loss = result.get('reconstruction_loss', result.get('recon_loss', None))
        kl_loss = result.get('kl_divergence', result.get('kl_loss', result.get('kld_loss', None)))
        flow_loss = result.get('flow_loss', None)
        if flow_loss is None and hasattr(result, 'get'):
            # Fallback for modular wrapper aliases
            flow_loss = result.get('kl_divergence_loss', None)
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
        if flow_loss is not None:
            self.log('train_flow_loss', flow_loss)
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
        try:
            # Enable detailed shape diagnostics for this step
            os.environ["RLVAE_DEBUG_SHAPES"] = "1"
            result = self.model(x)
            os.environ["RLVAE_DEBUG_SHAPES"] = "0"
        except RuntimeError as e:
            # Handle input shape mismatch for fallback MLP encoders from pythae
            if 'shape' in str(e) and 'invalid for input of size' in str(e):
                print(f"[SHAPE-GUARD] Caught shape error in model forward: {e}")
                print("[SHAPE-GUARD] Attempting a retry with sanitized input/encoder dims…")
                # Try to coerce encoder/decoder input_dim to observed shape and retry
                try:
                    x0 = x[:, 0]
                    obs_dim = tuple(x0.shape[1:])
                    if hasattr(self.model.encoder, 'input_dim'):
                        setattr(self.model.encoder, 'input_dim', obs_dim)
                    if hasattr(self.model.decoder, 'input_dim'):
                        setattr(self.model.decoder, 'input_dim', obs_dim)
                    result = self.model(x)
                except Exception as e2:
                    print(f"[SHAPE-GUARD] Retry failed: {e2}")
                    raise e
            else:
                raise e
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
        # Extract losses using the robust extraction
        total_loss = main_loss
        recon_loss = result.get('recon_loss', result.get('reconstruction_loss', None))
        kl_loss = result.get('kld_loss', result.get('kl_divergence', None))
        # Debug: Print loss values
        print(f"[DEBUG] Losses: total={total_loss.item():.4f}, recon={recon_loss.item():.4f}, kl={kl_loss.item():.4f}")
        # Log losses
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_recon_loss', recon_loss)
        self.log('val_kl_loss', kl_loss)
        if 'flow_loss' in result and result['flow_loss'] is not None:
            self.log('val_flow_loss', result['flow_loss'])
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
        if 'flow_loss' in result and result['flow_loss'] is not None:
            metrics['test_flow_loss'] = result['flow_loss']
        
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
        # Log recon vs real sequences snapshot each epoch
        try:
            if hasattr(self, 'validation_batch') and wandb.run is not None:
                with torch.no_grad():
                    x_val = self.validation_batch.to(self.device)
                    # Limit to a few sequences for grid
                    max_seqs = min(4, x_val.shape[0])
                    x_vis = x_val[:max_seqs]
                    result = self.model(x_vis)
                    # Handle different model output formats
                    if 'reconstruction' in result:
                        recon = result['reconstruction'].clamp(0, 1)
                    elif 'recon_x' in result:
                        recon = result['recon_x'].clamp(0, 1)
                    else:
                        print(f"[RECON-LOG] ⚠️ No reconstruction found in result keys: {list(result.keys())}")
                        return
                    # Take full sequence for comparison
                    # Shape: [B, T, C, H, W] -> [B*T, C, H, W]
                    B, T = x_vis.shape[0], x_vis.shape[1]
                    orig_frames = x_vis.reshape(B * T, *x_vis.shape[2:])
                    recon_frames = recon.reshape(B * T, *recon.shape[2:])
                    # Concatenate orig then recon, arrange rows by sequence with nrow=T
                    grid = vutils.make_grid(torch.cat([orig_frames, recon_frames], dim=0), nrow=T, normalize=False)
                    wandb.log({
                        "stageC/recon_vs_real_epoch": wandb.Image(grid),
                        "stageC/recon_sample_epoch": int(self.current_epoch)
                    })
        except Exception as e:
            print(f"[RECON-LOG] ⚠️ Failed to log recon vs real: {e}")
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
        metric_lr_scale = getattr(optimizer_config, 'metric_lr_scale', 0.25)
        # --- FLOW DIAGNOSTICS: Custom learning rate for all flows ---
        if hasattr(self.model, 'flow_manager'):
            all_flow_params = []
            for flow in self.model.flow_manager.flows:
                all_flow_params += list(flow.parameters())
            param_groups.append({'params': all_flow_params, 'lr': optimizer_config.lr * 0.1, 'weight_decay': optimizer_config.weight_decay})
            # Add all other model params (encoder, decoder, etc.)
            flow_param_ids = set([id(p) for p in all_flow_params])
            # Split metric params out if present
            metric_params = []
            metric_param_ids = set()
            metric_net = getattr(self.model, 'modular_metric', None)
            if metric_net is not None:
                metric_params = list(metric_net.parameters())
                metric_param_ids = set(id(p) for p in metric_params)
                if metric_params:
                    param_groups.append({'params': metric_params, 'lr': optimizer_config.lr * metric_lr_scale, 'weight_decay': optimizer_config.weight_decay})
            main_params = [p for n, p in self.model.named_parameters() if id(p) not in flow_param_ids and id(p) not in metric_param_ids]
            if main_params:
                param_groups.append({'params': main_params, 'lr': optimizer_config.lr, 'weight_decay': optimizer_config.weight_decay})
        else:
            # No explicit flows; optionally split metric params
            metric_params = []
            metric_param_ids = set()
            metric_net = getattr(self.model, 'modular_metric', None)
            if metric_net is not None:
                metric_params = list(metric_net.parameters())
                metric_param_ids = set(id(p) for p in metric_params)
                if metric_params:
                    param_groups.append({'params': metric_params, 'lr': optimizer_config.lr * metric_lr_scale, 'weight_decay': optimizer_config.weight_decay})
            main_params = [p for n, p in self.model.named_parameters() if id(p) not in metric_param_ids]
            if main_params:
                param_groups.append({'params': main_params, 'lr': optimizer_config.lr, 'weight_decay': optimizer_config.weight_decay})
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
            # Frequency controls from config with safe defaults
            try:
                viz_freq = int(getattr(self.config.training.logging, 'stage_c_metric_viz_every_n_epochs', 5))
            except Exception:
                viz_freq = 5
            try:
                snap_freq = int(getattr(self.config.training.logging, 'metric_snapshot_every_n_epochs', 5))
            except Exception:
                snap_freq = 5

            # Optionally save metric snapshots
            if snap_freq > 0 and (self.current_epoch % snap_freq == 0):
                save_dir = "metric_snapshots"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(metric_net.metric_net.state_dict(), f"{save_dir}/metric_epoch_{self.current_epoch}.pt")

            # Optionally plot and log metric visualization
            should_log_viz = (viz_freq > 0 and (self.current_epoch % viz_freq == 0) and wandb.run is not None)
            if should_log_viz:
                # Plot metric tensor for a fixed latent vector
                latent_dim = metric_net.latent_dim
                device = next(metric_net.metric_net.parameters()).device
                z = torch.zeros(1, latent_dim, device=device)
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
                plot_dir = os.path.join("metric_snapshots", "plots")
                os.makedirs(plot_dir, exist_ok=True)
                fig_path = os.path.join(plot_dir, f"metric_tensor_epoch_{self.current_epoch}.png")
                fig.savefig(fig_path)
                plt.close(fig)

                # Log image and stats
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                argb = np.frombuffer(renderer.tostring_argb(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                argb = argb.reshape((h, w, 4))
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                rgb[:, :, 0] = argb[:, :, 1]
                rgb[:, :, 1] = argb[:, :, 2]
                rgb[:, :, 2] = argb[:, :, 3]
                wandb.log({
                    "stageC/metric_tensor": wandb.Image(rgb, caption=f"Metric tensor at epoch {self.current_epoch}"),
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
        # Ensure encoder/decoder are on device (check if they are torch modules)
        if hasattr(self.model, 'encoder') and self.model.encoder is not None:
            if hasattr(self.model.encoder, 'to'):
                self.model.encoder = self.model.encoder.to(device)
        if hasattr(self.model, 'decoder') and self.model.decoder is not None:
            if hasattr(self.model.decoder, 'to'):
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