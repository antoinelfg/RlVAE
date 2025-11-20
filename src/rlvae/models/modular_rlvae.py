"""
Modular Riemannian Flow VAE
==========================

Enhanced version of RiemannianFlowVAE with:
- Hydra configuration support
- Modular architecture for easy comparisons
- Better experiment tracking and analysis
- Plug-and-play encoder/decoder architectures
- 100% modular components (metric, flow, loss, sampling)
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import wandb
from contextlib import nullcontext

from .riemannian_flow_vae import RiemannianFlowVAE, OfficialRHVAESampler
from .components.encoder_manager import EncoderManager
from .components.decoder_manager import DecoderManager
from .components.metric_tensor import MetricTensor
from .components.metric_loader import MetricLoader
from .components.flow_manager import FlowManager
from .components.loss_manager import LossManager
from .components.posterior_sampler import PosteriorSampler
from .components.riemannian_rhmc_posterior import RiemannianRHMCPosterior
from .components.riemannian_sampler import RiemannianSampler
from .components.sampler_manager import SamplerManager
try:
    # Use top-level samplers module for RHMC
    from src.models.samplers import RiemannianHMCSampler
except Exception:
    RiemannianHMCSampler = None  # Optional
from .components.native_inverse_metric import NativeInverseMetricTensor
from .components.manifold_sampler import ManifoldSampler
from .components.metric_update_manager import MetricUpdateManager
from pythae.models.base.base_utils import ModelOutput
from ..utils.stagec_debugger import stagec_debugger


class ModularRiemannianFlowVAE(RiemannianFlowVAE):
    """
    Fully modular version of RiemannianFlowVAE with all components modularized.
    
    Key improvements:
    - Configuration-driven initialization
    - Standardized metrics tracking
    - Comparison-friendly interface
    - Enhanced experiment logging
    - Plug-and-play encoder/decoder architectures
    - Modular metric tensor computations
    - Modular loss management
    - Modular sampling strategies
    """
    
    def __init__(self, config: DictConfig):
        """Initialize from Hydra configuration with all modular components."""
        config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        # Allow passing a full training config; extract model subsection if needed.
        if not hasattr(config, 'latent_dim') and hasattr(config, 'model'):
            config = OmegaConf.create(OmegaConf.to_container(config.model, resolve=True))
        # Ensure metric attributes exist before any parent/loader call paths
        # that might reference them.
        self.modular_metric = None
        self._metric_ready = False
        # Debug print for metric config
        print("[DEBUG] model.metric config at model init:", config.get('metric', {}))
        # Check if n_flows was explicitly set by checking if it's NOT the default auto value
        # Default auto value would be sequence_length - 1, so if it's different, user set it
        n_flows = getattr(config, 'n_flows', None)
        sequence_length = getattr(config, 'sequence_length', None)
        if sequence_length is not None:
            expected_auto_value = max(0, int(sequence_length) - 1)
            if n_flows is None:
                n_flows = expected_auto_value
                print(f"[AUTO] Setting n_flows = sequence_length - 1 = {n_flows}")
            elif int(n_flows) != expected_auto_value:
                print(f"[USER] Keeping user-specified n_flows = {n_flows} (auto would be {expected_auto_value})")
            else:
                print(f"[AUTO] Using n_flows = {n_flows} (matches sequence_length - 1)")
        else:
            print(f"[USER] Keeping n_flows = {n_flows if n_flows is not None else 'unknown'} (no sequence_length found)")
        if n_flows is None:
            n_flows = 0
        config = OmegaConf.merge(config, {"n_flows": int(n_flows)})
        # Extract core parameters
        # Map flow params to the original API (flow_hidden_dims)
        try:
            flow_hidden_size = int(getattr(config, 'flow_hidden_size'))
        except Exception:
            flow_hidden_size = 64
        try:
            flow_n_blocks = int(getattr(config, 'flow_n_blocks'))
        except Exception:
            flow_n_blocks = 2
        try:
            flow_n_hidden = int(getattr(config, 'flow_n_hidden'))
        except Exception:
            flow_n_hidden = 1
        flow_hidden_dims = [flow_hidden_size, flow_n_blocks, flow_n_hidden]

        # Forward original-features flags to the base implementation
        # so we replicate α/β ramps, LR warmup, phase toggles, EMA, etc.
        forward_keys = [
            # KL + posterior controls
            'kl_use_metric_normalization', 'kl_metric_norm_mode', 'posterior_local_alpha', 'kl_amp_safe',
            'posterior_alpha_ramp_enabled', 'posterior_alpha_start', 'posterior_alpha_end', 'posterior_alpha_ramp_epochs',
            'use_curvature_correction',
            # Beta ramp + warmup
            'beta_ramp_enabled', 'beta_start', 'beta_end', 'beta_ramp_epochs', 'beta_ramp_schedule',
            'lr_warmup_enabled', 'lr_warmup_epochs', 'lr_warmup_factor',
            # Phase toggles and regularizers
            'phase1_training', 'centroid_regularizer_enabled', 'centroid_regularizer_weight', 'centroid_regularizer_t0_only',
            'phase2_training', 'metric_learning_rate',
            # Metric constraints
            'spectral_penalty_enabled', 'spectral_penalty_weight', 'eigenval_min_bound', 'eigenval_max_bound',
            'smoothness_penalty_enabled', 'smoothness_penalty_weight',
            'anisotropy_alignment_enabled', 'anisotropy_alignment_weight',
            # EMA + validation
            'centroid_ema_enabled', 'centroid_ema_rate', 'centroid_ema_update_frequency',
            'eps_chol', 'identity_metric_mode', 'metric_validation_enabled',
            # Metric update / adaptive KL
            'update_metric_during_training', 'metric_update_frequency', 'metric_update_alpha',
            'metric_update_temperature', 'metric_update_regularization',
            'adaptive_kl_enabled', 'adaptive_kl_ramp_up_steps', 'adaptive_kl_alignment_weight'
        ]
        extra_kwargs = {}
        for k in forward_keys:
            if hasattr(config, k):
                extra_kwargs[k] = getattr(config, k)

        # Also forward pretrained and metric subsections when present
        if hasattr(config, 'pretrained'):
            extra_kwargs['pretrained'] = config.pretrained
        if hasattr(config, 'metric'):
            # Pass-through metric config so base can set up defaults;
            # we'll still replace G/G_inv with the modular metric below.
            extra_kwargs['metric'] = config.metric

        super().__init__(
            input_dim=tuple(config.input_dim),
            latent_dim=int(config.latent_dim),
            n_flows=int(config.n_flows),
            flow_hidden_dims=flow_hidden_dims,
            beta=float(config.beta),
            encoder=None,  # Will be created by manager
            decoder=None,  # Will be created by manager
            loop_mode=str(config.loop.mode),
            posterior_type=str(config.posterior.type),
            riemannian_beta=float(config.get('riemannian_beta', config.beta)),
            **extra_kwargs
        )
        self._debug_prev_logvar_stats: Optional[Dict[str, float]] = None
        # Handle case where encoder is None (will be created automatically)
        encoder_config = config.get('encoder', {})
        if encoder_config is None:
            encoder_config = {}
        encoder_arch = encoder_config.get('architecture', 'mlp') if isinstance(encoder_config, dict) else 'mlp'
        print(f"[MODEL INIT] input_dim: {config.input_dim}, encoder architecture: {encoder_arch}")
        # Store config for later use
        self.config = config
        
        # Create single source of truth for model config
        self.config_resolved = self._resolve_model_config(config)
        
        # Sanity check print (once)
        if not hasattr(self, '_config_sanity_printed'):
            print(
                f"[CONFIG] Resolved: latent_dim={self.config_resolved.get('latent_dim')}, "
                f"posterior.type={self.config_resolved.get('posterior', {}).get('type')}, "
                f"rhmc_steps={self.config_resolved.get('posterior', {}).get('rhmc_steps')}, "
                f"kl_metric_eval_point={self.config_resolved.get('kl_metric_eval_point')}, "
                f"riemannian_beta={self.config_resolved.get('riemannian_beta')}"
            )
            self._config_sanity_printed = True
        self.model_name = config.get('_target_', 'ModularRiemannianFlowVAE').split('.')[-1]

        # Initialize all modular components
        self._setup_modular_components()
        # Flag for visualization utilities: model expects sequence input
        self.expects_sequence_input = True
        
        # Initialize μ alignment settings
        self.mu_alignment_enabled = bool(getattr(self.config, 'mu_alignment_enabled', True))
        self._mu_align_ready = False
        self._mu_align_scale = None  # [D]
        self._mu_align_bias = None   # [D]
        # Target stats from metric centroids if available
        try:
            if hasattr(self, 'centroids_tens') and self.centroids_tens is not None:
                cm = self.centroids_tens.float()
                self._mu_target_mean = cm.mean(dim=0).to(self.device)
                self._mu_target_std = cm.std(dim=0).clamp_min(1e-6).to(self.device)
            else:
                self._mu_target_mean = torch.zeros(self.latent_dim, device=self.device)
                self._mu_target_std = torch.ones(self.latent_dim, device=self.device)
        except Exception:
            self._mu_target_mean = torch.zeros(self.latent_dim, device=self.device)
            self._mu_target_std = torch.ones(self.latent_dim, device=self.device)
        
        # Hook to ensure modular metric gets loaded when parent class loads metric
        self._setup_metric_transfer_hook()
        
        # Override G and G_inv methods to use modular metric with fallback loading
        self._setup_modular_metric_fallback()
    
    def _resolve_model_config(self, config):
        """Resolve model config with precedence: stage_c > model > training.model"""
        from omegaconf import OmegaConf
        
        # Start with base config
        resolved = OmegaConf.to_container(config, resolve=True)
        
        # Apply precedence: stage_c overrides > model.* > training.model.*
        if 'stage_c' in resolved:
            stage_c = resolved['stage_c']
            if isinstance(stage_c, dict):
                # Merge stage_c overrides
                for key, value in stage_c.items():
                    if key != 'model':  # Don't merge the nested model key
                        resolved[key] = value
        
        # Flatten nested model configs
        if 'model' in resolved:
            model_config = resolved['model']
            if isinstance(model_config, dict):
                for key, value in model_config.items():
                    resolved[key] = value
        
        if 'training' in resolved and 'model' in resolved['training']:
            training_model = resolved['training']['model']
            if isinstance(training_model, dict):
                for key, value in training_model.items():
                    if key not in resolved:  # Only use if not already set
                        resolved[key] = value
        
        return resolved

        # Apply optional posterior RHMC params from resolved config
        try:
            posterior_config = self.config_resolved.get('posterior', {})
            if posterior_config:
                print(f"[CONFIG] Applying posterior config: {posterior_config}")
                # Apply RHMC-specific configs if available
                if 'rhmc_steps' in posterior_config:
                    self.rhmc_steps = posterior_config['rhmc_steps']
                if 'rhmc_step_size' in posterior_config:
                    self.rhmc_step_size = posterior_config['rhmc_step_size']
                if 'rhmc_alpha' in posterior_config:
                    self.rhmc_alpha = posterior_config['rhmc_alpha']
                if 'rhmc_eps_reg' in posterior_config:
                    self.rhmc_eps_reg = posterior_config['rhmc_eps_reg']
        except Exception as e:
            print(f"[CONFIG] Warning: Could not apply posterior config: {e}")
        
        # Initialize manifold sampling if enabled
        self._setup_manifold_sampling()
        
        # Initialize metrics tracking
        self._setup_metrics_tracking()
        
    def _setup_modular_components(self):
        """Initialize all modular components."""
        
        # Create encoder and decoder managers
        self._setup_encoder_decoder()
        
        # 🚀 NEW: Initialize modular metric tensor
        metric_cfg = getattr(self.config, 'metric', {})
        trainable = metric_cfg.get('trainable', False)
        architecture = metric_cfg.get('architecture', 'mlp')
        arch_kwargs = metric_cfg.get('arch_kwargs', {})
        init_from_fixed = metric_cfg.get('init_from_fixed', False)
        fixed_metric_path = metric_cfg.get('fixed_metric_path', None)
        # Coalesce None overrides to safe defaults to avoid NaNs during init
        temp_override = metric_cfg.get('temperature_override', None)
        if temp_override is None:
            temp_override = 0.2
        reg_override = metric_cfg.get('regularization_override', None)
        if reg_override is None:
            reg_override = 1e-4

        self.riemannian_strict = bool(getattr(self.config, 'riemannian_strict', False) or os.environ.get('RLVAE_STRICT', '1') == '1')
        self.modular_metric = MetricTensor(
            latent_dim=self.config.latent_dim,
            device=self.device,
            trainable=trainable,
            architecture=architecture,
            arch_kwargs=arch_kwargs,
            temperature=temp_override,
            regularization=reg_override,
            init_from_fixed=init_from_fixed,
            fixed_metric_path=fixed_metric_path,
            normalize_weight_sum=metric_cfg.get('normalize_weight_sum', False),
            weight_kernel=metric_cfg.get('weight_kernel', 'mahalanobis_normed'),
            weight_metric_normalization=metric_cfg.get('weight_metric_normalization', 'trace'),
            topk_weights=metric_cfg.get('topk_weights', None),
            regularization_mode=metric_cfg.get('regularization_mode', 'precision'),
            use_background_identity=metric_cfg.get('use_background_identity', None),
        )
        # Optional: tighten spectral bounds/identity mixing from config to control condition number
        try:
            if 'eig_floor_abs' in metric_cfg and metric_cfg['eig_floor_abs'] is not None:
                self.modular_metric.eig_floor_abs = float(metric_cfg['eig_floor_abs'])
            if 'eig_ceiling' in metric_cfg and metric_cfg['eig_ceiling'] is not None:
                self.modular_metric.eig_ceiling = float(metric_cfg['eig_ceiling'])
            if 'bg_strength' in metric_cfg and metric_cfg['bg_strength'] is not None:
                self.modular_metric.bg_strength = float(metric_cfg['bg_strength'])
            if 'bg_floor' in metric_cfg and metric_cfg['bg_floor'] is not None:
                self.modular_metric.bg_floor = float(metric_cfg['bg_floor'])
            if 'use_background_identity' in metric_cfg and metric_cfg['use_background_identity'] is not None:
                self.modular_metric.use_background_identity = bool(metric_cfg['use_background_identity'])
        except Exception:
            pass
        
        # 🚀 NEW: Initialize modular metric loader
        self.metric_loader = MetricLoader(device=self.device)

        # Optionally load metric directly from config.metric.path to avoid
        # later ambiguous reloads via pretrained. Prefer this if provided.
        try:
            metric_cfg_path = None
            if hasattr(self.config, 'metric') and hasattr(self.config.metric, 'get'):
                metric_cfg_path = self.config.metric.get('path', None)
            if metric_cfg_path:
                md = self.metric_loader.load_from_file(
                    metric_cfg_path,
                    temperature_override=self.config.metric.get('temperature_override', None),
                    regularization_override=self.config.metric.get('regularization_override', None),
                )
                self.modular_metric.load_pretrained(**md)

                # Expose safe wrappers immediately
                def _G_impl(z: torch.Tensor) -> torch.Tensor:
                    return self.modular_metric.compute_metric(z)
                def _Ginv_impl(z: torch.Tensor) -> torch.Tensor:
                    return self.modular_metric.compute_inverse_metric(z)
                self.G = _G_impl
                self.G_inv = _Ginv_impl
                self._metric_ready = True
                # Backward-compatible buffers for other subsystems
                self.centroids_tens = self.modular_metric.centroids
                self.M_tens = self.modular_metric.metric_matrices
                self.temperature = getattr(self.modular_metric, 'temperature', torch.tensor(0.1, device=self.device))
                self.lbd = getattr(self.modular_metric, 'regularization', torch.tensor(0.01, device=self.device))
                print(f"✅ Loaded metric from config.metric.path: {metric_cfg_path}")
        except Exception as _e:
            print(f"[METRIC CONFIG LOAD] Warning: { _e }")
        
        # 🚀 NEW: Initialize modular loss manager
        metric_reg_weight = metric_cfg.get('metric_reg_weight', 0.0)
        metric_reg_type = metric_cfg.get('metric_reg_type', 'none')
        metric_reg_target = metric_cfg.get('metric_reg_target', 0.0)

        loss_cfg = getattr(self.config, 'losses', {}) if hasattr(self.config, 'losses') else {}
        # KL options sourced from losses config when available
        kl_use_metric_normalization = loss_cfg.get('kl_use_metric_normalization', getattr(self.config, 'kl_use_metric_normalization', True))
        kl_metric_norm_mode = loss_cfg.get('kl_metric_norm_mode', getattr(self.config, 'kl_metric_norm_mode', 'geomean'))
        kl_amp_safe = loss_cfg.get('kl_amp_safe', getattr(self.config, 'kl_amp_safe', True))
        kl_metric_eval_point = loss_cfg.get('kl_metric_eval_point', getattr(self.config, 'kl_metric_eval_point', 'z'))
        mu_l2_weight = float(loss_cfg.get('mu_l2_weight', getattr(self.config, 'mu_l2_weight', 0.0)))
        kl_prior_mode_cfg = loss_cfg.get('kl_prior_mode', getattr(self.config, 'kl_prior_mode', 'uniform'))
        volume_bias_weight = float(loss_cfg.get('volume_bias_weight', getattr(self.config, 'volume_bias_weight', 1.0)))
        volume_grad_scale = float(loss_cfg.get('volume_grad_scale', getattr(self.config, 'volume_grad_scale', 1.0)))
        use_pushforward_metric = loss_cfg.get('use_pushforward_metric', getattr(self.config, 'use_pushforward_metric', None))
        use_flow_corrections = loss_cfg.get('use_flow_corrections', getattr(self.config, 'use_flow_corrections', None))
        self._volume_bias_weight_cfg = volume_bias_weight
        self._volume_grad_scale_cfg = volume_grad_scale
        
        # Route tracing: LossManager config
        if not hasattr(self, '_loss_config_traced'):
            print(f"[ROUTE] LossManager: kl_metric_eval_point={self.config_resolved.get('kl_metric_eval_point', 'z')}, "
                  f"kl_prior_mode={self.config_resolved.get('kl_prior_mode', 'uniform')}, "
                  f"mu_l2_weight={self.config_resolved.get('mu_l2_weight', 0.0)}")
            self._loss_config_traced = True
        
        # Allow choosing metric representation from config (default 'g'), and pass mu regularizers from config
        metric_rep_pref = loss_cfg.get('metric_representation', 'g')
        mu_centroid_weight = float(loss_cfg.get('mu_centroid_weight', getattr(self.config, 'mu_centroid_weight', 5.0)))
        mu_volume_weight = float(loss_cfg.get('mu_volume_weight', getattr(self.config, 'mu_volume_weight', 0.0)))
        self.loss_manager = LossManager(
            beta=float(loss_cfg.get('beta', self.config_resolved.get('beta', 1.0))),
            riemannian_beta=float(loss_cfg.get('riemannian_beta', self.config_resolved.get('riemannian_beta', self.config_resolved.get('beta', 1.0)))),
            loop_penalty_weight=self.config_resolved.get('loop', {}).get('penalty', 0.0),
            device=self.device,
            metric_representation=metric_rep_pref,
            metric_reg_weight=metric_reg_weight,
            metric_reg_type=metric_reg_type,
            metric_reg_target=metric_reg_target,
            mu_l2_weight=mu_l2_weight,
            mu_centroid_weight=mu_centroid_weight,
            mu_volume_weight=mu_volume_weight,
            kl_prior_mode=kl_prior_mode_cfg,
            kl_use_metric_normalization=kl_use_metric_normalization,
            kl_metric_norm_mode=kl_metric_norm_mode,
            kl_amp_safe=kl_amp_safe,
            kl_metric_eval_point=kl_metric_eval_point,
            volume_bias_weight=volume_bias_weight,
            volume_grad_scale=volume_grad_scale,
            use_pushforward_metric=use_pushforward_metric,
            use_flow_corrections=use_flow_corrections,
            kl_include_logZ_in_loss=bool(loss_cfg.get('kl_include_logZ_in_loss', getattr(self.config, 'kl_include_logZ_in_loss', False))),
            kl_flip_logq_sign=bool(loss_cfg.get('kl_flip_logq_sign', getattr(self.config, 'kl_flip_logq_sign', False))),
        )
        # Route tracing: LossManager created successfully
        if not hasattr(self, '_loss_created_traced'):
            print(f"[ROUTE] LossManager created with resolved config")
            self._loss_created_traced = True
        
        # Resolve flow safety clamps (support both flattened and nested configs)
        flow_output_clip = getattr(self.config, "flow_output_clip", None)
        if flow_output_clip is None and hasattr(self.config, "flows"):
            try:
                flow_output_clip = getattr(self.config.flows, "output_clip", None)
            except Exception:
                flow_output_clip = None
        flow_output_clip = float(flow_output_clip) if flow_output_clip is not None else 50.0

        flow_logdet_clip = getattr(self.config, "flow_logdet_clip", None)
        if flow_logdet_clip is None and hasattr(self.config, "flows"):
            try:
                flow_logdet_clip = getattr(self.config.flows, "logdet_clip", None)
            except Exception:
                flow_logdet_clip = None
        flow_logdet_clip = float(flow_logdet_clip) if flow_logdet_clip is not None else 20.0

        # 🚀 NEW: Initialize modular flow manager (replace the one from parent)
        self.flow_manager = FlowManager(
            latent_dim=self.config.latent_dim,
            n_flows=self.config.n_flows,
            flow_hidden_size=self.config.flow_hidden_size,
            flow_n_blocks=self.config.flow_n_blocks,
            flow_n_hidden=self.config.flow_n_hidden,
            device=self.device,
            output_clip=flow_output_clip,
            logdet_clip=flow_logdet_clip,
        )
        # Enable a one-time autograd check of logdet direction in 2D to ensure consistency
        try:
            if int(self.config.latent_dim) == 2:
                self.flow_manager.enable_logdet_verify = True
        except Exception:
            pass

        # 🚀 NEW: Initialize posterior sampler and override base method to use component
        self.posterior_sampler = PosteriorSampler(self)
        # Ensure training uses modular sampler (affects super().forward)
        self.sample_metric_aware_posterior = self.posterior_sampler.sample_metric_aware_posterior

        # Metric update manager (modular replacement of original metric updates)
        self.modular_update_metric_enabled = bool(getattr(self.config, 'update_metric_during_training', False))
        if self.modular_update_metric_enabled:
            freq = int(getattr(self.config, 'metric_update_frequency', 100))
            reg = float(getattr(self.config, 'metric_update_regularization', 0.01))
            temp = float(getattr(self.config, 'metric_update_temperature', 0.1))
            self.metric_update_manager = MetricUpdateManager(
                metric_tensor=self.modular_metric,
                frequency=freq,
                regularization=reg,
                temperature=temp,
                device=self.device,
            )
            # Disable original in-base metric update to avoid double updates
            try:
                self.update_metric_during_training = False
            except Exception:
                pass
        else:
            self.metric_update_manager = None
        
        # 🚀 NEW: Initialize sampler manager for RHMC and other advanced sampling
        self._sampler_manager = SamplerManager(self)
        
        # 🚀 NEW: Initialize sampling components (including RHMC posterior)
        self._setup_sampling_components()
        
        print(f"✅ Initialized all modular components for {self.model_name}")
        
        # CRITICAL: Load pretrained components AFTER all setup to avoid overwriting
        self._load_pretrained_components_modular()
        
    def _setup_encoder_decoder(self):
        """Setup encoder and decoder using modular managers."""
        
        # Get encoder configuration
        encoder_config = self.config.get('encoder', {})
        print(f"🔍 DEBUG: Raw encoder config: {encoder_config}")
        
        # Handle different config types
        if isinstance(encoder_config, str):
            # Simple string configuration
            encoder_arch = encoder_config
            encoder_config = {'architecture': encoder_arch}
        elif hasattr(encoder_config, '__getitem__') and hasattr(encoder_config, 'get'):
            # Dict-like object (DictConfig, dict, etc.) - use as-is
            encoder_config = dict(encoder_config)  # Convert to regular dict
        else:
            encoder_config = {}
        
        # Default to MLP if not specified
        if 'architecture' not in encoder_config:
            encoder_config['architecture'] = 'mlp'
        
        print(f"🔍 DEBUG: Final encoder config: {encoder_config}")
        print(f"🔍 DEBUG: Encoder architecture: {encoder_config['architecture']}")
        
        # Get decoder configuration
        decoder_config = self.config.get('decoder', {})
        print(f"🔍 DEBUG: Raw decoder config: {decoder_config}")
        
        # Handle different config types
        if isinstance(decoder_config, str):
            # Simple string configuration
            decoder_arch = decoder_config
            decoder_config = {'architecture': decoder_arch}
        elif hasattr(decoder_config, '__getitem__') and hasattr(decoder_config, 'get'):
            # Dict-like object (DictConfig, dict, etc.) - use as-is
            decoder_config = dict(decoder_config)  # Convert to regular dict
        else:
            decoder_config = {}
        
        # Default to MLP if not specified
        if 'architecture' not in decoder_config:
            decoder_config['architecture'] = 'mlp'
        
        print(f"🔍 DEBUG: Final decoder config: {decoder_config}")
        print(f"🔍 DEBUG: Decoder architecture: {decoder_config['architecture']}")
        
        # Create encoder manager and encoder
        encoder_manager = EncoderManager(
            input_dim=tuple(self.config.input_dim),
            latent_dim=self.config.latent_dim,
            architecture=encoder_config['architecture'],
            config=encoder_config
        )
        # Create decoder manager and decoder
        decoder_manager = DecoderManager(
            input_dim=tuple(self.config.input_dim),
            latent_dim=self.config.latent_dim,
            architecture=decoder_config['architecture'],
            config=decoder_config
        )
        # Store managers for potential later use
        self.encoder_manager = encoder_manager
        self.decoder_manager = decoder_manager
        # Assign encoder/decoder
        self.encoder = encoder_manager.encoder
        self.decoder = decoder_manager.decoder
        # Move both encoder and decoder to the correct device
        device = getattr(self, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        # Also ensure the manager modules are on device (if they have parameters)
        self.encoder_manager = self.encoder_manager.to(device)
        self.decoder_manager = self.decoder_manager.to(device)
        print(f"✅ Created modular encoder: {encoder_config['architecture']}")
        print(f"✅ Created modular decoder: {decoder_config['architecture']}")
        
    def _setup_from_config(self):
        """Setup model components from configuration."""
        
        # Configure loop mode
        if hasattr(self, 'set_loop_mode'):
            self.set_loop_mode(
                self.config.loop.mode, 
                self.config.loop.penalty
            )
        
        # Load pretrained components using modular approach
        print(f"[DEBUG] Config structure: hasattr pretrained={hasattr(self.config, 'pretrained')}")
        if hasattr(self.config, 'pretrained'):
            print(f"[DEBUG] Pretrained config: {self.config.pretrained}")
            print(f"[DEBUG] Metric path: {getattr(self.config.pretrained, 'metric_path', 'NOT_FOUND')}")
        # Note: _load_pretrained_components_modular() is now called in __init__ after all setup
        
        # Configure Riemannian sampling
        if hasattr(self.config, 'sampling') and getattr(self.config.sampling, 'use_riemannian', False):
            method_cfg = getattr(self.config.sampling, 'method', 'custom')
            method = "custom" if method_cfg in ['geodesic', 'enhanced', 'basic'] else method_cfg
            self.enable_pure_rhvae(enable=True, method=method)
            self._riemannian_method = method_cfg
        else:
            self.enable_pure_rhvae(enable=False)
    
    def _load_pretrained_components_modular(self):
        """Load pretrained components using modular approach."""
        # Idempotency: avoid re-loading the same pretrained components multiple times
        if getattr(self, '_pretrained_loaded_once', False):
            print("[PRETRAINED DEBUG] Skipping pretrained load (already loaded once)")
            return
        # Debug: Check if pretrained config exists
        print(f"[PRETRAINED DEBUG] hasattr pretrained: {hasattr(self.config, 'pretrained')}")
        if hasattr(self.config, 'pretrained'):
            print(f"[PRETRAINED DEBUG] pretrained config: {self.config.pretrained}")
            encoder_path_val = getattr(self.config.pretrained, 'encoder_path', None)
            decoder_path_val = getattr(self.config.pretrained, 'decoder_path', None)
            print(f"[PRETRAINED DEBUG] encoder_path: {encoder_path_val}")
            print(f"[PRETRAINED DEBUG] decoder_path: {decoder_path_val}")
        
        # Load encoder and decoder using managers
        if hasattr(self.config, 'pretrained') and getattr(self.config.pretrained, 'encoder_path', None):
            encoder_path = Path(self.config.pretrained.encoder_path)
            print(f"[PRETRAINED DEBUG] encoder_path exists: {encoder_path.exists()}")
            if encoder_path.exists():
                print(f"🔧 Loading encoder from: {encoder_path}")
                
                # Debug: Show encoder architecture before loading
                print(f"[ENCODER DEBUG] Current encoder type: {type(self.encoder).__name__}")
                print(f"[ENCODER DEBUG] Current encoder architecture: {self.encoder_manager.architecture}")
                
                # Test reconstruction before loading pretrained weights
                print(f"[ENCODER DEBUG] Testing encoder BEFORE loading pretrained weights...")
                test_input = torch.randn(1, *self.config.input_dim).to(self.device)
                with torch.no_grad():
                    encoder_out = self.encoder(test_input)
                    decoder_out = self.decoder(encoder_out.embedding)
                    if isinstance(decoder_out, dict):
                        recon_before = decoder_out.get("reconstruction", next(iter(decoder_out.values())))
                    elif hasattr(decoder_out, 'reconstruction'):
                        recon_before = decoder_out.reconstruction
                    else:
                        recon_before = decoder_out
                    if isinstance(recon_before, torch.Tensor):
                        print(f"[ENCODER DEBUG] Recon BEFORE loading: min={recon_before.min():.4f}, max={recon_before.max():.4f}, mean={recon_before.mean():.4f}")
                
                self.encoder_manager.load_pretrained(encoder_path)
                
                # CRITICAL: Update the main model's encoder reference to use the loaded weights
                self.encoder = self.encoder_manager.encoder
                
                # Test reconstruction after loading pretrained weights
                print(f"[ENCODER DEBUG] Testing encoder AFTER loading pretrained weights...")
                with torch.no_grad():
                    encoder_out = self.encoder(test_input)
                    decoder_out = self.decoder(encoder_out.embedding)
                    if isinstance(decoder_out, dict):
                        recon_after = decoder_out.get("reconstruction", next(iter(decoder_out.values())))
                    elif hasattr(decoder_out, 'reconstruction'):
                        recon_after = decoder_out.reconstruction
                    else:
                        recon_after = decoder_out
                    if isinstance(recon_after, torch.Tensor):
                        print(f"[ENCODER DEBUG] Recon AFTER loading: min={recon_after.min():.4f}, max={recon_after.max():.4f}, mean={recon_after.mean():.4f}")
                    
                    # Debug: Check if encoder and decoder are the same objects
                    print(f"[ENCODER DEBUG] Encoder is same object: {self.encoder is self.encoder_manager.encoder}")
                    print(f"[ENCODER DEBUG] Decoder is same object: {self.decoder is self.decoder_manager.decoder}")
                
                print("✅ Loaded encoder weights")
        
        if hasattr(self.config, 'pretrained') and getattr(self.config.pretrained, 'decoder_path', None):
            decoder_path = Path(self.config.pretrained.decoder_path)
            print(f"[PRETRAINED DEBUG] decoder_path exists: {decoder_path.exists()}")
            if decoder_path.exists():
                print(f"🔧 Loading decoder from: {decoder_path}")
                
                # Debug: Show decoder architecture before loading
                print(f"[DECODER DEBUG] Current decoder type: {type(self.decoder).__name__}")
                print(f"[DECODER DEBUG] Current decoder architecture: {self.decoder_manager.architecture}")
                
                self.decoder_manager.load_pretrained(decoder_path)
                
                # CRITICAL: Update the main model's decoder reference to use the loaded weights
                self.decoder = self.decoder_manager.decoder
                
                # Test reconstruction after loading decoder weights
                print(f"[DECODER DEBUG] Testing AFTER loading decoder weights...")
                test_input = torch.randn(1, *self.config.input_dim).to(self.device)
                with torch.no_grad():
                    encoder_out = self.encoder(test_input)
                    decoder_out = self.decoder(encoder_out.embedding)
                    if isinstance(decoder_out, dict):
                        recon_final = decoder_out.get("reconstruction", next(iter(decoder_out.values())))
                    elif hasattr(decoder_out, 'reconstruction'):
                        recon_final = decoder_out.reconstruction
                    else:
                        recon_final = decoder_out
                    if isinstance(recon_final, torch.Tensor):
                        print(f"[DECODER DEBUG] Final recon: min={recon_final.min():.4f}, max={recon_final.max():.4f}, mean={recon_final.mean():.4f}")
                    # Debug: Check if encoder and decoder are the same objects
                    print(f"[DECODER DEBUG] Encoder is same object: {self.encoder is self.encoder_manager.encoder}")
                    print(f"[DECODER DEBUG] Decoder is same object: {self.decoder is self.decoder_manager.decoder}")
                
                print("✅ Loaded decoder weights")
        # Mark as loaded to prevent duplicate loads later in the pipeline
        self._pretrained_loaded_once = True
        
        # 🚀 NEW: Load metrics using modular approach
        print(f"[DEBUG] Checking metric loading conditions:")
        print(f"  - hasattr pretrained: {hasattr(self.config, 'pretrained')}")
        if hasattr(self.config, 'pretrained'):
            metric_path_val = getattr(self.config.pretrained, 'metric_path', None)
            print(f"  - metric_path: {metric_path_val}")
            if metric_path_val:
                metric_path = Path(metric_path_val)
                print(f"  - metric_path exists: {metric_path.exists()}")
                if metric_path.exists():
                    try:
                        # Use new modular metric loader
                        metric_data = self.metric_loader.load_from_file(
                            metric_path,
                            temperature_override=self.config.metric.get('temperature_override'),
                            regularization_override=self.config.metric.get('regularization_override')
                        )
                        
                        # ADAPT metric dimension to match model.latent_dim if needed
                        metric_data = self._adapt_metric_dims(metric_data)
                        # Load into modular metric tensor (no fallback)
                        self.modular_metric.load_pretrained(**metric_data)
                        
                        # Create backward-compatible interface functions
                        self._create_backward_compatible_interface()
                        
                        # Setup sampling components
                        self._setup_sampling_components()
                        
                        print("✅ Loaded metrics using modular components (2x faster!)")
                        try:
                
                            if os.environ.get('RLVAE_TRACE', '0') == '1':
                                print(f"Using Stage B metric: {metric_path}")
                        except Exception:
                            pass
                    
                    except Exception as e:
                        if self.riemannian_strict:
                            raise RuntimeError(f"Strict Riemannian mode: failed to load/adapt metric: {e}")
                        print(f"⚠️ Failed to load metrics with modular components: {e}")
                        print("🔄 Falling back to original implementation...")
                        self._fallback_to_original_metric_loading()
    
    def _create_backward_compatible_interface(self):
        """Create backward-compatible interface for existing code."""
        
        # Create G and G_inv functions that use modular components
        def G_modular(z: torch.Tensor) -> torch.Tensor:
            """Modular metric tensor computation."""
            return self.modular_metric.compute_metric(z)
        
        def G_inv_modular(z: torch.Tensor) -> torch.Tensor:
            """Modular inverse metric tensor computation."""
            return self.modular_metric.compute_inverse_metric(z)
        
        # Replace the original functions
        self.G = G_modular
        self.G_inv = G_inv_modular
        
        # Store modular metric parameters for backward compatibility
        self.centroids_tens = self.modular_metric.centroids
        self.M_tens = self.modular_metric.metric_matrices
        self.temperature = self.modular_metric.temperature
        self.lbd = self.modular_metric.regularization
        
        print(f"[MODULAR METRIC] Backward-compatible interface created")
        print(f"  - Modular metric loaded: {self.modular_metric._is_loaded}")
        print(f"  - Centroids shape: {self.centroids_tens.shape}")
        print(f"  - Metric matrices shape: {self.M_tens.shape}")
        
        # Debug: Test if metric is working
        print(f"[METRIC DEBUG] Metric loaded: {self.modular_metric.is_loaded()}")
        if self.modular_metric.is_loaded():
            test_z = torch.randn(1, self.latent_dim).to(self.device)
            try:
                G_test = self.G(test_z)
                print(f"[METRIC DEBUG] G(z) shape: {G_test.shape}, min: {G_test.min():.4f}, max: {G_test.max():.4f}")
                print(f"[METRIC DEBUG] Metric centroids: {self.centroids_tens.shape if self.centroids_tens is not None else 'None'}")
            except Exception as e:
                print(f"[METRIC DEBUG] Error testing metric: {e}")
        
        # Debug: Check model configuration
        print(f"[CONFIG DEBUG] Model configuration:")
        print(f"  - posterior_type: {getattr(self, 'posterior_type', 'NOT_SET')}")
        print(f"  - riemannian_kl_mode: {getattr(self, 'riemannian_kl_mode', 'NOT_SET')}")
        print(f"  - riemannian_beta: {getattr(self, 'riemannian_beta', 'NOT_SET')}")
        print(f"  - beta: {getattr(self, 'beta', 'NOT_SET')}")
        
        print("✅ Created backward-compatible metric interface")

    def _adapt_metric_dims(self, md: dict) -> dict:
        """Adapt a KxDsrc metric to model latent_dim Ddst via block-diagonal padding/slicing.
        - If Dsrc < Ddst: zero-pad centroids and block‑diag(Ginv, I)
        - If Dsrc > Ddst: slice first Ddst dims with warning
        """
        try:
            K, d_src = md['centroids'].shape
            d_dst = int(self.config.latent_dim)
            if d_src == d_dst:
                return md
            import torch
            if d_src < d_dst:
                pad = d_dst - d_src
                C = md['centroids']
                M = md['metric_matrices']
                C_pad = torch.zeros(K, d_dst, device=C.device, dtype=C.dtype)
                C_pad[:, :d_src] = C
                M_pad = torch.zeros(K, d_dst, d_dst, device=M.device, dtype=M.dtype)
                M_pad[:, :d_src, :d_src] = M
                eye = torch.eye(pad, device=M.device, dtype=M.dtype).unsqueeze(0).expand(K, pad, pad)
                M_pad[:, d_src:, d_src:] = eye
                print(f"[METRIC ADAPT] Padded metric from D={d_src} to D={d_dst} with identity on extra dims")
                return {
                    'centroids': C_pad,
                    'metric_matrices': M_pad,
                    'temperature': md['temperature'],
                    'regularization': md['regularization']
                }
            else:
                # d_src > d_dst: slice
                print(f"[METRIC ADAPT] Slicing metric from D={d_src} to D={d_dst}")
                return {
                    'centroids': md['centroids'][:, :d_dst],
                    'metric_matrices': md['metric_matrices'][:, :d_dst, :d_dst],
                    'temperature': md['temperature'],
                    'regularization': md['regularization']
                }
        except Exception as e:
            if self.riemannian_strict:
                raise
            print(f"⚠️ METRIC ADAPT failed: {e}. Proceeding without adaptation.")
            return md
    
    def _setup_sampling_components(self):
        """Setup sampling components using modular metric."""
        # Idempotent guard to avoid double initialization
        if getattr(self, '_sampling_components_initialized', False):
            print("[RHMC CONFIG] Sampling components already initialized; skipping re-create")
            return
        # Create multiple sampler options
        self._riemannian_sampler = RiemannianSampler(self)
        self._official_sampler = OfficialRHVAESampler(self)
        # Optional RHMC sampler for analysis/generation
        if RiemannianHMCSampler is not None:
            try:
                self._hmc_sampler = RiemannianHMCSampler(self)
            except Exception:
                self._hmc_sampler = None
        else:
            self._hmc_sampler = None
        # RHMC posterior component (training posterior alternative)
        try:
            # Pass the actual config dict, not the model object
            rhmc_config = {}
            if hasattr(self.config, 'posterior') and self.config.posterior is not None:
                try:
                    rhmc_config.update(OmegaConf.to_container(self.config.posterior, resolve=True))
                except Exception:
                    rhmc_config.update(dict(self.config.posterior))
            # Top-level fallbacks override if provided
            for k in (
                'rhmc_steps', 'rhmc_step_size', 'rhmc_alpha', 'rhmc_eps_reg',
                'max_momentum_norm', 'max_velocity_norm', 'max_position_step', 'max_position_norm',
                'min_cov_eig', 'eps_regularization', 'use_factorized_G_mu', 'kl_prior_mode'
            ):
                if hasattr(self.config, k):
                    rhmc_config[k] = getattr(self.config, k)
            rhmc_config.setdefault('volume_grad_scale', self._volume_grad_scale_cfg)
            rhmc_config.setdefault('volume_bias_weight', self._volume_bias_weight_cfg)
            # DEBUG: Show rhmc_eps_reg in config
            print(f"[RHMC CONFIG DEBUG] rhmc_eps_reg in rhmc_config: {rhmc_config.get('rhmc_eps_reg', 'NOT FOUND')}")
            if 'kl_prior_mode' not in rhmc_config:
                try:
                    mode = None
                    if hasattr(self.config, 'kl_prior_mode'):
                        mode = getattr(self.config, 'kl_prior_mode')
                    elif hasattr(self, 'config_resolved'):
                        mode = self.config_resolved.get('kl_prior_mode')
                    if mode is not None:
                        rhmc_config['kl_prior_mode'] = str(mode)
                except Exception:
                    pass
            print(f"[RHMC CONFIG] Creating RHMC posterior with config: {rhmc_config}")
            self.posterior_sampler_rhmc = RiemannianRHMCPosterior(self, rhmc_config)
            # Enforce config on instance in case sampler sets internal defaults
            try:
                for k in ('rhmc_steps','rhmc_step_size','rhmc_alpha','rhmc_eps_reg',
                          'max_momentum_norm','max_velocity_norm','max_position_step','max_position_norm',
                          'min_cov_eig','eps_regularization','use_factorized_G_mu',
                          'volume_grad_scale','volume_bias_weight'):
                    if k in rhmc_config:
                        setattr(self.posterior_sampler_rhmc, k if k != 'eps_regularization' else 'eps_reg', rhmc_config[k])
            except Exception:
                pass
        except Exception as e:
            print(f"[RHMC CONFIG] Failed to create RHMC posterior: {e}")
            raise
        
        print("✅ Setup modular sampling components")
        self._sampling_components_initialized = True

    # Convenience wrappers for RHMC (analysis/generation only)
    def sample_rhmc_prior(self, n_samples: int = 100, **kwargs):
        if getattr(self, '_hmc_sampler', None) is None:
            raise RuntimeError("RHMC sampler not available. Ensure src.models.samplers is importable.")
        return self._hmc_sampler.sample(n_samples=n_samples, **kwargs)

    def sample_rhmc_posterior(self, mu, log_var, **kwargs):
        if getattr(self, '_hmc_sampler', None) is None:
            raise RuntimeError("RHMC sampler not available. Ensure src.models.samplers is importable.")
        return self._hmc_sampler.sample_posterior(mu, log_var, **kwargs)

    # Override KL to use modular LossManager (ensures component edits affect training)
    def compute_riemannian_metric_kl_loss(self, mu, log_var, z_samples):
        metric_tensor = getattr(self, 'modular_metric', None)
        if metric_tensor is None:
            # Fallback to parent implementation if modular metric unavailable
            try:
                return super().compute_riemannian_metric_kl_loss(mu, log_var, z_samples)
            except Exception:
                # Last resort: standard KL
                log_var_clamped = torch.clamp(log_var, -10.0, 10.0)
                print(f"[KL DEBUG!!!!!????] log_var_clamped: {log_var_clamped.min():.4f}, {log_var_clamped.max():.4f}")
                return -0.5 * torch.sum(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp(), dim=1).mean()
        return self.loss_manager.compute_riemannian_kl_loss(mu, log_var, z_samples, metric_tensor)
    
    def _fallback_to_original_metric_loading(self):
        """Fallback to original metric loading if modular approach fails."""
        
        # Use the parent class's metric loading
        if hasattr(super(), 'load_pretrained_metrics'):
            super().load_pretrained_metrics(
                self.config.pretrained.metric_path,
                self.config.metric.get('temperature_override')
            )
        else:
            print("⚠️ No pretrained metric available, initializing with default values")
            # Initialize metric tensor with default values for Stage C
            self._initialize_default_metric()
        # Ensure sampling components are available even after fallback
        try:
            self._setup_sampling_components()
        except Exception:
            pass
    
    def _initialize_default_metric(self):
        """Initialize metric tensor with default values when no pretrained metric is available."""
        print("[METRIC INIT] Initializing default metric tensor for Stage C")
        
        # Create default centroids and metric matrices
        n_centroids = 10  # Default number of centroids
        centroids = torch.randn(n_centroids, self.latent_dim, device=self.device) * 0.5
        metric_matrices = torch.eye(self.latent_dim, device=self.device).unsqueeze(0).repeat(n_centroids, 1, 1)
        
        # Load into modular metric tensor
        self.modular_metric.load_pretrained(
            centroids=centroids,
            metric_matrices=metric_matrices,
            temperature=0.1,
            regularization=0.01
        )
        
        # Create backward-compatible interface
        self._create_backward_compatible_interface()
        
        print(f"[METRIC INIT] Default metric initialized with {n_centroids} centroids")
    
    def _setup_metric_transfer_hook(self):
        """Setup hook to transfer loaded metric data to modular metric tensor."""
        # Override the parent class's metric loading to also load modular metric
        original_load_pretrained_metrics = super().load_pretrained_metrics
        
        def load_pretrained_metrics_with_transfer(metric_path, temperature_override=None):
            print(f"[METRIC TRANSFER] Parent loading metric from: {metric_path}")
            # Call parent method first
            original_load_pretrained_metrics(metric_path, temperature_override)
            
            # Transfer loaded data to modular metric tensor
            if hasattr(self, 'centroids_tens') and hasattr(self, 'M_tens'):
                print(f"[METRIC TRANSFER] Transferring to modular metric tensor")
                print(f"  - Centroids shape: {self.centroids_tens.shape}")
                print(f"  - Metric matrices shape: {self.M_tens.shape}")
                
                # Load into modular metric tensor
                self.modular_metric.load_pretrained(
                    centroids=self.centroids_tens,
                    metric_matrices=self.M_tens,
                    temperature=getattr(self, 'temperature', 0.1),
                    regularization=getattr(self, 'lbd', 0.01)
                )
                
                # Create backward-compatible interface
                self._create_backward_compatible_interface()
                
                print(f"[METRIC TRANSFER] ✅ Modular metric loaded: {self.modular_metric._is_loaded}")
            else:
                print(f"[METRIC TRANSFER] ⚠️ Parent metric loading didn't create required variables")
        
        # Replace the parent method
        self.load_pretrained_metrics = load_pretrained_metrics_with_transfer
    
    def _setup_modular_metric_fallback(self):
        """Setup fallback to load modular metric when G/G_inv are accessed."""
        # If metric is already loaded/ready, do not override wrappers again.
        try:
            if getattr(self, '_metric_ready', False) and getattr(self, 'modular_metric', None) is not None \
               and getattr(self.modular_metric, '_is_loaded', False):
                print("[METRIC FALLBACK] Skipping override (metric already loaded)")
                return
        except Exception:
            pass
        def G_with_fallback(z: torch.Tensor) -> torch.Tensor:
            """G method with fallback to load modular metric if needed."""
            if not self.modular_metric._is_loaded:
                if hasattr(self, 'centroids_tens') and hasattr(self, 'M_tens'):
                    self.modular_metric.load_pretrained(
                        centroids=self.centroids_tens,
                        metric_matrices=self.M_tens,
                        temperature=getattr(self, 'temperature', 0.1),
                        regularization=getattr(self, 'lbd', 0.01)
                    )
            return self.modular_metric.compute_metric(z)
        
        def G_inv_with_fallback(z: torch.Tensor) -> torch.Tensor:
            """G_inv method with fallback to load modular metric if needed."""
            if not self.modular_metric._is_loaded:
                if hasattr(self, 'centroids_tens') and hasattr(self, 'M_tens'):
                    self.modular_metric.load_pretrained(
                        centroids=self.centroids_tens,
                        metric_matrices=self.M_tens,
                        temperature=getattr(self, 'temperature', 0.1),
                        regularization=getattr(self, 'lbd', 0.01)
                    )
            return self.modular_metric.compute_inverse_metric(z)
        
        # Override the G and G_inv methods
        self.G = G_with_fallback
        self.G_inv = G_inv_with_fallback
        
        print(f"[METRIC FALLBACK] G and G_inv methods overridden with fallback loading")
    
    def load_pretrained_metrics(self, metric_path, temperature_override=None):
        """Override parent method to load metrics into modular metric tensor."""
        print(f"[MODULAR METRIC] load_pretrained_metrics called with path: {metric_path}")
        print(f"[MODULAR METRIC] Loading pretrained metrics from: {metric_path}")
        # Avoid duplicate loads/overwrites if metric already attached
        try:
            if getattr(self, '_metric_ready', False) and getattr(self, 'modular_metric', None) is not None \
               and getattr(self.modular_metric, '_is_loaded', False):
                print("[MODULAR METRIC] Skipping reload (metric already attached)")
                return
        except Exception:
            pass
        
        try:
            self._metric_source = str(metric_path)
            # First call parent method to load metric into parent variables
            super().load_pretrained_metrics(metric_path, temperature_override)
            
            # Then transfer the loaded data to modular metric tensor
            if hasattr(self, 'centroids_tens') and hasattr(self, 'M_tens'):
                print(f"[MODULAR METRIC] Transferring loaded metric to modular tensor")
                print(f"  - Centroids shape: {self.centroids_tens.shape}")
                print(f"  - Metric matrices shape: {self.M_tens.shape}")

                # Lazily create modular metric if not present
                if getattr(self, 'modular_metric', None) is None:
                    from .components.metric_tensor import MetricTensor
                    ld = int(getattr(self, 'latent_dim', self.M_tens.shape[-1]))
                    self.modular_metric = MetricTensor(latent_dim=ld, device=self.device)

                # Freeze and eval for safety
                try:
                    self.modular_metric.eval()
                    for p in self.modular_metric.parameters():
                        p.requires_grad_(False)
                except Exception:
                    pass

                # Load into modular metric tensor
                self.modular_metric.load_pretrained(
                    centroids=self.centroids_tens,
                    metric_matrices=self.M_tens,
                    temperature=float(getattr(self, 'temperature', 0.1)),
                    regularization=float(getattr(self, 'lbd', 0.01))
                )

                # Expose safe wrappers immediately
                def _G_impl(z: torch.Tensor) -> torch.Tensor:
                    return self.modular_metric.compute_metric(z)
                def _Ginv_impl(z: torch.Tensor) -> torch.Tensor:
                    return self.modular_metric.compute_inverse_metric(z)
                self.G = _G_impl
                self.G_inv = _Ginv_impl
                self._metric_ready = True

                # Backward compatibility buffers
                self.centroids_tens = self.modular_metric.centroids
                self.M_tens = self.modular_metric.metric_matrices
                self.temperature = getattr(self.modular_metric, 'temperature', torch.tensor(0.1, device=self.device))
                self.lbd = getattr(self.modular_metric, 'regularization', torch.tensor(0.01, device=self.device))

                # Sanity check
                try:
                    with torch.no_grad():
                        z_test = torch.zeros(2, int(self.latent_dim), device=self.device)
                        Gz = self.G(z_test)
                        Ginv = self.G_inv(z_test)
                        eye = torch.eye(Gz.shape[-1], device=self.device).unsqueeze(0)
                        err = torch.linalg.norm(Gz @ Ginv - eye).item()
                        eig = torch.linalg.eigvalsh(Gz.float())
                        emin, emax = eig.min().item(), eig.max().item()
                        print(f"[METRIC CHECK] ||G G^-1 - I|| ≈ {err:.2e}  |  eig(G): [{emin:.3e}, {emax:.3e}]")
                        assert not torch.isnan(eig).any() and emin > 0.0, "Loaded G is not SPD"
                except Exception as _e:
                    print(f"[METRIC CHECK] Warning: sanity check failed: {_e}")

                print("✅ Loaded metrics into modular metric tensor")
                print(f"[METRIC SOURCE] {self._metric_source}")
            else:
                print("⚠️ Parent metric loading didn't create centroids_tens/M_tens")
                
        except Exception as e:
            print(f"⚠️ Failed to load metrics into modular tensor: {e}")
            # Fall back to parent method only
            super().load_pretrained_metrics(metric_path, temperature_override)
    
    def _setup_metrics_tracking(self):
        """Initialize comprehensive metrics tracking."""
        self.metrics_history = {
            'reconstruction_loss': [],
            'kl_divergence': [],
            'cyclicity_error': [],
            'riemannian_kl': [],
            'total_loss': []
        }
        
        # Model-specific metrics
        if hasattr(self.config, 'sampling') and getattr(self.config.sampling, 'use_riemannian', False):
            self.metrics_history.update({
                'geodesic_preservation': [],
                'metric_conditioning': [],
                'manifold_regularity': []
            })
    
    def forward(self, x: torch.Tensor, compute_metrics: bool = False) -> dict:
        """
        Enhanced forward pass with standardized output format.
        
        Args:
            x: Input tensor [B, T, C, H, W]
            compute_metrics: Whether to compute additional metrics
            
        Returns:
            Dictionary with standardized keys for easy comparison
        """
        # Optional: modular metric update before main forward
        try:
            if self.training and self.metric_update_manager is not None and self.modular_metric.is_loaded():
                with torch.no_grad():
                    x0 = x[:, 0]
                    enc_out = self.encoder(x0)
                    mu_now = enc_out.embedding
                    self.metric_update_manager.collect(mu_now)
                    if self.metric_update_manager.ready():
                        updated = self.metric_update_manager.update()
                        if updated:
                            # Refresh backward-compatible interface tensors
                            self.centroids_tens = self.modular_metric.centroids
                            self.M_tens = self.modular_metric.metric_matrices
        except Exception:
            pass

        # Ensure encoder/decoder see correct input shape (e.g., for pythae MLPs)
        try:
            x0 = x[:, 0]
            obs_dim = tuple(x0.shape[1:])
            if hasattr(self.encoder, 'input_dim') and tuple(getattr(self.encoder, 'input_dim')) != obs_dim:
                setattr(self.encoder, 'input_dim', obs_dim)
            if hasattr(self.decoder, 'input_dim') and tuple(getattr(self.decoder, 'input_dim')) != obs_dim:
                setattr(self.decoder, 'input_dim', obs_dim)
        except Exception:
            pass
        # Use the fully modular forward to ensure modular sampling + loss path
        return self.forward_modular(x, compute_metrics=compute_metrics)
    
    def forward_modular(self, x: torch.Tensor, compute_metrics: bool = False) -> Dict[str, torch.Tensor]:
        """
        Fully modular forward pass using all modular components.
        
        Args:
            x: Input tensor [B, T, C, H, W]
            compute_metrics: Whether to compute additional metrics
            
        Returns:
            Dictionary with standardized keys for easy comparison
        """
        batch_size, n_obs = x.shape[:2]

        # Route tracing: model state and config
        if not hasattr(self, '_route_traced'):
            print(f"[ROUTE] Model: {self.model_name}, posterior_type: {self.posterior_type}")
            print(f"[ROUTE] Has modular_metric: {hasattr(self, 'modular_metric')}, use_riem_kl: {self.posterior_type in ['riemannian_metric', 'riemannian_rhmc']}")
            self._route_traced = True
        
        # Encode initial observation using modular encoder
        x_0 = x[:, 0]
        grad_enabled = torch.is_grad_enabled()
        inference_mode_fn = getattr(torch, "is_inference_mode_enabled", None)
        inference_mode_active = inference_mode_fn() if inference_mode_fn else False
        encoder_ctx = nullcontext()
        ctx_label = None
        if self.training:
            if inference_mode_active and hasattr(torch, "inference_mode"):
                encoder_ctx = torch.inference_mode(False)
                ctx_label = "inference_mode_off"
            elif not grad_enabled:
                encoder_ctx = torch.enable_grad()
                ctx_label = "enable_grad"
        with encoder_ctx:
            encoder_out = self.encoder(x_0)
        mu = encoder_out.embedding
        log_var = encoder_out.log_covariance
        stagec_debugger.log_event(
            "encoder_forward",
            {
                "x_shape": list(x_0.shape),
                "mu_shape": list(mu.shape),
                "log_var_shape": list(log_var.shape),
                "mu_dtype": str(mu.dtype),
                "log_var_dtype": str(log_var.dtype),
                "training": bool(self.training),
            },
        )

        if os.environ.get("RLVAE_DEBUG", "0") == "1":
            log_var_min = log_var.min().item()
            log_var_max = log_var.max().item()
            log_var_mean = log_var.mean().item()
            log_var_std = log_var.std(unbiased=False).item()
            variance = log_var.exp()
            var_mean = variance.mean().item()
            var_min = variance.min().item()
            var_max = variance.max().item()
            delta_msg = ""
            if self._debug_prev_logvar_stats is not None:
                delta_mean = log_var_mean - self._debug_prev_logvar_stats["log_var_mean"]
                delta_var = var_mean - self._debug_prev_logvar_stats["var_mean"]
                delta_msg = f" (Δmean={delta_mean:+.4e}, Δvar_mean={delta_var:+.4e})"
            print(
                f"[ENCODER DEBUG] log_var: min={log_var_min:.4f}, max={log_var_max:.4f}, "
                f"mean={log_var_mean:.4f}, std={log_var_std:.4f}{delta_msg}"
            )
            print(
                f"[ENCODER DEBUG] variance: min={var_min:.4e}, max={var_max:.4e}, mean={var_mean:.4e}"
            )
            self._debug_prev_logvar_stats = {
                "log_var_mean": log_var_mean,
                "log_var_std": log_var_std,
                "var_mean": var_mean,
            }
        # If mu unexpectedly has no grad, try a guarded re-encode under grad-enabled context
        try:
            if self.training and not getattr(mu, 'requires_grad', True):
                if not hasattr(self, "_encoder_grad_ctx_reported"):
                    print("[GRAD DEBUG] mu.requires_grad=False after initial encode; attempting to re-enable gradients")
                    self._encoder_grad_ctx_reported = True
                reencode_ctx = None
                if inference_mode_active and hasattr(torch, "inference_mode"):
                    reencode_ctx = torch.inference_mode(False)
                elif not grad_enabled:
                    reencode_ctx = torch.enable_grad()
                if reencode_ctx is not None:
                    with reencode_ctx:
                        encoder_out = self.encoder(x_0)
                else:
                    encoder_out = self.encoder(x_0)
                mu = encoder_out.embedding
                log_var = encoder_out.log_covariance
                if getattr(mu, 'requires_grad', False) is False and not hasattr(self, "_encoder_grad_warning_printed"):
                    print("[GRAD WARNING] μ still lacks gradients after re-encode. Check for upstream inference/no_grad contexts.")
                    self._encoder_grad_warning_printed = True
        except Exception as _e:
            print(f"[GRAD DEBUG] re-encode check error: {_e}")
        if self.training and ctx_label and not hasattr(self, "_encoder_ctx_diag_printed"):
            print(f"[ROUTE] Encoder context: '{ctx_label}' (training={self.training}, grad_enabled={grad_enabled}, inference_mode={inference_mode_active})")
            self._encoder_ctx_diag_printed = True
        # Gradient diagnostics: ensure μ receives gradients
        try:
            if self.training and getattr(mu, 'requires_grad', False) and not hasattr(self, "_mu_grad_hook_set"):
                mu.retain_grad()
                def _mu_grad_hook(g):
                    try:
                        if not hasattr(self, "_mu_grad_printed"):
                            print(f"[GRAD DEBUG] mu grad norm: {g.norm().item():.6f}, mean {g.mean().item():.6f}, std {g.std().item():.6f}")
                            self._mu_grad_printed = True
                    except Exception:
                        pass
                    return g
                mu.register_hook(_mu_grad_hook)
                self._mu_grad_hook_set = True
        except Exception as _e:
            print(f"[GRAD DEBUG] Unable to attach mu grad hook: {_e}")
        # Optional: align μ distribution to metric centroid stats (first-batch calibration)
        # One‑time integrity probe (debug): record μ before/after any alignment and report Δ
        _mu_probe_before = None
        try:
            if os.environ.get('RLVAE_DEBUG', '0') == '1' and not hasattr(self, '_mu_align_probe_done'):
                _mu_probe_before = mu.detach().clone()
        except Exception:
            _mu_probe_before = None
        if self.mu_alignment_enabled:
            try:
                if not self._mu_align_ready:
                    mu_mean = mu.detach().float().mean(dim=0)
                    mu_std = mu.detach().float().std(dim=0).clamp_min(1e-6)
                    scale = (self._mu_target_std / mu_std).to(mu.device)
                    bias = (self._mu_target_mean - scale * mu_mean).to(mu.device)
                    self._mu_align_scale = scale
                    self._mu_align_bias = bias
                    self._mu_align_ready = True
                    if os.environ.get('RLVAE_TRACE','0') == '1':
                        print(f"TRACE MU ALIGN: target_mean={self._mu_target_mean.tolist()}, target_std={self._mu_target_std.tolist()}")
                        print(f"TRACE MU ALIGN: batch_mean={mu_mean.tolist()}, batch_std={mu_std.tolist()}")
                        print(f"TRACE MU ALIGN: scale={scale.tolist()}, bias={bias.tolist()}")
                if self._mu_align_scale is not None and self._mu_align_bias is not None:
                    mu = mu * self._mu_align_scale + self._mu_align_bias
            except Exception:
                pass
        # Report μ change (once) for transparency; no effect on training
        try:
            if os.environ.get('RLVAE_DEBUG', '0') == '1' and not hasattr(self, '_mu_align_probe_done'):
                mu_after = mu.detach()
                if _mu_probe_before is None:
                    _mu_probe_before = mu_after.clone()
                delta = (mu_after - _mu_probe_before).abs()
                mean_delta = float(delta.mean().item()) if delta.numel() else 0.0
                max_delta = float(delta.max().item()) if delta.numel() else 0.0
                print(
                    f"[MU ALIGN PROBE] enabled={bool(self.mu_alignment_enabled)} "
                    f"mean|Δμ|={mean_delta:.3e} max|Δμ|={max_delta:.3e}"
                )
                self._mu_align_probe_done = True
        except Exception:
            pass
        if stagec_debugger.enabled:
            try:
                with torch.no_grad():
                    mu_stats = {
                        "mu_alignment_enabled": bool(getattr(self, "mu_alignment_enabled", False)),
                        "mu_align_ready": bool(getattr(self, "_mu_align_ready", False)),
                        "mu_mean": float(mu.mean().item()),
                        "mu_std": float(mu.std(unbiased=False).item()),
                        "log_var_mean": float(log_var.mean().item()),
                        "log_var_std": float(log_var.std(unbiased=False).item()),
                    }
            except Exception:
                mu_stats = {
                    "mu_alignment_enabled": bool(getattr(self, "mu_alignment_enabled", False)),
                    "mu_align_ready": bool(getattr(self, "_mu_align_ready", False)),
                }
            stagec_debugger.log_event("mu_alignment_status", mu_stats)
        # TRACE encoder outputs
        try:

            if os.environ.get('RLVAE_TRACE', '0') == '1':
                print(f"TRACE ENCODER mu: dtype={mu.dtype}, shape={tuple(mu.shape)}, mean={mu.mean().item():.4g}, std={mu.std().item():.4g}, min={mu.min().item():.4g}, max={mu.max().item():.4g}")
                if isinstance(log_var, torch.Tensor):
                    print(f"TRACE ENCODER log_var: dtype={log_var.dtype}, shape={tuple(log_var.shape)}, mean={log_var.mean().item():.4g}, std={log_var.std().item():.4g}, min={log_var.min().item():.4g}, max={log_var.max().item():.4g}")
        except Exception:
            pass

        # One-time metric orientation and conditioning diagnostics (Stage C)
        try:
            if not hasattr(self, '_metric_diag_printed'):
                if hasattr(self, 'G') and hasattr(self, 'G_inv'):
                    with torch.no_grad():
                        mu2 = mu[: min(mu.shape[0], 8)].detach() if isinstance(mu, torch.Tensor) else None
                        if mu2 is None or mu2.numel() == 0:
                            mu2 = torch.zeros(2, self.latent_dim, device=self.device)
                        G_mu = self.G(mu2)
                        Ginv_mu = self.G_inv(mu2)
                        # Conditioning and SPD checks
                        evals_G = torch.linalg.eigvalsh(G_mu.float())
                        cond_G = (evals_G.max(dim=-1).values / evals_G.min(dim=-1).values.clamp_min(1e-12)).median().item()
                        emin = evals_G.min().item(); emax = evals_G.max().item()
                        # Orientation consistency
                        Ginv_from_G = torch.linalg.inv(G_mu.float())
                        diff_inv = torch.linalg.norm(Ginv_mu.float() - Ginv_from_G).item()
                        G_from_Ginv = torch.linalg.inv(Ginv_mu.float())
                        diff_G = torch.linalg.norm(G_mu.float() - G_from_Ginv).item()
                        # Metadata
                        nC = int(getattr(self, 'centroids_tens', torch.empty(0)).shape[0]) if hasattr(self, 'centroids_tens') else -1
                        Tval = float(getattr(self, 'temperature', torch.tensor(float('nan'), device=self.device)).item()) if hasattr(self, 'temperature') else float('nan')
                        Lval = float(getattr(self, 'lbd', torch.tensor(float('nan'), device=self.device)).item()) if hasattr(self, 'lbd') else float('nan')
                        print(f"[STAGE C METRIC] n_centroids={nC}, T={Tval:.3g}, λ={Lval:.3g}")
                        print(f"[STAGE C METRIC] eig(G): min={emin:.3e}, max={emax:.3e}, cond~{cond_G:.2e}")
                        print(f"[STAGE C METRIC] ||G_inv - inv(G)||_F = {diff_inv:.3e}; ||G - inv(G_inv)||_F = {diff_G:.3e}")
                        eye = torch.eye(G_mu.shape[-1], device=G_mu.device).unsqueeze(0)
                        err = torch.linalg.norm(G_mu @ Ginv_mu - eye).item()
                        print(f"[STAGE C METRIC] ||G(mu) G_inv(mu) - I|| = {err:.3e}")
                else:
                    print("[STAGE C METRIC] Missing G/G_inv wrappers on model")
                self._metric_diag_printed = True
        except Exception as _e:
            print(f"[STAGE C METRIC] Diagnostic error: {_e}")
        
        # DEBUG: Always print what we're using for posterior sampling
        if not hasattr(self, '_debug_printed'):
            print(f"[ModularRiemannianFlowVAE DEBUG] posterior_type={self.posterior_type}")
            print(f"[ModularRiemannianFlowVAE DEBUG] hasattr modular_metric: {hasattr(self, 'modular_metric')}")
            print(f"[ModularRiemannianFlowVAE DEBUG] hasattr posterior_sampler_rhmc: {hasattr(self, 'posterior_sampler_rhmc')}")
            self._debug_printed = True
        
        stage_z0 = None
        stage_zS = None
        stage_zF = None
        sigma_mu = None

        # Sample latents using modular sampling
        if self.posterior_type == "riemannian_metric" and hasattr(self, 'modular_metric'):
            try:
        
                if os.environ.get('RLVAE_TRACE', '0') == '1':
                    print('USING LOCAL METRIC-ALIGNED GAUSSIAN (forward)')
            except Exception:
                pass
            z_0 = self.sample_metric_aware_posterior(mu, log_var)
            stage_z0 = z_0
            stage_zS = z_0
        elif self.posterior_type == "riemannian_rhmc":
            # If model exposes a RHMC posterior sampler component, use it
            rhmc_log_q = None
            rhmc_z0 = None
            rhmc_traj = None
            try:
                if hasattr(self, 'posterior_sampler_rhmc'):
                    print('✅ USING RIEMANNIAN RHMC POSTERIOR (forward)')
                    # Pull overrides from config when available
                    # Check both top-level and posterior.* paths
                    alpha_override = getattr(self.config, 'rhmc_alpha', None)
                    if alpha_override is None and hasattr(self.config, 'posterior'):
                        posterior_cfg = getattr(self.config, 'posterior', {})
                        if hasattr(posterior_cfg, 'get'):
                            alpha_override = posterior_cfg.get('rhmc_alpha', None)
                        else:
                            alpha_override = getattr(posterior_cfg, 'rhmc_alpha', None)
                    
                    eps_override = getattr(self.config, 'rhmc_eps_reg', None)
                    if eps_override is None and hasattr(self.config, 'posterior'):
                        posterior_cfg = getattr(self.config, 'posterior', {})
                        if hasattr(posterior_cfg, 'get'):
                            eps_override = posterior_cfg.get('rhmc_eps_reg', None)
                        else:
                            eps_override = getattr(posterior_cfg, 'rhmc_eps_reg', None)
                    
                    # DEBUG: Show what overrides were found
                    if os.environ.get("RLVAE_DEBUG", "0") == "1":
                        print(f"[CONFIG OVERRIDE] alpha_override={alpha_override}, eps_override={eps_override}")
                    with torch.enable_grad():
                        rhmc_ret = self.posterior_sampler_rhmc.sample_riemannian_rhmc_posterior(
                            mu,
                            log_var,
                            return_log_prob=True,
                            return_traj=True,
                            return_initial=True,
                            with_jacobian=bool(getattr(self.config, 'rhmc_kl_jacobian', False)),
                            alpha=alpha_override,
                            eps_reg=eps_override,
                        )
                    # Normalize return tuple
                    if isinstance(rhmc_ret, tuple):
                        # Expected ordering: (zK, log_q?, z0?, traj_info?)
                        zK = rhmc_ret[0]
                        stage_zS = zK
                        if len(rhmc_ret) > 1 and isinstance(rhmc_ret[1], torch.Tensor) and rhmc_ret[1].dim() == 1:
                            rhmc_log_q = rhmc_ret[1]
                        if len(rhmc_ret) > 2 and isinstance(rhmc_ret[2], torch.Tensor) and rhmc_ret[2].shape == mu.shape:
                            rhmc_z0 = rhmc_ret[2]
                        if len(rhmc_ret) > 3 and isinstance(rhmc_ret[3], dict):
                            rhmc_traj = rhmc_ret[3]
                            sigma_candidate = rhmc_traj.get('Sigma_mu', None)
                            if isinstance(sigma_candidate, torch.Tensor):
                                sigma_mu = sigma_candidate
                        stage_z0 = rhmc_z0 if rhmc_z0 is not None else stage_zS
                        z_0 = stage_zS
                    else:
                        z_0 = rhmc_ret
                        zK = z_0
                        stage_zS = zK
                        stage_z0 = zK
                    print(f"[RHMC DEBUG] zK shape: {z_0.shape}, mean: {z_0.mean().item():.4f}, std: {z_0.std().item():.4f}")
                else:
                    if getattr(self, 'riemannian_strict', False) or os.environ.get('RLVAE_STRICT','0') == '1':
                        raise RuntimeError('Strict Riemannian mode: RHMC posterior component not wired')
                    # Fall back to metric-aware Gaussian when RHMC component not wired (non-strict)
                    print('⚠️ RHMC posterior not wired; USING LOCAL METRIC-ALIGNED GAUSSIAN')
                    z_0 = self.sample_metric_aware_posterior(mu, log_var)
                    zK = z_0
            except Exception as e:
                if getattr(self, 'riemannian_strict', False) or os.environ.get('RLVAE_STRICT','0') == '1':
                    raise
                print(f"⚠️ RHMC posterior sampling failed in forward: {e}; falling back to standard reparam")
                eps = torch.randn_like(mu)
                z_0 = mu + eps * torch.exp(0.5 * log_var)
                zK = z_0
                stage_zS = zK
                stage_z0 = z_0
        else:
            print("⚠️ DEBUG: Falling back to standard Gaussian posterior sampling (no modular metric available or incorrect posterior_type)")
            eps = torch.randn_like(mu)
            z_0 = mu + eps * torch.exp(0.5 * log_var)
            zK = z_0
            stage_zS = zK
            stage_z0 = z_0
        # Route tracing: sampling method and z_0 stats
        if not hasattr(self, '_sampling_traced'):
            print(f"[ROUTE] Sampling: {self.posterior_type}, z_0 stats: mean={z_0.mean().item():.4f}, std={z_0.std().item():.4f}")
            self._sampling_traced = True
        
        # Build latent sequence
        if self.n_flows > 0:
            z_seq = [z_0]
            z_seq_out, log_det_jacobians = self.flow_manager.apply_flows(z_seq, n_obs=n_obs)
            z_seq = z_seq_out
            z_seq_tensor = torch.stack(z_seq_out, dim=1)
            stage_zF = z_seq_out[-1]
        else:
            # No flows: tile z_0 across observed timesteps to keep shapes consistent
            log_det_jacobians = []
            z_seq = [z_0]
            z_seq_tensor = z_0.unsqueeze(1).expand(-1, n_obs, -1).contiguous()
            stage_zF = z_0

        # Sanitize Stage-C latents for non-finite values before loss usage
        try:
            if 'mu' in locals():
                mu_ref = mu
            else:
                mu_ref = None
            for name, tensor in [("stage_z0", stage_z0), ("stage_zS", stage_zS), ("stage_zF", stage_zF)]:
                if isinstance(tensor, torch.Tensor):
                    bad = ~torch.isfinite(tensor)
                    if bad.any():
                        repl = mu_ref if (mu_ref is not None and mu_ref.shape == tensor.shape) else torch.zeros_like(tensor)
                        tensor = torch.where(bad, repl, tensor)
                        if name == "stage_z0":
                            stage_z0 = tensor
                        elif name == "stage_zS":
                            stage_zS = tensor
                        elif name == "stage_zF":
                            stage_zF = tensor
        except Exception:
            pass
        if self.loop_mode == "closed":
            z_seq_tensor[:, -1] = z_seq_tensor[:, 0]
        z_flat = z_seq_tensor.reshape(-1, self.latent_dim)
        decoder_out = self.decoder(z_flat)
        if isinstance(decoder_out, dict):
            recon_x = decoder_out.get("reconstruction", next(iter(decoder_out.values())))
        elif hasattr(decoder_out, 'reconstruction'):
            recon_x = decoder_out.reconstruction
        else:
            recon_x = decoder_out
        recon_x = recon_x.view(batch_size, n_obs, *self.input_dim)
        # Sanitize NaNs/Infs before clamping/log-losses
        recon_x = torch.nan_to_num(recon_x, nan=0.5, posinf=1.0, neginf=0.0)
        # Route tracing: reconstruction stats (once)
        if not hasattr(self, '_recon_traced'):
            print(f"[ROUTE] Reconstruction: min={recon_x.min().item():.4f}, max={recon_x.max().item():.4f}, mean={recon_x.mean().item():.4f}")
            self._recon_traced = True
        # Clamp recon_x if using BCE loss (optional, here for safety)
        recon_x = torch.clamp(recon_x, min=1e-6, max=1-1e-6)
        # Compute losses using modular loss manager
        # If reconstruction_mode is t0_only, restrict tensors to t=0 to avoid unintended averaging
        x_for_loss = x
        recon_for_loss = recon_x
        recon_mode = getattr(self.config, 'reconstruction_mode', None)
        if recon_mode == 't0_only':
            x_for_loss = x[:, :1]
            recon_for_loss = recon_x[:, :1]

        # Attach gradient hooks for μ and zS once to inspect non‑RHMC gradients
        try:
            if self.training and getattr(mu, 'requires_grad', False) and not hasattr(self, '_mu_loss_hook_set'):
                mu.retain_grad()
                def _mu_loss_hook(g):
                    try:
                        if not hasattr(self, '_mu_loss_grad_printed') and os.environ.get('RLVAE_GRAD_TRACE','0') == '1':
                            print(f"[LOSS TRACE] hook ∂Total/∂μ norm={g.norm().item():.4e} mean={g.mean().item():+.4e}")
                            self._mu_loss_grad_printed = True
                    except Exception:
                        pass
                    return g
                mu.register_hook(_mu_loss_hook)
                self._mu_loss_hook_set = True
            if self.training and isinstance(stage_zS, torch.Tensor) and getattr(stage_zS, 'requires_grad', False) and not hasattr(self, '_zS_loss_hook_set'):
                stage_zS.retain_grad()
                def _zS_loss_hook(g):
                    try:
                        if not hasattr(self, '_zS_loss_grad_printed') and os.environ.get('RLVAE_GRAD_TRACE','0') == '1':
                            print(f"[LOSS TRACE] hook ∂Total/∂zS norm={g.norm().item():.4e} mean={g.mean().item():+.4e}")
                            self._zS_loss_grad_printed = True
                    except Exception:
                        pass
                    return g
                stage_zS.register_hook(_zS_loss_hook)
                self._zS_loss_hook_set = True
        except Exception:
            pass

        losses = self.loss_manager.compute_total_loss(
            x=x_for_loss,
            x_recon=recon_for_loss,
            mu=mu,
            log_var=log_var,
            z_samples=stage_zS,
            log_det_jacobians=log_det_jacobians,
            z_seq=z_seq,
            flow_manager=self.flow_manager,
            loop_mode=self.loop_mode,
            metric_tensor=self.modular_metric if hasattr(self, 'modular_metric') else None,
            use_riemannian_kl=(self.posterior_type in ["riemannian_metric", "riemannian_rhmc"]),
            z0=stage_z0,
            zS=stage_zS,
            zF=stage_zF,
            Sigma_mu=sigma_mu,
            # RHMC KL Monte‑Carlo wiring
            rhmc_z0=rhmc_z0 if self.posterior_type == "riemannian_rhmc" else None,
            rhmc_zK=zK if self.posterior_type == "riemannian_rhmc" else None,
            rhmc_log_q=rhmc_log_q if self.posterior_type == "riemannian_rhmc" else None,
            rhmc_traj_info=rhmc_traj if self.posterior_type == "riemannian_rhmc" else None,
            rhmc_posterior=getattr(self, 'posterior_sampler_rhmc', None) if self.posterior_type == "riemannian_rhmc" else None,
            rhmc_kl_mode=str(getattr(self.config, 'rhmc_kl_mode', getattr(getattr(self.config, 'posterior', {}), 'rhmc_kl_mode', 'mc'))).lower(),
            rhmc_kl_source=str(getattr(self.config, 'rhmc_kl_source', getattr(getattr(self.config, 'posterior', {}), 'rhmc_kl_source', 'z0'))).lower(),
            rhmc_kl_jacobian=bool(getattr(self.config, 'rhmc_kl_jacobian', False)),
        )
        # One-time check: encoder params require_grad
        try:
            if not hasattr(self, "_enc_require_grad_printed"):
                enc_params = [(n, p.requires_grad) for n, p in self.encoder.named_parameters(recurse=True)]
                true_count = sum(1 for _, r in enc_params if r)
                total_count = len(enc_params)
                print(f"[GRAD DEBUG] Encoder params require_grad: {true_count}/{total_count}")
                self._enc_require_grad_printed = True
        except Exception:
            pass
        # Route tracing: loss composition (once)
        if not hasattr(self, '_loss_traced'):
            print(f"[ROUTE] Losses: recon={losses['reconstruction_loss'].item():.4f}, kl={losses['kl_divergence_loss'].item():.4f}, flow={losses['flow_loss'].item():.4f}")
            self._loss_traced = True
        
        # Prepare result
        result = {
            'reconstruction': recon_x,
            'latent_samples': z_seq_tensor,
            'mu': mu,
            'log_var': log_var,
            'reconstruction_loss': losses['reconstruction_loss'],
            'kl_divergence_loss': losses['kl_divergence_loss'],
            'flow_loss': losses['flow_loss'],
            'loop_penalty': losses['loop_penalty'],
            'total_loss': losses['total_loss']
        }
        loss_details = losses.get('loss_details', {})
        if isinstance(loss_details, dict):
            result.update(loss_details)

        # Add aliases for Lightning trainer compatibility
        result['kl_divergence'] = result['kl_divergence_loss']
        result['total_loss'] = result['total_loss']
        result['reconstruction_loss'] = result['reconstruction_loss']

        if compute_metrics:
            result.update(self._compute_additional_metrics(x, result))
        
        return result
    
    def _compute_additional_metrics(self, x: torch.Tensor, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute additional metrics for analysis."""
        metrics = {}
        
        # Cyclicity error (for temporal sequences)
        if len(x.shape) == 5:  # [B, T, C, H, W]
            with torch.no_grad():
                recon = output['reconstruction']
                cyclicity_error = torch.mean((recon[:, 0] - recon[:, -1]) ** 2)
                metrics['cyclicity_error'] = cyclicity_error
        
        # Latent space metrics
        z = output['latent_samples']
        metrics['latent_norm'] = torch.mean(torch.norm(z, dim=-1))
        metrics['latent_variance'] = torch.var(z)
        
        # Riemannian-specific metrics
        if self.config.sampling.use_riemannian and hasattr(self, 'G_inv'):
            metrics.update(self._compute_riemannian_metrics(z))
        
        return metrics
    
    def _compute_riemannian_metrics(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute Riemannian geometry metrics."""
        with torch.no_grad():
            try:
                # Metric conditioning
                G_inv = self.G_inv(z)
                eigenvals = torch.linalg.eigvals(G_inv)
                condition_number = torch.mean(torch.max(eigenvals.real, dim=-1)[0] / torch.min(eigenvals.real, dim=-1)[0])
                
                # Manifold regularity (determinant stability)
                det_G_inv = torch.det(G_inv)
                regularity = torch.std(torch.log(det_G_inv + 1e-8))
                
                return {
                    'metric_conditioning': condition_number,
                    'manifold_regularity': regularity,
                    'metric_determinant': torch.mean(det_G_inv)
                }
            except:
                return {
                    'metric_conditioning': torch.tensor(0.0),
                    'manifold_regularity': torch.tensor(0.0),
                    'metric_determinant': torch.tensor(1.0)
                }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary for comparison."""
        summary = {
            'model_name': self.model_name,
            'architecture': {
                'latent_dim': self.latent_dim,
                'n_flows': self.config.n_flows,
                'input_dim': self.config.input_dim
            },
            'configuration': {
                'posterior_type': self.config.posterior.type,
                'sampling_method': self.config.sampling.method,
                'loop_mode': self.config.loop.mode,
                'uses_riemannian': self.config.sampling.use_riemannian
            },
            'hyperparameters': {
                'beta': self.config.beta,
                'riemannian_beta': self.config.get('riemannian_beta', 1.0),
                'cycle_penalty': self.config.loop.penalty
            },
            'modular_components': {
                'encoder': type(self.encoder).__name__,
                'decoder': type(self.decoder).__name__,
                'metric_tensor': 'MetricTensor' if hasattr(self, 'modular_metric') else 'Legacy',
                'flow_manager': 'FlowManager' if hasattr(self, 'flow_manager') else 'Legacy',
                'loss_manager': 'LossManager' if hasattr(self, 'loss_manager') else 'Legacy',
                'metric_loader': 'MetricLoader' if hasattr(self, 'metric_loader') else 'Legacy'
            }
        }
        
        return summary
    
    def get_modular_summary(self) -> Dict[str, Any]:
        """Get detailed summary of all modular components."""
        summary = {
            'model_name': self.model_name,
            'modularity_level': '100%' if self._is_fully_modular() else 'Partial',
            'components': {}
        }
        
        # Encoder/Decoder info
        if hasattr(self, 'encoder_manager'):
            summary['components']['encoder'] = {
                'type': type(self.encoder).__name__,
                'architecture': self.encoder_manager.architecture,
                'parameters': sum(p.numel() for p in self.encoder.parameters())
            }
        
        if hasattr(self, 'decoder_manager'):
            summary['components']['decoder'] = {
                'type': type(self.decoder).__name__,
                'architecture': self.decoder_manager.architecture,
                'parameters': sum(p.numel() for p in self.decoder.parameters())
            }
        
        # Metric tensor info
        if hasattr(self, 'modular_metric'):
            summary['components']['metric_tensor'] = {
                'type': 'MetricTensor',
                'is_loaded': self.modular_metric.is_loaded(),
                'config': self.modular_metric.get_config()
            }
        
        # Flow manager info
        if hasattr(self, 'flow_manager'):
            summary['components']['flow_manager'] = {
                'type': 'FlowManager',
                'n_flows': self.flow_manager.n_flows,
                'parameters': sum(p.numel() for p in self.flow_manager.parameters()),
                'config': self.flow_manager.get_flow_params()
            }
        
        # Loss manager info
        if hasattr(self, 'loss_manager'):
            summary['components']['loss_manager'] = {
                'type': 'LossManager',
                'config': self.loss_manager.get_config()
            }
        
        # Metric loader info
        if hasattr(self, 'metric_loader'):
            summary['components']['metric_loader'] = {
                'type': 'MetricLoader',
                'device': str(self.metric_loader.device)
            }
        
        return summary
    
    def _is_fully_modular(self) -> bool:
        """Check if the model is 100% modular."""
        required_components = [
            'encoder_manager',
            'decoder_manager', 
            'modular_metric',
            'metric_loader',
            'flow_manager',
            'loss_manager'
        ]
        
        return all(hasattr(self, component) for component in required_components)
    
    def log_to_wandb(self, metrics: Dict[str, torch.Tensor], epoch: int, prefix: str = ""):
        """Log metrics to wandb with standardized naming."""
        if wandb.run is None:
            return
        
        log_dict = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.detach().cpu().numpy()
            log_dict[f"{prefix}{key}"] = value
        
        log_dict['epoch'] = epoch
        wandb.log(log_dict)
    
    @classmethod
    def create_comparison_models(cls, base_config: DictConfig, model_variants: List[str]) -> Dict[str, 'ModularRiemannianFlowVAE']:
        """
        Create multiple model variants for comparison.
        
        Args:
            base_config: Base configuration
            model_variants: List of model configuration names
            
        Returns:
            Dictionary of model_name -> model instances
        """
        models = {}
        
        for variant in model_variants:
            # Load variant config (this would be implemented with Hydra compose API)
            variant_config = base_config.copy()  # Simplified for now
            
            # Create model with variant config
            models[variant] = cls(variant_config)
        
        return models


    # === NEW: Generation and Evaluation Integration ===
    
    def create_generator(self, config=None):
        """Create a generator interface for this model, optionally using a Hydra config."""
        from src.generation.generator import create_generator
        return create_generator(self, config=config)
    
    def create_inference_pipeline(self, config=None):
        """Create an inference pipeline for this model, optionally using a Hydra config."""
        from src.inference.inference_pipeline import create_inference_pipeline
        return create_inference_pipeline(self, config=config)
    
    def create_evaluator(self):
        """Create an evaluator for comprehensive model assessment."""
        from src.evaluation.evaluator import create_evaluator
        return create_evaluator(self)
    
    def generate_samples(self, num_samples: int = 64, method: str = "geodesic", 
                        sampler_type: str = "working", **kwargs) -> Dict[str, torch.Tensor]:
        """
        Convenient method to generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            method: Sampling method ("geodesic", "enhanced", "basic", "standard")
            sampler_type: Sampler type ("working", "hmc", "official")
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated images and metadata
        """
        from src.generation.generator import GenerationConfig
        
        generator = self.create_generator()
        config = GenerationConfig(
            num_samples=num_samples,
            sampling_method=method,
            sampler_type=sampler_type,
            **kwargs
        )
        
        return generator.generate_from_prior(config)
    
    def compute_fid_score(self, real_images: torch.Tensor, num_generated: int = 1000,
                         cache_key: str = "evaluation", **kwargs) -> Dict[str, float]:
        """
        Compute FID score against real images.
        
        Args:
            real_images: Real images for comparison [N, C, H, W]
            num_generated: Number of samples to generate for FID computation
            cache_key: Cache key for real image statistics
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing FID score and related metrics
        """
        from src.evaluation.fid_scorer import create_fid_scorer
        from src.generation.generator import GenerationConfig
        
        # Initialize FID scorer
        fid_scorer = create_fid_scorer(device=self.device)
        
        # Cache real statistics
        fid_scorer.cache_real_statistics(real_images, cache_key)
        
        # Generate samples
        generator = self.create_generator()
        config = GenerationConfig(
            num_samples=num_generated,
            sequence_length=1,  # Single images for FID
            **kwargs
        )
        
        generation_result = generator.generate_from_prior(config)
        generated_images = generation_result['images']
        
        # Remove sequence dimension if present
        if generated_images.dim() == 5:
            # If shape is [B, S, C, H, W], flatten to [B*S, C, H, W]
            if generated_images.shape[1] == 1:
                generated_images = generated_images[:, 0]
            else:
                B, S, C, H, W = generated_images.shape
                generated_images = generated_images.reshape(B * S, C, H, W)
        
        # Compute FID
        fid_result = fid_scorer.evaluate_with_cached_real(generated_images, cache_key)
        
        return fid_result
    
    def evaluate_reconstruction(self, test_images: torch.Tensor,
                              batch_size: int = 32) -> Dict[str, Any]:
        """
        Evaluate reconstruction quality on test images.
        
        Args:
            test_images: Test images [N, C, H, W]
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing reconstruction metrics
        """
        from src.inference.inference_pipeline import InferenceConfig
        
        inference_pipeline = self.create_inference_pipeline()
        config = InferenceConfig(batch_size=batch_size, return_uncertainties=True)
        
        result = inference_pipeline.encode_and_reconstruct(test_images, config)
        
        return {
            'reconstruction_metrics': result['reconstruction_metrics'],
            'latent_statistics': {
                'mean_norm': torch.norm(result['latents'], dim=-1).mean().item(),
                'std_norm': torch.norm(result['latents'], dim=-1).std().item(),
            },
            'uncertainty_analysis': result.get('uncertainties'),
        }
    
    def comprehensive_evaluation(self, real_images: torch.Tensor,
                               **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of the model.
        
        Args:
            real_images: Real images for evaluation [N, C, H, W]
            **kwargs: Additional evaluation parameters
            
        Returns:
            Complete evaluation results
        """
        from src.evaluation.evaluator import EvaluationConfig
        
        evaluator = self.create_evaluator()
        config = EvaluationConfig(**kwargs)
        
        return evaluator.evaluate_comprehensive(real_images, config)

    def decode(self, z):
        """
        Decode latent vectors z into images using the model's decoder.
        Args:
            z: Tensor of shape (N, latent_dim)
        Returns:
            Decoded images as a tensor
        """
        return self.decoder(z)

    def _setup_manifold_sampling(self):
        """Setup manifold sampling if enabled in configuration."""
        manifold_config = self.config.get('manifold_sampling', {})
        
        if manifold_config.get('enabled', False):
            print("🎯 Setting up manifold sampling")
            
            # Check if we should use native G⁻¹ 
            use_native = manifold_config.get('native_g_inverse', {}).get('use_native', True)
            
            if use_native:
                # Replace traditional metric with native G⁻¹ metric
                self._setup_native_inverse_metric(manifold_config)
            
            # Initialize manifold sampler
            self.manifold_sampler = ManifoldSampler(
                metric_tensor=self.modular_metric,
                method=manifold_config.get('method', 'combined'),
                step_size_base=manifold_config.get('step_size_base', 0.25),
                exploration_ratio=manifold_config.get('exploration_ratio', 0.6),
                direction_change_frequency=manifold_config.get('direction_change_frequency', 3),
                random_component_scale=manifold_config.get('random_component_scale', 0.1),
                bounds=(manifold_config.get('bounds', {}).get('min', -5.0), 
                       manifold_config.get('bounds', {}).get('max', 5.0))
            )
            
            self.manifold_sampling_enabled = True
            print("✅ Manifold sampling initialized")
        else:
            self.manifold_sampler = None
            self.manifold_sampling_enabled = False
            print("⏭️  Manifold sampling disabled")
    
    def _setup_native_inverse_metric(self, manifold_config):
        """Replace traditional metric with native G⁻¹ metric."""
        print("🔄 Converting to native G⁻¹ metric system")
        
        # Extract native G⁻¹ parameters
        native_config = manifold_config.get('native_g_inverse', {})
        temperature = native_config.get('temperature', 2.0)
        regularization = native_config.get('regularization', 1e-4)
        
        # Create native inverse metric tensor
        native_metric = NativeInverseMetricTensor(
            latent_dim=self.latent_dim,
            device=self.device
        )
        
        # If we have existing centroids and metrics, convert them
        if (hasattr(self, 'modular_metric') and hasattr(self.modular_metric, 'centroids') and 
            self.modular_metric.centroids is not None and len(self.modular_metric.centroids) > 0):
            print("🔄 Converting existing traditional metric to native G⁻¹")
            
            # Get existing components
            centroids = self.modular_metric.centroids
            
            if hasattr(self.modular_metric, 'metric_matrices'):
                # Traditional G matrices - compute inverse for native G⁻¹
                G_matrices = self.modular_metric.metric_matrices
                inverse_metrics = torch.linalg.inv(G_matrices)
            else:
                # Fallback: create identity matrices
                n_centroids = len(centroids)
                inverse_metrics = torch.eye(self.latent_dim, device=self.device).unsqueeze(0).repeat(n_centroids, 1, 1)
            
            # Load into native metric
            native_metric.load_inverse_metrics(centroids, inverse_metrics, temperature, regularization)
        else:
            print("⚠️  No existing metric found - native G⁻¹ will be initialized later")
        
        # Replace the metric tensor
        self.modular_metric = native_metric
        print("✅ Successfully converted to native G⁻¹ metric system")
    
    def sample_manifold_points(self, method: str = None, n_samples: int = 100, **kwargs):
        """
        Sample points using manifold sampling.
        
        Args:
            method: Sampling method to use
            n_samples: Number of samples to generate
            **kwargs: Additional sampling parameters
            
        Returns:
            Dictionary with sampled points
        """
        if not self.manifold_sampling_enabled:
            raise ValueError("Manifold sampling is not enabled. Set manifold_sampling.enabled=true in config.")
        
        return self.manifold_sampler.sample(method=method, n_samples=n_samples, **kwargs)
    
    def create_manifold_visualization(self, samples=None, latent_data=None, title=None, save_path=None):
        """
        Create manifold sampling visualization.
        
        Args:
            samples: Samples to visualize (if None, will generate new samples)
            latent_data: Background latent data points
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        if not self.manifold_sampling_enabled:
            raise ValueError("Manifold sampling is not enabled. Set manifold_sampling.enabled=true in config.")
        
        if samples is None:
            samples = self.sample_manifold_points()
        
        if title is None:
            title = f"Manifold Sampling Analysis - {self.model_name}"
        
        return self.manifold_sampler.create_visualization(
            samples=samples,
            latent_data=latent_data,
            title=title,
            save_path=save_path
        )



class ModelFactory:
    """Factory for creating models from configurations."""
    
    @staticmethod
    def create_model(config: DictConfig) -> ModularRiemannianFlowVAE:
        """Create model from configuration."""
        return ModularRiemannianFlowVAE(config)
    
    @staticmethod
    def create_comparison_suite(config: DictConfig) -> Dict[str, ModularRiemannianFlowVAE]:
        """Create a suite of models for comparison."""
        base_model_cfg = OmegaConf.create(OmegaConf.to_container(config.model, resolve=True))
        if hasattr(config.experiment, 'models'):
            models = {}
            for model_name in config.experiment.models:
                model_config = OmegaConf.create(OmegaConf.to_container(base_model_cfg, resolve=True))
                # Apply model-specific overrides
                if model_name == 'vanilla_vae':
                    overrides = {
                        "n_flows": 0,
                        "riemannian_beta": 0.0,
                        "posterior": {"type": 'gaussian'},
                        "sampling": {"use_riemannian": False, "method": 'standard'},
                        "loop": {"mode": 'open', "penalty": 0.0},
                    }
                    model_config = OmegaConf.merge(model_config, overrides)
                
                models[model_name] = ModularRiemannianFlowVAE(model_config)
            
            return models
        else:
            return {'main': ModularRiemannianFlowVAE(base_model_cfg)}


class MetricsCollector:
    """Collect and analyze metrics across different models."""
    
    def __init__(self):
        self.model_metrics = {}
    
    def add_model_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Add metrics for a specific model."""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = []
        self.model_metrics[model_name].append(metrics)
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get summary comparing all models."""
        summary = {}
        
        for model_name, metrics_list in self.model_metrics.items():
            if not metrics_list:
                continue
                
            # Aggregate metrics
            aggregated = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    aggregated[f"{key}_mean"] = sum(values) / len(values)
                    aggregated[f"{key}_std"] = (sum((x - aggregated[f"{key}_mean"])**2 for x in values) / len(values))**0.5
                    aggregated[f"{key}_final"] = values[-1]
            
            summary[model_name] = aggregated
        
        return summary
    
    def log_comparison_to_wandb(self):
        """Log comparison results to wandb."""
        if wandb.run is None:
            return
        
        summary = self.get_comparison_summary()
        
        # Create comparison table
        table_data = []
        for model_name, metrics in summary.items():
            row = {'model': model_name}
            row.update(metrics)
            table_data.append(row)
        
        wandb.log({"model_comparison": wandb.Table(data=table_data, columns=list(table_data[0].keys()))})
    
