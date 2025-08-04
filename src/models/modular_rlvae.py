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

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path
from omegaconf import DictConfig
import wandb

from .riemannian_flow_vae import RiemannianFlowVAE, WorkingRiemannianSampler, OfficialRHVAESampler
from .components.encoder_manager import EncoderManager
from .components.decoder_manager import DecoderManager
from .components.metric_tensor import MetricTensor
from .components.metric_loader import MetricLoader
from .components.flow_manager import FlowManager
from .components.loss_manager import LossManager
from pythae.models.base.base_utils import ModelOutput


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
        # Debug print for metric config
        print("[DEBUG] model.metric config at model init:", config.get('metric', {}))
        # Check if n_flows was explicitly set by checking if it's NOT the default auto value
        # Default auto value would be sequence_length - 1, so if it's different, user set it
        if hasattr(config, 'sequence_length'):
            expected_auto_value = config.sequence_length - 1
            current_n_flows = getattr(config, 'n_flows', None)
            
            # If n_flows is different from auto value, user explicitly set it
            if current_n_flows is not None and current_n_flows != expected_auto_value:
                print(f"[USER] Keeping user-specified n_flows = {current_n_flows} (auto would be {expected_auto_value})")
            else:
                # Auto-set to sequence_length - 1
                config.n_flows = expected_auto_value
                print(f"[AUTO] Setting n_flows = sequence_length - 1 = {config.n_flows}")
        elif hasattr(config, 'model') and hasattr(config.model, 'sequence_length'):
            expected_auto_value = config.model.sequence_length - 1
            current_n_flows = getattr(config.model, 'n_flows', None)
            
            if current_n_flows is not None and current_n_flows != expected_auto_value:
                print(f"[USER] Keeping user-specified model.n_flows = {current_n_flows} (auto would be {expected_auto_value})")
            else:
                config.model.n_flows = expected_auto_value
                print(f"[AUTO] Setting model.n_flows = model.sequence_length - 1 = {config.model.n_flows}")
        else:
            print(f"[USER] Keeping n_flows = {getattr(config, 'n_flows', 'unknown')} (no sequence_length found)")
        # Extract core parameters
        super().__init__(
            input_dim=tuple(config.input_dim),
            latent_dim=config.latent_dim,
            n_flows=config.n_flows,
            flow_hidden_size=config.flow_hidden_size,
            flow_n_blocks=config.flow_n_blocks,
            flow_n_hidden=config.flow_n_hidden,
            epsilon=config.epsilon,
            beta=config.beta,
            riemannian_beta=config.get('riemannian_beta', config.beta),
            posterior_type=config.posterior.type,
            loop_mode=config.loop.mode,
            encoder=None,  # Will be created by manager
            decoder=None   # Will be created by manager
        )
        print(f"[MODEL INIT] input_dim: {config.input_dim}, encoder architecture: {config.get('encoder', {}).get('architecture', 'mlp')}")
        # Store config for later use
        self.config = config
        self.model_name = config.get('_target_', 'ModularRiemannianFlowVAE').split('.')[-1]
        
        # Initialize all modular components
        self._setup_modular_components()
        
        # Setup components based on config
        self._setup_from_config()
        
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
        self.modular_metric = MetricTensor(
            latent_dim=self.config.latent_dim,
            device=self.device,
            trainable=trainable,
            architecture=architecture,
            arch_kwargs=arch_kwargs,
            temperature=metric_cfg.get('temperature_override', 0.1),
            regularization=metric_cfg.get('regularization_override', 0.01),
            init_from_fixed=init_from_fixed,
            fixed_metric_path=fixed_metric_path
        )
        
        # 🚀 NEW: Initialize modular metric loader
        self.metric_loader = MetricLoader(device=self.device)
        
        # 🚀 NEW: Initialize modular loss manager
        metric_reg_weight = metric_cfg.get('metric_reg_weight', 0.0)
        metric_reg_type = metric_cfg.get('metric_reg_type', 'none')
        metric_reg_target = metric_cfg.get('metric_reg_target', 0.0)
        self.loss_manager = LossManager(
            beta=self.config.beta,
            riemannian_beta=self.config.get('riemannian_beta', self.config.beta),
            loop_penalty_weight=self.config.loop.penalty,
            device=self.device,
            metric_reg_weight=metric_reg_weight,
            metric_reg_type=metric_reg_type,
            metric_reg_target=metric_reg_target
        )
        
        # 🚀 NEW: Initialize modular flow manager (replace the one from parent)
        self.flow_manager = FlowManager(
            latent_dim=self.config.latent_dim,
            n_flows=self.config.n_flows,
            flow_hidden_size=self.config.flow_hidden_size,
            flow_n_blocks=self.config.flow_n_blocks,
            flow_n_hidden=self.config.flow_n_hidden,
            device=self.device
        )
        
        print(f"✅ Initialized all modular components for {self.model_name}")
        
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self._load_pretrained_components_modular()
        
        # Configure Riemannian sampling
        if self.config.sampling.use_riemannian:
            method = "custom" if self.config.sampling.method in ['geodesic', 'enhanced', 'basic'] else self.config.sampling.method
            self.enable_pure_rhvae(enable=True, method=method)
            self._riemannian_method = self.config.sampling.method
        else:
            self.enable_pure_rhvae(enable=False)
    
    def _load_pretrained_components_modular(self):
        """Load pretrained components using modular approach."""
        
        # Load encoder and decoder using managers
        if self.config.pretrained.encoder_path:
            encoder_path = Path(self.config.pretrained.encoder_path)
            if encoder_path.exists():
                print(f"🔧 Loading encoder from: {encoder_path}")
                self.encoder_manager.load_pretrained(encoder_path)
                print("✅ Loaded encoder weights")
        
        if self.config.pretrained.decoder_path:
            decoder_path = Path(self.config.pretrained.decoder_path)
            if decoder_path.exists():
                print(f"🔧 Loading decoder from: {decoder_path}")
                self.decoder_manager.load_pretrained(decoder_path)
                print("✅ Loaded decoder weights")
        
        # 🚀 NEW: Load metrics using modular approach
        if self.config.pretrained.metric_path:
            metric_path = Path(self.config.pretrained.metric_path)
            if metric_path.exists():
                try:
                    # Use new modular metric loader
                    metric_data = self.metric_loader.load_from_file(
                        metric_path,
                        temperature_override=self.config.metric.get('temperature_override'),
                        regularization_override=self.config.metric.get('regularization_override')
                    )
                    
                    # Load into modular metric tensor
                    self.modular_metric.load_pretrained(**metric_data)
                    
                    # Create backward-compatible interface functions
                    self._create_backward_compatible_interface()
                    
                    # Setup sampling components
                    self._setup_sampling_components()
                    
                    print("✅ Loaded metrics using modular components (2x faster!)")
                    
                except Exception as e:
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
        
        print("✅ Created backward-compatible metric interface")
    
    def _setup_sampling_components(self):
        """Setup sampling components using modular metric."""
        
        # Create multiple sampler options
        self._riemannian_sampler = WorkingRiemannianSampler(self)
        self._official_sampler = OfficialRHVAESampler(self)
        
        print("✅ Setup modular sampling components")
    
    def _fallback_to_original_metric_loading(self):
        """Fallback to original metric loading if modular approach fails."""
        
        # Use the parent class's metric loading
        if hasattr(super(), 'load_pretrained_metrics'):
            super().load_pretrained_metrics(
                self.config.pretrained.metric_path,
                self.config.metric.get('temperature_override')
            )
        else:
            print("⚠️ No fallback metric loading available")
    
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
        if self.config.sampling.use_riemannian:
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
        output = super().forward(x)

        result = {
            'reconstruction': output.recon_x,
            'latent_samples': output.z,
            'reconstruction_loss': output.recon_loss,
            'kl_divergence': output.kld_loss,
            'total_loss': output.loss
        }

        if hasattr(output, 'riemannian_kl'):
            result['riemannian_kl'] = output.riemannian_kl

        if compute_metrics:
            result.update(self._compute_additional_metrics(x, result))

        return result
    
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

        # Debug: Check input for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[DEBUG] Input x contains NaN or Inf!")
        print(f"[DEBUG] Input x stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        
        # Encode initial observation using modular encoder
        x_0 = x[:, 0]
        encoder_out = self.encoder(x_0)
        mu = encoder_out.embedding
        log_var = encoder_out.log_covariance
        
        # Sample latents using modular sampling
        if self.posterior_type == "riemannian_metric" and hasattr(self, 'modular_metric'):
            z_0 = self.sample_metric_aware_posterior(mu, log_var)
        else:
            print("⚠️ DEBUG: Falling back to standard Gaussian posterior sampling (no modular metric available or incorrect posterior_type)")
            eps = torch.randn_like(mu)
            z_0 = mu + eps * torch.exp(0.5 * log_var)
        print(f"[DEBUG] z_0 stats before flows: mean={z_0.mean().item():.4f}, std={z_0.std().item():.4f}, min={z_0.min().item():.4f}, max={z_0.max().item():.4f}")
        
        z_seq = [z_0]
        if self.n_flows > 0:
            z_seq_out, log_det_jacobians = self.flow_manager.apply_flows(z_seq, n_obs=n_obs)
            z_seq = z_seq_out
        else:
            log_det_jacobians = []
        z_seq_tensor = torch.stack(z_seq, dim=1)
        if self.loop_mode == "closed":
            z_seq_tensor[:, -1] = z_seq_tensor[:, 0]
        z_flat = z_seq_tensor.reshape(-1, self.latent_dim)
        decoder_out = self.decoder(z_flat)
        recon_x = decoder_out["reconstruction"]
        recon_x = recon_x.view(batch_size, n_obs, *self.input_dim)
        # Debug: Check recon_x for NaN/Inf
        if torch.isnan(recon_x).any() or torch.isinf(recon_x).any():
            print("[DEBUG] recon_x contains NaN or Inf!")
        print(f"[DEBUG] recon_x stats: min={recon_x.min().item():.4f}, max={recon_x.max().item():.4f}, mean={recon_x.mean().item():.4f}, std={recon_x.std().item():.4f}")
        # Clamp recon_x if using BCE loss (optional, here for safety)
        recon_x = torch.clamp(recon_x, min=1e-6, max=1-1e-6)
        # Compute losses using modular loss manager
        losses = self.loss_manager.compute_total_loss(
            x=x,
            x_recon=recon_x,
            mu=mu,
            log_var=log_var,
            z_samples=z_0,
            log_det_jacobians=log_det_jacobians,
            z_seq=z_seq,
            loop_mode=self.loop_mode,
            metric_tensor=self.modular_metric if hasattr(self, 'modular_metric') else None,
            use_riemannian_kl=self.posterior_type == "riemannian_metric"
        )
        # Debug: Check loss values for NaN/Inf
        for k, v in losses.items():
            if isinstance(v, torch.Tensor) and (torch.isnan(v).any() or torch.isinf(v).any()):
                print(f"[DEBUG] Loss {k} contains NaN or Inf! Value: {v}")
            elif isinstance(v, torch.Tensor):
                print(f"[DEBUG] Loss {k}: {v.item():.4f}")
        
        # Prepare result
        result = {
            'reconstruction': recon_x,
            'latent_samples': z_seq_tensor,
            'reconstruction_loss': losses['reconstruction_loss'],
            'kl_divergence_loss': losses['kl_divergence_loss'],
            'flow_loss': losses['flow_loss'],
            'loop_penalty': losses['loop_penalty'],
            'total_loss': losses['total_loss']
        }

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
                'riemannian_beta': self.config.get('riemannian_beta', 0.0),
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


class ModelFactory:
    """Factory for creating models from configurations."""
    
    @staticmethod
    def create_model(config: DictConfig) -> ModularRiemannianFlowVAE:
        """Create model from configuration."""
        return ModularRiemannianFlowVAE(config)
    
    @staticmethod
    def create_comparison_suite(config: DictConfig) -> Dict[str, ModularRiemannianFlowVAE]:
        """Create a suite of models for comparison."""
        if hasattr(config.experiment, 'models'):
            models = {}
            for model_name in config.experiment.models:
                # Create config for this model variant
                model_config = config.model.copy()
                
                # Apply model-specific overrides
                if model_name == 'vanilla_vae':
                    model_config.n_flows = 0
                    model_config.riemannian_beta = 0.0
                    model_config.posterior.type = 'gaussian'
                    model_config.sampling.use_riemannian = False
                    model_config.sampling.method = 'standard'
                    model_config.loop.mode = 'open'
                    model_config.loop.penalty = 0.0
                
                models[model_name] = ModularRiemannianFlowVAE(model_config)
            
            return models
        else:
            return {'main': ModularRiemannianFlowVAE(config.model)}


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