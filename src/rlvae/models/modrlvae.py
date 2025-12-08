"""
Fully Modular Riemannian Flow VAE (ModRLVAE)
===========================================

This model mirrors the structure of the original RiemannianFlowVAE but is built
exclusively from modular components (encoder/decoder/flows/metric/posterior/loss).

Key properties:
- No inheritance from the monolithic implementation
- Uses EncoderManager / DecoderManager
- Uses MetricTensor + MetricLoader
- Uses FlowManager for temporal latent evolution (IAF)
- Uses PosteriorSampler for local metric–aligned Gaussian posterior
- Uses LossManager for recon + (Riemannian|Euclidean) KL + flow + loop
- Exposes G and G_inv for downstream samplers/visualizations

File paths of used components:
- EncoderManager: src/rlvae/models/components/encoder_manager.py
- DecoderManager: src/rlvae/models/components/decoder_manager.py
- MetricTensor: src/rlvae/models/components/metric_tensor.py
- MetricLoader: src/rlvae/models/components/metric_loader.py
- FlowManager: src/rlvae/models/components/flow_manager.py
- PosteriorSampler: src/rlvae/models/components/posterior_sampler.py
- LossManager: src/rlvae/models/components/loss_manager.py
- RiemannianSampler (optional prior): src/rlvae/models/components/riemannian_sampler.py
"""

from typing import Optional, Dict, Any, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
from omegaconf import DictConfig

from pythae.models.base.base_utils import ModelOutput

from .components.encoder_manager import EncoderManager
from .components.decoder_manager import DecoderManager
from .components.metric_tensor import MetricTensor
from .components.metric_loader import MetricLoader
from .components.flow_manager import FlowManager
from .components.loss_manager import LossManager
from .components.posterior_sampler import PosteriorSampler
from .components.riemannian_sampler import RiemannianSampler
from .components.manifold_sampler import ManifoldSampler
from .components.regularizers import RegularizerManager
from .components.sampler_manager import SamplerManager


class ModRLVAE(nn.Module):
    """Fully modular Riemannian Flow VAE built from components only."""

    def __init__(self, config: DictConfig):
        super().__init__()

        # --- Core hyperparameters from config ---
        self.config = config
        self.model_name = "ModRLVAE"

        # Input/latent
        self.input_dim: Tuple[int, ...] = tuple(config.input_dim)
        self.latent_dim: int = int(config.latent_dim)

        # Posterior type (default to riemannian_metric)
        # Careful: avoid str(None) which is truthy and breaks the fallback
        _post_type_attr = getattr(config, "posterior_type", None)
        if _post_type_attr is not None:
            self.posterior_type = str(_post_type_attr)
        else:
            _post_cfg = getattr(config, "posterior", None)
            if _post_cfg is not None:
                try:
                    if hasattr(_post_cfg, 'get'):
                        self.posterior_type = str(_post_cfg.get('type', 'riemannian_metric'))
                    else:
                        self.posterior_type = str(getattr(_post_cfg, 'type', 'riemannian_metric'))
                except Exception:
                    self.posterior_type = 'riemannian_metric'
            else:
                self.posterior_type = 'riemannian_metric'

        # Loop/cycle settings
        self.loop_mode: str = str(getattr(config.loop, "mode", "open")) if hasattr(config, "loop") else "open"
        self.loop_lambda: float = float(getattr(config.loop, "penalty", 0.0)) if hasattr(config, "loop") else 0.0

        # Flows (temporal evolution across observations)
        # Auto default: n_flows = sequence_length - 1 when sequence_length provided
        n_flows_cfg = getattr(config, "n_flows", None)
        if n_flows_cfg is None and hasattr(config, "sequence_length"):
            self.n_flows = int(config.sequence_length) - 1
        else:
            self.n_flows = int(n_flows_cfg or 0)

        self.flow_hidden_size = int(getattr(config, "flow_hidden_size", 64))
        self.flow_n_blocks = int(getattr(config, "flow_n_blocks", 2))
        self.flow_n_hidden = int(getattr(config, "flow_n_hidden", 1))
        # Sequence-based model flag for visualization helpers
        self.expects_sequence_input = True
        # Optional flow clamps (default match FlowManager defaults)
        flow_output_clip_cfg = getattr(config, "flow_output_clip", None)
        if flow_output_clip_cfg is None and hasattr(config, "flows"):
            try:
                flow_output_clip_cfg = getattr(config.flows, "output_clip", None)
            except Exception:
                flow_output_clip_cfg = None
        self.flow_output_clip = float(flow_output_clip_cfg) if flow_output_clip_cfg is not None else 50.0

        flow_logdet_clip_cfg = getattr(config, "flow_logdet_clip", None)
        if flow_logdet_clip_cfg is None and hasattr(config, "flows"):
            try:
                flow_logdet_clip_cfg = getattr(config.flows, "logdet_clip", None)
            except Exception:
                flow_logdet_clip_cfg = None
        self.flow_logdet_clip = float(flow_logdet_clip_cfg) if flow_logdet_clip_cfg is not None else 20.0

        # Loss scalars
        self.beta: float = float(getattr(config, "beta", 1.0))
        self.riemannian_beta: float = float(getattr(config, "riemannian_beta", self.beta))

        # KL/posterior options
        self.kl_use_metric_normalization = bool(getattr(config, "kl_use_metric_normalization", True))
        self.kl_metric_norm_mode = str(getattr(config, "kl_metric_norm_mode", "geomean"))
        self.posterior_local_alpha = float(getattr(config, "posterior_local_alpha", 0.5))
        self.kl_amp_safe = bool(getattr(config, "kl_amp_safe", True))
        self.kl_metric_eval_point = str(getattr(config, "kl_metric_eval_point", "z"))
        # Backward-compat toggle: use_curvature_correction -> choose eval point
        if hasattr(config, 'use_curvature_correction'):
            self.kl_metric_eval_point = 'z' if bool(getattr(config, 'use_curvature_correction')) else 'z'

        # Optional posterior alpha ramp schedule (by epoch)
        self.posterior_alpha_ramp_enabled = bool(getattr(config, "posterior_alpha_ramp_enabled", False))
        self.posterior_alpha_start = float(getattr(config, "posterior_alpha_start", 0.25))
        self.posterior_alpha_end = float(getattr(config, "posterior_alpha_end", 1.0))
        self.posterior_alpha_ramp_epochs = int(getattr(config, "posterior_alpha_ramp_epochs", 10))
        self._current_epoch: Optional[int] = None

        # Posterior sampling stability controls (for outlier suppression)
        # - posterior_cov_norm_mode: normalize G used in Σ = α·G(μ) by geomean/trace/none
        # - posterior_step_clip_scale: clamp ||ε|| ≤ c·√D to cut 3σ‑plus excursions
        self.posterior_cov_norm_mode = str(getattr(config, "posterior_cov_norm_mode", "geomean"))
        self.posterior_step_clip_scale = float(getattr(config, "posterior_step_clip_scale", 3.0))
        # Optional Mahalanobis clamp radius (in units of Σ^{-1} at μ). None/<=0 disables.
        self.posterior_maha_clip = float(getattr(config, "posterior_maha_clip", 0.0))

        # --- Build components ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Encoder / Decoder managers
        enc_cfg = getattr(config, "encoder", {})
        dec_cfg = getattr(config, "decoder", {})
        # Normalize configs to plain dicts when needed (DictConfig -> dict)
        enc_cfg_norm = dict(enc_cfg) if hasattr(enc_cfg, '__getitem__') and hasattr(enc_cfg, 'get') else (enc_cfg or {})
        dec_cfg_norm = dict(dec_cfg) if hasattr(dec_cfg, '__getitem__') and hasattr(dec_cfg, 'get') else (dec_cfg or {})
        self.encoder_manager = EncoderManager(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            architecture=(enc_cfg_norm.get("architecture", enc_cfg_norm) if isinstance(enc_cfg_norm, dict) else enc_cfg_norm) or "mlp",
            config=enc_cfg_norm if isinstance(enc_cfg_norm, dict) else {"architecture": str(enc_cfg_norm) if enc_cfg_norm else "mlp"},
            device=device,
        )
        self.decoder_manager = DecoderManager(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            architecture=(dec_cfg_norm.get("architecture", dec_cfg_norm) if isinstance(dec_cfg_norm, dict) else dec_cfg_norm) or "mlp",
            config=dec_cfg_norm if isinstance(dec_cfg_norm, dict) else {"architecture": str(dec_cfg_norm) if dec_cfg_norm else "mlp"},
            device=device,
        )
        self.encoder = self.encoder_manager.encoder
        self.decoder = self.decoder_manager.decoder

        # Metric tensor + loader
        metric_cfg = getattr(config, "metric", {})
        self.mod_metric = MetricTensor(
            latent_dim=self.latent_dim,
            device=device,
            trainable=bool(metric_cfg.get("trainable", False)),
            architecture=metric_cfg.get("architecture", "mlp"),
            arch_kwargs=metric_cfg.get("arch_kwargs", {}),
            temperature=float(metric_cfg.get("temperature_override", 0.2) or 0.2),
            regularization=float(metric_cfg.get("regularization_override", 1e-4) or 1e-4),
            init_from_fixed=bool(metric_cfg.get("init_from_fixed", False)),
            fixed_metric_path=metric_cfg.get("fixed_metric_path", None),
            normalize_weight_sum=metric_cfg.get("normalize_weight_sum", False),
            weight_kernel=metric_cfg.get("weight_kernel", "mahalanobis_normed"),
            weight_metric_normalization=metric_cfg.get("weight_metric_normalization", "trace"),
            topk_weights=metric_cfg.get("topk_weights", None),
            regularization_mode=metric_cfg.get("regularization_mode", "precision"),
            use_background_identity=metric_cfg.get("use_background_identity", False),
        )
        self.metric_loader = MetricLoader(device=device)

        # Load metric data if provided via pretrained.metric_path (fixed snapshot)
        pretrained_cfg = getattr(config, "pretrained", {})
        # Normalize to mapping-like
        if hasattr(pretrained_cfg, 'get'):
            metric_path = pretrained_cfg.get("metric_path")
        else:
            metric_path = None
        if metric_path:
            md = self.metric_loader.load_from_file(
                metric_path,
                temperature_override=metric_cfg.get("temperature_override", None),
                regularization_override=metric_cfg.get("regularization_override", None),
            )
            self.mod_metric.load_pretrained(**md)

        # Optionally load pretrained encoder/decoder
        try:
            pretrained_cfg = getattr(config, 'pretrained', {})
            if hasattr(pretrained_cfg, 'get'):
                enc_path = pretrained_cfg.get('encoder_path', None)
                dec_path = pretrained_cfg.get('decoder_path', None)
                if enc_path:
                    try:
                        self.encoder_manager.load_pretrained(enc_path)
                        print(f"✅ Loaded pretrained encoder from {enc_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to load pretrained encoder: {e}")
                if dec_path:
                    try:
                        self.decoder_manager.load_pretrained(dec_path)
                        print(f"✅ Loaded pretrained decoder from {dec_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to load pretrained decoder: {e}")
        except Exception:
            pass

        # Backward‑compatible metric accessors expected by samplers/visuals
        def _G(z: torch.Tensor) -> torch.Tensor:
            return self.mod_metric.compute_metric(z)

        def _G_inv(z: torch.Tensor) -> torch.Tensor:
            return self.mod_metric.compute_inverse_metric(z)

        self.G = _G
        self.G_inv = _G_inv
        # Backward/compatibility alias expected by training/eval plugins
        self.modular_metric = self.mod_metric

        # Expose centroid/temperature/regularization buffers if available
        self.centroids_tens = getattr(self.mod_metric, "centroids", None)
        self.M_tens = getattr(self.mod_metric, "metric_matrices", None)
        self.temperature = getattr(self.mod_metric, "temperature", torch.tensor(0.1, device=device))
        self.lbd = getattr(self.mod_metric, "regularization", torch.tensor(0.01, device=device))

        # Flow manager (IAF temporal evolution)
        self.flow_manager = FlowManager(
            latent_dim=self.latent_dim,
            n_flows=self.n_flows,
            flow_hidden_size=self.flow_hidden_size,
            flow_n_blocks=self.flow_n_blocks,
            flow_n_hidden=self.flow_n_hidden,
            device=device,
            output_clip=self.flow_output_clip,
            logdet_clip=self.flow_logdet_clip,
        )

        # Loss manager
        # Flow loss mode: 'relu' (default), 'l2', 'abs', 'frobenius', 'none'
        flow_loss_mode = str(getattr(config, "flow_loss_mode", "relu"))
        self.loss_manager = LossManager(
            beta=self.beta,
            riemannian_beta=self.riemannian_beta,
            loop_penalty_weight=self.loop_lambda,
            device=device,
            kl_use_metric_normalization=self.kl_use_metric_normalization,
            kl_metric_norm_mode=self.kl_metric_norm_mode,
            kl_amp_safe=self.kl_amp_safe,
            kl_metric_eval_point=self.kl_metric_eval_point,
            flow_loss_mode=flow_loss_mode,
        )

        # Posterior sampler (local metric‑aligned Gaussian)
        self.posterior_sampler = PosteriorSampler(self)

        # Optional prior/training sampler manager
        self.sampler_manager = SamplerManager(self)
        self._sampler_manager = self.sampler_manager  # legacy forward expects this alias
        # Optional prior sampler (geodesic/enhanced/basic)
        self.riemannian_sampler = RiemannianSampler(self)

        # Optional manifold sampler (visualization)
        self.manifold_sampler: Optional[ManifoldSampler] = None

        # Regularizers + EMA configuration (mirrors original flags)
        self.phase1_training = bool(getattr(config, 'phase1_training', False))
        self.phase2_training = bool(getattr(config, 'phase2_training', False))
        self.reg_manager = RegularizerManager(
            model=self,
            centroid_enabled=bool(getattr(config, 'centroid_regularizer_enabled', False)),
            centroid_weight=float(getattr(config, 'centroid_regularizer_weight', 0.01)),
            centroid_t0_only=bool(getattr(config, 'centroid_regularizer_t0_only', True)),
            spectral_enabled=bool(getattr(config, 'spectral_penalty_enabled', False)),
            spectral_weight=float(getattr(config, 'spectral_penalty_weight', 0.1)),
            eigenval_min_bound=float(getattr(config, 'eigenval_min_bound', 1e-2)),
            eigenval_max_bound=float(getattr(config, 'eigenval_max_bound', 1e2)),
            smoothness_enabled=bool(getattr(config, 'smoothness_penalty_enabled', False)),
            smoothness_weight=float(getattr(config, 'smoothness_penalty_weight', 0.01)),
            anisotropy_enabled=bool(getattr(config, 'anisotropy_alignment_enabled', False)),
            anisotropy_weight=float(getattr(config, 'anisotropy_alignment_weight', 0.05)),
            ema_enabled=bool(getattr(config, 'centroid_ema_enabled', False)),
            ema_rate=float(getattr(config, 'centroid_ema_rate', 0.01)),
            ema_update_frequency=int(getattr(config, 'centroid_ema_update_frequency', 10)),
        )

        # Training step counter for EMA
        self._global_step = 0

        # Sampling method from config (optional)
        s_cfg = getattr(config, 'sampling', {})
        self.sampling_method = None
        if isinstance(s_cfg, dict):
            self.sampling_method = s_cfg.get('method', None)
            if (self.sampling_method or '').lower() == 'official':
                try:
                    self.sampler_manager.setup_official()
                except Exception:
                    pass

        # Do not call self.to(device) here to avoid recursion; components already moved

    # ---------------------- utils / configuration ----------------------
    def set_current_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)

    def set_loop_mode(self, mode: str = "open", penalty_weight: float = 0.0) -> None:
        self.loop_mode = str(mode)
        self.loop_lambda = float(penalty_weight)
        if hasattr(self.loss_manager, "loop_penalty_weight"):
            self.loss_manager.loop_penalty_weight = self.loop_lambda

    # Exposed for PosteriorSampler
    def get_current_posterior_alpha(self, current_epoch: Optional[int] = None) -> float:
        if not self.posterior_alpha_ramp_enabled:
            return float(self.posterior_local_alpha)
        epoch = current_epoch if current_epoch is not None else (self._current_epoch or 0)
        if self.posterior_alpha_ramp_epochs <= 0:
            return float(self.posterior_alpha_end)
        t = max(0.0, min(1.0, float(epoch) / float(self.posterior_alpha_ramp_epochs)))
        return float((1 - t) * self.posterior_alpha_start + t * self.posterior_alpha_end)

    # ----------------------------- forward -----------------------------
    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass: encode x_0, sample z_0 from posterior, evolve with flows,
        decode z_t, compute losses via LossManager. Shapes: x [B,T,C,H,W].
        """
        assert x.dim() >= 4, "Expected x to be [B,T,C,H,W] or [B,C,H,W] with T=1"

        if x.dim() == 4:
            # Insert T=1
            x = x.unsqueeze(1)

        batch_size, n_obs = x.shape[0], x.shape[1]

        # Encode first observation
        x0 = x[:, 0]
        enc_out = self.encoder(x0)
        mu = enc_out.embedding
        log_var = enc_out.log_covariance

        # DEBUG: Always print what we're using for posterior sampling
        if not hasattr(self, '_debug_printed'):
            print(f"[ModRLVAE DEBUG] posterior_type={self.posterior_type}, sampling_method={self.sampling_method}")
            print(f"[ModRLVAE DEBUG] Using {'metric_aware_posterior' if self.posterior_type == 'riemannian_metric' else 'sampler_manager'}")
            self._debug_printed = True
        
        # Posterior sampling (with optional sampler manager methods)
        if self.posterior_type == "riemannian_metric":
            z0 = self.posterior_sampler.sample_metric_aware_posterior(mu, log_var)
        else:
            z0 = self.sampler_manager.sample_training(
                mu, log_var, posterior_type=self.posterior_type, method=self.sampling_method
            )

        # Build latent sequence via flows
        z_seq_list = [z0]
        log_det_jacobians = []
        if self.n_flows > 0:
            z_seq_out, log_det_jacobians = self.flow_manager.apply_flows(z_seq_list, n_obs=n_obs)
            z_seq_list = z_seq_out

        # Ensure sequence length matches n_obs
        if len(z_seq_list) != n_obs:
            # If no flows, tile z0
            if self.n_flows == 0:
                z_seq_tensor = z0.unsqueeze(1).expand(-1, n_obs, -1).contiguous()
            else:
                raise RuntimeError(f"z sequence length {len(z_seq_list)} != n_obs {n_obs}")
        else:
            z_seq_tensor = torch.stack(z_seq_list, dim=1)

        # Closed loop constraint
        if self.loop_mode == "closed":
            z_seq_tensor = z_seq_tensor.clone()
            z_seq_tensor[:, -1] = z_seq_tensor[:, 0]

        # Decode sequence
        z_flat = z_seq_tensor.reshape(batch_size * n_obs, self.latent_dim)
        dec_out = self.decoder(z_flat)
        recon_x = dec_out["reconstruction"] if isinstance(dec_out, dict) else getattr(dec_out, "reconstruction", dec_out)
        recon_x = recon_x.view(batch_size, n_obs, *self.input_dim)

        # Losses
        use_riem_kl = (str(self.posterior_type).lower() == "riemannian_metric")
        losses = self.loss_manager.compute_total_loss(
            x=x,
            x_recon=recon_x,
            mu=mu,
            log_var=log_var,
            z_samples=z0,
            log_det_jacobians=log_det_jacobians,
            z_seq=z_seq_list,
            flow_manager=self.flow_manager,
            loop_mode=self.loop_mode,
            metric_tensor=self.mod_metric,
            use_riemannian_kl=use_riem_kl,
        )

        # Regularizers (modular): Phase 1/2
        reg_terms = {
            'centroid_regularizer': torch.tensor(0.0, device=recon_x.device),
            'spectral_penalty': torch.tensor(0.0, device=recon_x.device),
            'smoothness_penalty': torch.tensor(0.0, device=recon_x.device),
            'anisotropy_penalty': torch.tensor(0.0, device=recon_x.device),
        }
        if self.training:
            self.reg_manager.step()
            self._global_step += 1
            # Phase 1: light centroid regularizer
            if self.phase1_training:
                reg_terms['centroid_regularizer'] = self.reg_manager.compute_centroid_regularizer(mu, t=0)
            # Phase 2: spectral / smoothness / anisotropy + EMA
            if self.phase2_training:
                reg_terms['spectral_penalty'] = self.reg_manager.compute_spectral_penalty(z0)
                reg_terms['smoothness_penalty'] = self.reg_manager.compute_smoothness_penalty(z0)
                reg_terms['anisotropy_penalty'] = self.reg_manager.compute_anisotropy_alignment(mu)
                self.reg_manager.maybe_update_centroids_ema(mu)

        # Add regularizers to total loss
        total_reg = sum(reg_terms.values())
        losses['total_loss'] = losses['total_loss'] + total_reg

        # Map to ModelOutput for trainer compatibility
        out = ModelOutput(
            recon_x=recon_x,
            z=z_seq_tensor,
            loss=losses["total_loss"],
            recon_loss=losses["reconstruction_loss"],
            kld_loss=losses["kl_divergence_loss"],
            flow_loss=losses["flow_loss"],
            loop_penalty=losses["loop_penalty"],
        )
        # Add explicit total_loss field for trainer compatibility
        out["total_loss"] = losses["total_loss"]
        # Attach extra fields commonly inspected
        out["riemannian_kl"] = losses["kl_divergence_loss"] if use_riem_kl else torch.tensor(0.0, device=recon_x.device)
        out["metric_reg"] = losses.get("metric_reg", torch.tensor(0.0, device=recon_x.device))
        # Attach modular regularizers
        out["centroid_regularizer"] = reg_terms['centroid_regularizer']
        out["spectral_penalty"] = reg_terms['spectral_penalty']
        out["smoothness_penalty"] = reg_terms['smoothness_penalty']
        out["anisotropy_penalty"] = reg_terms['anisotropy_penalty']
        return out

    # ------------------------- sampling helpers ------------------------
    @torch.no_grad()
    def sample_prior(self, num_samples: int = 64, method: str = "geodesic") -> torch.Tensor:
        return self.riemannian_sampler.sample_prior(num_samples, method=method)

    def enable_manifold_sampling(self, **kwargs) -> None:
        self.manifold_sampler = ManifoldSampler(self.mod_metric, device=next(self.parameters()).device, **kwargs)

    def sample_manifold_points(self, method: str = "combined", n_samples: int = 100, **kwargs) -> Dict[str, torch.Tensor]:
        if self.manifold_sampler is None:
            self.enable_manifold_sampling()
        assert self.manifold_sampler is not None
        return self.manifold_sampler.sample(method=method, n_samples=n_samples, **kwargs)

    # -------------------------- integration helpers --------------------------
    # Expose posterior sampler for plugins expecting a direct method on the model
    def sample_metric_aware_posterior(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return self.posterior_sampler.sample_metric_aware_posterior(mu, log_var)

    # Generation / inference / evaluation integration to match ModularRiemannianFlowVAE API
    def create_generator(self, config=None):
        from src.generation.generator import create_generator
        return create_generator(self, config=config)

    def create_inference_pipeline(self, config=None):
        from src.inference.inference_pipeline import create_inference_pipeline
        return create_inference_pipeline(self, config=config)

    def create_evaluator(self):
        from src.evaluation.evaluator import create_evaluator
        return create_evaluator(self)

    # -------------------------- reporting helpers ----------------------------
    def get_model_summary(self) -> Dict[str, Any]:
        try:
            enc_name = type(self.encoder).__name__
        except Exception:
            enc_name = 'UnknownEncoder'
        try:
            dec_name = type(self.decoder).__name__
        except Exception:
            dec_name = 'UnknownDecoder'
        summary = {
            'model_name': self.model_name,
            'architecture': {
                'latent_dim': self.latent_dim,
                'n_flows': self.n_flows,
                'input_dim': list(self.input_dim),
            },
            'configuration': {
                'posterior_type': self.posterior_type,
                'loop_mode': self.loop_mode,
            },
            'modular_components': {
                'encoder': enc_name,
                'decoder': dec_name,
                'metric_tensor': 'MetricTensor',
                'flow_manager': 'FlowManager',
                'loss_manager': 'LossManager',
            },
        }
        return summary
