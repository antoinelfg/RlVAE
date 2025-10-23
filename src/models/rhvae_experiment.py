"""
RHVAE Experiment for Sprites dataset with comprehensive logging and visualization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import wandb
from tqdm import tqdm
import os

from pythae.models import RHVAE, RHVAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
from pythae.models.nn.base_architectures import BaseEncoder, BaseDecoder
from pythae.models.base import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput
from .samplers.hmc_sampler import RiemannianHMCSampler, RHVAEVolumeElementHMCSampler


class RHVAEMetricAdapter:
    """Adapter exposing RHVAE metric API expected by our samplers."""

    def __init__(self, model: RHVAE, temperature: float, regularization: float,
                 weight_kernel: str = "isotropic", weight_metric_normalization: str = "trace",
                 normalize_weight_sum: bool = False, topk_weights: Optional[int] = None):
        self._model = model
        self.temperature = float(temperature)
        self.regularization = float(regularization)
        # Weighting kernel for centroid influence: isotropic | mahalanobis | mahalanobis_normed
        self.weight_kernel = str(weight_kernel)
        # Normalization for S used in Mahalanobis weights when using _normed
        self.weight_metric_normalization = str(weight_metric_normalization)
        # Optional weight normalization and top-k pruning
        # Enforce RAW weights globally (user request): ignore config normalization
        # This makes all metric uses (visuals and samplers) use unnormalized weights
        self.normalize_weight_sum = False
        self.topk_weights = None if topk_weights is None else int(topk_weights)
        # Mirror common attributes
        self.latent_dim = int(model.model_config.latent_dim)
        self.device = next(model.parameters()).device

    def parameters(self):
        return self._model.parameters()

    @property
    def centroids_tens(self):
        return getattr(self._model, 'centroids_tens')

    @property
    def M_tens(self):
        return getattr(self._model, 'M_tens')

    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        # Weighted sum of M matrices à la RHVAE
        # z: [B, D], centroids: [K, D], M: [K, D, D]
        centroids = self.centroids_tens  # [K, D]
        M = self.M_tens  # [K, D, D]
        if centroids is None or M is None:
            raise RuntimeError("RHVAE model does not expose centroids_tens or M_tens.")

        z_exp = z.unsqueeze(1)  # [B, 1, D]
        c_exp = centroids.unsqueeze(0)  # [1, K, D]
        diff = c_exp - z_exp  # [B, K, D]

        if self.weight_kernel == "isotropic":
            dist = torch.sum(diff * diff, dim=-1)  # [B, K]
        else:
            # Mahalanobis distance per centroid using S_j
            # Choose S_j
            if self.weight_kernel == "mahalanobis_normed":
                # Normalize M to control scale in weights
                if self.weight_metric_normalization == "trace":
                    traces = torch.einsum('kii->k', M).clamp_min(1e-12)
                    S = M / (traces.view(-1, 1, 1) / float(self.latent_dim))
                elif self.weight_metric_normalization == "det":
                    dets = torch.linalg.det(M).abs().clamp_min(1e-24)
                    scales = dets.pow(1.0 / float(self.latent_dim)).clamp_min(1e-12)
                    S = M / scales.view(-1, 1, 1)
                else:
                    S = M
            else:  # "mahalanobis"
                S = M

            # Compute (z-c)^T S (z-c) for each centroid j
            # diff: [B, K, D], S: [K, D, D]
            # Use einsum to batch over B and align K properly
            Sd = torch.einsum('bkd,kde->bke', diff, S)  # [B, K, E=D]
            dist = torch.einsum('bke,bke->bk', Sd, diff)  # [B, K]

        weights = torch.exp(-dist / (self.temperature ** 2))  # [B, K]
        # Optional top-k pruning
        if self.topk_weights is not None and self.topk_weights > 0:
            k = min(self.topk_weights, weights.size(1))
            topv, topi = torch.topk(weights, k=k, dim=1, largest=True, sorted=False)
            mask = torch.zeros_like(weights)
            mask.scatter_(1, topi, 1.0)
            weights = weights * mask
        # Optional normalization so that sum_j w_j(z)=1
        if self.normalize_weight_sum:
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            weights = weights / denom
        # Broadcast multiply and sum over centroids
        weighted_M = M.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)  # [B, K, D, D]
        G_inv = weighted_M.sum(dim=1)  # [B, D, D]
        # Regularization for stability
        if self.regularization > 0:
            eye = torch.eye(G_inv.size(-1), device=G_inv.device, dtype=G_inv.dtype)
            G_inv = G_inv + self.regularization * eye.unsqueeze(0)
        return G_inv

    def G(self, z: torch.Tensor) -> torch.Tensor:
        # Invert G_inv stably
        G_inv = self.G_inv(z)
        try:
            G = torch.linalg.inv(G_inv)
        except RuntimeError:
            # Fallback via eigh
            evals, evecs = torch.linalg.eigh(G_inv)
            evals = torch.clamp(evals, min=1e-8)
            G = evecs @ torch.diag_embed(1.0 / evals) @ evecs.transpose(-2, -1)
        return G

    def G_inv_first2(self, z2: torch.Tensor) -> torch.Tensor:
        """Compute a 2x2 restriction of G^{-1} using only first two latent dims.

        This is for visualization over (z1, z2) slices to avoid domination by
        off-slice dimensions. Distances and matrices are restricted to 2D.
        """
        assert z2.dim() == 2 and z2.size(1) == 2, "z2 must be [B,2]"
        if self.centroids_tens is None or self.M_tens is None:
            raise RuntimeError("RHVAE model does not expose centroids_tens or M_tens.")
        C2 = self.centroids_tens[:, :2]               # [K,2]
        M2 = self.M_tens[:, :2, :2]                   # [K,2,2]
        z_exp = z2.unsqueeze(1)                        # [B,1,2]
        c_exp = C2.unsqueeze(0)                        # [1,K,2]
        diff = c_exp - z_exp                           # [B,K,2]

        # Distances for weights in 2D
        if self.weight_kernel == "isotropic":
            dist = torch.sum(diff * diff, dim=-1)      # [B,K]
        else:
            if self.weight_kernel == "mahalanobis_normed":
                # Normalize M2 by trace or det in 2D
                if self.weight_metric_normalization == "trace":
                    traces = torch.einsum('kii->k', M2).clamp_min(1e-12)
                    S = M2 / (traces.view(-1, 1, 1) / 2.0)
                elif self.weight_metric_normalization == "det":
                    dets = torch.linalg.det(M2).abs().clamp_min(1e-24)
                    scales = dets.pow(1.0 / 2.0).clamp_min(1e-12)
                    S = M2 / scales.view(-1, 1, 1)
                else:
                    S = M2
            else:  # mahalanobis
                S = M2
            Sd = torch.einsum('bkd,kde->bke', diff, S)  # [B,K,2]
            dist = torch.einsum('bke,bke->bk', Sd, diff)

        weights = torch.exp(-dist / (self.temperature ** 2))
        if self.topk_weights is not None and self.topk_weights > 0:
            k = min(self.topk_weights, weights.size(1))
            topv, topi = torch.topk(weights, k=k, dim=1, largest=True, sorted=False)
            mask = torch.zeros_like(weights)
            mask.scatter_(1, topi, 1.0)
            weights = weights * mask
        if self.normalize_weight_sum:
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            weights = weights / denom
        G_inv2 = (M2.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        if self.regularization > 0:
            eye2 = torch.eye(2, device=G_inv2.device, dtype=G_inv2.dtype)
            G_inv2 = G_inv2 + self.regularization * eye2.unsqueeze(0)
        return G_inv2

    def G_inv_subspace(self, z2: torch.Tensor, U: torch.Tensor, mean: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute a 2x2 restriction of G^{-1} in an arbitrary 2D subspace.

        Args:
            z2: [B,2] coordinates in the subspace basis
            U:  [D,2] column-orthonormal matrix whose columns span the subspace
            mean: optional [D] vector. If provided, centroids are centered as (C-mean)
                  before projection so that coordinates match PCA(2) convention
        """
        assert z2.dim() == 2 and z2.size(1) == 2, "z2 must be [B,2]"
        if self.centroids_tens is None or self.M_tens is None:
            raise RuntimeError("RHVAE model does not expose centroids_tens or M_tens.")
        # Project centroids and M onto the subspace: C2 = (C-mean) U, M2 = U^T M U
        C = self.centroids_tens  # [K,D]
        M = self.M_tens          # [K,D,D]
        if mean is not None:
            C2 = (C - mean) @ U  # [K,2]
        else:
            C2 = C @ U           # [K,2]
        MU = torch.einsum('kde,ej->kdj', M, U)       # [K,D,2]
        M2 = torch.einsum('id,kdj->kij', U.t(), MU)  # [K,2,2]

        z_exp = z2.unsqueeze(1)            # [B,1,2]
        c_exp = C2.unsqueeze(0)            # [1,K,2]
        diff = c_exp - z_exp               # [B,K,2]

        # Distances for weights
        if self.weight_kernel == "isotropic":
            dist = torch.sum(diff * diff, dim=-1)  # [B,K]
        else:
            if self.weight_kernel == "mahalanobis_normed":
                if self.weight_metric_normalization == "trace":
                    traces = torch.einsum('kii->k', M2).clamp_min(1e-12)
                    S = M2 / (traces.view(-1, 1, 1) / 2.0)
                elif self.weight_metric_normalization == "det":
                    dets = torch.linalg.det(M2).abs().clamp_min(1e-24)
                    scales = dets.pow(1.0 / 2.0).clamp_min(1e-12)
                    S = M2 / scales.view(-1, 1, 1)
                else:
                    S = M2
            else:
                S = M2
            Sd = torch.einsum('bkd,kde->bke', diff, S)  # [B,K,2]
            dist = torch.einsum('bke,bke->bk', Sd, diff)

        weights = torch.exp(-dist / (self.temperature ** 2))
        if self.topk_weights is not None and self.topk_weights > 0:
            k = min(self.topk_weights, weights.size(1))
            topv, topi = torch.topk(weights, k=k, dim=1, largest=True, sorted=False)
            mask = torch.zeros_like(weights)
            mask.scatter_(1, topi, 1.0)
            weights = weights * mask
        if self.normalize_weight_sum:
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            weights = weights / denom
        G_inv2 = (M2.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        if self.regularization > 0:
            eye2 = torch.eye(2, device=G_inv2.device, dtype=G_inv2.dtype)
            G_inv2 = G_inv2 + self.regularization * eye2.unsqueeze(0)
        return G_inv2

    # Compatibility API expected by some samplers/loggers
    def compute_metric_tensor(self, z: torch.Tensor, t: int = 0) -> torch.Tensor:
        return self.G(z)

class RGBEncoder(BaseEncoder):
    """RGB encoder for RHVAE."""
    
    def __init__(self, args):
        super().__init__()
        
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        
        # Encoder architecture for RGB images
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        
        # Calculate the size after convolutions
        conv_output_size = 512 * 4 * 4  # 64x64 -> 4x4 after 4 conv layers
        
        # FC layers for mean and log variance
        self.embedding = nn.Linear(conv_output_size, self.latent_dim)
        self.log_var = nn.Linear(conv_output_size, self.latent_dim)
        
    def forward(self, x):
        # x shape: [batch_size, 3, 64, 64]
        x = F.relu(self.conv1(x))  # -> [batch_size, 64, 32, 32]
        x = F.relu(self.conv2(x))  # -> [batch_size, 128, 16, 16]
        x = F.relu(self.conv3(x))  # -> [batch_size, 256, 8, 8]
        x = F.relu(self.conv4(x))  # -> [batch_size, 512, 4, 4]
        
        # Flatten
        x = x.reshape(x.size(0), -1)  # -> [batch_size, 512*4*4]
        
        # Get embedding and log variance
        embedding = self.embedding(x)
        log_var = self.log_var(x)
        
        # Return ModelOutput
        output = ModelOutput()
        output["embedding"] = embedding
        output["log_covariance"] = log_var
        
        return output

class RGBDecoder(BaseDecoder):
    """RGB decoder for RHVAE."""
    
    def __init__(self, args):
        super().__init__()
        
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        
        # Calculate the size after convolutions
        conv_output_size = 512 * 4 * 4  # 64x64 -> 4x4 after 4 conv layers
        
        # FC layer from latent to conv features
        self.fc = nn.Linear(self.latent_dim, conv_output_size)
        
        # Decoder architecture for RGB images
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        
    def forward(self, z):
        # z shape: [batch_size, latent_dim]
        x = self.fc(z)  # -> [batch_size, 512*4*4]
        x = x.reshape(x.size(0), 512, 4, 4)  # -> [batch_size, 512, 4, 4]
        
        x = F.relu(self.deconv1(x))  # -> [batch_size, 256, 8, 8]
        x = F.relu(self.deconv2(x))  # -> [batch_size, 128, 16, 16]
        x = F.relu(self.deconv3(x))  # -> [batch_size, 64, 32, 32]
        # Use sigmoid but avoid hard saturation to exact 0 which leads to all-black visuals
        x = torch.sigmoid(self.deconv4(x))  # -> [batch_size, 3, 64, 64]
        x = torch.clamp(x, 1e-6, 1.0)
        
        # Return ModelOutput
        output = ModelOutput()
        output["reconstruction"] = x
        
        return output

class RHVAEExperiment:
    """
    RHVAE Experiment with comprehensive logging and visualization.
    """
    
    def __init__(
        self,
        input_dim: List[int],
        latent_dim: int,
        n_lf: int = 3,
        eps_lf: float = 0.001,
        beta_zero: float = 0.3,
        temperature: float = 1.5,
        regularization: float = 0.001,
        encoder: Optional[Dict] = None,
        decoder: Optional[Dict] = None,
        device: str = "auto",
        seed: int = 42,
        max_centroids: Optional[int] = None,
        centroid_subsample_method: str = "fps",
        # Metric post-processing and alignment controls
        align_with_knn_cov: bool = False,
        knn_k: int = 300,
        alpha_align: float = 0.5,
        metric_normalization: str = "none",  # none|trace|det
        target_mean_eig: float = 1.0,
        # Weighting kernel controls
        weight_kernel: str = "isotropic",  # isotropic|mahalanobis|mahalanobis_normed
        weight_metric_normalization: str = "trace",
        normalize_weight_sum: bool = False,
        topk_weights: Optional[int] = None,
        # Decoder-Jacobian post-estimation of M
        reestimate_metric_from_decoder_jacobian: bool = False,
        jacobian_alpha: float = 0.5,
        jacobian_h: float = 1e-3,
        jacobian_stride: int = 4,
        # Global post-processing scale on M to control overall metric strength
        metric_scale: float = 1.0,
        # Optional: realign centroids to latent distribution after training
        realign_centroids: bool = False,
        centroid_realign_method: str = "kmeans",
    ):
        self.input_dim = tuple(input_dim)
        self.latent_dim = latent_dim
        self.n_lf = n_lf
        self.eps_lf = eps_lf
        self.beta_zero = beta_zero
        self.temperature = temperature
        self.regularization = regularization
        self.encoder_config = encoder or {}
        self.decoder_config = decoder or {}
        self.seed = seed
        self.max_centroids = int(max_centroids) if max_centroids is not None else None
        self.centroid_subsample_method = centroid_subsample_method
        # Post-processing
        self.align_with_knn_cov = bool(align_with_knn_cov)
        self.knn_k = int(knn_k)
        self.alpha_align = float(alpha_align)
        self.metric_normalization = str(metric_normalization)
        self.target_mean_eig = float(target_mean_eig)
        self.weight_kernel = str(weight_kernel)
        self.weight_metric_normalization = str(weight_metric_normalization)
        self.normalize_weight_sum = bool(normalize_weight_sum)
        self.topk_weights = None if topk_weights is None else int(topk_weights)
        self.reestimate_metric_from_decoder_jacobian = bool(reestimate_metric_from_decoder_jacobian)
        self.jacobian_alpha = float(jacobian_alpha)
        self.jacobian_h = float(jacobian_h)
        self.jacobian_stride = int(jacobian_stride)
        self.metric_scale = float(metric_scale)
        self.realign_centroids = bool(realign_centroids)
        self.centroid_realign_method = str(centroid_realign_method)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize model
        self._setup_model()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'metric_loss': []
        }
        
    def _setup_model(self):
        """Setup RHVAE model with proper configuration."""
        # Create RHVAE config
        model_config = RHVAEConfig(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            n_lf=self.n_lf,
            eps_lf=self.eps_lf,
            beta_zero=self.beta_zero,
            temperature=self.temperature,
            regularization=self.regularization
        )
        
        # Create encoder/decoder depending on channel count
        in_channels = int(self.input_dim[0]) if isinstance(self.input_dim, (list, tuple)) else 3
        if in_channels == 1:
            # Use Pythae's MNIST ResNet blocks for grayscale
            encoder = Encoder_ResNet_VAE_MNIST(model_config)
            decoder = Decoder_ResNet_AE_MNIST(model_config)
        else:
            # Custom RGB path
            encoder = RGBEncoder(model_config)
            decoder = RGBDecoder(model_config)
        
        # Create RHVAE model
        self.model = RHVAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder
        ).to(self.device)

        # Store architecture labels for later logging
        self._encoder_type = "MNIST_ResNet" if in_channels == 1 else "RGBEncoder"
        self._decoder_type = "MNIST_ResNet" if in_channels == 1 else "RGBDecoder"
        
        print(f"✅ RHVAE model created:")
        print(f"   Input dim: {self.input_dim}")
        print(f"   Latent dim: {self.latent_dim}")
        print(f"   Device: {self.device}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create metric adapter for RHMC sampling
        self.metric_adapter = RHVAEMetricAdapter(
            model=self.model,
            temperature=self.temperature,
            regularization=self.regularization,
            weight_kernel=getattr(self, 'weight_kernel', 'isotropic'),
            weight_metric_normalization=getattr(self, 'weight_metric_normalization', 'trace'),
            normalize_weight_sum=getattr(self, 'normalize_weight_sum', False),
            topk_weights=getattr(self, 'topk_weights', None),
        )
        # Default RHMC sampler: RHVAE-style volume-element HMC (empirically robust)
        self.rhmc_sampler = RHVAEVolumeElementHMCSampler(
            model=self.metric_adapter,
            mcmc_steps_nbr=50,
            n_lf=self.n_lf,
            eps_lf=self.eps_lf,
            beta_zero=self.beta_zero,
        )
        # Optional: Dual sampler can be enabled later if needed
        self.dual_rhmc_sampler = None
        
    def load_data(self, train_path: str, test_path: str, batch_size: int = 32):
        """Load and prepare data for training."""
        print(f"📊 Loading data from {train_path} and {test_path}")
        
        # Load data
        obj_train = torch.load(train_path)
        obj_test = torch.load(test_path)

        def _to_chw(t):
            # Accept [N,T,C,H,W], [N,C,H,W], or [N,H,W]
            if t.dim() == 5:
                t = t[:, 0]
            elif t.dim() == 3:
                t = t.unsqueeze(1)
            if t.max() > 1.5:
                t = t.float() / 255.0
            else:
                t = t.float()
            return t

        train_data = _to_chw(obj_train if isinstance(obj_train, torch.Tensor) else obj_train.get('data', obj_train))
        test_data = _to_chw(obj_test if isinstance(obj_test, torch.Tensor) else obj_test.get('data', obj_test))

        print(f"   Train data shape: {train_data.shape}")
        print(f"   Test data shape: {test_data.shape}")
        
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        
        return train_data, test_data
        
    def train(
        self,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        log_every: int = 10,
        save_every: int = 10,
        output_dir: str = "outputs/rhvae_sprites",
        use_wandb: bool = True,
        wandb_config: Optional[Dict] = None,
    ):
        """Train the RHVAE model with comprehensive logging."""
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup WandB
        if use_wandb:
            wandb_config = wandb_config or {}
            wandb.init(
                project=wandb_config.get("project", "rlvae_experiments"),
                name=wandb_config.get("name", "rhvae_sprites"),
                config={
                    "model": "RHVAE",
                    "input_dim": self.input_dim,
                    "latent_dim": self.latent_dim,
                    "n_lf": self.n_lf,
                    "eps_lf": self.eps_lf,
                    "beta_zero": self.beta_zero,
                    "temperature": self.temperature,
                    "regularization": self.regularization,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "batch_size": self.batch_size,
                    "device": str(self.device),
                }
            )
        
        # Setup training pipeline
        training_config = BaseTrainerConfig(
            output_dir=str(output_path),
            learning_rate=learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_epochs=epochs,
            optimizer_cls="Adam",
            optimizer_params={"weight_decay": weight_decay},
        )
        
        pipeline = TrainingPipeline(
            training_config=training_config,
            model=self.model
        )
        
        print(f"🚀 Starting RHVAE training for {epochs} epochs...")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Output dir: {output_path}")
        
        try:
            # Train the model
            pipeline(
                train_data=self.train_data,
                eval_data=self.test_data
            )
            
            print("✅ Training completed successfully!")
            
            # Optional centroid/M subsampling to reduce over-smoothing
            try:
                self._maybe_subsample_metric_centroids()
            except Exception as e:
                print(f"⚠️ Metric subsampling skipped: {e}")

            # Optional post-processing/alignment of metric to latent geometry
            try:
                self._postprocess_metric()
            except Exception as e:
                print(f"⚠️ Metric post-processing skipped: {e}")
            
            # Log final metrics
            if use_wandb:
                self._log_final_metrics()
                
            # Log periodic metrics during training
            if use_wandb and log_every > 0:
                self._log_periodic_metrics()

            # RHMC sampling and quick metric diagnostics
            if use_wandb:
                self._log_rhmc_prior_samples()
                self._log_metric_det_heatmap_2d()
                self._log_posterior_rhmc_reconstructions()
                self._log_weight_vs_distance()
                self._log_geodesic_like_interpolation()
                self._log_geodesic_streamlines_2d()
                self._log_random_pair_geodesics_pca2(num_pairs=6, steps=60, h=0.06)
                # Synthetic manifold sanity check for RHMC with RAW weights
                self._log_synthetic_manifold_demo()
                # New comprehensive figure and diagnostic variants
                # Prefer PCA(2)-aligned visuals; drop raw z1z2 to avoid misleading blur
                self._log_comprehensive_ginv_panel()
                self._log_det_variants_panels()
                self._log_det_G_and_Ginv_maps()
                self._log_centroids_pca_overlay()
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            if use_wandb:
                wandb.finish()
            raise e
            
    def _log_final_metrics(self):
        """Log comprehensive final metrics to WandB."""
        print("📊 Logging comprehensive metrics to WandB...")
        self.model.eval()
        
        # 1. Log training statistics (always works)
        self._log_training_statistics()
        
        # 2. Log metric analysis (RHVAE-specific, doesn't need forward pass)
        self._log_metric_analysis()
        
        # 3. Try to get reconstructions using encoder/decoder directly
        try:
            self._log_reconstructions_direct()
        except Exception as e:
            print(f"⚠️ Reconstruction logging failed: {e}")
            
        # 4. Try to get latent space using encoder directly
        try:
            self._log_latent_space_direct()
        except Exception as e:
            print(f"⚠️ Latent space logging failed: {e}")
            
        # 5. Try to generate samples using decoder directly
        try:
            self._log_generated_samples_direct()
        except Exception as e:
            print(f"⚠️ Generated samples logging failed: {e}")
            
        # 6. Try interpolation using decoder directly
        try:
            self._log_interpolation_direct()
        except Exception as e:
            print(f"⚠️ Interpolation logging failed: {e}")
            
    def _log_reconstructions_direct(self):
        """Log reconstructions using encoder/decoder directly to avoid gradient issues."""
        with torch.no_grad():
            # Get sample data
            sample_batch = self.test_data[:16].to(self.device)
            
            # Encode
            encoder_output = self.model.encoder(sample_batch)
            z = encoder_output["embedding"]
            
            # Decode
            decoder_output = self.model.decoder(z)
            reconstructions = decoder_output["reconstruction"]
            
            # Log reconstructions
            self._log_reconstructions(sample_batch, reconstructions, "final")
            
    def _log_latent_space_direct(self):
        """Log latent space using encoder directly."""
        with torch.no_grad():
            # Get sample data
            sample_batch = self.test_data[:32].to(self.device)
            
            # Encode to get latent representations
            encoder_output = self.model.encoder(sample_batch)
            z = encoder_output["embedding"]
            
            # Log latent space analysis
            self._log_latent_space_analysis(z, "final")
            
    def _log_generated_samples_direct(self):
        """Log generated samples using decoder directly."""
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(16, self.latent_dim).to(self.device)
            
            # Decode
            decoder_output = self.model.decoder(z)
            samples = decoder_output["reconstruction"]
            
            # Create a grid of generated samples
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            
            for i in range(16):
                row = i // 4
                col = i % 4
                
                img = samples[i].permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"Generated {i+1}", fontsize=10)
                axes[row, col].axis('off')
            
            plt.tight_layout()
            wandb.log({"generated_samples": wandb.Image(fig)})
            plt.close()
            
    def _log_interpolation_direct(self):
        """Log interpolation using decoder directly."""
        with torch.no_grad():
            # Get two random latent points
            z1 = torch.randn(1, self.latent_dim).to(self.device)
            z2 = torch.randn(1, self.latent_dim).to(self.device)
            
            # Interpolate between them
            alphas = torch.linspace(0, 1, 8).to(self.device)
            interpolated = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                decoder_output = self.model.decoder(z_interp)
                interpolated.append(decoder_output["reconstruction"])
                
            interpolated = torch.stack(interpolated)
            
            # Create interpolation grid
            fig, axes = plt.subplots(2, 8, figsize=(24, 6))
            
            for i in range(8):
                img = interpolated[i]
                # Handle tensor dimensions properly
                if img.dim() == 4:  # [1, 3, 64, 64]
                    img = img.squeeze(0)  # Remove batch dimension
                img = img.permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                axes[0, i].imshow(img)
                axes[0, i].set_title(f"Step {i+1}", fontsize=10)
                axes[0, i].axis('off')
                
                # Show the latent interpolation
                t = i / 7.0
                z_interp = (1 - t) * z1 + t * z2
                z_interp_np = z_interp.cpu().numpy().flatten()
                axes[1, i].bar(range(len(z_interp_np)), z_interp_np)
                axes[1, i].set_title(f"Latent {i+1}", fontsize=8)
                axes[1, i].set_ylim([-3, 3])
            
            plt.tight_layout()
            wandb.log({"latent_interpolation": wandb.Image(fig)})
            plt.close()
                
    def _log_reconstructions(self, original: torch.Tensor, reconstructed: torch.Tensor, suffix: str = ""):
        """Log detailed reconstruction comparisons to WandB."""
        # Create a comprehensive grid of original vs reconstructed images
        num_samples = min(16, len(original))
        fig, axes = plt.subplots(4, 8, figsize=(24, 12))
        
        for i in range(num_samples):
            row = i // 8
            col = i % 8
            
            # Original
            img_orig = original[i].permute(1, 2, 0).cpu().numpy()
            img_orig = np.clip(img_orig, 0, 1)
            axes[row*2, col].imshow(img_orig)
            axes[row*2, col].set_title(f"Original {i}", fontsize=8)
            axes[row*2, col].axis('off')
            
            # Reconstructed
            img_recon = reconstructed[i].permute(1, 2, 0).cpu().numpy()
            img_recon = np.clip(img_recon, 0, 1)
            axes[row*2+1, col].imshow(img_recon)
            axes[row*2+1, col].set_title(f"Reconstructed {i}", fontsize=8)
            axes[row*2+1, col].axis('off')
        
        plt.tight_layout()
        wandb.log({f"reconstructions_{suffix}": wandb.Image(fig)})
        plt.close()
        
        # Log reconstruction quality metrics
        mse = torch.mean((original - reconstructed) ** 2).item()
        wandb.log({f"reconstruction_mse_{suffix}": mse})
        
    def _log_latent_space_analysis(self, z: torch.Tensor, suffix: str = ""):
        """Log comprehensive latent space visualizations to WandB."""
        z_np = z.cpu().numpy()
        
        # 1. 2D scatter plot (first 2 dimensions)
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(z_np[:, 0], z_np[:, 1], alpha=0.7, s=50)
        ax.set_xlabel("Latent dim 1", fontsize=12)
        ax.set_ylabel("Latent dim 2", fontsize=12)
        ax.set_title(f"Latent Space 2D Projection {suffix}", fontsize=14)
        ax.grid(True, alpha=0.3)
        wandb.log({f"latent_space_2d_{suffix}": wandb.Image(fig)})
        plt.close()
        
        # 2. Heatmap of all dimensions
        fig, ax = plt.subplots(figsize=(16, 10))
        im = ax.imshow(z_np.T, aspect='auto', cmap='viridis')
        ax.set_xlabel("Samples", fontsize=12)
        ax.set_ylabel("Latent dimensions", fontsize=12)
        ax.set_title(f"Latent Space Heatmap {suffix}", fontsize=14)
        plt.colorbar(im, ax=ax)
        wandb.log({f"latent_heatmap_{suffix}": wandb.Image(fig)})
        plt.close()
        
        # 3. Distribution of each latent dimension
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(min(8, z_np.shape[1])):
            axes[i].hist(z_np[:, i], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f"Dim {i+1} Distribution", fontsize=10)
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")
        
        plt.tight_layout()
        wandb.log({f"latent_distributions_{suffix}": wandb.Image(fig)})
        plt.close()
        
        # 4. Correlation matrix of latent dimensions
        if z_np.shape[1] > 1:
            corr_matrix = np.corrcoef(z_np.T)
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title(f"Latent Dimensions Correlation {suffix}", fontsize=14)
            plt.colorbar(im, ax=ax)
            wandb.log({f"latent_correlation_{suffix}": wandb.Image(fig)})
            plt.close()
        
    def _log_metric_analysis(self):
        """Log metric matrix and centroids analysis."""
        try:
            if hasattr(self.model, 'M_tens') and hasattr(self.model, 'centroids_tens'):
                M = self.model.M_tens.detach().cpu().numpy()
                centroids = self.model.centroids_tens.detach().cpu().numpy()
                
                # 1. Metric matrix heatmap
                fig, ax = plt.subplots(figsize=(12, 10))
                im = ax.imshow(M[0], cmap='viridis')
                ax.set_title("RHVAE Metric Matrix", fontsize=14)
                ax.set_xlabel("Dimension", fontsize=12)
                ax.set_ylabel("Dimension", fontsize=12)
                plt.colorbar(im, ax=ax)
                wandb.log({"metric_matrix": wandb.Image(fig)})
                plt.close()
                
                # 2. Metric matrix determinant evolution
                if len(M.shape) == 3:  # Multiple time steps
                    det_values = [np.linalg.det(M[i]) for i in range(len(M))]
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.plot(det_values)
                    ax.set_title("Metric Matrix Determinant Evolution", fontsize=14)
                    ax.set_xlabel("Training Step", fontsize=12)
                    ax.set_ylabel("Determinant", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    wandb.log({"metric_determinant_evolution": wandb.Image(fig)})
                    plt.close()
                else:  # Single time step
                    det_value = np.linalg.det(M)
                    wandb.log({"metric_determinant": det_value})
                
                # 3. Centroids visualization
                if len(centroids.shape) == 3:  # Multiple time steps
                    fig, ax = plt.subplots(figsize=(12, 10))
                    scatter = ax.scatter(centroids[0, :, 0], centroids[0, :, 1], 
                                       alpha=0.7, s=100, c=range(len(centroids[0])), cmap='tab10')
                    ax.set_xlabel("Centroid dim 1", fontsize=12)
                    ax.set_ylabel("Centroid dim 2", fontsize=12)
                    ax.set_title("RHVAE Metric Centroids", fontsize=14)
                    plt.colorbar(scatter, ax=ax)
                    wandb.log({"metric_centroids": wandb.Image(fig)})
                    plt.close()
                else:  # Single time step
                    fig, ax = plt.subplots(figsize=(12, 10))
                    scatter = ax.scatter(centroids[:, 0], centroids[:, 1], 
                                       alpha=0.7, s=100, c=range(len(centroids)), cmap='tab10')
                    ax.set_xlabel("Centroid dim 1", fontsize=12)
                    ax.set_ylabel("Centroid dim 2", fontsize=12)
                    ax.set_title("RHVAE Metric Centroids", fontsize=14)
                    plt.colorbar(scatter, ax=ax)
                    wandb.log({"metric_centroids": wandb.Image(fig)})
                    plt.close()
                
                # 4. Centroids evolution
                if len(centroids.shape) == 3 and centroids.shape[0] > 1:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    for i in range(min(5, centroids.shape[1])):  # Show first 5 centroids
                        ax.plot(centroids[:, i, 0], label=f'Centroid {i+1}')
                    ax.set_title("Centroids Evolution (dim 1)", fontsize=14)
                    ax.set_xlabel("Training Step", fontsize=12)
                    ax.set_ylabel("Centroid Value", fontsize=12)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    wandb.log({"centroids_evolution": wandb.Image(fig)})
                    plt.close()

                # 5. Off-diagonal energy and eigen-spread diagnostics for M and G_inv
                try:
                    import numpy as _np
                    def offdiag_ratio(A: _np.ndarray) -> float:
                        d = _np.eye(A.shape[0], dtype=A.dtype)
                        off = A * (1.0 - d)
                        return float(_np.sum(off**2) / (_np.sum(A**2) + 1e-12))

                    K = M.shape[0] if M.ndim == 3 else 1
                    idxs = _np.linspace(0, K-1, num=min(24, K), dtype=int)

                    m_off, m_spread, g_off, g_spread = [], [], [], []
                    C_t = self.metric_adapter.centroids_tens.detach() if hasattr(self.metric_adapter, 'centroids_tens') else None
                    for i in idxs:
                        A = M[i] if M.ndim == 3 else M
                        m_off.append(offdiag_ratio(A))
                        eig = _np.linalg.eigvalsh(A)
                        eig = _np.clip(eig, 1e-8, None)
                        m_spread.append(float(_np.max(eig) / _np.min(eig)))

                        try:
                            if C_t is not None:
                                zi = C_t[i:i+1].to(self.device)
                                Ginv_i = self.metric_adapter.G_inv(zi)[0].detach().cpu().numpy()
                                g_off.append(offdiag_ratio(Ginv_i))
                                ge = _np.linalg.eigvalsh(Ginv_i)
                                ge = _np.clip(ge, 1e-8, None)
                                g_spread.append(float(_np.max(ge) / _np.min(ge)))
                        except Exception:
                            pass

                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].hist(m_off, bins=20); axes[0].set_title('M off-diagonal ratio')
                    axes[1].hist(m_spread, bins=20); axes[1].set_title('M eigen spread (cond)')
                    plt.tight_layout(); wandb.log({"M_offdiag_and_spread": wandb.Image(fig)}); plt.close()

                    if len(g_off) > 0:
                        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                        axes[0].hist(g_off, bins=20); axes[0].set_title('G_inv(c) off-diagonal ratio')
                        axes[1].hist(g_spread, bins=20); axes[1].set_title('G_inv(c) eigen spread (cond)')
                        plt.tight_layout(); wandb.log({"Ginv_offdiag_and_spread": wandb.Image(fig)}); plt.close()
                except Exception as _e:
                    print(f"⚠️ Metric diagnostics failed: {_e}")

                # 6. Panels: a few M_j and corresponding G_inv at centroid
                try:
                    K = M.shape[0] if M.ndim == 3 else 1
                    sel = np.linspace(0, K-1, num=min(6, K), dtype=int)
                    fig, axes = plt.subplots(2, len(sel), figsize=(4*len(sel), 6))
                    for c, i in enumerate(sel):
                        A = M[i] if M.ndim == 3 else M
                        im = axes[0, c].imshow(A, cmap='viridis'); axes[0, c].set_title(f"M[{i}]")
                        plt.colorbar(im, ax=axes[0, c])
                    if hasattr(self.metric_adapter, 'centroids_tens') and self.metric_adapter.centroids_tens is not None:
                        for c, i in enumerate(sel):
                            zi = self.metric_adapter.centroids_tens[i:i+1].to(self.device)
                            Ginv_i = self.metric_adapter.G_inv(zi)[0].detach().cpu().numpy()
                            im = axes[1, c].imshow(Ginv_i, cmap='magma'); axes[1, c].set_title(f"G_inv(c[{i}])")
                            plt.colorbar(im, ax=axes[1, c])
                    plt.tight_layout(); wandb.log({"metric_matrix_panels": wandb.Image(fig)}); plt.close()
                except Exception as _e:
                    print(f"⚠️ Metric panel logging failed: {_e}")

                # 7. Console-based tracing of metric SPD statistics
                try:
                    C_tensor = torch.as_tensor(centroids, dtype=torch.float32)
                    M_tensor = torch.as_tensor(M, dtype=torch.float32)
                    self._trace_metric_statistics(
                        centroids=C_tensor,
                        matrices=M_tensor,
                        stage="metric_analysis"
                    )
                except Exception as trace_exc:
                    print(f"[METRIC TRACE] Failed during metric analysis: {trace_exc}")
                    
        except Exception as e:
            print(f"⚠️ Metric analysis logging failed: {e}")

    def _trace_metric_statistics(
        self,
        *,
        centroids: Optional[torch.Tensor],
        matrices: Optional[torch.Tensor],
        stage: str = "export",
        sample_points: int = 16,
    ) -> None:
        """Print detailed statistics about the learned metric tensors."""
        if centroids is None or matrices is None:
            print(f"[METRIC TRACE][{stage}] Missing centroids or metric matrices; skipping trace.")
            return

        with torch.no_grad():
            mats = matrices.reshape(-1, matrices.shape[-2], matrices.shape[-1]).float()
            if mats.numel() == 0:
                print(f"[METRIC TRACE][{stage}] Empty metric tensor payload; skipping.")
                return

            mats = 0.5 * (mats + mats.transpose(-1, -2))  # symmetrize
            eigvals = torch.linalg.eigvalsh(mats)
            eig_min = float(eigvals.min().item())
            eig_max = float(eigvals.max().item())
            eig_mean = float(eigvals.mean().item())
            eig_median = float(eigvals.median().item())
            eig_std = float(eigvals.std(unbiased=False).item())
            per_cond = eigvals.max(dim=-1).values / eigvals.min(dim=-1).values.clamp_min(1e-12)
            cond_min = float(per_cond.min().item())
            cond_max = float(per_cond.max().item())
            cond_mean = float(per_cond.mean().item())

            diag = torch.diagonal(mats, dim1=-2, dim2=-1)
            diag_min = float(diag.min().item())
            diag_max = float(diag.max().item())
            diag_mean = float(diag.mean().item())

            sign, logabsdet = torch.linalg.slogdet(mats)
            finite_mask = torch.isfinite(logabsdet)
            if finite_mask.any():
                logdet_min = float(logabsdet[finite_mask].min().item())
                logdet_max = float(logabsdet[finite_mask].max().item())
                logdet_mean = float(logabsdet[finite_mask].mean().item())
            else:
                logdet_min = logdet_max = logdet_mean = float('nan')

            print(f"[METRIC TRACE][{stage}] matrices={mats.shape[0]}, latent_dim={mats.shape[-1]}")
            print(f"[METRIC TRACE][{stage}] eigen min={eig_min:.6f} max={eig_max:.6f} mean={eig_mean:.6f} median={eig_median:.6f} std={eig_std:.6f}")
            print(f"[METRIC TRACE][{stage}] condition number range=[{cond_min:.2e}, {cond_max:.2e}] mean={cond_mean:.2e}")
            print(f"[METRIC TRACE][{stage}] diagonal min={diag_min:.6f} max={diag_max:.6f} mean={diag_mean:.6f}")
            print(f"[METRIC TRACE][{stage}] log|det| min={logdet_min:.6f} max={logdet_max:.6f} mean={logdet_mean:.6f}")

            try:
                num_centroids = int(centroids.shape[0])
            except Exception:
                num_centroids = 0
            if num_centroids > 0:
                print(f"[METRIC TRACE][{stage}] centroids={num_centroids} (latent_dim={centroids.shape[-1]})")

            # Optional: sample a few points to evaluate assembled G^{-1}(z)
            if (
                num_centroids > 0
                and hasattr(self, 'metric_adapter')
                and getattr(self.metric_adapter, 'G_inv', None) is not None
            ):
                try:
                    sample_n = min(sample_points, num_centroids)
                    if sample_n > 0:
                        idx = torch.randperm(num_centroids)[:sample_n]
                        z = centroids[idx].to(self.device)
                        Ginv = self.metric_adapter.G_inv(z).detach().cpu().float()
                        Ginv = 0.5 * (Ginv + Ginv.transpose(-1, -2))
                        eigs_g = torch.linalg.eigvalsh(Ginv)
                        print(
                            f"[METRIC TRACE][{stage}] assembled G_inv eigen min={eigs_g.min().item():.6f} "
                            f"max={eigs_g.max().item():.6f} mean={eigs_g.mean().item():.6f}"
                        )
                except Exception as sample_exc:
                    print(f"[METRIC TRACE][{stage}] Failed to trace assembled G_inv: {sample_exc}")

    def _collect_all_latents(self, max_samples: Optional[int] = 6000) -> torch.Tensor:
        """Encode the dataset to collect latent means μ(x). Optionally limit samples for speed."""
        self.model.eval()
        device = self.device
        with torch.no_grad():
            # Concatenate train and test for coverage
            data = torch.cat([self.train_data, self.test_data], dim=0)
            if max_samples is not None and data.shape[0] > max_samples:
                idx = torch.randperm(data.shape[0])[: max_samples]
                data = data[idx]
            batch_size = min(256, data.shape[0])
            mus = []
            for i in range(0, data.shape[0], batch_size):
                batch = data[i : i + batch_size].to(device)
                enc = self.model.encoder(batch)
                mus.append(enc["embedding"].detach().cpu())
            return torch.cat(mus, dim=0)

    def _normalize_metric_matrices(self, M: torch.Tensor) -> torch.Tensor:
        """Apply scale normalization to SPD matrices as configured."""
        if self.metric_normalization.lower() == "none":
            return M
        d = M.shape[-1]
        if self.metric_normalization.lower() == "trace":
            traces = torch.einsum('kii->k', M)
            # Avoid division by zero
            scales = (traces / float(d)).clamp_min(1e-12)
            M = M / scales.view(-1, 1, 1) * float(self.target_mean_eig)
        elif self.metric_normalization.lower() == "det":
            dets = torch.linalg.det(M).abs().clamp_min(1e-24)
            # Scale so geometric mean eigenvalue equals target_mean_eig
            scales = dets.pow(1.0 / float(d)).clamp_min(1e-12)
            M = M / scales.view(-1, 1, 1) * float(self.target_mean_eig)
        return M

    def _postprocess_metric(self):
        """Optional alignment of metric with latent KNN covariance and normalization.

        This improves anisotropy and off-diagonal structure using local latent geometry.
        """
        if not (hasattr(self.model, 'M_tens') and hasattr(self.model, 'centroids_tens')):
            return
        M = self.model.M_tens
        C = self.model.centroids_tens
        if M is None or C is None:
            return
        if C.ndim != 2 or M.ndim != 3:
            return

        device = C.device
        K, D = C.shape

        # Snapshot before-edit metrics for logging
        M_before = M.detach().cpu().clone()

        # Optionally align with local latent KNN covariance
        if self.align_with_knn_cov:
            print(f"🔧 Aligning metric with KNN covariance: k={self.knn_k}, alpha={self.alpha_align:.2f}")
            all_mu = self._collect_all_latents(max_samples=8000).to(device)
            # Compute distances from centroids to all latents
            with torch.no_grad():
                dists = torch.cdist(C, all_mu)  # [K, N]
            knn_k = min(int(self.knn_k), all_mu.shape[0])
            M_aligned = []
            eye = torch.eye(D, device=device, dtype=all_mu.dtype)
            for i in range(K):
                idx = torch.topk(dists[i], k=knn_k, largest=False).indices
                local = all_mu[idx]
                mean = local.mean(dim=0, keepdim=True)
                diffs = local - mean
                # Unbiased covariance
                cov = (diffs.t() @ diffs) / max(1, local.shape[0] - 1)
                # Regularize for numerical stability
                cov = cov + (1e-6 * eye)
                # Blend original M and local covariance
                M_new = (1.0 - self.alpha_align) * M[i] + self.alpha_align * cov
                # Ensure SPD via eig clamping
                evals, evecs = torch.linalg.eigh(M_new)
                evals = torch.clamp(evals, min=1e-8)
                M_new = evecs @ torch.diag(evals) @ evecs.t()
                M_aligned.append(M_new)
            M = torch.stack(M_aligned, dim=0)

        # Optional: refine using decoder Jacobian pullback metric at centroids
        if self.reestimate_metric_from_decoder_jacobian:
            print(f"🔧 Blending metric with decoder-Jacobian pullback: alpha={self.jacobian_alpha:.2f}, stride={self.jacobian_stride}, h={self.jacobian_h}")
            M_j_list = []
            self.model.eval()
            with torch.no_grad():
                for i in range(K):
                    z = C[i]
                    D = self.latent_dim
                    h = self.jacobian_h
                    cols = []
                    for k in range(D):
                        e = torch.zeros(D, device=device, dtype=z.dtype)
                        e[k] = 1.0
                        zp = z + h * e
                        zm = z - h * e
                        yp = self.model.decoder(zp.unsqueeze(0))["reconstruction"][0]
                        ym = self.model.decoder(zm.unsqueeze(0))["reconstruction"][0]
                        # Optionally subsample spatial grid for speed
                        if yp.dim() == 3 and self.jacobian_stride > 1:
                            s = self.jacobian_stride
                            yp = yp[:, ::s, ::s]
                            ym = ym[:, ::s, ::s]
                        vp = (yp - ym).reshape(-1) / (2.0 * h)
                        cols.append(vp)
                    J = torch.stack(cols, dim=1)  # [out_dim, D]
                    M_jac = J.t().matmul(J) / max(1, J.shape[0])
                    # Regularize and clamp to SPD
                    M_jac = M_jac + 1e-6 * torch.eye(D, device=device, dtype=J.dtype)
                    evals, evecs = torch.linalg.eigh(M_jac)
                    evals = torch.clamp(evals, min=1e-8)
                    M_jac = evecs @ torch.diag(evals) @ evecs.t()
                    M_j_list.append(M_jac)
            M_j = torch.stack(M_j_list, dim=0)
            # Blend
            M = (1.0 - self.jacobian_alpha) * M + self.jacobian_alpha * M_j

        # Apply scale normalization if requested (after all blends)
        M = self._normalize_metric_matrices(M)

        # Global scale
        if self.metric_scale != 1.0:
            M = M * self.metric_scale

        # Optional centroid realignment to current latent distribution using KMeans
        if getattr(self, "realign_centroids", False) and getattr(self, "centroid_realign_method", "kmeans") == "kmeans":
            try:
                from sklearn.cluster import KMeans
                with torch.no_grad():
                    all_mu = self._collect_all_latents(max_samples=12000).to(device)
                K_now = C.shape[0]
                km = KMeans(n_clusters=K_now, n_init=10, random_state=42)
                km.fit(all_mu.detach().cpu().numpy())
                C_new = torch.tensor(km.cluster_centers_, device=device, dtype=C.dtype)
                # Map each new centroid to a unique closest old centroid to reuse its M
                used = torch.zeros(K_now, dtype=torch.bool, device=device)
                order = []
                for j in range(K_now):
                    dists = torch.cdist(C_new[j:j+1], C)[0]
                    dists[used] = float("inf")
                    idx = int(torch.argmin(dists).item())
                    order.append(idx)
                    used[idx] = True
                order_t = torch.tensor(order, device=device)
                M = M[order_t]
                C = C_new
                print("🔧 Realigned centroids to KMeans on μ(x) and reassigned M by nearest old centroid")
            except Exception as _e:
                print(f"⚠️ Centroid realignment skipped: {_e}")

        # Write back
        self.model.M_tens = M
        self.model.centroids_tens = C

        # Log before/after summary
        try:
            import numpy as _np
            def _offdiag_ratio_np(A: _np.ndarray) -> float:
                d = _np.eye(A.shape[0], dtype=A.dtype)
                off = A * (1.0 - d)
                return float(_np.sum(off**2) / (_np.sum(A**2) + 1e-12))
            Mb = M_before.numpy()
            Ma = M.detach().cpu().numpy()
            r_before = _np.array([_offdiag_ratio_np(m) for m in Mb])
            r_after = _np.array([_offdiag_ratio_np(m) for m in Ma])
            if wandb.run is not None:
                wandb.log({
                    "metric_offdiag_ratio_before_mean": float(r_before.mean()),
                    "metric_offdiag_ratio_after_mean": float(r_after.mean()),
                })
        except Exception:
            pass
            
    def _log_generated_samples(self):
        """Log generated samples from the trained model."""
        try:
            samples = self.sample(num_samples=16)
            
            # Create a grid of generated samples
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            
            for i in range(16):
                row = i // 4
                col = i % 4
                
                img = samples[i].permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"Generated {i+1}", fontsize=10)
                axes[row, col].axis('off')
            
            plt.tight_layout()
            wandb.log({"generated_samples": wandb.Image(fig)})
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Generated samples logging failed: {e}")

    def _log_rhmc_prior_samples(self):
        """Sample latents via RHMC prior and decode; log to WandB."""
        try:
            self.model.eval()
            # Sampling needs gradients; decode without gradients
            z = self.rhmc_sampler.sample_prior(num_samples=16, method='hmc')
            with torch.no_grad():
                dec = self.model.decoder(z)
                samples = dec["reconstruction"]

                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                for i in range(16):
                    r, c = divmod(i, 4)
                    img = samples[i].permute(1, 2, 0).cpu().numpy()
                    img = np.clip(img, 0, 1)
                    axes[r, c].imshow(img)
                    axes[r, c].axis('off')
                plt.tight_layout()
                wandb.log({"rhmc_prior_samples": wandb.Image(fig)})
                plt.close()
        except Exception as e:
            print(f"⚠️ RHMC prior sampling log failed: {e}")

    def _log_metric_det_heatmap_2d(self):
        """Log a 2D determinant heatmap of G_inv over first two latent dims."""
        try:
            if self.latent_dim < 2:
                return
            grid_lin = torch.linspace(-3, 3, 50, device=self.device)
            X, Y = torch.meshgrid(grid_lin, grid_lin, indexing='ij')
            Z = torch.zeros(X.shape + (self.latent_dim,), device=self.device)
            Z[..., 0] = X
            Z[..., 1] = Y
            Z = Z.reshape(-1, self.latent_dim)
            with torch.no_grad():
                Ginv = self.metric_adapter.G_inv(Z)
                dets = torch.linalg.det(Ginv)
                dets = torch.clamp(dets, min=1e-12)
                H = dets.reshape(50, 50).sqrt().cpu().numpy()
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(H, origin='lower', extent=[-3, 3, -3, 3], cmap='viridis')
            ax.set_title('sqrt(det(G^{-1})) over (z1,z2)')
            ax.set_xlabel('z1'); ax.set_ylabel('z2')
            plt.colorbar(im, ax=ax)
            # Overlay centroids if available
            try:
                if hasattr(self.metric_adapter, 'centroids_tens') and self.metric_adapter.centroids_tens is not None:
                    C = self.metric_adapter.centroids_tens.detach().cpu().numpy()
                    if C.shape[1] >= 2:
                        ax.scatter(C[:, 0], C[:, 1], s=20, c='red', marker='x', label='centroids')
                        ax.legend(loc='upper right')
            except Exception:
                pass
            wandb.log({"metric_sqrt_det_Ginv_heatmap": wandb.Image(fig)})
            plt.close()
        except Exception as e:
            print(f"⚠️ Metric heatmap log failed: {e}")

    def _log_posterior_rhmc_reconstructions(self):
        """Encode test images, RHMC-sample from posterior, decode, and log comparisons."""
        try:
            self.model.eval()
            with torch.no_grad():
                original = self.test_data[:16].to(self.device)
                enc_out = self.model.encoder(original)
                mu = enc_out["embedding"]
                log_var = enc_out["log_covariance"]
            # RHMC posterior refinement (use volume-element sampler over metric adapter)
            vol_sampler = RHVAEVolumeElementHMCSampler(
                model=self.metric_adapter,
                mcmc_steps_nbr=30,
                n_lf=max(5, self.n_lf - 1),
                eps_lf=max(1e-5, float(self.eps_lf) * 0.7),
                beta_zero=self.beta_zero,
            )
            z = vol_sampler.sample_riemannian_latents(mu, log_var)
            with torch.no_grad():
                dec_out = self.model.decoder(z)
                recon = dec_out["reconstruction"]

            # Grid log
            fig, axes = plt.subplots(4, 8, figsize=(20, 10))
            for i in range(16):
                r, c = divmod(i, 8)
                img_o = original[i].permute(1, 2, 0).cpu().numpy()
                img_o = np.clip(img_o, 0, 1)
                axes[r*2, c].imshow(img_o); axes[r*2, c].axis('off')
                img_r = recon[i].permute(1, 2, 0).cpu().numpy()
                img_r = np.clip(img_r, 0, 1)
                axes[r*2+1, c].imshow(img_r); axes[r*2+1, c].axis('off')
            plt.tight_layout()
            wandb.log({"rhmc_posterior_recon": wandb.Image(fig)})
            plt.close()

            # MSE metric
            mse = torch.mean((original - recon) ** 2).item()
            wandb.log({"rhmc_posterior_recon_mse": mse})
        except Exception as e:
            print(f"⚠️ RHMC posterior recon log failed: {e}")

    def _log_weight_vs_distance(self):
        """Plot centroid weight and sqrt(det(G^{-1})) as a function of distance from centroids."""
        try:
            if not hasattr(self.metric_adapter, 'centroids_tens') or self.metric_adapter.centroids_tens is None:
                return
            C = self.metric_adapter.centroids_tens.detach()
            if C.shape[0] == 0:
                return
            device = self.device
            num_centroids_to_show = int(min(5, C.shape[0]))
            indices = torch.linspace(0, C.shape[0]-1, steps=num_centroids_to_show).long()
            radii = torch.linspace(0, 3.0, 60, device=device)

            fig, axes = plt.subplots(num_centroids_to_show, 2, figsize=(10, 3 * num_centroids_to_show))
            if num_centroids_to_show == 1:
                axes = np.array([axes])

            for row, idx in enumerate(indices):
                c = C[idx].to(device)
                # Directions along z1 and z2 axes
                e1 = torch.zeros(self.latent_dim, device=device); e1[0] = 1.0
                e2 = torch.zeros(self.latent_dim, device=device); e2[1] = 1.0 if self.latent_dim > 1 else 0.0

                # Build rays points: c + r * e1
                Z1 = c.unsqueeze(0) + radii.unsqueeze(1) * e1.unsqueeze(0)
                Z2 = c.unsqueeze(0) + radii.unsqueeze(1) * e2.unsqueeze(0)
                Z = torch.cat([Z1, Z2], dim=0)

                with torch.no_grad():
                    Ginv = self.metric_adapter.G_inv(Z)
                    dets = torch.linalg.det(Ginv).clamp(min=1e-12).sqrt().cpu().numpy()

                # Weights to this centroid only
                with torch.no_grad():
                    diff1 = Z1 - c.unsqueeze(0)
                    diff2 = Z2 - c.unsqueeze(0)
                    dists1 = torch.linalg.norm(diff1, dim=-1)
                    dists2 = torch.linalg.norm(diff2, dim=-1)
                    w1 = torch.exp(-(dists1 ** 2) / (self.temperature ** 2)).cpu().numpy()
                    w2 = torch.exp(-(dists2 ** 2) / (self.temperature ** 2)).cpu().numpy()

                d = radii.cpu().numpy()
                # Left: weights; Right: sqrt(det(G_inv))
                axes[row, 0].plot(d, w1, label='axis z1')
                axes[row, 0].plot(d, w2, label='axis z2')
                axes[row, 0].set_title(f'Centroid {int(idx)} weight vs dist')
                axes[row, 0].set_xlabel('distance'); axes[row, 0].set_ylabel('weight')
                axes[row, 0].legend()

                axes[row, 1].plot(d, dets[: len(d)], label='axis z1')
                axes[row, 1].plot(d, dets[len(d):], label='axis z2')
                axes[row, 1].set_title(f'Centroid {int(idx)} sqrt(det(G^-1)) vs dist')
                axes[row, 1].set_xlabel('distance'); axes[row, 1].set_ylabel('sqrt(det(G^-1))')
                axes[row, 1].legend()

            plt.tight_layout()
            wandb.log({"weight_and_sqrt_det_vs_distance": wandb.Image(fig)})
            plt.close()
        except Exception as e:
            print(f"⚠️ Weight vs distance log failed: {e}")

    def _log_comprehensive_ginv_panel(self):
        """Log a 2x2 panel: data+centroids, det(G^{-1}) heatmap, points colored by det(G^{-1}), anisotropy map.

        Also overlays KMeans centroids computed from current latent means to verify adaptation.
        """
        try:
            if wandb.run is None:
                return
            # Latent means of a subset
            with torch.no_grad():
                data = self.train_data[:3000].to(self.device)
                enc = self.model.encoder(data)
                mu = enc["embedding"].detach()
            if mu.shape[1] < 2:
                return
            # Choose subspace: PCA(2) for visual alignment with centroid PCA overlay
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            Zp = pca.fit_transform(mu.cpu().numpy())
            Up = torch.tensor(pca.components_.T, device=self.device, dtype=mu.dtype)  # [D,2]
            mean = torch.tensor(pca.mean_, device=self.device, dtype=mu.dtype)        # [D]
            z = Zp
            C = None
            C_km = None
            if hasattr(self.metric_adapter, 'centroids_tens') and self.metric_adapter.centroids_tens is not None and self.metric_adapter.centroids_tens.shape[1] >= 2:
                C_full = self.metric_adapter.centroids_tens.detach().to(self.device, dtype=mu.dtype)
                # Center by PCA mean then project
                C = ((C_full - mean) @ Up).cpu().numpy()

            # Optional: recompute centroids on all latents (KMeans) to check adaptation
            try:
                from sklearn.cluster import KMeans
                all_mu = self._collect_all_latents(max_samples=8000)
                n_clusters = int(C.shape[0]) if C is not None else min(32, all_mu.shape[0])
                km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                km.fit(all_mu.cpu().numpy())
                C_km_full = km.cluster_centers_
                C_km = np.asarray(C_km_full)[:, :2]
            except Exception:
                C_km = None

            import matplotlib.pyplot as plt
            import numpy as np
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            # 1. scatter
            ax = axes[0, 0]
            ax.scatter(z[:, 0], z[:, 1], s=5, alpha=0.2, color='skyblue', label='Data Points')
            if C is not None:
                ax.scatter(C[:, 0], C[:, 1], s=40, marker='*', color='crimson', label='Model Centroids (PCA2)')
            if C_km is not None:
                ax.scatter(C_km[:, 0], C_km[:, 1], s=30, marker='x', color='black', label='KMeans on μ')
            ax.set_title('1. Centroids vs Data\n(Model centroids and K-Means overlay)')
            ax.set_xlabel('z1'); ax.set_ylabel('z2'); ax.legend(loc='upper right', fontsize=8)

            # Grid
            xmin, xmax = z[:, 0].min()-0.5, z[:, 0].max()+0.5
            ymin, ymax = z[:, 1].min()-0.5, z[:, 1].max()+0.5
            gx, gy = np.meshgrid(np.linspace(xmin, xmax, 140), np.linspace(ymin, ymax, 140))
            grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
            grid_t = torch.tensor(grid, device=self.device, dtype=mu.dtype)
            # For this panel, switch to RAW-weight metric to match data density when desired
            alt_adapter = RHVAEMetricAdapter(
                model=self.model,
                temperature=self.temperature,
                regularization=self.regularization,
                weight_kernel=self.weight_kernel,
                weight_metric_normalization=self.weight_metric_normalization,
                normalize_weight_sum=False,
                topk_weights=self.topk_weights,
            )
            with torch.no_grad():
                Ginv2 = alt_adapter.G_inv_subspace(grid_t, Up, mean)
                det = torch.linalg.det(Ginv2).cpu().numpy().reshape(gx.shape)
                evals, _ = torch.linalg.eigh(Ginv2)
                aniso = (evals[:, -1] - evals[:, -2]).cpu().numpy().reshape(gx.shape)

            # 2. det heatmap (chunked computation and float32 to reduce memory)
            ax = axes[0, 1]
            im = ax.imshow(det.astype(np.float32), origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto')
            ax.set_title('2. G^{-1} Determinant\n(Manifold Structure)')
            ax.set_xlabel('z1'); ax.set_ylabel('z2')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='det(G^{-1})')

            # 3. points colored by det
            ax = axes[1, 0]
            # Plot true data points faintly
            ax.scatter(z[:, 0], z[:, 1], s=3, alpha=0.05, color='lightgray', label='Data (μ)')
            # Draw RHMC samples (volume-element sampler) from the RAW-weight metric
            alt_sampler = RHVAEVolumeElementHMCSampler(
                model=alt_adapter,
                mcmc_steps_nbr=50,
                n_lf=self.n_lf,
                eps_lf=self.eps_lf,
                beta_zero=self.beta_zero,
            )
            # Important: RHMC sampling needs gradients enabled
            # Reduce memory pressure: sample in smaller batches then concat
            try:
                z_chunks = []
                need = 1600
                bs = 400
                while need > 0:
                    take = min(bs, need)
                    z_chunks.append(alt_sampler.sample_prior(num_samples=take, method='hmc'))
                    need -= take
                z_rh = torch.cat(z_chunks, dim=0)
            except Exception:
                z_rh = alt_sampler.sample_prior(num_samples=800, method='hmc')
            z_rh2 = ((z_rh - mean) @ Up).detach().cpu().numpy()
            pts = torch.tensor(z_rh2, device=self.device, dtype=mu.dtype)
            with torch.no_grad():
                det_pts = torch.linalg.det(alt_adapter.G_inv_subspace(pts, Up, mean)).cpu().numpy()
            sc = ax.scatter(z_rh2[:, 0], z_rh2[:, 1], c=det_pts, s=8, cmap='viridis', label='RHMC prior (raw)')
            ax.set_title('3. RHMC prior samples (RAW metric)\n(Colored by det(G^{-1}))')
            ax.set_xlabel('z1'); ax.set_ylabel('z2')
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='det(G^{-1})')

            # 4. anisotropy
            ax = axes[1, 1]
            im2 = ax.imshow(aniso, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='coolwarm', aspect='auto')
            if C is not None:
                ax.scatter(C[:, 0], C[:, 1], s=10, facecolors='none', edgecolors='k', alpha=0.5)
            ax.set_title('4. Anisotropy (λ1 - λ2) (RAW weights)\n(Stretching/Compression)')
            ax.set_xlabel('z1'); ax.set_ylabel('z2')
            fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='Anisotropy')

            plt.tight_layout()
            wandb.log({"comprehensive_ginv_panel_pca2": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"⚠️ Comprehensive G^-1 panel log failed: {e}")

    def _log_comprehensive_ginv_panel_z12(self):
        """Deprecated: prefer PCA-aligned comprehensive panel."""
        return

    def _log_det_G_and_Ginv_maps(self):
        """Log side-by-side heatmaps of det(G^{-1}) and det(G) over (z1, z2)."""
        try:
            if self.latent_dim < 2 or wandb.run is None:
                return
            import numpy as np
            from sklearn.decomposition import PCA
            with torch.no_grad():
                mu = self.model.encoder(self.train_data[:4096].to(self.device))["embedding"].detach()
            pca = PCA(n_components=2, random_state=42)
            Zp = pca.fit_transform(mu.cpu().numpy())
            Up = torch.tensor(pca.components_.T, device=self.device, dtype=mu.dtype)
            mean = torch.tensor(pca.mean_, device=self.device, dtype=mu.dtype)
            z2 = Zp
            xmin, xmax = float(np.percentile(z2[:, 0], 1) - 0.2), float(np.percentile(z2[:, 0], 99) + 0.2)
            ymin, ymax = float(np.percentile(z2[:, 1], 1) - 0.2), float(np.percentile(z2[:, 1], 99) + 0.2)
            gx, gy = np.meshgrid(np.linspace(xmin, xmax, 140), np.linspace(ymin, ymax, 140))
            grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
            grid_t = torch.tensor(grid, device=self.device, dtype=mu.dtype)

            with torch.no_grad():
                Ginv2 = self.metric_adapter.G_inv_subspace(grid_t, Up, mean)
                detGinv = torch.linalg.det(Ginv2)
                evals, _ = torch.linalg.eigh(Ginv2)
                logdet_inv = torch.sum(torch.log(evals.clamp_min(1e-12)), dim=1)
                detG = torch.exp(-logdet_inv)

            detGinv_img = detGinv.cpu().numpy().reshape(gx.shape)
            detG_img = detG.cpu().numpy().reshape(gx.shape)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            im1 = axes[0].imshow(detGinv_img, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto')
            axes[0].set_title('det(G^{-1})'); axes[0].set_xlabel('z1'); axes[0].set_ylabel('z2')
            fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

            im2 = axes[1].imshow(detG_img, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='magma', aspect='auto')
            axes[1].set_title('det(G)'); axes[1].set_xlabel('z1'); axes[1].set_ylabel('z2')
            fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

            plt.tight_layout()
            wandb.log({"det_G_and_det_Ginv_pca2": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"⚠️ det(G) vs det(G^-1) panel log failed: {e}")

    def _log_det_G_and_Ginv_maps_z12(self):
        """Deprecated: prefer PCA-aligned determinant maps."""
        return

    def _log_det_variants_panels(self):
        """Log det(G^{-1}) vs det(G) with and without weight normalization, and a weight-sum map.

        Helps diagnose why det(G^{-1}) may peak away from centroids when weights are normalized.
        """
        try:
            if wandb.run is None or self.latent_dim < 2:
                return
            with torch.no_grad():
                mu = self.model.encoder(self.train_data[:4096].to(self.device))["embedding"].detach()
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            Zp = pca.fit_transform(mu.cpu().numpy())
            Up = torch.tensor(pca.components_.T, device=self.device, dtype=mu.dtype)
            mean = torch.tensor(pca.mean_, device=self.device, dtype=mu.dtype)
            z2 = Zp
            xmin, xmax = float(np.percentile(z2[:, 0], 1) - 0.2), float(np.percentile(z2[:, 0], 99) + 0.2)
            ymin, ymax = float(np.percentile(z2[:, 1], 1) - 0.2), float(np.percentile(z2[:, 1], 99) + 0.2)
            gx, gy = np.meshgrid(np.linspace(xmin, xmax, 140), np.linspace(ymin, ymax, 140))
            grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
            grid_t = torch.tensor(grid, device=self.device, dtype=mu.dtype)

            # Current adapter (possibly normalized)
            with torch.no_grad():
                Ginv_norm2 = self.metric_adapter.G_inv_subspace(grid_t, Up, mean)
                detGinv_norm = torch.linalg.det(Ginv_norm2).cpu().numpy().reshape(gx.shape)
                G_norm2 = torch.linalg.inv(Ginv_norm2)
                detG_norm = torch.linalg.det(G_norm2).cpu().numpy().reshape(gx.shape)

            # Unnormalized adapter for comparison
            alt_adapter = RHVAEMetricAdapter(
                model=self.model,
                temperature=self.temperature,
                regularization=self.regularization,
                weight_kernel=self.weight_kernel,
                weight_metric_normalization=self.weight_metric_normalization,
                normalize_weight_sum=False,
                topk_weights=self.topk_weights,
            )
            with torch.no_grad():
                Ginv_raw2 = alt_adapter.G_inv_subspace(grid_t, Up, mean)
                detGinv_raw = torch.linalg.det(Ginv_raw2).cpu().numpy().reshape(gx.shape)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            im = axes[0, 0].imshow(detGinv_norm, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto')
            axes[0, 0].set_title('det(G^{-1}) (normalized weights)'); axes[0, 0].set_xlabel('z1'); axes[0, 0].set_ylabel('z2')
            fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

            im = axes[0, 1].imshow(detGinv_raw, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto')
            axes[0, 1].set_title('det(G^{-1}) (raw weights sum)'); axes[0, 1].set_xlabel('z1'); axes[0, 1].set_ylabel('z2')
            fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

            im = axes[1, 0].imshow(detG_norm, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='magma', aspect='auto')
            axes[1, 0].set_title('det(G) (normalized weights)'); axes[1, 0].set_xlabel('z1'); axes[1, 0].set_ylabel('z2')
            fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

            # Weight-sum map (diagnostic). Recompute weights without normalization using robust BK batching.
            C = self.metric_adapter.centroids_tens.detach()  # [K,D]
            M = self.metric_adapter.M_tens.detach()          # [K,D,D]
            B = grid_t.shape[0]
            K = C.shape[0]
            D = C.shape[1]
            # Project into PCA(2)
            C2 = (C - mean) @ Up  # [K,2]
            diff2 = C2.unsqueeze(0) - grid_t.unsqueeze(1)  # [B,K,2]
            if self.weight_kernel == 'isotropic':
                dist = torch.sum(diff2 * diff2, dim=-1)
            else:
                # M projected: U^T M U
                MU = torch.einsum('kde,ej->kdj', M, Up)       # [K,D,2]
                M2 = torch.einsum('id,kdj->kij', Up.t(), MU)  # [K,2,2]
                S = M2
                if self.weight_kernel == 'mahalanobis_normed':
                    if self.weight_metric_normalization == 'trace':
                        traces = torch.einsum('kii->k', M2).clamp_min(1e-12)
                        S = M2 / (traces.view(-1, 1, 1) / 2.0)
                    elif self.weight_metric_normalization == 'det':
                        dets = torch.linalg.det(S).abs().clamp_min(1e-24)
                        scales = dets.pow(0.5).clamp_min(1e-12)
                        S = S / scales.view(-1, 1, 1)
                v = diff2.reshape(B * K, 2)
                S_bk = S.repeat(B, 1, 1).reshape(B * K, 2, 2)
                S_v = torch.bmm(v.unsqueeze(1), S_bk).squeeze(1)
                dist = (S_v * v).sum(dim=1).reshape(B, K)
            w_raw = torch.exp(-dist / (self.temperature ** 2))
            wsum = w_raw.sum(dim=1).cpu().numpy().reshape(gx.shape)
            im = axes[1, 1].imshow(wsum, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='plasma', aspect='auto')
            axes[1, 1].set_title('Weight sum Σ_j w_j(z) (raw)'); axes[1, 1].set_xlabel('z1'); axes[1, 1].set_ylabel('z2')
            fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

            plt.tight_layout()
            wandb.log({"det_variants_panel": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"⚠️ det variants panel log failed: {e}")

    def _log_centroids_pca_overlay(self):
        """PCA overlay of μ(x) and centroids to verify adaptation when raw (z1,z2) projection is misleading."""
        try:
            if wandb.run is None:
                return
            with torch.no_grad():
                all_mu = self._collect_all_latents(max_samples=8000)
            if all_mu.shape[1] < 3:
                return
            C = None
            if hasattr(self.metric_adapter, 'centroids_tens') and self.metric_adapter.centroids_tens is not None:
                C = self.metric_adapter.centroids_tens.detach().cpu()
            if C is None:
                return
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            Zp = pca.fit_transform(all_mu.cpu().numpy())
            Cp = pca.transform(C.cpu().numpy())
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            ax.scatter(Zp[:, 0], Zp[:, 1], s=5, alpha=0.15, color='skyblue', label='μ(x) (PCA)')
            ax.scatter(Cp[:, 0], Cp[:, 1], s=40, marker='*', color='crimson', label='Centroids (PCA)')
            ax.set_title('Centroids vs Data in PCA(2) space')
            ax.legend(loc='upper right', fontsize=8)
            wandb.log({"centroids_pca_overlay": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"⚠️ PCA centroid overlay failed: {e}")

    def _log_geodesic_like_interpolation(self):
        """Heuristic geodesic-like interpolation and overlay on det(G^{-1}) PCA(2) heatmap."""
        try:
            if not hasattr(self.metric_adapter, 'centroids_tens') or self.metric_adapter.centroids_tens is None:
                return
            C = self.metric_adapter.centroids_tens.detach().to(self.device)
            if C.shape[0] < 2:
                return
            idx = torch.randperm(C.shape[0])[:2]
            z = C[idx[0]].clone()
            z_target = C[idx[1]].clone()

            steps = 16
            path_images = []
            with torch.no_grad():
                for k in range(steps):
                    # Decode current point
                    img = self.model.decoder(z.unsqueeze(0))["reconstruction"][0]
                    path_images.append(img)
                    # Compute local guided step
                    d = (z_target - z)
                    Ginv = self.metric_adapter.G_inv(z.unsqueeze(0))[0]
                    v = Ginv @ d
                    # Normalize and step
                    step = 0.15 * v / (torch.norm(v) + 1e-8)
                    z = z + step

            # Log grid
            fig, axes = plt.subplots(2, steps, figsize=(2 * steps, 4))
            for i in range(steps):
                img = path_images[i].permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                axes[0, i].imshow(img); axes[0, i].axis('off')
                # Bar of latent for reference
                axes[1, i].bar(range(min(16, self.latent_dim)), z.cpu().numpy()[: min(16, self.latent_dim)])
                axes[1, i].set_ylim([-3, 3]); axes[1, i].set_xticks([])
            plt.tight_layout()
            wandb.log({"geodesic_like_interpolation": wandb.Image(fig)})
            plt.close()

            # New subfigure: overlay trajectory on det(G^{-1}) PCA2 heatmap
            from sklearn.decomposition import PCA
            with torch.no_grad():
                mu = self.model.encoder(self.train_data[:4096].to(self.device))["embedding"].detach()
            pca = PCA(n_components=2, random_state=42)
            Zp = pca.fit_transform(mu.cpu().numpy())
            Up = torch.tensor(pca.components_.T, device=self.device, dtype=mu.dtype)
            mean = torch.tensor(pca.mean_, device=self.device, dtype=mu.dtype)
            # Build grid
            import numpy as _np
            xmin, xmax = float(_np.percentile(Zp[:, 0], 1) - 0.2), float(_np.percentile(Zp[:, 0], 99) + 0.2)
            ymin, ymax = float(_np.percentile(Zp[:, 1], 1) - 0.2), float(_np.percentile(Zp[:, 1], 99) + 0.2)
            gx, gy = _np.meshgrid(_np.linspace(xmin, xmax, 160), _np.linspace(ymin, ymax, 160))
            grid = _np.stack([gx.ravel(), gy.ravel()], axis=1)
            grid_t = torch.tensor(grid, device=self.device, dtype=mu.dtype)
            with torch.no_grad():
                Ginv2 = self.metric_adapter.G_inv_subspace(grid_t, Up, mean)
                det = torch.linalg.det(Ginv2).cpu().numpy().reshape(gx.shape)
            # Build the same latent path in PCA2
            path_pts = []
            zc = C[idx[0]].clone()
            for _ in range(steps):
                path_pts.append(((zc - mean) @ Up).detach().cpu().numpy())
                d = (z_target - zc)
                Ginv = self.metric_adapter.G_inv(zc.unsqueeze(0))[0]
                v = Ginv @ d
                zc = zc + 0.15 * v / (torch.norm(v) + 1e-8)
            path_pts = _np.stack(path_pts)
            import matplotlib.pyplot as _plt
            fig2, ax2 = _plt.subplots(1, 1, figsize=(6, 5))
            im = ax2.imshow(det, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto')
            ax2.plot(path_pts[:, 0], path_pts[:, 1], color='crimson', linewidth=2.0)
            ax2.scatter(path_pts[0, 0], path_pts[0, 1], color='yellow', s=40, label='start')
            ax2.set_title('Geodesic path over det(G^{-1}) (PCA2)')
            ax2.set_xlabel('z1'); ax2.set_ylabel('z2'); fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            wandb.log({"geodesic_over_det_ginv_pca2": wandb.Image(fig2)})
            _plt.close(fig2)
        except Exception as e:
            print(f"⚠️ Geodesic-like interpolation log failed: {e}")

    def _log_synthetic_manifold_demo(self):
        """Sanity-check RHMC on a simple ring manifold with raw weights."""
        try:
            if self.latent_dim < 2 or wandb.run is None:
                return
            D = self.latent_dim
            K = 60
            theta = torch.linspace(0, 2 * torch.pi, K + 1, device=self.device)[:-1]
            radius = 2.0
            C = torch.zeros(K, D, device=self.device)
            C[:, 0] = radius * torch.cos(theta)
            C[:, 1] = radius * torch.sin(theta)
            M = torch.zeros(K, D, D, device=self.device)
            for k in range(K):
                t = torch.tensor([[-torch.sin(theta[k])], [torch.cos(theta[k])]], device=self.device)
                n = torch.tensor([[torch.cos(theta[k])], [torch.sin(theta[k])]], device=self.device)
                B = torch.eye(D, device=self.device)
                B[:2, :2] = 0.2 * (t @ t.T) + 0.02 * (n @ n.T)
                M[k] = B
            # Temporarily inject synthetic metric into model
            old_C, old_M = getattr(self.model, 'centroids_tens', None), getattr(self.model, 'M_tens', None)
            self.model.centroids_tens = C
            self.model.M_tens = M
            # Use isotropic kernel for a clean, analyzable ring potential and
            # sharper attraction to the annulus in the synthetic demo
            tmp_adapter = RHVAEMetricAdapter(
                model=self.model,
                temperature=0.5,
                regularization=self.regularization,
                weight_kernel='isotropic',
                weight_metric_normalization=self.weight_metric_normalization,
                normalize_weight_sum=False,
                topk_weights=self.topk_weights,
            )
            import numpy as np
            xmin, xmax = -3.5, 3.5
            ymin, ymax = -3.5, 3.5
            gx, gy = np.meshgrid(np.linspace(xmin, xmax, 160), np.linspace(ymin, ymax, 160))
            grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
            grid_t = torch.tensor(grid, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                Ginv2 = tmp_adapter.G_inv_first2(grid_t)
                det = torch.linalg.det(Ginv2).cpu().numpy().reshape(gx.shape)
            sampler = RHVAEVolumeElementHMCSampler(model=tmp_adapter, mcmc_steps_nbr=200, n_lf=max(15, int(self.n_lf)), eps_lf=max(0.02, float(self.eps_lf)), beta_zero=self.beta_zero)
            z = sampler.sample_prior(1200)
            z2 = z[:, :2].detach().cpu().numpy()
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            ax = axes[0]
            im = ax.imshow(det, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='equal')
            ax.scatter(C[:, 0].cpu(), C[:, 1].cpu(), s=10, c='r')
            ax.set_title('Synthetic det(G^{-1}) heatmap')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax = axes[1]
            ax.scatter(z2[:, 0], z2[:, 1], s=4, c='k', alpha=0.5)
            ax.set_title('RHMC samples (should follow ring)')
            ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])
            plt.tight_layout()
            wandb.log({"synthetic_manifold_rhmc": wandb.Image(fig)})
            plt.close(fig)
            # Restore original metric
            self.model.centroids_tens = old_C
            self.model.M_tens = old_M
        except Exception as e:
            print(f"⚠️ Synthetic manifold demo failed: {e}")
    def _log_geodesic_streamlines_2d(self):
        """Log geodesic streamlines over (z1, z2) by following steepest descent of a surrogate energy under G.

        We integrate ODE: dz/dt = v where v solves G(z) v = -∇phi(z). We use phi(z)=||z||^2/2 to visualize
        how the metric bends shortest paths. This is heuristic but reveals flow lines of the manifold.
        """
        try:
            if self.latent_dim < 2 or wandb.run is None:
                return
            import numpy as np
            import matplotlib.pyplot as plt

            # Grid of seeds
            with torch.no_grad():
                mu = self.model.encoder(self.train_data[:4096].to(self.device))["embedding"].detach()
                z_np = mu[:, :2].detach().cpu().numpy()
            xmin, xmax = float(np.percentile(z_np[:, 0], 1) - 0.2), float(np.percentile(z_np[:, 0], 99) + 0.2)
            ymin, ymax = float(np.percentile(z_np[:, 1], 1) - 0.2), float(np.percentile(z_np[:, 1], 99) + 0.2)
            seeds_x = np.linspace(xmin, xmax, 18)
            seeds_y = np.linspace(ymin, ymax, 18)
            seeds = np.stack(np.meshgrid(seeds_x, seeds_y), axis=-1).reshape(-1, 2)

            def step(z, h=0.04):
                z_full = torch.zeros(1, self.latent_dim, device=self.device)
                z_full[0, :2] = z
                z_full.requires_grad_(True)
                phi = 0.5 * torch.sum(z_full[:, :2] * z_full[:, :2])
                grad = torch.autograd.grad(phi, z_full)[0]  # [1,D]
                Gz = self.metric_adapter.G(z_full)[0]       # [D,D]
                # Solve G v = -grad for v in the first 2 dims (pad others to 0)
                G2 = Gz[:2, :2]
                g2 = grad[0, :2]
                v2 = torch.linalg.solve(G2 + 1e-8 * torch.eye(2, device=self.device), -g2)
                new = z + h * v2
                return new

            # Trace few steps for each seed
            paths = []
            for s in seeds:
                z = torch.tensor(s, device=self.device, dtype=torch.float32)
                path = [z.detach().cpu().numpy().copy()]
                for _ in range(35):
                    z = step(z)
                    path.append(z.detach().cpu().numpy().copy())
                paths.append(np.stack(path))

            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            ax.scatter(z_np[:, 0], z_np[:, 1], s=5, alpha=0.05, color='lightgray')
            for p in paths[::3]:
                ax.plot(p[:, 0], p[:, 1], color='tab:orange', linewidth=1.0, alpha=0.8)
            ax.set_title('Geodesic-like Streamlines over (z1, z2)')
            ax.set_xlabel('z1'); ax.set_ylabel('z2')
            wandb.log({"geodesic_streamlines": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"⚠️ Geodesic streamlines log failed: {e}")
            
    def _log_random_pair_geodesics_pca2(self, num_pairs: int = 6, steps: int = 60, h: float = 0.06):
        """Trace geodesic-like trajectories between random point pairs in PCA(2) using G_inv_subspace.

        We integrate in PCA(2): z_{t+1} = z_t + h * v_t with v_t = G^{-1}(z_t) * (z_b - z_t),
        which follows shortest paths under the local metric approximation. This is a pragmatic
        path-tracing heuristic that is stable and reveals bending due to anisotropy.
        """
        try:
            if wandb.run is None or self.latent_dim < 2:
                return
            import numpy as _np
            from sklearn.decomposition import PCA
            with torch.no_grad():
                mu = self.model.encoder(self.train_data[:4096].to(self.device))["embedding"].detach()
            pca = PCA(n_components=2, random_state=42)
            Zp = pca.fit_transform(mu.cpu().numpy())
            Up = torch.tensor(pca.components_.T, device=self.device, dtype=mu.dtype)
            mean = torch.tensor(pca.mean_, device=self.device, dtype=mu.dtype)

            z2 = torch.tensor(Zp, device=self.device, dtype=mu.dtype)
            n = z2.shape[0]
            idx = torch.randperm(n)[: max(2 * num_pairs, 2)]
            pairs = idx.view(-1, 2)

            paths = []
            for a, b in pairs:
                za = z2[a].clone()
                zb = z2[b].clone()
                path = [za.detach().cpu().numpy().copy()]
                zt = za
                for _ in range(steps):
                    d = (zb - zt).unsqueeze(0)  # [1,2]
                    Ginv = self.metric_adapter.G_inv_subspace(zt.unsqueeze(0), Up, mean)[0]  # [2,2]
                    v = Ginv @ d[0]
                    v = v / (torch.norm(v) + 1e-8)
                    zt = zt + h * v
                    path.append(zt.detach().cpu().numpy().copy())
                paths.append(_np.stack(path))

            import matplotlib.pyplot as _plt
            fig, ax = _plt.subplots(1, 1, figsize=(7, 6))
            ax.scatter(Zp[:, 0], Zp[:, 1], s=4, alpha=0.05, color='lightgray')
            colors = _plt.cm.tab10(_np.linspace(0, 1, len(paths)))
            for c, p in zip(colors, paths):
                ax.plot(p[:, 0], p[:, 1], color=c, linewidth=1.5)
                ax.scatter([p[0, 0], p[-1, 0]], [p[0, 1], p[-1, 1]], c=[c, c], s=18)
            ax.set_title('Geodesic-like trajectories between random pairs (PCA2)')
            ax.set_xlabel('z1'); ax.set_ylabel('z2')
            wandb.log({"geodesic_pairs_pca2": wandb.Image(fig)})
            _plt.close(fig)
        except Exception as e:
            print(f"⚠️ Pair geodesics PCA2 log failed: {e}")

    def _log_interpolation_examples(self):
        """Log interpolation examples between latent points."""
        try:
            # Get two random latent points
            z1 = torch.randn(1, self.latent_dim).to(self.device)
            z2 = torch.randn(1, self.latent_dim).to(self.device)
            
            # Interpolate between them
            interpolated = self.interpolate(z1, z2, num_steps=8)
            
            # Create interpolation grid
            fig, axes = plt.subplots(2, 8, figsize=(24, 6))
            
            for i in range(8):
                img = interpolated[i].permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                axes[0, i].imshow(img)
                axes[0, i].set_title(f"Step {i+1}", fontsize=10)
                axes[0, i].axis('off')
                
                # Show the latent interpolation
                t = i / 7.0
                z_interp = (1 - t) * z1 + t * z2
                z_interp_np = z_interp.cpu().numpy().flatten()
                axes[1, i].bar(range(len(z_interp_np)), z_interp_np)
                axes[1, i].set_title(f"Latent {i+1}", fontsize=8)
                axes[1, i].set_ylim([-3, 3])
            
            plt.tight_layout()
            wandb.log({"latent_interpolation": wandb.Image(fig)})
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Interpolation logging failed: {e}")
            
    def _log_training_statistics(self):
        """Log training statistics and model information."""
        try:
            # Model parameters count
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            wandb.log({
                "model_total_parameters": total_params,
                "model_trainable_parameters": trainable_params,
                "model_latent_dimension": self.latent_dim,
                "model_input_dimension": self.input_dim,
            })
            
            # Log model architecture info
            # Use values set during _setup_model; fall back on input channels
            if not hasattr(self, "_encoder_type") or not hasattr(self, "_decoder_type"):
                in_ch = int(self.input_dim[0]) if isinstance(self.input_dim, (list, tuple)) else 3
                self._encoder_type = "MNIST_ResNet" if in_ch == 1 else "RGBEncoder"
                self._decoder_type = "MNIST_ResNet" if in_ch == 1 else "RGBDecoder"
            wandb.log({
                "model_architecture": {
                    "encoder_type": self._encoder_type,
                    "decoder_type": self._decoder_type,
                    "latent_dim": self.latent_dim,
                    "input_dim": self.input_dim,
                }
            })
            
        except Exception as e:
            print(f"⚠️ Training statistics logging failed: {e}")
            
    def _log_periodic_metrics(self):
        """Log metrics periodically during training."""
        try:
            self.model.eval()
            with torch.no_grad():
                # Get a small batch for periodic logging
                sample_batch = self.test_data[:8].to(self.device)
                
                # Encode directly
                encoder_output = self.model.encoder(sample_batch)
                z = encoder_output["embedding"]
                
                # Decode directly
                decoder_output = self.model.decoder(z)
                reconstructions = decoder_output["reconstruction"]
                
                # Log basic reconstructions
                self._log_reconstructions(sample_batch, reconstructions, "periodic")
                
                # Log basic latent space
                z_np = z.cpu().numpy()
                
                # Simple 2D scatter plot
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(z_np[:, 0], z_np[:, 1], alpha=0.7, s=50)
                ax.set_xlabel("Latent dim 1")
                ax.set_ylabel("Latent dim 2")
                ax.set_title("Latent Space (Periodic)")
                ax.grid(True, alpha=0.3)
                wandb.log({"latent_space_periodic": wandb.Image(fig)})
                plt.close()
                
                # Log latent statistics
                wandb.log({
                    "latent_mean": float(z_np.mean()),
                    "latent_std": float(z_np.std()),
                    "latent_min": float(z_np.min()),
                    "latent_max": float(z_np.max()),
                })
                
        except Exception as e:
            print(f"⚠️ Periodic logging failed: {e}")
             
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'n_lf': self.n_lf,
                'eps_lf': self.eps_lf,
                'beta_zero': self.beta_zero,
                'temperature': self.temperature,
                'regularization': self.regularization,
            },
            'training_history': self.training_history,
        }, path)
        print(f"✅ Model saved to {path}")

    def export_metric(self, path: str):
        """Export centroids and M matrices to a portable torch file."""
        try:
            if not (hasattr(self.model, 'centroids_tens') and hasattr(self.model, 'M_tens')):
                print("⚠️ RHVAE model does not expose centroids_tens/M_tens; skipping export")
                return
            centroids = self.model.centroids_tens.detach().cpu() if self.model.centroids_tens is not None else None
            M = self.model.M_tens.detach().cpu() if self.model.M_tens is not None else None
            try:
                self._trace_metric_statistics(
                    centroids=centroids,
                    matrices=M,
                    stage="export_metric"
                )
            except Exception as trace_exc:
                print(f"[METRIC TRACE] Export trace failed: {trace_exc}")
            payload = {
                'centroids': centroids,
                'M_tens': M,
                'temperature': float(self.temperature),
                'regularization': float(self.regularization),
                'latent_dim': int(self.latent_dim),
            }
            torch.save(payload, path)
            print(f"✅ Metric exported to {path}")
            if wandb.run is not None:
                wandb.save(path)
        except Exception as e:
            print(f"⚠️ Metric export failed: {e}")
        
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        print(f"✅ Model loaded from {path}")
        
    def sample(self, num_samples: int = 16) -> torch.Tensor:
        """Generate samples from the trained model."""
        self.model.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            
            # Decode
            decoder_output = self.model.decoder(z)
            samples = decoder_output["reconstruction"]
            
        return samples
        
    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, num_steps: int = 8) -> torch.Tensor:
        """Interpolate between two latent points."""
        self.model.eval()
        with torch.no_grad():
            # Linear interpolation
            alphas = torch.linspace(0, 1, num_steps).to(self.device)
            interpolated = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                decoder_output = self.model.decoder(z_interp)
                interpolated.append(decoder_output["reconstruction"])
                
        return torch.stack(interpolated) 

    def _maybe_subsample_metric_centroids(self):
        """Reduce number of centroids/M matrices to mitigate over-smoothing.
        Uses either farthest point sampling (fps) or k-means in latent space on centroids.
        """
        if self.max_centroids is None:
            return
        if not hasattr(self.model, 'centroids_tens') or not hasattr(self.model, 'M_tens'):
            return
        C = self.model.centroids_tens
        M = self.model.M_tens
        if C is None or M is None:
            return
        K = C.shape[0]
        if K <= self.max_centroids:
            return
        device = C.device
        keep = None
        if self.centroid_subsample_method == 'kmeans':
            try:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=self.max_centroids, init='k-means++', n_init=10, random_state=42)
                km.fit(C.detach().cpu().numpy())
                centers = torch.tensor(km.cluster_centers_, device=device, dtype=C.dtype)
                dists = torch.cdist(centers, C)
                keep = torch.argmin(dists, dim=1)
            except Exception as e:
                print(f"⚠️ k-means subsampling failed ({e}); falling back to FPS")
                self.centroid_subsample_method = 'fps'
        if keep is None:
            # Farthest Point Sampling on centroids
            with torch.no_grad():
                chosen = []
                start_idx = torch.randint(0, K, (1,), device=device).item()
                chosen.append(start_idx)
                min_d = torch.cdist(C[start_idx:start_idx+1], C).squeeze(0)
                for _ in range(1, self.max_centroids):
                    far_idx = torch.argmax(min_d).item()
                    chosen.append(far_idx)
                    new_d = torch.cdist(C[far_idx:far_idx+1], C).squeeze(0)
                    min_d = torch.minimum(min_d, new_d)
                keep = torch.tensor(chosen, device=device)
        # Apply selection
        self.model.centroids_tens = C[keep]
        self.model.M_tens = M[keep]
        print(f"🔧 Subsampled centroids/M: {K} -> {self.model.centroids_tens.shape[0]} ({self.centroid_subsample_method})")
