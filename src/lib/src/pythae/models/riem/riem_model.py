import os
from typing import Optional

import sys
from tqdm import tqdm
sys.path.append("......")
sys.path.append(".....")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.src.pythae.data.datasets import BaseDataset
from lib.src.pythae.models.base.base_utils import ModelOutput
from lib.src.pythae.models.nn import BaseDecoder, BaseEncoder
from lib.src.pythae.models.normalizing_flows import IAF, IAFConfig
from lib.src.pythae.models.vae import VAE, VAEConfig
from lib.src.pythae.models.lvae_iaf.lvae_iaf_config import LVAE_IAF_Config
from diffusion.stable_diffusion.latent_diffusion import MyLatentDiffusion
from diffusion.stable_diffusion.sampler.ddim import DDIMSampler
from geometric_perspective_on_vaes.sampling import hmc_sampling

# ------------------------------------------------------
# Helper functions for metric integration
# ------------------------------------------------------
def compute_metric(z, centroids, M_tens, lbd, temperature):
    """
    Compute the inverse metric G_inv(z) at the latent position z.
    Args:
        z (torch.Tensor): Latent variable of shape [B, D].
        centroids (torch.Tensor): Tensor of centroids [N, D].
        M_tens (torch.Tensor): Local metric matrices [N, D, D].
        lbd (float): Regularization constant.
        temperature (float): Temperature parameter.
    Returns:
        torch.Tensor: Inverse metric at z, of shape [B, D, D].
    """

    diff_norm = torch.norm(z.unsqueeze(1) - centroids.unsqueeze(0), dim=-1) ** 2
    weights = torch.exp(- diff_norm / (temperature ** 2))
    weighted_matrices = (M_tens.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
    D = z.shape[-1]
    eye = torch.eye(D, device=z.device).unsqueeze(0).expand_as(weighted_matrices)
    G_inv_z = weighted_matrices + lbd * eye
    return G_inv_z

def compute_G(z, centroids, M_tens, lbd, temperature):
    """
    Compute the metric G(z) (the inverse of compute_metric) at the latent position z.
    """
    G_inv_z = compute_metric(z, centroids, M_tens, lbd, temperature)
    G_z = torch.inverse(G_inv_z)
    return G_z

def sample_with_metric(z, centroids, M_tens, lbd, temperature, beta_zero_sqrt):
    """
    Adjust the latent sample z based on the local Riemannian metric.
    Args:
        z (torch.Tensor): Initial latent sample [B, D].
        centroids, M_tens, lbd, temperature: Metric parameters.
        beta_zero_sqrt (float or torch.Tensor): Scaling factor.
    Returns:
        torch.Tensor: Metric-adjusted latent sample.
    """
    G_z = compute_G(z, centroids, M_tens, lbd, temperature)
    # Compute lower triangular matrix via Cholesky decomposition.
    L = torch.linalg.cholesky(G_z)
    gamma = torch.randn_like(z)
    adjusted = (L @ (gamma / beta_zero_sqrt).unsqueeze(-1)).squeeze(-1)
    return z + adjusted

# ------------------------------------------------------
# New function: Sample uniformly with respect to the Riemannian measure.
# ------------------------------------------------------
def _sample_uniform_riemann(self, n_samples, latent_dim, domain_low=None, domain_high=None, M=10.0):
    """
    Sample from a uniform distribution with respect to the Riemannian measure on a bounded domain.
    
    Args:
        n_samples (int): Number of samples to draw.
        latent_dim (int): Dimensionality of the latent space.
        domain_low (torch.Tensor, optional): Lower bounds for each dimension. Defaults to -1.
        domain_high (torch.Tensor, optional): Upper bounds for each dimension. Defaults to 1.
        M (float): An upper bound on the volume element in the domain.
    
    Returns:
        (torch.Tensor, None): Samples of shape [n_samples, latent_dim] and a dummy value.
    """
    if domain_low is None:
        domain_low = -torch.ones(latent_dim, device=self.device)
    if domain_high is None:
        domain_high = torch.ones(latent_dim, device=self.device)
    
    samples = []
    while len(samples) < n_samples:
        # Draw candidate uniformly from the hypercube.
        candidate = torch.rand(latent_dim, device=self.device) * (domain_high - domain_low) + domain_low
        candidate = candidate.unsqueeze(0)  # shape: [1, latent_dim]
        # Compute the volume element using the metric.
        G_candidate = compute_G(candidate, self.GM.centroids, self.GM.M_tens, self.GM.lbd, self.temperature)
        vol_candidate = torch.sqrt(torch.det(G_candidate))  # shape: [1]
        # Rejection sampling: accept with probability vol_candidate / M.
        u = torch.rand(1, device=self.device)
        if u < vol_candidate / M:
            samples.append(candidate.squeeze(0))
    samples = torch.stack(samples, dim=0)
    return samples, None

# ------------------------------------------------------
# LLDM_BIS Model with Metric Integration and Riemannian U sampling for z0
# ------------------------------------------------------
class RIEM(VAE):
    """
    RIEM model that integrates a Riemannian metric in its latent sampling and prior density estimation.
    In this variant the initial latent sample z₀ is drawn from a uniform distribution with respect to the Riemannian measure.
    """
    def __init__(
        self,
        model_config: LVAE_IAF_Config,
        encoder: Optional[BaseEncoder],
        decoder: Optional[BaseDecoder],
        pretrained_vae: Optional[VAE],  # Ensure to call retrieveG on the VAE beforehand!
        pretrained_ldm: Optional[MyLatentDiffusion],
        ddim_sampler: Optional[DDIMSampler],
        precomputed_zT_samples = None,
        GM = None,  # Metric module containing centroids, M_tens, and lbd
        temp: Optional[float] = 1.,
        verbose = False
    ):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "RIEM_IAF"
        self.input_dim = model_config.input_dim
        self.n_obs = model_config.n_obs_per_ind
        self.warmup = model_config.warmup
        self.context_dim = model_config.context_dim
        self.beta = model_config.beta
        self.flows = nn.ModuleList()

        self.linear_scheduling = self.model_config.linear_scheduling_steps
        self.prior = model_config.prior
        self.posterior = model_config.posterior

        self.pretrained_vae = pretrained_vae
        self.pretrained_ldm = pretrained_ldm
        self.zT_samples = precomputed_zT_samples
        self.ddim_sampler = ddim_sampler
        self.diff_t_steps = np.flip(ddim_sampler.time_steps)
        print('Diffusion time steps ', self.diff_t_steps)
        self.device = self.pretrained_ldm.device
        if verbose:
            print('Running on ', self.device)

        self.temperature = temp
        # Compute beta_zero_sqrt from beta for scaling in metric-based sampling.
        self.beta_zero_sqrt = torch.sqrt(torch.tensor(self.beta, device=self.device)) if self.beta is not None else 1.0

        # Flag to use uniform (Riemannian) sampling for z0.
        self.use_uniform_sample = True

        if self.posterior == "iaf":
            self.posterior_iaf_config = IAFConfig(
                input_dim=(model_config.latent_dim,),
                n_blocks=3,
                n_hidden_in_made=2,
                hidden_size=model_config.hidden_size,
                context_dim=model_config.context_dim,
                include_batch_norm=False,
            )
            self.posterior_iaf_flow = IAF(self.posterior_iaf_config).to(self.device)

        if verbose:
            print('Freezing pre-trained VAE and pre-trained LDM...')
        for p in self.pretrained_vae.parameters():
            p.requires_grad = False
        for p in self.pretrained_ldm.parameters():
            p.requires_grad = False
        if verbose:
            print('Freezing done.')
            print('Number of trainable parameters: {:.1e}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))
            print('Number of total parameters: {:.1e}'.format(sum(p.numel() for p in self.parameters())))

        self.GM = GM

        self.res_dict = {}
        for i in range(self.n_obs):
            self.res_dict[i] = {'rec_loss': 0, 'reg_loss': 0, 'count': 0}


    def _sample_uniform_riemann(self, n_samples, latent_dim, domain_low=None, domain_high=None, M=10.0):
        # This is the same as the standalone function above but attached as a method.
        if domain_low is None:
            domain_low = -torch.ones(latent_dim, device=self.device)
        if domain_high is None:
            domain_high = torch.ones(latent_dim, device=self.device)
        
        samples = []
        while len(samples) < n_samples:
            candidate = torch.rand(latent_dim, device=self.device) * (domain_high - domain_low) + domain_low
            candidate = candidate.unsqueeze(0)
            G_candidate = compute_G(candidate, self.GM.centroids, self.GM.M_tens, self.GM.lbd, self.temperature)
            vol_candidate = torch.sqrt(torch.det(G_candidate))
            u = torch.rand(1, device=self.device)
            if u < vol_candidate / M:
                samples.append(candidate.squeeze(0))
        samples = torch.stack(samples, dim=0)
        return samples, None

    def forward(self, inputs: BaseDataset, vi_index, **kwargs):
        device = self.device
        # Expecting inputs to have a "data" key that is already [B, n_obs, C, H, W]
        if isinstance(inputs, dict):
            x = inputs["data"].to(device)
        else:
            x = inputs.to(device)
        # If x has an extra leading dimension (for instance, [1, 32, 10, 1, 64, 64]) squeeze it.
        if x.shape[0] == 1 and x.shape[1] > 1:
            x = x.squeeze(0)
        # Now x should have shape [B, n_obs, C, H, W].
        # (Optionally, check here and print debug info)
        # print("Forward: x shape:", x.shape)

        epoch = kwargs.pop("epoch", 100)
        # Use provided sequence and pixel masks if available; otherwise assume ones.
        if hasattr(inputs, "seq_mask"):
            seq_mask = inputs["seq_mask"].to(device)
        else:
            seq_mask = torch.ones(x.shape[0], self.n_obs, device=device)
        if hasattr(inputs, "pix_mask"):
            pix_mask = inputs["pix_mask"].to(device)
        else:
            pix_mask = torch.ones_like(x)
        # If necessary, apply pixel and sequence masks (this part is left as in your original code).
        if len(x.shape) != 3:
            x = x * pix_mask * seq_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        batch_size = x.shape[0]

        # --- Warmup Branch ---
        if epoch < self.warmup:
            encoder_output = self.encoder(x)
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            std = torch.exp(0.5 * log_var)
            # Sample normally (or use uniform Riemannian if flagged)
            z, _ = self._sample_gauss(mu, std)
            if self.GM is not None:
                z = sample_with_metric(z, self.GM.centroids, self.GM.M_tens, self.GM.lbd,
                                    self.temperature, self.beta_zero_sqrt)
            z0 = z
            log_abs_det_jac_posterior = 0
            if self.posterior == "iaf":
                if self.posterior_iaf_config.context_dim is not None:
                    try:
                        h = encoder_output.context
                    except Exception as e:
                        raise AttributeError(f"Cannot get context from encoder outputs: {e}")
                    flow_output = self.posterior_iaf_flow.inverse(z, h=h)
                else:
                    flow_output = self.posterior_iaf_flow.inverse(z)
                z = flow_output.out
                log_abs_det_jac_posterior += flow_output.log_abs_det_jac
            z_seq = z
            recon_x = self.decoder(z_seq)["reconstruction"]
            # IMPORTANT: Do not reshape x here. Both x and recon_x must align.
            loss, recon_loss, kld = self.vae_loss_function(
                recon_x=recon_x,
                x=x,  # x is already [B, n_obs, C, H, W]
                mu=mu,
                log_var=log_var,
                z0=z0,
                zk=z,
                log_abs_det_jac_posterior=log_abs_det_jac_posterior,
                epoch=epoch,
                seq_mask=seq_mask,
                pix_mask=pix_mask
            )
        else:
            # --- Non-Warmup Branch ---
            if vi_index is None:
                # Select observation index based on seq_mask if available.
                if seq_mask[0].sum() < self.n_obs:
                    probs = seq_mask[0].cpu().numpy().flatten()
                    probs /= probs.sum()
                    vi_index = int(np.random.choice(np.arange(self.n_obs), p=probs))
                else:
                    vi_index = np.random.randint(0, self.n_obs)
            encoder_output = self.encoder(x[:, vi_index])
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            std = torch.exp(0.5 * log_var)
            if self.use_uniform_sample:
                z, _ = self._sample_uniform_riemann(batch_size, self.model_config.latent_dim)
            else:
                z, _ = self._sample_gauss(mu, std)
            if self.GM is not None:
                z = sample_with_metric(z, self.GM.centroids, self.GM.M_tens, self.GM.lbd,
                                    self.temperature, self.beta_zero_sqrt)
            z0 = z
            log_abs_det_jac_posterior = 0
            if self.posterior == "iaf":
                if self.posterior_iaf_config.context_dim is not None:
                    try:
                        h = encoder_output.context
                    except Exception as e:
                        raise AttributeError(f"Cannot get context from encoder outputs: {e}")
                    flow_output = self.posterior_iaf_flow.inverse(z, h=h)
                else:
                    flow_output = self.posterior_iaf_flow.inverse(z)
                z = flow_output.out
                log_abs_det_jac_posterior += flow_output.log_abs_det_jac
            # Build latent sequence using diffusion steps (as before).
            z_seq_list = []
            z_rev = z
            for i in range(vi_index - 1, -1, -1):
                t1 = self.diff_t_steps[(i+1)*np.ones(batch_size).astype(int)]
                t2 = self.diff_t_steps[i*np.ones(batch_size).astype(int)]
                z_rev = self.pretrained_ldm.sequential_diffusion(x=z_rev, t1=t1, t2=t2).to(self.pretrained_ldm.device).float()
                z_seq_list.append(z_rev)
            z_seq_list.reverse()
            z_seq_list.append(z.to(self.pretrained_ldm.device))
            z_for = z
            for i in range(vi_index, self.n_obs - 1):
                t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float()
                z_for = z_for.reshape(batch_size, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w).float().to(self.pretrained_ldm.device)
                noise_pred = self.pretrained_ldm(z_for, t)
                z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(
                    e_t=noise_pred,
                    index=self.n_obs - 1 - i,
                    x=z_for,
                    temperature=self.temperature,
                    repeat_noise=False
                )
                z_for = z_for.reshape(batch_size, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w).to(self.pretrained_ldm.device)
                z_seq_list.append(z_for)
            z_seq = torch.cat(z_seq_list, dim=-1)
            recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]
            z_seq = z_seq.reshape(x.shape[0], self.n_obs, self.model_config.latent_dim)
            loss, recon_loss, kld = self.loss_function(
                recon_x=recon_x,
                x=x,  # x remains [B, n_obs, C, H, W]
                mu=mu,
                log_var=log_var,
                z0=z0,
                z_seq=z_seq,
                vi_index=vi_index,
                z_vi_index=z,
                log_abs_det_jac_posterior=log_abs_det_jac_posterior,
                epoch=epoch,
                seq_mask=seq_mask,
                pix_mask=pix_mask
            )
            # Update internal result dictionary.
            self.res_dict[vi_index]['rec_loss'] += recon_loss.item()
            self.res_dict[vi_index]['reg_loss'] += kld.item()
            self.res_dict[vi_index]['count'] += 1

        # Create the output ModelOutput exactly as in the original lldm_model.
        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x.reshape_as(x),  # Ensure recon_x is reshaped exactly as input x.
            z=z,
            z_seq=z_seq,
            x=x,
            log_abs_det_jac_posterior=log_abs_det_jac_posterior
        )
        return output

    # (The remainder of the methods—vae_loss_function, _vae_loss_function, _vae_iaf_loss_function,
    # _vamp_loss_function, loss_function, _log_p_z, log_p_j_hat, reconstruct, oversample, predict, and get_nll
    # remain unchanged.)
    
    def _sample_gauss(self, mu, std):
        # Original reparameterization trick (kept here for compatibility if needed)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def vae_loss_function(self, recon_x, x, mu, log_var, z0, epoch, zk=None, 
                         log_abs_det_jac_posterior=None, seq_mask=None, pix_mask=None):
        """
        Unified loss function that selects one of the variants based on the model's prior/posterior.
        """
        if self.prior == "standard" and self.posterior == "gaussian":
            loss, recon_loss, kld = self._vae_loss_function(recon_x, x, mu, log_var, z0, seq_mask, pix_mask)
        elif self.prior == "vamp" and self.posterior == "gaussian":
            loss, recon_loss, kld = self._vamp_loss_function(recon_x, x, mu, log_var, z0, epoch, seq_mask, pix_mask)
        elif self.posterior == "iaf":
            loss, recon_loss, kld = self._vae_iaf_loss_function(recon_x, x, mu, log_var, z0, zk, 
                                                               log_abs_det_jac_posterior, epoch, seq_mask, pix_mask)
        else:
            raise ValueError(f"Unknown prior/posterior combination: {self.prior}/{self.posterior}")
        return loss, recon_loss, kld

    def _vae_loss_function(self, recon_x, x, mu, log_var, z, seq_mask=None, pix_mask=None):
        """
        Compute VAE loss with proper tensor reshaping.
        """
        batch_size = x.shape[0]
        
        # Reshape input and reconstruction to match dimensions
        x_flat = x.reshape(batch_size, -1)
        recon_x_flat = recon_x.reshape(batch_size, -1)
        
        if pix_mask is not None:
            pix_mask_flat = pix_mask.reshape(batch_size, -1)
        else:
            pix_mask_flat = torch.ones_like(x_flat)
            
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = 0.5 * (F.mse_loss(recon_x_flat, x_flat, reduction="none") * pix_mask_flat).sum(dim=-1)
        elif self.model_config.reconstruction_loss == "bce":
            recon_loss = (F.binary_cross_entropy(recon_x_flat, x_flat, reduction="none") * pix_mask_flat).sum(dim=-1)
        else:
            raise ValueError(f"Unknown reconstruction loss type: {self.model_config.reconstruction_loss}")
            
        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        
        if seq_mask is not None:
            loss = ((recon_loss + KLD) * seq_mask).mean()
        else:
            loss = (recon_loss + KLD).mean()
            
        return loss, recon_loss.mean(), KLD.mean()

    def _vae_iaf_loss_function(self, recon_x, x, mu, log_var, z0, zk, log_abs_det_jac, 
                               epoch, seq_mask=None, pix_mask=None):
        """
        Loss function for the IAF posterior case.
        Computes the reconstruction loss and adjusts the KL divergence using the flow.
        """
        # Compute dynamic number of observations per sample.
        n_obs_dyn = recon_x.shape[0] // x.shape[0]
        recon_x_flat = recon_x.reshape(x.shape[0] * n_obs_dyn, -1)
        x_flat = x.repeat_interleave(n_obs_dyn, dim=0).reshape(x.shape[0] * n_obs_dyn, -1)
        
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = 0.5 * (F.mse_loss(recon_x_flat, x_flat, reduction="none") *
                                pix_mask.repeat_interleave(n_obs_dyn, dim=0).reshape(x.shape[0] * n_obs_dyn, -1)
                               ).sum(dim=-1)
        elif self.model_config.reconstruction_loss == "bce":
            recon_loss = (F.binary_cross_entropy(recon_x_flat, x_flat, reduction="none") *
                          pix_mask.repeat_interleave(n_obs_dyn, dim=0).reshape(x.shape[0] * n_obs_dyn, -1)
                         ).sum(dim=-1)
        else:
            raise ValueError("Unknown reconstruction loss type")
        
        log_prob_z0 = (-0.5 * (log_var + torch.pow(z0 - mu, 2) / torch.exp(log_var))).sum(dim=1)
        log_prob_zk = self._log_p_z(zk)
        KLD = log_prob_z0 - log_prob_zk - log_abs_det_jac
        # Apply sequence mask if available.
        if seq_mask is not None:
            seq_mask_flat = seq_mask.reshape(x.shape[0], -1)
            loss = ((recon_loss + KLD) * seq_mask_flat).mean(dim=0)
        else:
            loss = (recon_loss + KLD).mean()
        return loss, recon_loss.mean(), KLD.mean()

    def _vamp_loss_function(self, recon_x, x, mu, log_var, z, epoch, seq_mask=None, pix_mask=None):
        """
        Loss function for the VAMP prior case.
        """
        # Compute dynamic number of observations per sample.
        n_obs_dyn = recon_x.shape[0] // x.shape[0]
        recon_x_flat = recon_x.reshape(x.shape[0] * n_obs_dyn, -1)
        x_flat = x.repeat_interleave(n_obs_dyn, dim=0).reshape(x.shape[0] * n_obs_dyn, -1)
        
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = 0.5 * (F.mse_loss(recon_x_flat, x_flat, reduction="none") *
                                pix_mask.repeat_interleave(n_obs_dyn, dim=0).reshape(x.shape[0] * n_obs_dyn, -1)
                               ).sum(dim=-1)
        elif self.model_config.reconstruction_loss == "bce":
            recon_loss = (F.binary_cross_entropy(recon_x_flat, x_flat, reduction="none") *
                          pix_mask.repeat_interleave(n_obs_dyn, dim=0).reshape(x.shape[0] * n_obs_dyn, -1)
                         ).sum(dim=-1)
        else:
            raise ValueError("Unknown reconstruction loss type")
        
        log_p_z = self._log_p_z(z)
        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=1)
        KLD = -(log_p_z - log_q_z)
        
        beta = 1.0 * epoch / self.linear_scheduling if self.linear_scheduling > 0 else 1.0
        if beta > self.beta or not self.training:
            beta = self.beta
        # Apply sequence mask if provided.
        if seq_mask is not None:
            seq_mask_flat = seq_mask.reshape(x.shape[0], -1)
            loss = ((recon_loss + beta * KLD) * seq_mask_flat).mean(dim=0)
        else:
            loss = (recon_loss + beta * KLD).mean()
        return loss, recon_loss.mean(), KLD.mean()


    def loss_function(self, recon_x, x, mu, log_var, z_0_vi_index, z_seq, vi_index, 
                  z_vi_index, log_abs_det_jac_posterior, epoch, seq_mask=None, pix_mask=None):
        """
        Compute the overall loss as the sum of reconstruction loss and a KL divergence term,
        using the collapsed input target.
        
        Expected shapes:
        - x: [B, n_obs, C, H, W]   e.g. [32, 10, 1, 64, 64]
        - recon_x: [B * n_obs, C, H, W]   e.g. [320, 1, 64, 64]
        
        Args:
            recon_x (torch.Tensor): The decoder output, shape [B*n_obs, C, H, W].
            x (torch.Tensor): The original input tensor, shape [B, n_obs, C, H, W].
            mu, log_var (torch.Tensor): Encoder output (latent mean and log variance), shape [B, latent_dim].
            z_0_vi_index (torch.Tensor): The initial latent sample.
            z_seq (torch.Tensor): The full latent sequence (expected to be reshaped to [B, n_obs, latent_dim]).
            vi_index (int): The selected observation index.
            z_vi_index (torch.Tensor): The latent sample at a particular index.
            log_abs_det_jac_posterior (float or torch.Tensor): Log absolute determinant (if using IAF), scalar.
            epoch (int): Current epoch (for dynamic beta scheduling).
            seq_mask (torch.Tensor, optional): Mask over sequence elements, shape [B, n_obs] if provided.
            pix_mask (torch.Tensor, optional): Pixel mask with shape matching x (i.e. [B, n_obs, C, H, W]).
        
        Returns:
            tuple: (total_loss, mean_recon_loss, mean_KL_loss)
        """
        # Ensure no NaNs.
        assert not torch.isnan(x).any(), "Input x has NaNs."

        if x.shape[0] == 1:
            x = x.squeeze(0) 

        B = x.shape[0]
        n_obs_expected = x.shape[1]  # Expected number of observations (e.g., 10)
        
        # Collapse x along its temporal dimension.
        # This converts x from [B, n_obs, C, H, W] -> [B*n_obs, C, H, W]
        x_target = x.reshape(B * n_obs_expected, x.shape[-3], x.shape[-2], x.shape[-1])
        x_flat = x_target.reshape(B * n_obs_expected, -1)
        
        # For recon_x: assume it is produced with shape [B * n_obs_out, C, H, W].
        n_obs_out = recon_x.shape[0] // B
        if n_obs_out != n_obs_expected:
            print(f"Warning: Expected {n_obs_expected} observations from x, but decoder produced {n_obs_out}.")
        recon_x_flat = recon_x.reshape(B * n_obs_out, -1)
        
        
        # Process pixel mask similarly: assume pix_mask is [B, n_obs, C, H, W].
        if pix_mask is not None:
            pix_mask_target = pix_mask.reshape(B * n_obs_expected, pix_mask.shape[-3], pix_mask.shape[-2], pix_mask.shape[-1])
            pix_mask_flat = pix_mask_target.reshape(B * n_obs_expected, -1)
        else:
            pix_mask_flat = torch.ones_like(x_flat)
        
        prod_dim = np.prod(self.input_dim)  # Expected product, e.g. 1*64*64 = 4096
        
        # Compute reconstruction loss.
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = 0.5 * (F.mse_loss(recon_x_flat, x_flat, reduction="none") * pix_mask_flat).sum(dim=-1)
        elif self.model_config.reconstruction_loss == "bce":
            recon_loss = (F.binary_cross_entropy(recon_x_flat, x_flat, reduction="none") * pix_mask_flat).sum(dim=-1)
        else:
            raise ValueError("Unknown reconstruction loss type")
        
        # Compute the standard KL divergence (one value per sample).
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)  # shape: [B]
        
        # Refined KL term using the first latent sample from z_seq.
        # Assuming z_seq is [B, n_obs, latent_dim] we take the first observation.
        z0_state = z_seq[:, 0]  
        log_prob_z = (-0.5 * (log_var + torch.pow(z_0_vi_index - mu, 2) / (torch.exp(log_var) + 1e-7))).sum(dim=1) - log_abs_det_jac_posterior
        if vi_index == 0 or vi_index == self.n_obs - 1:
            log_prior = self.log_p_j_hat(j=vi_index, z=z_vi_index).to(z_vi_index.device)
            KL_term = log_prob_z - log_prior
            KL_term = torch.clamp(KL_term, min=-2, max=500)
        else:
            KL_term = torch.zeros_like(log_prob_z)
            KL_term = torch.clamp(KL_term, min=-2, max=500)
        
        # Dynamic KL beta scheduling.
        if self.linear_scheduling > 0:
            beta_val = self.beta * epoch / self.linear_scheduling if epoch > 0 else self.beta
            if beta_val > self.beta or not self.training:
                beta_val = self.beta
        else:
            beta_val = self.beta

        # Combine reconstruction loss (which is per observation, shape: [B*n_obs]) with KL divergence (shape: [B]).
        # To combine them, we first reshape recon_loss as [B, n_obs] and take mean across the temporal dimension.
        recon_loss_per_sample = recon_loss.reshape(B, -1).mean(dim=1)
        total_loss = (recon_loss_per_sample + beta_val * KL_term).mean()

        return total_loss, recon_loss.mean(), KL_term.mean()



    def _log_p_z(self, z):
        if self.prior == "standard":
            log_p_z = (-0.5 * torch.pow(z, 2)).sum(dim=1)
        elif self.prior == "vamp":
            C = self.vamp_number_components
            x = self.pseudo_inputs(self.idle_input.to(z.device)).reshape((C,) + self.model_config.input_dim)
            encoder_output = self.encoder(x)
            prior_mu, prior_log_var = encoder_output.embedding, encoder_output.log_covariance
            z_expand = z.unsqueeze(1)
            prior_mu = prior_mu.unsqueeze(0)
            prior_log_var = prior_log_var.unsqueeze(0)
            log_p_z = (torch.sum(-0.5 * (prior_log_var + (z_expand - prior_mu) ** 2 / torch.exp(prior_log_var)), dim=2)
                    - torch.log(torch.tensor(C, dtype=torch.float))).to(z.device)
            log_p_z = torch.logsumexp(log_p_z, dim=1)
        return log_p_z

    def log_p_j_hat(self, j, z):
        """
        Prior on z_j incorporating the metric if available.
        """
        assert j >= 0 and j < self.n_obs
        if j == 0:
            return (-0.5 * torch.pow(z, 2)).sum(dim=1)
        else:
            if self.GM is not None:
                G_z = compute_G(z, self.GM.centroids, self.GM.M_tens, self.GM.lbd, self.temperature)
                volume_term = 0.5 * torch.logdet(G_z)
                base_log_prior = (-0.5 * torch.pow(z, 2)).sum(dim=1)
                return base_log_prior + volume_term
            else:
                return self.pretrained_vae.log_pi(z)

    def reconstruct(self, input, vi_index, z_vi_index=None):
        device = self.device
        # Check if input is a dictionary or a tensor.
        if isinstance(input, dict):
            x = input["data"].to(device)
        else:
            x = input.to(device)
        
        if x.shape[0] == 1:
            x = x.squeeze(0) 

        # If the tensor does not have the correct number of dimensions, adjust as needed.
        # For example, if x is 4D (B, C, H, W), no need to unsqueeze;
        # but if it is 2D or a different shape, you might need to adjust.
        #if len(x.shape) == 4 or len(x.shape) == 2:
        #    pass
        #else:
        #    x = x.unsqueeze(0)
        
        if hasattr(input, 'seq_mask'):
            seq_mask = input['seq_mask'].to(device)
        else:
            seq_mask = torch.ones(x.shape[0], self.n_obs).to(device)
        
        if hasattr(input, 'pix_mask'):
            pix_mask = input['pix_mask'].to(device)
        else:
            pix_mask = torch.ones_like(x)
        
        # If x is 3D, replace NaNs (or as needed)
        if len(x.shape) == 3:
            x = torch.nan_to_num(x)
        else:
            x = x * pix_mask * seq_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        batch_size = x.shape[0]
        encoder_output = self.encoder(x[:, vi_index])
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        if self.use_uniform_sample:
            z, _ = self._sample_uniform_riemann(batch_size, self.model_config.latent_dim)
        else:
            z, _ = self._sample_gauss(mu, std)
        if self.GM is not None:
            z = sample_with_metric(z, self.GM.centroids, self.GM.M_tens, self.GM.lbd, self.temperature, self.beta_zero_sqrt)
        z0 = z
        log_abs_det_jac_posterior = 0
        if self.posterior == 'iaf':
            if self.posterior_iaf_config.context_dim is not None:
                try:
                    h = encoder_output.context
                except AttributeError as e:
                    raise AttributeError("Cannot get context from encoder outputs. Exception: {}".format(e))
                flow_output = self.posterior_iaf_flow.inverse(z, h=h)
            else:
                flow_output = self.posterior_iaf_flow.inverse(z)
            z = flow_output.out
            log_abs_det_jac_posterior += flow_output.log_abs_det_jac
        z_vi_index = z if z_vi_index is None else z_vi_index
        z_seq = []
        z_rev = z_vi_index
        for i in range(vi_index - 1, -1, -1):
            t1 = self.diff_t_steps[(i+1)*np.ones(batch_size).astype(int)]
            t2 = self.diff_t_steps[i*np.ones(batch_size).astype(int)]
            z_rev = self.pretrained_ldm.sequential_diffusion(x=z_rev, t1=t1, t2=t2).to(self.pretrained_ldm.device).float()
            z_seq.append(z_rev)
        z_seq.reverse()
        z_seq.append(z_vi_index.to(self.pretrained_ldm.device))
        z_for = z_vi_index
        for i in range(vi_index, self.n_obs - 1):
            t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float()
            z_for = z_for.reshape(batch_size, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w).float().to(self.pretrained_ldm.device)
            noise_pred = self.pretrained_ldm(z_for, t)
            z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(e_t=noise_pred,
                                                                index=self.n_obs - 1 - i,
                                                                x=z_for,
                                                                temperature=self.temperature,
                                                                repeat_noise=False)
            z_for = z_for.reshape(batch_size, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w).to(self.pretrained_ldm.device)
            z_seq.append(z_for)
        z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.model_config.latent_dim)
        recon_x = self.decoder(z_seq)["reconstruction"].cpu().detach()
        return z_seq.cpu().detach(), recon_x


    def oversample(self, x, vi_index=0, z_vi_index=None, sampler=None, num_supp_steps=None, verbose=False):
        """
        Performs oversampling of the latent code via additional diffusion steps.
        This method generates an extended latent sequence by propagating the latent variable
        backwards (noising) and then forwards (denoising) using the pretrained LDM.
        
        Args:
            x (dict): A dictionary that contains the key "data" with input tensor.
            vi_index (int): The observation index to condition on.
            z_vi_index (torch.Tensor, optional): Optionally provided latent variable.
            sampler (DDIMSampler, optional): A DDIM sampler instance; if None, one is constructed.
            num_supp_steps (int, optional): Additional support steps to add to the diffusion timeline.
            verbose (bool): Whether to print debug information.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The latent sequence and its reconstruction.
        """
        device = self.device
        x = x["data"].to(device)
        x = x.unsqueeze(0) if len(x.shape) == 4 else x
        batch_size = x.shape[0]

        # Obtain encoding from the selected observation.
        encoder_output = self.encoder(x[:, vi_index])
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        if self.GM is not None:
            z = sample_with_metric(z, self.GM.centroids, self.GM.M_tens, self.GM.lbd,
                                   self.temperature, self.beta_zero_sqrt)
        z_0_vi_index = z
        log_abs_det_jac_posterior = 0
        if self.posterior == 'iaf':
            if self.posterior_iaf_config.context_dim is not None:
                try:
                    h = encoder_output.context
                except AttributeError as e:
                    raise AttributeError("Cannot get context from encoder outputs. Exception: {}".format(e))
                flow_output = self.posterior_iaf_flow.inverse(z, h=h)
            else:
                flow_output = self.posterior_iaf_flow.inverse(z)
            z = flow_output.out
            log_abs_det_jac_posterior += flow_output.log_abs_det_jac
        z_vi_index = z

        # Build DDIM sampler if not provided.
        if sampler is None:
            assert num_supp_steps is not None, "num_supp_steps must be provided if sampler is None."
            sampler = DDIMSampler(self.pretrained_ldm, n_steps=self.n_obs - 1 + num_supp_steps, ddim_eta=1)
        self.oversampling_diff_t_steps = np.flip(np.sort(sampler.time_steps))
        if verbose:
            print('Updating the DDIM sampler...')
            print('Previous diffusion timeline: ', self.diff_t_steps)
            print('New diffusion timeline: ', self.oversampling_diff_t_steps)
        matching_index = np.argmin(np.abs(self.oversampling_diff_t_steps - self.diff_t_steps[vi_index]))

        # Generate latent sequence: propagate backwards.
        z_seq = []
        z_rev = z_vi_index
        for i in range(matching_index - 1, -1, -1):
            t1 = self.oversampling_diff_t_steps[(i+1) * np.ones(batch_size).astype(int)]
            t2 = self.oversampling_diff_t_steps[i * np.ones(batch_size).astype(int)]
            z_rev = self.pretrained_ldm.sequential_diffusion(x=z_rev, t1=t1, t2=t2).to(self.pretrained_ldm.device).float()
            z_seq.append(z_rev)
        z_seq.reverse()
        z_seq.append(z_vi_index.to(self.pretrained_ldm.device))

        # Propagate forwards: denoising (backward diffusion).
        z_for = z_vi_index
        for i in range(matching_index, len(self.oversampling_diff_t_steps) - 1):
            t = torch.tensor(self.oversampling_diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float()
            z_for = z_for.reshape(batch_size, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w)\
                      .float().to(self.pretrained_ldm.device)
            noise_pred = self.pretrained_ldm(z_for, t)
            z_for, _ = sampler.get_x_prev_and_pred_x0(e_t=noise_pred,
                                                       index=len(self.oversampling_diff_t_steps) - 1 - i,
                                                       x=z_for,
                                                       temperature=self.temperature,
                                                       repeat_noise=False)
            z_for = z_for.reshape(batch_size, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w)\
                      .to(self.pretrained_ldm.device)
            z_seq.append(z_for)
        z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.latent_dim)
        recon_x = self.decoder(z_seq)["reconstruction"]
        return z_seq, recon_x

    def generate(self, train_data, num_gen_seq=1, vi_index=0, T_multiplier=0.5, batch_size=128, freeze=False, device='cpu', verbose=True):
        """
        Generate new samples (i.e. latent sequences and their reconstructions) 
        from the given train_data using HMC-based sampling over the latent manifold.
        
        This method uses a temporary VAE instance (final_vae) that is built using the
        same encoder and decoder as the current LLDM_BIS model. It then extracts the 
        latent embedding statistics (mu and log_var) via the final_vae.retrieveG method.
        Finally, it uses HMC sampling (hmc_sampling) to sample latent variables and 
        applies the diffusion steps to generate a complete latent sequence which is then decoded.
        
        Args:
            train_data (torch.Tensor): Input data tensor (or slice thereof) used for conditioning.
            num_gen_seq (int): Number of generation sequences to sample.
            vi_index (int): Index of the observation to condition on.
            T_multiplier (float): Multiplier used in retrieveG.
            batch_size (int): Batch size for processing (to avoid memory issues).
            freeze (bool): If True, the latent embedding of vi_index is "frozen" (repeated); otherwise new samples are drawn.
            device (str): Device to use for computation.
            verbose (bool): Whether to print progress.
            
        Returns:
            full_recon_x (torch.Tensor): Generated reconstructions.
            full_z_seq (torch.Tensor): Corresponding latent sequences.
        """
        # Move self to the specified device.
        self = self.to(device)
        # Create a temporary VAE using the same encoder and decoder.
        model_config = VAEConfig(
            input_dim=self.input_dim,
            latent_dim=self.model_config.latent_dim,
            uses_default_encoder=False,
            uses_default_decoder=False,
            reconstruction_loss='mse'
        )
        final_vae = VAE(model_config=model_config, encoder=self.encoder, decoder=self.decoder)
        # Select the observation data from train_data at vi_index.
        obs_data = train_data[:, vi_index]
        # Retrieve latent statistics from the final_vae.
        _, mu, log_var = final_vae.retrieveG(obs_data, verbose=verbose, T_multiplier=T_multiplier, device=device, addStdNorm=False)

        # Ensure batch_size does not exceed num_gen_seq.
        batch_size = num_gen_seq if num_gen_seq <= batch_size else batch_size
        all_z_vi = []

        final_vae = final_vae.to(device)
        mu = mu.to(device)

        if verbose:
            if freeze:
                print(f'Freezing the {vi_index}th observation...')
                print(f'Sampling 1 point on the {vi_index}th manifold...')
            else:
                print(f'Sampling {num_gen_seq} points on the {vi_index}th manifold...')

        if not freeze:
            # Use HMC sampling to obtain latent samples.
            for j in range(0, num_gen_seq // batch_size):
                z, p = hmc_sampling(final_vae, mu, n_samples=batch_size, mcmc_steps_nbr=100)
                all_z_vi.append(z)
            if num_gen_seq % batch_size != 0:
                z, p = hmc_sampling(final_vae, mu, n_samples=num_gen_seq % batch_size, mcmc_steps_nbr=100)
                all_z_vi.append(z)
        else:
            z, p = hmc_sampling(final_vae, mu, n_samples=1, mcmc_steps_nbr=100)
            all_z_vi = [z] * num_gen_seq

        full_recon_x, full_z_seq = [], []
        # Process batches to generate full latent sequences and reconstructions.
        for j in range(0, num_gen_seq // batch_size):
            if verbose:
                print(f'Processing batch {j+1}/{num_gen_seq // batch_size}')
            # When freeze is True, all latent samples are the same; otherwise take the j-th sample.
            z_vi_index = torch.cat(all_z_vi[j*batch_size:(j+1)*batch_size], dim=0) if freeze else all_z_vi[j]
            z_seq = []
            z_rev = z_vi_index
            # Propagate backwards (forward diffusion / adding noise)
            if verbose and vi_index > 0:
                print('Propagating in the past...')
            for i in range(vi_index - 1, -1, -1):
                t1 = self.diff_t_steps[(i+1) * np.ones(z_rev.shape[0]).astype(int)]
                t2 = self.diff_t_steps[i * np.ones(z_rev.shape[0]).astype(int)]
                z_rev = self.pretrained_ldm.sequential_diffusion(x=z_rev, t1=t1, t2=t2).to(self.pretrained_ldm.device).float()
                z_seq.append(z_rev)
            z_seq.reverse()
            z_seq.append(z_vi_index.to(self.pretrained_ldm.device))
            # Propagate forwards (backward diffusion / denoising)
            z_for = z_vi_index
            if verbose and vi_index < self.n_obs - 1:
                print('Propagating in the future...')
            for i in range(vi_index, self.n_obs - 1):
                t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float()
                z_for = z_for.reshape(z_vi_index.shape[0], self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w).float().to(self.pretrained_ldm.device)
                noise_pred = self.pretrained_ldm(z_for, t)
                z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(
                    e_t=noise_pred,
                    index=self.n_obs - 1 - i,
                    x=z_for,
                    temperature=self.temperature,
                    repeat_noise=False
                )
                z_for = z_for.reshape(z_vi_index.shape[0], self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w).to(self.pretrained_ldm.device)
                z_seq.append(z_for)
            z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.model_config.latent_dim)
            if verbose:
                print('Decoding generated latent sequences...')
            if len(self.input_dim) == 3:
                recon_x = self.decoder(z_seq)["reconstruction"].reshape(-1, self.n_obs, self.input_dim[0], self.input_dim[1], self.input_dim[2])
            elif len(self.input_dim) == 2:
                recon_x = self.decoder(z_seq)["reconstruction"].reshape(-1, self.n_obs, self.input_dim[1])
            z_seq = z_seq.reshape(-1, self.n_obs, self.model_config.latent_dim)
            full_recon_x.append(recon_x.detach().cpu())
            full_z_seq.append(z_seq.detach().cpu())

        if num_gen_seq % batch_size != 0:
            if verbose:
                print(f'Remainder batch of size {num_gen_seq % batch_size}')
            rem = num_gen_seq % batch_size
            z_vi_index = torch.cat(all_z_vi[-rem:], dim=0) if freeze else all_z_vi[-1]
            z_seq = []
            z_rev = z_vi_index
            if verbose and vi_index > 0:
                print('Propagating in the past for remainder batch...')
            for i in range(vi_index - 1, -1, -1):
                t1 = self.diff_t_steps[(i+1) * np.ones(z_rev.shape[0]).astype(int)]
                t2 = self.diff_t_steps[i * np.ones(z_rev.shape[0]).astype(int)]
                z_rev = self.pretrained_ldm.sequential_diffusion(x=z_rev, t1=t1, t2=t2).to(self.pretrained_ldm.device).float()
                z_seq.append(z_rev)
            z_seq.reverse()
            z_seq.append(z_vi_index.to(self.pretrained_ldm.device))
            z_for = z_vi_index
            if verbose and vi_index < self.n_obs - 1:
                print('Propagating in the future for remainder batch...')
            for i in range(vi_index, self.n_obs - 1):
                t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float()
                z_for = z_for.reshape(z_vi_index.shape[0], self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w).float().to(self.pretrained_ldm.device)
                noise_pred = self.pretrained_ldm(z_for, t)
                z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(
                    e_t=noise_pred,
                    index=self.n_obs - 1 - i,
                    x=z_for,
                    temperature=self.temperature,
                    repeat_noise=False
                )
                z_for = z_for.reshape(z_vi_index.shape[0], self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w).to(self.pretrained_ldm.device)
                z_seq.append(z_for)
            z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.model_config.latent_dim)
            if verbose:
                print('Decoding remainder batch...')
            if len(self.input_dim) == 3:
                recon_x = self.decoder(z_seq)["reconstruction"].reshape(-1, self.n_obs, self.input_dim[0], self.input_dim[1], self.input_dim[2])
            else:
                recon_x = self.decoder(z_seq)["reconstruction"].reshape(-1, self.n_obs, self.input_dim[1])
            z_seq = z_seq.reshape(-1, self.n_obs, self.model_config.latent_dim)
            full_recon_x.append(recon_x.detach().cpu())
            full_z_seq.append(z_seq.detach().cpu())

        full_recon_x = torch.cat(full_recon_x, dim=0)
        full_z_seq = torch.cat(full_z_seq, dim=0)

        return full_recon_x, full_z_seq

    def predict(self, x, vi_index, num_gen_seq=1, batch_size=100, device='cuda'):
        """
        Generate model predictions given input x using DDIM sampling.
        
        Args:
            x (torch.Tensor): The input tensor.
            vi_index (int): The observation index used by the encoder.
            num_gen_seq (int): Total number of generation sequences.
            batch_size (int): Maximum batch size.
            device (str): Device string (e.g. 'cuda').
        
        Returns:
            torch.Tensor: Generated predictions.
        """
        self = self.to(device)
        batch_size = num_gen_seq if num_gen_seq <= batch_size else batch_size
        x = x.unsqueeze(0) if len(x.shape) == 4 else x
        n_seq = x.shape[0]
        x_vi_index = x[:, vi_index].to(device)
        z_vi_index = self.encoder(x_vi_index.unsqueeze(0)).embedding  # Expect shape [1, latent_dim]
        # Duplicate the latent representation to match the desired number of sequences.
        z_vi_index = torch.cat(num_gen_seq * [z_vi_index], dim=1).reshape(n_seq, num_gen_seq, self.latent_dim)
        all_pred_x = []
        for j in range(0, num_gen_seq // batch_size):
            z_seq = []
            z_for = z_vi_index[:, j * batch_size: (j+1) * batch_size]
            for i in range(vi_index, self.n_obs - 1):
                t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float()
                z_for = z_for.reshape(-1, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w)\
                        .float().to(self.pretrained_ldm.device)
                noise_pred = self.pretrained_ldm(z_for, t)
                z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(e_t=noise_pred,
                                                                     index=self.n_obs - 1 - i,
                                                                     x=z_for,
                                                                     temperature=self.temperature,
                                                                     repeat_noise=False)
                z_for = z_for.reshape(-1, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w)\
                        .to(self.pretrained_ldm.device)
                z_seq.append(z_for)
            z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.latent_dim)
            if len(self.input_dim) == 3:
                pred_x = self.decoder(z_seq)["reconstruction"].reshape(n_seq, batch_size, self.n_obs - vi_index - 1, 
                                                                       self.input_dim[0], self.input_dim[1], self.input_dim[2])
            elif len(self.input_dim) == 2:
                pred_x = self.decoder(z_seq)["reconstruction"].reshape(n_seq, batch_size, self.n_obs - vi_index - 1, 
                                                                       self.input_dim[1])
            all_pred_x.append(pred_x)
        if num_gen_seq % batch_size != 0:
            rem = num_gen_seq % batch_size
            z_seq = []
            z_for = z_vi_index[:, -rem:]
            for i in range(vi_index, self.n_obs - 1):
                t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float()
                z_for = z_for.reshape(n_seq, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w)\
                        .float().to(self.pretrained_ldm.device)
                noise_pred = self.pretrained_ldm(z_for, t)
                z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(e_t=noise_pred,
                                                                     index=self.n_obs - 1 - i,
                                                                     x=z_for,
                                                                     temperature=self.temperature,
                                                                     repeat_noise=False)
                z_for = z_for.reshape(n_seq, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w)\
                        .to(self.pretrained_ldm.device)
                z_seq.append(z_for)
            z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.latent_dim)
            if len(self.input_dim) == 3:
                pred_x = self.decoder(z_seq)["reconstruction"].reshape(n_seq, rem, self.n_obs - vi_index - 1,
                                                                       self.input_dim[0], self.input_dim[1], self.input_dim[2])
            else:
                pred_x = self.decoder(z_seq)["reconstruction"].reshape(n_seq, rem, self.n_obs - vi_index - 1,
                                                                       self.input_dim[1])
            all_pred_x.append(pred_x)
        all_pred_x = torch.cat(all_pred_x, dim=1).to(device)
        return all_pred_x





    def get_nll(self, data, vi_index, n_samples=1, batch_size=100):
        """
        Estimate the negative log-likelihood (NLL) of the model via importance sampling.
        
        Args:
            data (torch.Tensor): Input data tensor with shape [Dataset, ...].
            vi_index (int): The observation index to use for evaluation.
            n_samples (int): Total number of importance samples.
            batch_size (int): Batch size for processing samples.
        
        Returns:
            float: Estimated negative log-likelihood.
        """
        from tqdm import tqdm  # for progress tracking
        # Compute number of full batches (if n_samples > batch_size).
        n_full_batches = n_samples // batch_size if n_samples > batch_size else 1
        
        log_p = []
        
        # Process each datum separately.
        for i in tqdm(range(len(data))):
            # Get a single sample and move it to the model's device.
            x = data[i].unsqueeze(0).to(self.device)  # shape: [1, ...]
            log_p_x_samples = []
            
            for _ in range(n_full_batches):
                # Replicate x to form a batch.
                x_rep = x.repeat(batch_size, *([1] * (len(x.shape) - 1)))  # shape: [batch_size, ...]
                
                # --- Encoder and Sampling ---
                # Use the selected observation index to produce latent statistics.
                encoder_output = self.encoder(x_rep[:, vi_index])
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                std = torch.exp(0.5 * log_var)
                
                # Sample latent z using the Gaussian reparameterization.
                z, _ = self._sample_gauss(mu, std)
                if self.GM is not None:
                    z = sample_with_metric(z, self.GM.centroids, self.GM.M_tens, self.GM.lbd,
                                        self.temperature, self.beta_zero_sqrt)
                z_0 = z
                log_abs_det_jac_posterior = 0
                if self.posterior == 'iaf':
                    if self.posterior_iaf_config.context_dim is not None:
                        try:
                            h = encoder_output.context
                        except Exception as e:
                            raise AttributeError("Cannot get context from encoder outputs: " + str(e))
                        flow_output = self.posterior_iaf_flow.inverse(z, h=h)
                    else:
                        flow_output = self.posterior_iaf_flow.inverse(z)
                    z = flow_output.out
                    log_abs_det_jac_posterior += flow_output.log_abs_det_jac
                z_rep = z  # final latent state after possible flow.
                
                # --- Construct Latent Sequence via Diffusion Steps ---
                z_seq_list = []
                z_rev = z_rep
                # Propagate backwards: add noise from earlier steps.
                for k in range(vi_index - 1, -1, -1):
                    t1 = self.diff_t_steps[(k+1)*np.ones(batch_size).astype(int)]
                    t2 = self.diff_t_steps[k*np.ones(batch_size).astype(int)]
                    z_rev = self.pretrained_ldm.sequential_diffusion(x=z_rev, t1=t1, t2=t2)\
                            .to(self.pretrained_ldm.device).float()
                    z_seq_list.append(z_rev)
                z_seq_list.reverse()  # now in temporal order
                # Append current latent state.
                z_seq_list.append(z_rep.to(self.pretrained_ldm.device))
                # Propagate forwards: denoising using DDIM steps.
                z_for = z_rep
                for k in range(vi_index, self.n_obs - 1):
                    t = torch.tensor(self.diff_t_steps[k]).reshape(1).to(self.pretrained_ldm.device).float()
                    z_for = z_for.reshape(batch_size, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w)\
                            .float().to(self.pretrained_ldm.device)
                    noise_pred = self.pretrained_ldm(z_for, t)
                    z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(e_t=noise_pred,
                                                                        index=self.n_obs - 1 - k,
                                                                        x=z_for,
                                                                        temperature=self.temperature,
                                                                        repeat_noise=False)
                    z_for = z_for.reshape(batch_size, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w)\
                            .to(self.pretrained_ldm.device)
                    z_seq_list.append(z_for)
                
                # Concatenate the sequence. Assume that the final sequence is 2D: [batch_size * n_seq, latent_dim]
                z_seq = torch.cat(z_seq_list, dim=-1)
                
                # Decode using the decoder.
                recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]
                
                # --- Compute Reconstruction Log-Likelihood ---
                # Determine the number of reconstructed observations per input sample:
                n_obs_rec = recon_x.shape[0] // x_rep.shape[0]  # dynamic
                if n_obs_rec <= 0:
                    raise RuntimeError("Reconstruction output shape does not match input batch size.")
                # Repeat x_rep to match the reconstruction.
                x_rep_target = x_rep.repeat_interleave(n_obs_rec, dim=0)
                # Flatten both tensors.
                recon_x_flat = recon_x.reshape(x_rep.shape[0] * n_obs_rec, -1)
                x_target_flat = x_rep_target.reshape(x_rep.shape[0] * n_obs_rec, -1)
                prod_dim = np.prod(self.input_dim)
                if self.model_config.reconstruction_loss == "mse":
                    log_p_x_given_z = (-0.5 * F.mse_loss(recon_x_flat, x_target_flat, reduction="none")
                                    .sum(dim=-1)
                                    - torch.tensor(prod_dim / 2 * np.log(2 * np.pi)).to(x_rep.device())
                                    ).reshape(x_rep.shape[0], -1).mean(dim=-1)
                elif self.model_config.reconstruction_loss == "bce":
                    log_p_x_given_z = (-F.binary_cross_entropy(recon_x_flat, x_target_flat, reduction="none")
                                    .sum(dim=-1)
                                    ).reshape(x_rep.shape[0], -1).mean(dim=-1)
                else:
                    raise ValueError("Unknown reconstruction loss type")
                
                # --- Compute Prior Log-Density ---
                # Reshape the latent sequence to [batch_size, self.n_obs, latent_dim] and use the first frame.
                z_seq_reshaped = z_seq.reshape(x_rep.shape[0], self.n_obs, self.latent_dim)
                z0_state = z_seq_reshaped[:, 0]
                log_prob_z = (-0.5 * (log_var + torch.pow(z_0 - mu, 2) / torch.exp(log_var))).sum(dim=1) - log_abs_det_jac_posterior
                log_prior = self._log_p_z(z0_state)
                
                # Combine likelihood and prior.
                log_p_sample = log_p_x_given_z + log_prior
                log_p_x_samples.append(log_p_sample.detach().cpu())
            
            # Concatenate all samples for this datum.
            log_p_x_samples = torch.cat(log_p_x_samples, dim=0)
            n_total = log_p_x_samples.numel()
            # Compute importance-sampled log likelihood with log-sum-exp.
            log_p_value = torch.logsumexp(log_p_x_samples, dim=0) - np.log(n_total)
            log_p.append(log_p_value.item())
            
            if i % 100 == 0:
                print(f"Current NLL after processing sample {i}: {np.mean(log_p)}")
        return np.mean(log_p)