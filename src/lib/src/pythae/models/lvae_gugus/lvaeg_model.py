import os
from typing import Optional

import sys
from tqdm import tqdm
sys.path.append("......")
sys.path.append(".....")
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..normalizing_flows import IAF, IAFConfig
from ..vae import VAE, VAEConfig
from .lvae_gugus_config import LVAE_GUGUS_Config
from diffusion.stable_diffusion.latent_diffusion import MyLatentDiffusion
from diffusion.stable_diffusion.sampler.ddim import DDIMSampler
from geometric_perspective_on_vaes.sampling import hmc_sampling


class LVAE_GUGUS(VAE):
    """Longitudinal Variational Auto Encoder with Inverse Autoregressive Flows
    (:class:`~pythae.models.normalizing_flows.IAF`).

    Args:
        model_config(VAE_IAF_Config): The Variational Autoencoder configuration seting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: LVAE_GUGUS_Config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        super().__init__(model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "LVAEG_IAF"
        self.n_obs = model_config.n_obs_per_ind
        self.warmup = model_config.warmup
        self.context_dim = model_config.context_dim
        self.beta = model_config.beta
        self.linear_scheduling = model_config.linear_scheduling_steps

        self.prior = model_config.prior
        self.posterior = model_config.posterior

        # Riemannian prior usage flag and metric info
        self.use_riemann_prior = getattr(model_config, 'use_riemann_prior', False)
        # ─── Riemannian metric (t = 0) support ───────────────────────────
        self.G0: Optional[torch.Tensor] = None   # (1, D, D) persistent metric at t = 0
        self.GM: Optional[torch.Tensor] = None   # mean latent at t = 0 for HMC sampling
        self.use_metric_trace: bool = True       # set False to disable expensive tracking
        # ──────────────────────────────────────────────────────────────────

        # Setup flows for longitudinal structure
        self.flows = nn.ModuleList()
        iaf_config = IAFConfig(
            input_dim=(model_config.latent_dim,),
            n_blocks=model_config.n_made_blocks,
            n_hidden_in_made=model_config.n_hidden_in_made,
            hidden_size=model_config.hidden_size,
            include_batch_norm=False,
            context_dim=model_config.context_dim
        )
        for _ in range(self.n_obs - 1):
            self.flows.append(IAF(iaf_config))

        # VAMP prior configuration
        if self.prior == "vamp":
            self.vamp_number_components = model_config.vamp_number_components
            linear_layer = nn.Linear(
                self.vamp_number_components, int(np.prod(model_config.input_dim))
            )
            self.pseudo_inputs = nn.Sequential(linear_layer, nn.Hardtanh(0.0, 1.0))
            self.idle_input = torch.eye(
                self.vamp_number_components, requires_grad=False
            ).to(self.device)

        # Posterior IAF configuration
        if self.posterior == "iaf":
            self.posterior_iaf_config = IAFConfig(
                input_dim=(model_config.latent_dim,),
                n_blocks=3,
                n_hidden_in_made=2,
                hidden_size=model_config.hidden_size,
                context_dim=model_config.context_dim,
                include_batch_norm=False,
            )
            self.posterior_iaf_flow = IAF(self.posterior_iaf_config)

        
    def forward(self, inputs: BaseDataset, **kwargs):
        device = self.device
        x = inputs['data'].to(device)
        x = x.unsqueeze(0) if len(x.shape) == 4 else x
        epoch = kwargs.pop("epoch", 100)

        seq_mask = inputs['seq_mask'].to(device) if hasattr(inputs, 'seq_mask') else torch.ones(x.shape[0], self.n_obs).to(device)
        pix_mask = inputs['pix_mask'].to(device) if hasattr(inputs, 'pix_mask') else torch.ones_like(x)

        if len(x.shape) == 3:
            x = torch.nan_to_num(x)
        else:
            x = x * pix_mask * seq_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        batch_size = x.shape[0]

        if epoch < self.warmup:
            encoder_output = self.encoder(x)
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            std = torch.exp(0.5 * log_var)
            z, _ = self._sample_gauss(mu, std)
            z0 = z
            log_abs_det_jac_posterior = 0

            if self.posterior == 'iaf':
                h = encoder_output.context if self.posterior_iaf_config.context_dim is not None else None
                flow_output = self.posterior_iaf_flow.inverse(z, h=h) if h is not None else self.posterior_iaf_flow.inverse(z)
                z = flow_output.out
                log_abs_det_jac_posterior += flow_output.log_abs_det_jac

            z_seq = z
            recon_x = self.decoder(z_seq)["reconstruction"]

            loss, recon_loss, kld = self.vae_loss_function(
                recon_x=recon_x,
                x=x.reshape((x.shape[0] * self.n_obs,) + x.shape[2:]),
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
            seq_mask = seq_mask.reshape(-1, self.n_obs)
            vi_index = np.random.choice(np.arange(self.n_obs), p=(seq_mask[0].cpu().numpy() / seq_mask[0].sum())) \
                if seq_mask[0].sum() < self.n_obs else np.random.randint(self.n_obs)

            log_abs_det_jac_posterior = 0

            if self.use_riemann_prior and vi_index == 0:
                assert self.GM is not None, "Riemannian metric (GM) not set. Call retrieveG first."
                mu = self.GM.to(device)
                z, _ = hmc_sampling(self, mu, n_samples=batch_size, mcmc_steps_nbr=100)
                z_0_vi_index = z
            else:
                encoder_output = self.encoder(x[:, vi_index])
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)
                z_0_vi_index = z

                if self.posterior == 'iaf':
                    h = encoder_output.context if self.posterior_iaf_config.context_dim is not None else None
                    flow_output = self.posterior_iaf_flow.inverse(z, h=h) if h is not None else self.posterior_iaf_flow.inverse(z)
                    z = flow_output.out
                    log_abs_det_jac_posterior += flow_output.log_abs_det_jac

            z_vi_index = z

            z_seq = []
            z_rev = z_vi_index
            log_abs_det_jac = 0
            for i in range(vi_index - 1, -1, -1):
                flow_output = self.flows[i](z_rev)
                z_rev = flow_output.out
                log_abs_det_jac += flow_output.log_abs_det_jac
                z_seq.append(z_rev)

            z_seq.reverse()
            z_seq.append(z_vi_index)

            z_for = z_vi_index
            for i in range(vi_index, self.n_obs - 1):
                flow_output = self.flows[i].inverse(z_for)
                z_for = flow_output.out
                z_seq.append(z_for)

            z_seq = torch.cat(z_seq, dim=-1).reshape(x.shape[0], self.n_obs, self.latent_dim)
            recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]

            loss, recon_loss, kld = self.loss_function(
                recon_x=recon_x,
                x=x,
                mu=mu,
                log_var=log_var,
                z_0_vi_index=z_0_vi_index,
                z_seq=z_seq,
                z_vi_index=z_vi_index,
                log_abs_det_jac_posterior=log_abs_det_jac_posterior,
                log_abs_det_jac=log_abs_det_jac,
                epoch=epoch,
                seq_mask=seq_mask,
                pix_mask=pix_mask
            )

        return ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x.reshape_as(x),
            z=z,
            z_seq=z_seq,
            x=x,
            z_vi_index=z_vi_index if 'z_vi_index' in locals() else None,
            vi_index=vi_index if 'vi_index' in locals() else None
        )


    def vae_loss_function(self, recon_x, x, mu, log_var, z0, epoch, zk=None, log_abs_det_jac_posterior=None, seq_mask=None, pix_mask=None):
        if self.prior == "standard" and self.posterior == "gaussian":
            loss, recon_loss, kld = self._vae_loss_function(recon_x, x, mu, log_var, z0, seq_mask, pix_mask)

        elif self.prior == "vamp" and self.posterior == "gaussian":
            loss, recon_loss, kld = self._vamp_loss_function(recon_x, x, mu, log_var, z0, epoch, seq_mask, pix_mask)

        elif self.posterior == "iaf":
            loss, recon_loss, kld = self._vae_iaf_loss_function(recon_x, x, mu, log_var, z0, zk, log_abs_det_jac_posterior, epoch, seq_mask, pix_mask)

        return loss, recon_loss, kld


    def _vae_loss_function(self, recon_x, x, mu, log_var, z, seq_mask=None, pix_mask=None):
        recon_loss = self._compute_recon_loss(recon_x, x, pix_mask)
        std = torch.exp(0.5 * log_var)
        z_sample, _ = self._sample_gauss(mu, std)
        log_q = (-0.5 * (log_var + torch.pow(z_sample - mu, 2) / log_var.exp())).sum(dim=1)
        log_p = self._log_p_z(z)
        KLD = log_q - log_p

        return ((recon_loss + KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


    def _vae_iaf_loss_function(self, recon_x, x, mu, log_var, z0, zk, log_abs_det_jac, epoch, seq_mask=None, pix_mask=None):
        recon_loss = self._compute_recon_loss(recon_x, x, pix_mask)
        log_prob_z0 = (-0.5 * (log_var + torch.pow(z0 - mu, 2) / torch.exp(log_var))).sum(dim=1)
        log_prob_zk = self._log_p_z(zk)
        KLD = log_prob_z0 - log_prob_zk - log_abs_det_jac

        return ((recon_loss + KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


    def _vamp_loss_function(self, recon_x, x, mu, log_var, z, epoch, seq_mask=None, pix_mask=None):
        recon_loss = self._compute_recon_loss(recon_x, x, pix_mask)
        log_p_z = self._log_p_z(z)
        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=1)
        KLD = -(log_p_z - log_q_z)

        beta = self._compute_beta(epoch)
        return ((recon_loss + beta * KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


    def loss_function(self, recon_x, x, mu, log_var, z_0_vi_index, z_seq, z_vi_index, log_abs_det_jac, log_abs_det_jac_posterior, epoch, seq_mask=None, pix_mask=None):
        recon_loss = self._compute_recon_loss(recon_x, x, pix_mask, sequential=True, seq_mask=seq_mask)
        z0 = z_seq[:, 0]
        log_prob_z_vi_index = (
            -0.5 * (log_var + torch.pow(z_0_vi_index - mu, 2) / (torch.exp(log_var) + 1e-7))
        ).sum(dim=1) - log_abs_det_jac_posterior

        log_p_z = self._log_p_z(z0)
        log_prior_z_vi_index = log_p_z + log_abs_det_jac
        KLD = log_prob_z_vi_index - log_prior_z_vi_index

        beta = self._compute_beta(epoch)
        return (recon_loss + beta * KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


    def _log_p_z(self, z, t=0):
        """
        Computes log p(z) for a batch of latents z at time t using the Riemannian metric G_t.
        Args:
            z (torch.Tensor): (batch, latent_dim)
            t (int): time index
        Returns:
            log_p_z (torch.Tensor): (batch,)
        """
        if self.prior == "standard":
            G_t = self.metric_tensor_field(z, t)  # (batch, D, D)
            log_det_Gt = torch.logdet(G_t)
            Ginv = torch.linalg.inv(G_t)
            quad = torch.einsum('bi,bij,bj->b', z, Ginv, z)
            D = z.shape[1]
            log_p_z = -0.5 * (log_det_Gt + quad + D * np.log(2 * np.pi))
            return log_p_z
        elif self.prior == "vamp":
            # Existing VAMP logic unchanged
            C = self.vamp_number_components
            x = self.pseudo_inputs(self.idle_input.to(z.device)).reshape((C,) + self.model_config.input_dim)
            encoder_output = self.encoder(x)
            prior_mu, prior_log_var = encoder_output.embedding, encoder_output.log_covariance

            z_expand = z.unsqueeze(1)
            prior_mu = prior_mu.unsqueeze(0)
            prior_log_var = prior_log_var.unsqueeze(0)

            log_p_z = (
                torch.sum(
                    -0.5 * (
                        prior_log_var + (z_expand - prior_mu) ** 2 / torch.exp(prior_log_var)
                    ),
                    dim=2,
                ) - torch.log(torch.tensor(C).type(torch.float))
            )
            log_p_z = torch.logsumexp(log_p_z, dim=1)
            return log_p_z


    def _compute_beta(self, epoch):
        if self.linear_scheduling > 0:
            beta = self.beta * epoch / self.linear_scheduling
            if beta > self.beta or not self.training:
                beta = self.beta
        else:
            beta = self.beta
        return beta


    def _compute_recon_loss(self, recon_x, x, pix_mask, sequential=False, seq_mask=None):
        if sequential:
            batch_size = x.shape[0]
            n_total = batch_size * self.n_obs
            recon_x = recon_x.reshape(n_total, -1)
            x = x.reshape(n_total, -1)
            mask = pix_mask.reshape(n_total, -1)
            loss = F.mse_loss(recon_x, x, reduction="none") if self.model_config.reconstruction_loss == "mse" else F.binary_cross_entropy(recon_x, x, reduction="none")
            recon_loss = (loss * mask).sum(dim=-1).reshape(batch_size, self.n_obs)
            seq_mask = seq_mask.reshape(batch_size, self.n_obs)
            recon_loss = recon_loss * seq_mask
            return recon_loss.mean(dim=-1)
        else:
            recon_x = recon_x.reshape(x.shape[0], -1)
            x = x.reshape(x.shape[0], -1)
            mask = pix_mask.reshape(x.shape[0], -1)
            loss = F.mse_loss(recon_x, x, reduction="none") if self.model_config.reconstruction_loss == "mse" else F.binary_cross_entropy(recon_x, x, reduction="none")
            recon_loss = (loss * mask).sum(dim=-1)
            return recon_loss


    def reconstruct(self, x, vi_index, z_vi_index=None, device='cpu'):
        self = self.to(device)
        x = x["data"].to(device)
        x = x.unsqueeze(0) if len(x.shape) == 4 else x

        # Use Riemannian sampling if available for vi_index = 0
        if self.use_riemann_prior and vi_index == 0 and self.GM is not None:
            mu = self.GM.to(device)
            z = hmc_sampling(self, mu, n_samples=x.shape[0], mcmc_steps_nbr=100)[0]
        else:
            encoder_output = self.encoder(x[:, vi_index])
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            std = torch.exp(0.5 * log_var)
            z = self._sample_gauss(mu, std)[0]

            if self.posterior == 'iaf':
                h = getattr(encoder_output, "context", None)
                if self.posterior_iaf_config.context_dim is not None and h is None:
                    raise AttributeError("Ensure encoder outputs a context vector.")
                flow_output = self.posterior_iaf_flow.inverse(z, h=h) if h is not None else self.posterior_iaf_flow.inverse(z)
                z = flow_output.out

        z_vi_index = z if z_vi_index is None else z_vi_index

        # Temporal propagation
        z_seq = []
        z_rev = z_vi_index
        for i in range(vi_index - 1, -1, -1):
            flow_output = self.flows[i](z_rev)
            z_rev = flow_output.out
            z_seq.append(z_rev)
        z_seq.reverse()
        z_seq.append(z_vi_index)

        z_for = z_vi_index
        for i in range(vi_index, self.n_obs - 1):
            flow_output = self.flows[i].inverse(z_for)
            z_for = flow_output.out
            z_seq.append(z_for)

        z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.latent_dim)
        recon_x = self.decoder(z_seq)["reconstruction"]

        return z_seq, recon_x


    def generate(
        self,
        train_data=None,
        random_normal=False,
        num_gen_seq=1,
        vi_index=0,
        T_multiplier=0.5,
        batch_size=128,
        freeze=False,
        device='cpu',
        verbose=True,
        detach=True
    ):
        self = self.to(device)

        # --- Sampling step ---
        if not random_normal:
            assert train_data is not None, 'train_data must be provided if random_normal is False'
            assert self.use_riemann_prior and self.GM is not None, "GM must be set for Riemannian generation"

            mu = self.GM.to(device)
            all_z_vi = []

            if verbose:
                msg = f'Freezing the {vi_index}th obs, sampling 1 point...' if freeze else f'Sampling {num_gen_seq} points on the {vi_index}th manifold...'
                print(msg)

            if not freeze:
                for j in range(0, num_gen_seq, batch_size):
                    n = min(batch_size, num_gen_seq - j)
                    z, _ = hmc_sampling(self, mu, n_samples=n, mcmc_steps_nbr=100)
                    all_z_vi.append(z)
            else:
                z, _ = hmc_sampling(self, mu, n_samples=1, mcmc_steps_nbr=100)
                all_z_vi = [z] * num_gen_seq

            all_z_vi = torch.cat(all_z_vi, dim=0)

        else:
            assert vi_index == 0, 'Random normal sampling only supported for vi_index = 0'
            all_z_vi = torch.randn(num_gen_seq, self.latent_dim).to(device)

        # --- Temporal decoding ---
        full_recon_x, full_z_seq = [], []

        for j in range(0, num_gen_seq, batch_size):
            z_vi_index = all_z_vi[j:j + batch_size]
            z_seq = []

            z_rev = z_vi_index
            for i in range(vi_index - 1, -1, -1):
                flow_output = self.flows[i](z_rev)
                z_rev = flow_output.out
                z_seq.append(z_rev)
            z_seq.reverse()
            z_seq.append(z_vi_index)

            z_for = z_vi_index
            for i in range(vi_index, self.n_obs - 1):
                flow_output = self.flows[i].inverse(z_for)
                z_for = flow_output.out
                z_seq.append(z_for)

            z_seq = torch.cat(z_seq, dim=-1)
            recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]

            # Reshape reconstruction
            if len(self.input_dim) == 3:
                recon_x = recon_x.reshape(-1, self.n_obs, *self.input_dim)
            else:
                recon_x = recon_x.reshape(-1, self.n_obs, self.input_dim[1])

            z_seq = z_seq.reshape(-1, self.n_obs, self.latent_dim)

            if detach:
                recon_x = recon_x.detach()
                z_seq = z_seq.detach()

            full_recon_x.append(recon_x.cpu())
            full_z_seq.append(z_seq.cpu())

        full_recon_x = torch.cat(full_recon_x, dim=0)
        full_z_seq = torch.cat(full_z_seq, dim=0)

        return full_recon_x, full_z_seq


    def predict(self, x, vi_index, num_gen_seq=1, batch_size=100, device='cpu'):
        self = self.to(device)
        x = x.unsqueeze(0) if len(x.shape) == 4 else x  # Ensure batch dim
        x = x.to(device)
        n_seq = x.shape[0]

        x_vi_index = x[:, vi_index]

        # Use Riemannian prior if active and vi_index == 0
        if self.use_riemann_prior and vi_index == 0 and self.GM is not None:
            mu = self.GM.to(device)
            z = hmc_sampling(self, mu, n_samples=n_seq, mcmc_steps_nbr=100)[0]
        else:
            encoder_output = self.encoder(x_vi_index)
            mu = encoder_output.embedding
            z = mu  # no stochasticity here for deterministic forecasting

        # Repeat latent for multiple generations
        z_vi_index = z.unsqueeze(1).repeat(1, num_gen_seq, 1)  # [n_seq, num_gen_seq, latent_dim]

        all_pred_x = []

        for j in range(0, num_gen_seq, batch_size):
            cur_batch_size = min(batch_size, num_gen_seq - j)
            z_for = z_vi_index[:, j:j + cur_batch_size]  # [n_seq, cur_batch_size, latent_dim]

            z_seq = []
            for i in range(vi_index, self.n_obs - 1):
                flow_output = self.flows[i].inverse(z_for)
                z_for = flow_output.out
                z_seq.append(z_for)

            z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.latent_dim)
            recon = self.decoder(z_seq)["reconstruction"]

            if len(self.input_dim) == 3:
                pred_x = recon.reshape(n_seq, cur_batch_size, self.n_obs - vi_index - 1, *self.input_dim)
            else:
                pred_x = recon.reshape(n_seq, cur_batch_size, self.n_obs - vi_index - 1, self.input_dim[1])

            all_pred_x.append(pred_x)

        return torch.cat(all_pred_x, dim=1)  # [n_seq, num_gen_seq, future_obs, ...]


    def infer_missing(self, x, seq_mask, pix_mask):
        self = self.to(x.device)
        x = x * pix_mask * seq_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        p_x_given_z = []
        reconstructions = []
        vi_idx = []

        for vi_index in range(self.n_obs):
            if seq_mask[0, vi_index] == 0:
                continue

            # Riemannian prior if available for vi_index == 0
            if self.use_riemann_prior and vi_index == 0 and self.GM is not None:
                mu = self.GM.to(x.device)
                z = hmc_sampling(self, mu, n_samples=x.shape[0], mcmc_steps_nbr=100)[0]
            else:
                encoder_output = self.encoder(x[:, vi_index])
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                std = torch.exp(0.5 * log_var)
                z = self._sample_gauss(mu, std)[0]

                if self.posterior == 'iaf':
                    h = getattr(encoder_output, "context", None)
                    if self.posterior_iaf_config.context_dim is not None and h is None:
                        raise AttributeError("Ensure encoder outputs a context vector.")
                    flow_output = self.posterior_iaf_flow.inverse(z, h=h) if h is not None else self.posterior_iaf_flow.inverse(z)
                    z = flow_output.out

            z_vi_index = z

            # Time propagation
            z_seq = []
            z_rev = z_vi_index
            for i in range(vi_index - 1, -1, -1):
                flow_output = self.flows[i](z_rev)
                z_rev = flow_output.out
                z_seq.append(z_rev)
            z_seq.reverse()
            z_seq.append(z_vi_index)

            z_for = z_vi_index
            for i in range(vi_index, self.n_obs - 1):
                flow_output = self.flows[i].inverse(z_for)
                z_for = flow_output.out
                z_seq.append(z_for)

            z_seq = torch.cat(z_seq, dim=-1)

            recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]
            z_seq = z_seq.reshape(x.shape[0], self.n_obs, self.latent_dim)

            if self.model_config.reconstruction_loss == "mse":
                recon_loss = (
                    0.5 * F.mse_loss(
                        recon_x.reshape(x.shape[0]*self.n_obs, -1),
                        x.reshape(x.shape[0]*self.n_obs, -1),
                        reduction="none"
                    ) * pix_mask.reshape(x.shape[0]*self.n_obs, -1)
                ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask

            elif self.model_config.reconstruction_loss == "bce":
                recon_loss = (
                    F.binary_cross_entropy(
                        recon_x.reshape(x.shape[0]*self.n_obs, -1),
                        x.reshape(x.shape[0]*self.n_obs, -1),
                        reduction="none"
                    ) * pix_mask.reshape(x.shape[0]*self.n_obs, -1)
                ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask

            recon_loss = recon_loss.mean(dim=-1)  # Mean over time
            p_x_given_z.append(-recon_loss)
            reconstructions.append(recon_x)
            vi_idx.append(vi_index)

        # Stack and select best recon
        all_p = torch.stack(p_x_given_z)  # [num_vi_idx, batch]
        best_idx = torch.argsort(all_p, dim=0, descending=True)  # [num_vi_idx, batch]
        all_recon = torch.stack(reconstructions).reshape(-1, x.shape[0], self.n_obs, *x.shape[2:])  # [num_vi_idx, B, n_obs, ...]

        return all_recon, best_idx, vi_idx
        


    #def infer_traj(self, x):


    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    # ───────────────────────── Metric helpers ────────────────────────────
    @torch.no_grad()
    def retrieveG(self, data, vi_index=0, verbose=False, T_multiplier=1.0, device='cpu', addStdNorm=True):
        """
        Estimates the Riemannian metric tensor field from the latent space geometry at a specific vi_index.

        Stores the average posterior mean (mu) as `self.GM`, for Riemannian sampling.

        Args:
            data (torch.Tensor): Input data of shape (N, n_obs, ...) with or without batch dim.
            vi_index (int): Index of observation to condition on.
            verbose (bool): Whether to print metric stats.
            T_multiplier (float): Temperature multiplier for metric smoothing (if relevant).
            device (str): Device for computation.
            addStdNorm (bool): If True, includes stddev in `G(z)` (used for normalizing flows).
        
        Returns:
            mu_avg (torch.Tensor): The average latent mean vector.
            G0 (torch.Tensor): The metric tensor at t = 0 (diag‑cov form).
        """
        self = self.to(device)
        data = data.to(device)
        N = data.shape[0]
        x_vi = data[:, vi_index]

        encoder_output = self.encoder(x_vi)
        mu = encoder_output.embedding  # (N, latent_dim)
        log_var = encoder_output.log_covariance  # (N, latent_dim)

        mu_avg = mu.mean(dim=0, keepdim=True).to(device)      # (1, D)
        self.GM = mu_avg                           # store mean for sampling

        # Build diagonal metric G0 from posterior std at t = 0
        if addStdNorm:
            std_avg = torch.exp(0.5 * log_var).mean(dim=0, keepdim=True).to(device)  # (1, D)
            G0 = torch.diag_embed(std_avg.squeeze() ** 2)                 # (D, D)
        else:
            G0 = torch.eye(self.latent_dim, device=device)

        self.G0 = G0.unsqueeze(0).to(device)  # (1, D, D) persistent

        if verbose:
            print(f"Retrieved G0 (diag).  det(G0)={torch.det(G0):.3e}")

        return mu_avg, G0

    @staticmethod
    def is_rank0():
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def metric_trajectory(self, z0: torch.Tensor, vi_index: int = 0):
        """
        Transport G0 along the longitudinal IAF chain and return a list
        [(z_t, G_t)] for t = 0 … n_obs‑1.
        """
        # Only run on rank 0 and not during sanity check
        if hasattr(self, 'trainer') and getattr(self.trainer, 'sanity_checking', False):
            return None
        if not self.is_rank0():
            return None
        assert self.G0 is not None, "Call retrieveG first to set G0."
        device = z0.device
        G0 = self.G0.to(device)
        traj = []
        z, G = z0, G0.expand(z0.size(0), -1, -1)
        for t in range(vi_index, -1, -1):
            traj.insert(0, (z, G))
            if t > 0:
                z, G = self._push_metric(z, G, self.flows[t-1])

        z, G = traj[vi_index]
        # forward in time
        for t in range(vi_index, self.n_obs - 1):
            z, G = self._push_metric(z, G, self.flows[t].inverse)
            traj.append((z, G))

        return traj

    def _push_metric(self, z: torch.Tensor, G: torch.Tensor, flow):
        """
        Push‑forward a metric tensor G through the given normalising‑flow block.
        """
        device = z.device
        G = G.to(device)
        z = z.detach().requires_grad_(True)
        # choose correct mapping (flow or its inverse)
        if callable(flow):
            f = lambda y: flow(y).out if hasattr(flow(y), "out") else flow(y)
        else:
            raise RuntimeError("flow must be callable")
        z_next = f(z)
        batch_size, latent_dim = z.shape
        J = []
        for i in range(batch_size):
            J_i = torch.autograd.functional.jacobian(
                lambda y: f(y.unsqueeze(0)).squeeze(0), z[i], vectorize=True
            )  # shape: [latent_dim, latent_dim]
            J.append(J_i)
        J = torch.stack(J, dim=0)  # shape: [batch_size, latent_dim, latent_dim]
        J = J.to(device)
        Jinv = torch.linalg.inv(J)
        G_next = Jinv.transpose(-1, -2) @ G @ Jinv
        return z_next.detach(), G_next.detach()

    def log_metric_movie(self, traj):
        """Quick text summary of metric evolution (min/max eigenvalues & det)."""
        for t, (_, G) in enumerate(traj):
            eig = torch.linalg.eigvalsh(G)
            print(f"t={t:02d}  λmin={eig.min():.2e}  λmax={eig.max():.2e}  det={torch.det(G).mean():.2e}")
    # ─────────────────────────────────────────────────────────────────────


    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Estimates the negative log-likelihood via importance sampling using the approximate posterior q(z|x).
        """

        self = self.to(data.device)
        data = data.to(self.device)
        log_p = []

        n_full_batch = max(1, n_samples // batch_size)
        actual_batch_size = min(n_samples, batch_size)

        for i in range(len(data)):
            x = data[i].unsqueeze(0)
            log_p_x = []

            for _ in range(n_full_batch):
                vi_index = np.random.randint(self.n_obs)
                x_rep = torch.cat([x] * actual_batch_size)

                encoder_output = self.encoder(x_rep[:, vi_index])
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)
                z_0_vi_index = z

                log_abs_det_jac_posterior = 0
                if self.posterior == 'iaf':
                    h = getattr(encoder_output, "context", None)
                    if self.posterior_iaf_config.context_dim is not None and h is None:
                        raise AttributeError("Ensure encoder outputs a context vector.")
                    flow_output = self.posterior_iaf_flow.inverse(z, h=h) if h is not None else self.posterior_iaf_flow.inverse(z)
                    z = flow_output.out
                    log_abs_det_jac_posterior += flow_output.log_abs_det_jac

                z_vi_index = z

                # Propagate latent trajectory
                z_seq = []
                z_rev = z_vi_index
                log_abs_det_jac = 0
                for j in range(vi_index - 1, -1, -1):
                    flow_output = self.flows[j](z_rev)
                    z_rev = flow_output.out
                    log_abs_det_jac += flow_output.log_abs_det_jac
                    z_seq.append(z_rev)
                z_seq.reverse()
                z_seq.append(z_vi_index)

                z_for = z_vi_index
                for j in range(vi_index, self.n_obs - 1):
                    flow_output = self.flows[j].inverse(z_for)
                    z_for = flow_output.out
                    z_seq.append(z_for)

                z_seq = torch.cat(z_seq, dim=-1).reshape(x_rep.shape[0], self.n_obs, self.latent_dim)
                recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]

                # Log p(x | z)
                if self.model_config.reconstruction_loss == "mse":
                    log_p_x_given_z = (
                        -0.5 * F.mse_loss(
                            recon_x.reshape(x_rep.shape[0]*self.n_obs, -1),
                            x_rep.reshape(x_rep.shape[0]*self.n_obs, -1),
                            reduction="none",
                        ).sum(dim=-1) -
                        0.5 * np.prod(self.input_dim) * np.log(2 * np.pi)
                    ).reshape(x_rep.shape[0], -1).sum(dim=-1)

                elif self.model_config.reconstruction_loss == "bce":
                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0]*self.n_obs, -1),
                        x_rep.reshape(x_rep.shape[0]*self.n_obs, -1),
                        reduction="none"
                    ).sum(dim=-1).reshape(x_rep.shape[0], -1).sum(dim=-1)

                # Log q(z | x)
                log_prob_z_vi_index = (
                    -0.5 * (log_var + torch.pow(z_0_vi_index - mu, 2) / torch.exp(log_var))
                ).sum(dim=1) - log_abs_det_jac_posterior

                # Log p(z)
                z0 = z_seq[:, 0]
                log_p_z = self._log_p_z(z0)
                log_prior_z_vi_index = log_p_z + log_abs_det_jac

                log_p_x.append(log_p_x_given_z + log_prior_z_vi_index - log_prob_z_vi_index)

            log_p_x = torch.cat(log_p_x)
            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())

            if i % 100 == 0:
                print(f"Current NLL at {i}: {np.mean(log_p):.4f}")

        return np.mean(log_p)


    @torch.no_grad()
    def retrieveG(self, data, vi_index=0, verbose=False, T_multiplier=1.0, device='cpu', addStdNorm=True):
        """
        Estimates the Riemannian metric tensor field from the latent space geometry at a specific vi_index.

        Stores the average posterior mean (mu) as `self.GM`, for Riemannian sampling.

        Args:
            data (torch.Tensor): Input data of shape (N, n_obs, ...) with or without batch dim.
            vi_index (int): Index of observation to condition on.
            verbose (bool): Whether to print metric stats.
            T_multiplier (float): Temperature multiplier for metric smoothing (if relevant).
            device (str): Device for computation.
            addStdNorm (bool): If True, includes stddev in `G(z)` (used for normalizing flows).
        
        Returns:
            mu_avg (torch.Tensor): The average latent mean vector.
            G0 (torch.Tensor): The metric tensor at t = 0 (diag‑cov form).
        """
        self = self.to(device)
        data = data.to(device)
        N = data.shape[0]
        x_vi = data[:, vi_index]

        encoder_output = self.encoder(x_vi)
        mu = encoder_output.embedding  # (N, latent_dim)
        log_var = encoder_output.log_covariance  # (N, latent_dim)

        mu_avg = mu.mean(dim=0, keepdim=True)      # (1, D)
        self.GM = mu_avg                           # store mean for sampling

        # Build diagonal metric G0 from posterior std at t = 0
        if addStdNorm:
            std_avg = torch.exp(0.5 * log_var).mean(dim=0, keepdim=True)  # (1, D)
            G0 = torch.diag_embed(std_avg.squeeze() ** 2)                 # (D, D)
        else:
            G0 = torch.eye(self.latent_dim, device=device)

        self.G0 = G0.unsqueeze(0)  # (1, D, D) persistent

        if verbose:
            print(f"Retrieved G0 (diag).  det(G0)={torch.det(G0):.3e}")

        return mu_avg, G0

    def metric_tensor_field(self, z, t):
        """
        Returns the Riemannian metric at time t for latent z.
        G0 is pushed forward through the flows up to time t.
        Args:
            z (torch.Tensor): Latent tensor of shape (batch, latent_dim)
            t (int): Time index (0-based)
        Returns:
            G (torch.Tensor): Metric tensor of shape (batch, latent_dim, latent_dim)
        """
        assert self.G0 is not None, "G0 not set. Call retrieveG first."
        G = self.G0.to(z.device).expand(z.shape[0], -1, -1)
        z_t = z
        if t == 0:
            return G
        # Push G0 through flows up to time t
        for i in range(t):
            z_t, G = self._push_metric(z_t, G, self.flows[i].inverse)
        return G