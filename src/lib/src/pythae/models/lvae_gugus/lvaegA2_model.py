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
import torch.linalg
import math
import contextlib  # for mixed‑precision autocast wrappers
from torch.utils.data import DataLoader
import itertools

# ───────────────────────── cache helper ──────────────────────────────
class _ForwardCache(dict):
    """Dictionary flushed at the start of every forward pass.
    Keys are tensor.data_ptr() IDs pointing to the underlying storage.
    """
    pass

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..normalizing_flows import IAF, IAFConfig
from ..vae import VAE
from .lvae_gugus_config import LVAE_GUGUS_Config
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

        # ─── optional numerical‑stability / efficiency toggles ─────────────
        self.use_weight_norm      = getattr(model_config, "weight_norm_flows", True)
        self.use_mixed_precision  = getattr(model_config, "mixed_precision", False)
        self.autocast = torch.cuda.amp.autocast if (self.use_mixed_precision and torch.cuda.is_available()) else contextlib.nullcontext

        # Riemannian prior usage flag and metric info
        self.use_riemann_prior = getattr(model_config, 'use_riemann_prior', False)
        # ─── Riemannian metric (t = 0) support ───────────────────────────
        self.G0: Optional[torch.Tensor] = None   # (1, D, D) persistent metric at t = 0
        self.GM: Optional[torch.Tensor] = None   # mean latent at t = 0 for HMC sampling
        self.use_metric_trace: bool = True       # set False to disable expensive tracking
        # dynamic metric mode – no cached G_list; recomputed every call
        # forward‑pass cache (encoder stats, metrics, …)
        self._fwd_cache: _ForwardCache = _ForwardCache()
        # ─── geometry debug / evaluation flags ────────────────────────────
        self.metric_mode: str = getattr(model_config, "metric_mode", "push")  # "push" or "empirical"
        # if >0 : compare empirical vs. push‑forward metrics every *n* forward calls
        self.metric_eval_period: int = getattr(model_config, "metric_eval_period", 0)
        self._fwd_counter: int = 0        # counts forward passes for periodic metric eval

        # Setup flows for longitudinal structure
        self.flows = nn.ModuleList()
        iaf_config = IAFConfig(
            input_dim=(model_config.latent_dim,),
            n_blocks=model_config.n_made_blocks,
            n_hidden_in_made=model_config.n_hidden_in_made,
            hidden_size=model_config.hidden_size,
            include_batch_norm=True,
            context_dim=model_config.context_dim
        )
        for _ in range(self.n_obs - 1):
            flow = IAF(iaf_config)
            self._apply_weight_norm(flow)
            self.flows.append(flow)

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
                include_batch_norm=True,
            )
            self.posterior_iaf_flow = IAF(self.posterior_iaf_config)
            self._apply_weight_norm(self.posterior_iaf_flow)

        
    def forward(self, inputs: BaseDataset, **kwargs):
        device = self.device
        # new forward pass ⇒ flush per‑batch cache
        self._fwd_cache.clear()
        # periodic metric evaluation
        self._fwd_counter += 1
        if self.metric_eval_period > 0 and (self._fwd_counter % self.metric_eval_period == 0):
            with torch.no_grad():
                try:
                    stats = self.compare_metrics(inputs['data'][:, 0], t=0)   # check first time‑step
                    # simple stdout; user can redirect to logger
                    print(f"[Metric‑Diag] step={self._fwd_counter} "
                          f"fro={stats['fro_gap']:.3e} "
                          f"specCos={stats['spectral_cos']:.3f} "
                          f"condR={stats['cond_ratio']:.2f}")
                except Exception as exc:  # never crash training on diagnostics
                    print("[Metric‑Diag] skipped due to", exc)
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

        with self.autocast():
            if epoch < self.warmup:
                encoder_output = self.encoder(x)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)
                z0 = z
                log_abs_det_jac_posterior = 0

                if self.posterior == 'iaf':
                    h = encoder_output.context if self.posterior_iaf_config.context_dim is not None else None
                    z, log_det = self._apply_flow(self.posterior_iaf_flow, z, inverse=True, context=h)
                    log_abs_det_jac_posterior += log_det

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
                    pix_mask=pix_mask,
                    vi_index=0
                )

            else:
                seq_mask = seq_mask.reshape(-1, self.n_obs)
                mask_sum = seq_mask[0].sum()
                if mask_sum < self.n_obs:
                    weights = (seq_mask[0] / mask_sum).to(dtype=torch.float32)
                    vi_index = torch.multinomial(weights, 1).item()
                else:
                    vi_index = torch.randint(self.n_obs, (1,), device=seq_mask.device).item()

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
                        z, log_det = self._apply_flow(self.posterior_iaf_flow, z, inverse=True, context=h)
                        log_abs_det_jac_posterior += log_det

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


    def vae_loss_function(self, recon_x, x, mu, log_var, z0, epoch, vi_index, zk=None, log_abs_det_jac_posterior=None, seq_mask=None, pix_mask=None):
        if self.prior == "standard" and self.posterior == "gaussian":
            loss, recon_loss, kld = self._vae_loss_function(recon_x, x, mu, log_var, z0, vi_index, seq_mask, pix_mask)

        elif self.prior == "vamp" and self.posterior == "gaussian":
            loss, recon_loss, kld = self._vamp_loss_function(recon_x, x, mu, log_var, z0, epoch, vi_index, seq_mask, pix_mask)

        elif self.posterior == "iaf":
            loss, recon_loss, kld = self._vae_iaf_loss_function(recon_x, x, mu, log_var, z0, zk, log_abs_det_jac_posterior, epoch, vi_index, seq_mask, pix_mask)

        return loss, recon_loss, kld


    def _vae_loss_function(self, recon_x, x, mu, log_var, z, vi_index, seq_mask=None, pix_mask=None):
        recon_loss = self._compute_recon_loss(recon_x, x, pix_mask)
        std = torch.exp(0.5 * log_var)
        z_sample, _ = self._sample_gauss(mu, std)
        log_q = (-0.5 * (log_var + torch.pow(z_sample - mu, 2) / log_var.exp())).sum(dim=1)
        log_p = self._log_p_z(z, t=vi_index)
        KLD = log_q - log_p

        return ((recon_loss + KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


    def _vae_iaf_loss_function(self, recon_x, x, mu, log_var, z0, zk, log_abs_det_jac, epoch, vi_index, seq_mask=None, pix_mask=None):
        recon_loss = self._compute_recon_loss(recon_x, x, pix_mask)
        log_prob_z0 = (-0.5 * (log_var + torch.pow(z0 - mu, 2) / torch.exp(log_var))).sum(dim=1)
        log_prob_zk = self._log_p_z(zk, t=vi_index)
        KLD = log_prob_z0 - log_prob_zk - log_abs_det_jac

        return ((recon_loss + KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


    def _vamp_loss_function(self, recon_x, x, mu, log_var, z, epoch, vi_index, seq_mask=None, pix_mask=None):
        recon_loss = self._compute_recon_loss(recon_x, x, pix_mask)
        log_p_z = self._log_p_z(z, t=vi_index)
        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=1)
        KLD = -(log_p_z - log_q_z)

        beta = self._compute_beta(epoch)
        return ((recon_loss + beta * KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


    def loss_function(
            self, recon_x, x, *unused,
            epoch, seq_mask=None, pix_mask=None, **__
        ):
            # ───────────────────────── reconstruction term ─────────────────────────
            recon_loss = self._compute_recon_loss(
                recon_x, x, pix_mask, sequential=True, seq_mask=seq_mask
            )                    # (batch,)

            # ───────────────────────── KL term: Σ_t KL(q_t || p_t) ─────────────────
            B, T = x.size(0), self.n_obs
            kl_terms = []

            for t in range(T):
                enc = self._cached_encoder(x[:, t])
                mu_t, log_var_t = enc.embedding, enc.log_covariance
                std_t = torch.exp(0.5 * log_var_t)
                z_t, _ = self._sample_gauss(mu_t, std_t)      # reparam sample

                # log q_t(z_t | x_t)
                log_q_t = (
                    -0.5 * (log_var_t + (z_t - mu_t).pow(2) / torch.exp(log_var_t))
                ).sum(dim=1)

                # log p_t(z_t) using pushed metric G_t
                log_p_t = self._log_p_z(z_t, t=t)

                kl_t = log_q_t - log_p_t                      # (batch,)
                if seq_mask is not None:
                    kl_t = kl_t * seq_mask[:, t]              # mask un‑observed t
                kl_terms.append(kl_t)

            KLD = torch.stack(kl_terms, dim=1).sum(dim=1)     # (batch,)

            beta = self._compute_beta(epoch)
            total = recon_loss + beta * KLD                   # (batch,)

            # return batch‑averaged metrics
            return total.mean(), recon_loss.mean(), KLD.mean()


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
            # ── ensure FP32 for stable logdet / inverse ──
            if G_t.dtype != torch.float32:
                Gt32 = G_t.float()
                z32  = z.float()
            else:
                Gt32, z32 = G_t, z

            # numerical stabilisation: add εI to avoid logdet/ inverse blow‑ups
            eps = 1e-6
            D = z.shape[1]
            Gt32 = Gt32 + torch.eye(D, device=Gt32.device, dtype=Gt32.dtype) * eps

            log_det_Gt = torch.logdet(Gt32)
            Ginv = torch.linalg.inv(Gt32)
            quad = torch.einsum('bi,bij,bj->b', z32, Ginv, z32)
            # D = z.shape[1]  # <-- REMOVE this line to avoid redefinition
            log_p_z = -0.5 * (log_det_Gt + quad + D * np.log(2 * np.pi))
            return log_p_z
        elif self.prior == "vamp":
            # Existing VAMP logic unchanged
            C = self.vamp_number_components
            x = self.pseudo_inputs(self.idle_input.to(self.device)).reshape((C,) + self.model_config.input_dim)
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


    def reconstruct(self, x, vi_index, z_vi_index=None, device=None):
        device = device or self.device
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
                z, _ = self._apply_flow(self.posterior_iaf_flow, z, inverse=True, context=h)

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
        device=None,
        verbose=True,
        detach=True
    ):
        device = device or self.device
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


    def predict(self, x, vi_index, num_gen_seq=1, batch_size=100, device=None):
        device = device or self.device
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
        self = self.to(self.device)
        x = x * pix_mask * seq_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        p_x_given_z = []
        reconstructions = []
        vi_idx = []

        for vi_index in range(self.n_obs):
            if seq_mask[0, vi_index] == 0:
                continue

            # Riemannian prior if available for vi_index == 0
            if self.use_riemann_prior and vi_index == 0 and self.GM is not None:
                mu = self.GM.to(self.device)
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
                    z, _ = self._apply_flow(self.posterior_iaf_flow, z, inverse=True, context=h)

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


    # ───────────────────── cached encoder helper ──────────────────────
    def _cached_encoder(self, x: torch.Tensor):
        """Run the encoder once per unique input tensor and cache the result."""
        key = x.data_ptr()
        if key not in self._fwd_cache:
            self._fwd_cache[key] = self.encoder(x)
        return self._fwd_cache[key]

    
    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    # ───────────────────────── Flow helper ────────────────────────────────
    def _apply_flow(
        self,
        flow,
        z: torch.Tensor,
        *,
        inverse: bool = False,
        context: Optional[torch.Tensor] = None,
    ):
        """
        Apply a (possibly inverse) autoregressive flow *once* and return
        both the transformed latent and its log‑|det Jacobian|.

        Parameters
        ----------
        flow : IAF
            The flow module.
        z : torch.Tensor
            Input latent, shape (B, D).
        inverse : bool, default False
            If True, use flow.inverse; else use forward direction.
        context : torch.Tensor | None
            Optional conditioning vector for the flow.

        Returns
        -------
        z_out : torch.Tensor  # (B, D)
        log_det : torch.Tensor  # (B,)
        """
        if inverse:
            out = flow.inverse(z, h=context) if context is not None else flow.inverse(z)
        else:
            out = flow(z, h=context) if context is not None else flow(z)

        # `pythae` flows return a struct‑like object
        if hasattr(out, "out"):
            return out.out, out.log_abs_det_jac
        # fallback to tuple
        if isinstance(out, tuple) and len(out) == 2:
            return out
        # identity fallback
        return out, torch.zeros(z.size(0), device=z.device, dtype=z.dtype)

    def _apply_weight_norm(self, module: nn.Module):
        """Apply weight‑normalisation to all Linear layers of *module* (in‑place)."""
        if not self.use_weight_norm:
            return
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.utils.weight_norm(m, name="weight", dim=0)

    # ───────────────────── geometry diagnostics ────────────────────────
    @torch.no_grad()
    def _empirical_metric(self, log_var: torch.Tensor) -> torch.Tensor:
        """Return a diagonal empirical metric from posterior variances.

        Parameters
        ----------
        log_var : torch.Tensor, shape (B, D)
            Log‑variance output of the encoder.

        Returns
        -------
        G_emp : torch.Tensor, shape (B, D, D)
            Diagonal metric diag(σ²).
        """
        var = torch.exp(log_var)               # (B, D)
        return torch.diag_embed(var)           # (B, D, D)

    @torch.no_grad()
    def compare_metrics(self, x_t: torch.Tensor, t: int = 0) -> dict:
        """Compute discrepancy between **empirical** metric and push‑forward metric at time *t*.

        Returns a dict with
            "fro_gap", "spectral_cos", "cond_ratio"
        """
        enc = self.encoder(x_t)                # encode single time slice
        mu, log_var = enc.embedding, enc.log_covariance
        z, _ = self._sample_gauss(mu, torch.exp(0.5 * log_var))
        G_emp  = self._empirical_metric(log_var)          # (B, D, D)
        G_push = self.metric_tensor_field(z, t)           # (B, D, D)

        # ── Frobenius relative gap ──
        fro_gap = (G_emp - G_push).norm(p='fro', dim=(1, 2)) / \
                  (G_emp.norm(p='fro', dim=(1, 2)) + 1e-8)

        # ── spectral cosine between eigen‑spectra ──
        λ_emp  = torch.linalg.eigvalsh(G_emp)
        λ_push = torch.linalg.eigvalsh(G_push)
        spectral_cos = F.cosine_similarity(λ_emp, λ_push, dim=1)

        # ── condition number ratio ──
        cond_emp  = λ_emp.max(dim=1).values / (λ_emp.min(dim=1).values + 1e-12)
        cond_push = λ_push.max(dim=1).values / (λ_push.min(dim=1).values + 1e-12)
        cond_ratio = cond_push / (cond_emp + 1e-12)

        stats = {
            "fro_gap": fro_gap.mean().item(),
            "spectral_cos": spectral_cos.mean().item(),
            "cond_ratio": cond_ratio.mean().item(),
        }
        return stats


        # ─────────────────── downstream task probing ────────────────────────
    @torch.no_grad()
    def _latent_repr(self, x: torch.Tensor, t: int = 0, sample: bool = False):
        """Return latent representation at time‑step *t*.

        Parameters
        ----------
        x : torch.Tensor, shape (B, n_obs, …)
            Input sequence.
        t : int, default 0
            Which observation index to encode.
        sample : bool, default False
            If True, draw a stochastic sample; otherwise use the posterior mean.

        Returns
        -------
        z : torch.Tensor, shape (B, D)
        """
        enc = self.encoder(x[:, t])
        mu, log_var = enc.embedding, enc.log_covariance
        if sample:
            z, _ = self._sample_gauss(mu, torch.exp(0.5 * log_var))
        else:
            z = mu
        if self.posterior == "iaf":
            z, _ = self._apply_flow(self.posterior_iaf_flow, z, inverse=True,
                                    context=getattr(enc, "context", None))
        return z

    # ---------- k‑NN retrieval --------------------------------------------------
    @torch.no_grad()
    def knn_probe(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        *,
        k: int = 10,
        t: int = 0,
        distance: str = "euclidean",
    ):
        """Simple k‑NN retrieval accuracy / NDCG in latent space.

        Parameters
        ----------
        data    : torch.Tensor, (N, n_obs, …)
        labels  : torch.Tensor, (N,)
                  Class/category id for each sample.
        k       : int
        t       : int, which observation index to encode
        distance: "euclidean" | "geodesic"
        """
        device = self.device
        z = self._latent_repr(data.to(device), t, sample=False)  # (N, D)

        if distance == "euclidean":
            dist = torch.cdist(z, z)                # (N, N)
        elif distance == "geodesic":
            # approximate geodesic using Mahalanobis under pushed metric
            G = self.metric_tensor_field(z, t)      # (N, D, D)
            # efficient pairwise (z_i - z_j)^T G_i^{-1} (z_i - z_j)
            Ginv = torch.linalg.inv(G)              # (N, D, D)
            diff = z.unsqueeze(1) - z.unsqueeze(0)  # (N, N, D)
            dist = torch.einsum("nij,njk,nik->ni", diff, Ginv, diff)  # (N, N)
            dist = torch.sqrt(torch.clamp(dist, min=0))
        else:
            raise ValueError("distance must be 'euclidean' or 'geodesic'")

        # exclude self‑distance
        dist.fill_diagonal_(float("inf"))
        knn_idx = dist.topk(k, largest=False).indices         # (N, k)
        knn_lab = labels.to(device)[knn_idx]                  # (N, k)

        # retrieval accuracy: at least one of k neighbours shares the label
        correct_any = (knn_lab == labels.to(device).unsqueeze(1)).any(dim=1)
        acc = correct_any.float().mean().item()

        # NDCG@k
        rel = (knn_lab == labels.unsqueeze(1)).float()        # (N, k)
        denom = 1.0 / torch.log2(torch.arange(2, k + 2, device=device).float())
        dcg   = (rel * denom).sum(dim=1)
        ideal = denom[:1].expand_as(rel)                      # only 1 relevant per row
        idcg  = ideal.sum(dim=1)
        ndcg  = (dcg / idcg).mean().item()

        return {"acc": acc, "ndcg": ndcg}

    # ---------- linear probe ----------------------------------------------------
    @torch.no_grad()
    def linear_probe(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        *,
        t: int = 0,
        metric: bool = False,
        lr: float = 1e-2,
        epochs: int = 200,
    ):
        """Fit a linear classifier (log‑loss) on frozen latents; report accuracy.

        If *metric* is True, inputs are whitened wrt the averaged metric.
        """
        device = self.device
        z_tr = self._latent_repr(train_data.to(device), t, sample=False)
        z_te = self._latent_repr(test_data.to(device), t, sample=False)

        if metric:
            G = self.metric_tensor_field(z_tr, t)          # (B, D, D)
            Gmean = G.mean(dim=0)                          # (D, D)
            L = torch.linalg.cholesky(Gmean + 1e-6 * torch.eye(Gmean.size(0), device=device))
            z_tr = torch.linalg.solve(L, z_tr.T).T
            z_te = torch.linalg.solve(L, z_te.T).T

        # simple logistic regression via torch
        num_cls = int(train_labels.max().item() + 1)
        W = torch.zeros(z_tr.size(1), num_cls, device=device, requires_grad=True)
        optim = torch.optim.SGD([W], lr=lr)

        y_tr = train_labels.to(device).long()
        for _ in range(epochs):
            logits = z_tr @ W
            loss = F.cross_entropy(logits, y_tr)
            optim.zero_grad()
            loss.backward()
            optim.step()

        with torch.no_grad():
            acc = (z_te @ W).argmax(dim=1).eq(test_labels.to(device)).float().mean().item()
        return acc

    # ───────────── curvature regularity indicator ───────────────────────
    @torch.no_grad()
    def curvature_stats(self, z: torch.Tensor, t: int = 0) -> dict:
        """
        Simple surrogate of sectional curvature:
            * eigenvalue dispersion of G
            * norm of ∇² log det G (approximated via finite diffs on batch)

        Returns dictionary of statistics for monitoring.
        """
        G = self.metric_tensor_field(z, t)          # (B, D, D)
        λ = torch.linalg.eigvalsh(G)                # (B, D)
        disp = (λ.std(dim=1) / (λ.mean(dim=1) + 1e-12)).mean().item()

        logdet = torch.logdet(G)
        # crude Laplacian via batch variance
        lap = logdet.var(unbiased=False).item()

        return {"eig_disp": disp, "lap_logdet": lap}


    # ───────────────────────── Metric helpers ────────────────────────────
    @torch.no_grad()
    def retrieveG(self, data, vi_index=0, verbose=False, T_multiplier=1.0, device=None, addStdNorm=True):
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
        device = device or self.device
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
        Push‑forward a Riemannian metric G through an invertible flow in a *vectorised*
        manner.

        Args
        ----
        z   : torch.Tensor, shape (B, D)
              Latent samples at which the metric is currently defined.
        G   : torch.Tensor, shape (B, D, D)
              Riemannian metric at the same points z.
        flow: callable
              A bijective mapping (or its inverse) that supports batched input and
              returns either an `IAFOutput` with attribute `.out` **or** the
              transformed tensor directly.

        Returns
        -------
        z_next : torch.Tensor, shape (B, D)
                 Latent samples after applying the flow.
        G_next : torch.Tensor, shape (B, D, D)
                 Corresponding pushed‑forward metric given by

                        G_next = (J^{-T}) G (J^{-1})

                 where J = ∂f/∂z is the Jacobian of the flow.
        """
        device = z.device
        z = z.clone().requires_grad_(True)      # keep autograd graph
        G = G.to(device)

        # ─── apply flow on the full batch ────────────────────────────────────────
        def flow_batch(y):
            out = flow(y)
            return out.out if hasattr(out, "out") else out

        z_next = flow_batch(z)                  # (B, D)

        # ─── Per-sample Jacobian loop (GUGUS style) ─────────────────────────────
        J = []
        for i in range(z.shape[0]):
            J_i = torch.autograd.functional.jacobian(
                lambda y: flow_batch(y.unsqueeze(0)).squeeze(0),
                z[i],
                vectorize=False
            )
            J.append(J_i)
        J = torch.stack(J, dim=0)  # shape: (B, D, D)

        # ─── push‑forward metric  G_next = J^{-T} G J^{-1}  ─────────────────────
        J_inv = torch.linalg.inv(J)             # (B, D, D)
        G_next = J_inv.transpose(-1, -2) @ G @ J_inv

        # Detach to avoid holding large graphs further up the chain
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
                    z, log_det = self._apply_flow(self.posterior_iaf_flow, z, inverse=True, context=h)
                    log_abs_det_jac_posterior += log_det

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
    def retrieveG(self, data, vi_index=0, verbose=False, T_multiplier=1.0, device=None, addStdNorm=True):

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
        device = device or self.device
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

    def _latent_repr(self, x: torch.Tensor, t: int = 0, sample: bool = False):
        """Return latent representation at time‑step *t*.

        Parameters
        ----------
        x : torch.Tensor, shape (B, n_obs, …)
            Input sequence.
        t : int, default 0
            Which observation index to encode.
        sample : bool, default False
            If True, draw a stochastic sample; otherwise use the posterior mean.

        Returns
        -------
        z : torch.Tensor, shape (B, D)
        """
        enc = self.encoder(x[:, t])
        mu, log_var = enc.embedding, enc.log_covariance
        if sample:
            z, _ = self._sample_gauss(mu, torch.exp(0.5 * log_var))
        else:
            z = mu
        if self.posterior == "iaf":
            z, _ = self._apply_flow(self.posterior_iaf_flow, z, inverse=True,
                                    context=getattr(enc, "context", None))
        return z

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
        if self.metric_mode == "empirical":
            # fall back to identity‑diag based on encoder variance at time t
            # require caller to provide x_t in kwargs else default to G0.
            # (kept simple for now)
            return self.G0.to(self.device).expand(z.shape[0], -1, -1)
        G = self.G0.to(self.device).expand(z.shape[0], -1, -1)
        if t == 0:
            return G
        for i in range(t):
            z, G = self._push_metric(z, G, self.flows[i].inverse)
        return G