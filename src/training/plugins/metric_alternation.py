"""
Metric Alternation Plugin (Stage C)
===================================

Extracts the Stage C logic (alternating metric-only vs RLVAE epochs) from the
Lightning module into a reusable helper class.

Responsibilities:
- Decide metric-only vs RLVAE epochs given config
- Freeze/unfreeze parameter subsets
- Build and refresh anchor sets
- Provide metric-only loss computation
"""

from __future__ import annotations

from typing import Optional
import torch
import random


class MetricAlternationPlugin:
    def __init__(self, trainer: "LightningRlVAETrainer"):
        self.trainer = trainer
        cfg = getattr(trainer.config, 'training', None)
        self.cfg = cfg.metric_alternation if (cfg and hasattr(cfg, 'metric_alternation')) else None
        # Derived flags
        self.enabled = bool(self.cfg and getattr(self.cfg, 'enabled', False))
        self.warmup_epochs = int(getattr(self.cfg, 'warmup_epochs', 5)) if self.enabled else 0
        self.k_rlvae_epochs = int(getattr(self.cfg, 'k_rlvae_epochs', 3)) if self.enabled else 0
        self.metric_step_epochs = int(getattr(self.cfg, 'metric_step_epochs', 1)) if self.enabled else 0
        self.logdet_clip = float(getattr(self.cfg, 'logdet_clip', 50.0)) if self.enabled else 50.0
        self.anchor_size = int(getattr(self.cfg, 'anchor_size', 20000)) if self.enabled else 0
        self.anchor_refresh_frac = float(getattr(self.cfg, 'anchor_refresh_frac', 0.1)) if self.enabled else 0.1
        self.consistency_weight = float(getattr(self.cfg, 'consistency_weight', 0.0)) if self.enabled else 0.0
        # State
        self.metric_only_epoch = False
        self._anchors_z: Optional[torch.Tensor] = None  # CPU tensor

    # ---------------- scheduling ----------------
    def on_train_epoch_start(self):
        if not self.enabled:
            self.metric_only_epoch = False
            return
        e = self.trainer.current_epoch
        if e < self.warmup_epochs:
            self.metric_only_epoch = False
            self._freeze_for_rlvae_step()
            self._wandb_phase('warmup_rlvae', e)
            return
        period = self.k_rlvae_epochs + self.metric_step_epochs
        in_metric_block = ((e - self.warmup_epochs) % period) >= self.k_rlvae_epochs
        # Only if metric is trainable/available
        metric_net = getattr(self.trainer.model, 'modular_metric', None)
        can_train_metric = (
            metric_net is not None and
            getattr(metric_net, '_is_loaded', True) and
            getattr(metric_net, 'trainable', False) and
            len(list(metric_net.parameters())) > 0
        )
        self.metric_only_epoch = (in_metric_block and can_train_metric)
        if self.metric_only_epoch:
            self._freeze_for_metric_step()
            self._ensure_anchor_set()
            self._refresh_anchor_subset()
            self._wandb_phase('metric', e)
        else:
            self._freeze_for_rlvae_step()
            self._wandb_phase('rlvae', e)

    # ---------------- model freezing ----------------
    def _set_requires_grad(self, module: torch.nn.Module, requires: bool) -> None:
        for p in module.parameters(recurse=True):
            p.requires_grad = requires

    def _freeze_for_metric_step(self) -> None:
        model = self.trainer.model
        # Freeze everything
        self._set_requires_grad(model, False)
        # Unfreeze metric network
        metric_net = getattr(model, 'modular_metric', None)
        if metric_net is not None:
            self._set_requires_grad(metric_net, True)

    def _freeze_for_rlvae_step(self) -> None:
        model = self.trainer.model
        # Unfreeze all
        self._set_requires_grad(model, True)
        # Keep metric frozen if present
        metric_net = getattr(model, 'modular_metric', None)
        if metric_net is not None:
            self._set_requires_grad(metric_net, False)

    # ---------------- anchors management ----------------
    @torch.no_grad()
    def _collect_anchor_samples(self, max_needed: int) -> torch.Tensor:
        dm = self.trainer.data_module
        assert dm is not None, "Data module required to build anchor set"
        loader = dm.train_dataloader()
        collected = []
        device = self.trainer.device
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            elif isinstance(batch, dict):
                x = batch.get('x', next(iter(batch.values())))
            else:
                x = batch
            x = x.to(device)
            x0 = x[:, 0]
            enc_out = self.trainer.model.encoder(x0)
            mu = enc_out.embedding.detach().cpu()
            collected.append(mu)
            if sum(t.shape[0] for t in collected) >= max_needed:
                break
        if not collected:
            raise RuntimeError("Failed to collect anchors: empty dataset?")
        return torch.cat(collected, dim=0)[:max_needed]

    def _ensure_anchor_set(self) -> None:
        if self._anchors_z is None or self._anchors_z.shape[0] < self.anchor_size:
            anchors = self._collect_anchor_samples(self.anchor_size)
            self._anchors_z = anchors

    def _refresh_anchor_subset(self) -> None:
        if self._anchors_z is None:
            return
        refresh_n = max(1, int(self.anchor_size * self.anchor_refresh_frac))
        pool_size = self._anchors_z.shape[0]
        refresh_n = min(refresh_n, pool_size)
        new_samples = self._collect_anchor_samples(refresh_n)
        if new_samples.shape[0] > refresh_n:
            new_samples = new_samples[:refresh_n]
        idxs = random.sample(range(pool_size), k=refresh_n)
        self._anchors_z[idxs] = new_samples

    # ---------------- losses ----------------
    @torch.no_grad()
    def _prepare_batch_z0(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0]
        enc_out = self.trainer.model.encoder(x0)
        mu = enc_out.embedding
        log_var = enc_out.log_covariance if hasattr(enc_out, 'log_covariance') else torch.zeros_like(mu)
        try:
            z0 = self.trainer.model.sample_metric_aware_posterior(mu, log_var)
        except Exception:
            eps = torch.randn_like(mu)
            z0 = mu + eps * torch.exp(0.5 * log_var)
        return z0.detach()

    def metric_only_loss(self, x: torch.Tensor) -> torch.Tensor:
        model = self.trainer.model
        metric_net = getattr(model, 'modular_metric', None)
        if (
            metric_net is None or
            not getattr(metric_net, 'trainable', False) or
            len(list(metric_net.parameters())) == 0
        ):
            return torch.zeros((), device=self.trainer.device, requires_grad=True)

        # Prepare z batch
        with torch.no_grad():
            z_batch = self._prepare_batch_z0(x)

        # Ensure anchors
        self._ensure_anchor_set()
        anchors = self._anchors_z.to(self.trainer.device)

        # Use the trainable metric network when available so gradients flow to ψ
        has_net = hasattr(metric_net, 'metric_net') and any(p.requires_grad for p in metric_net.metric_net.parameters())
        if has_net:
            G_b = metric_net.metric_net(z_batch)                 # [B, D, D]
            sign_b, logdetG_b = torch.linalg.slogdet(G_b)
            logdetG_b = torch.clamp(logdetG_b, min=-self.logdet_clip, max=self.logdet_clip)
            d_b = -0.5 * logdetG_b                               # since log det G^{-1} = - log det G

            # Anchors
            G_a = metric_net.metric_net(anchors)
            sign_a, logdetG_a = torch.linalg.slogdet(G_a)
            logdetG_a = torch.clamp(logdetG_a, min=-self.logdet_clip, max=self.logdet_clip)
            d_a = -0.5 * logdetG_a
        else:
            # Fallback to fixed metric buffers (no gradient to ψ)
            Ginv_b = model.modular_metric.compute_inverse_metric(z_batch)
            _, logdet_b = torch.linalg.slogdet(Ginv_b)
            logdet_b = torch.clamp(logdet_b, min=-self.logdet_clip, max=self.logdet_clip)
            d_b = 0.5 * logdet_b

            Ginv_a = model.modular_metric.compute_inverse_metric(anchors)
            _, logdet_a = torch.linalg.slogdet(Ginv_a)
            logdet_a = torch.clamp(logdet_a, min=-self.logdet_clip, max=self.logdet_clip)
            d_a = 0.5 * logdet_a

        logZ = torch.logsumexp(d_a, dim=0) - torch.log(torch.tensor(float(d_a.shape[0]), device=self.trainer.device))
        loss = -(d_b.mean()) + logZ
        if self.consistency_weight > 0.0:
            loss = loss + 0.0 * self.consistency_weight
        return loss

    # ---------------- utils ----------------
    def _wandb_phase(self, name: str, epoch: int):
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"phase": name, "epoch": epoch})
        except Exception:
            pass
