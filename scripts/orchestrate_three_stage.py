#!/usr/bin/env python3
"""
Three-Stage Orchestration Script (Vanilla/RHVAE -> Metric(t0) -> ModRLVAE)
=======================================================================

Drives the full pipeline with W&B logging:
1) Stage A: Train Vanilla VAE on full data (or RHVAE via Lightning if requested)
2) Stage B: Extract metric at t=0 (centroids via KMeans + per-cluster precision)
3) Stage C: Train ModRLVAE with pretrained metric + encoder/decoder

Usage example:
  python scripts/orchestrate_three_stage.py \
    --experiment conf/experiment/global_vanilla_modrlvae_pipeline.yaml \
    --project rlvae-three-stage --run_name demo --epochs_a 5 --epochs_c 5 \
    --stage_a_model vanilla

Notes:
- This script uses existing configs: conf/config.yaml and the experiment yaml.
- It writes encoder/decoder weights and metric files to checkpoint dirs from config.
"""

import argparse
import os
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from omegaconf import OmegaConf, DictConfig
import wandb

# Ensure repo root on sys.path so 'src' package is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Repo-local imports
from data.cyclic_dataset import CyclicSpritesDataModule
from models.modular_vanilla_vae import ModularVanillaVAE
from rlvae.models.modrlvae import ModRLVAE


def load_configs(experiment_yaml: str, training_yaml: str | None = None) -> DictConfig:
    base = OmegaConf.load('conf/config.yaml')
    exp = OmegaConf.load(experiment_yaml)
    cfg = OmegaConf.merge(base, OmegaConf.create({'experiment': exp}))
    # Merge data/training defaults explicitly (base already references defaults)
    data_cfg = OmegaConf.load('conf/data/cyclic_sprites.yaml')
    train_cfg = OmegaConf.load(training_yaml or 'conf/training/full_data.yaml')
    ckpt_cfg = OmegaConf.load('conf/checkpoint/default.yaml')
    cfg.data = data_cfg
    cfg.training = train_cfg
    cfg.checkpoint = ckpt_cfg
    return cfg


def init_wandb(cfg: DictConfig, run_name: str = None, project: str = None):
    proj = project or cfg.wandb.project
    wandb.init(project=proj,
               name=run_name or cfg.experiment.name,
               config=OmegaConf.to_container(cfg, resolve=False),
               mode=cfg.wandb.mode)


def ensure_dirs(cfg: DictConfig):
    out = Path(cfg.output_dir)
    (out / 'metrics').mkdir(parents=True, exist_ok=True)
    for k in ['stageA_dir', 'stageB_dir', 'stageC_dir', 'metric_dir', 'dir']:
        p = Path(OmegaConf.to_container(cfg.checkpoint)[k])
        p.mkdir(parents=True, exist_ok=True)


def stage_a_train_vanilla(cfg: DictConfig, epochs: int, device: torch.device) -> dict:
    """Train ModularVanillaVAE on full data using only t=0 frames."""
    dm = CyclicSpritesDataModule(cfg.data)
    # Use training config for splits
    dm.setup(stage='fit', training_config=cfg.training)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    C = cfg.data.channels
    H, W = cfg.data.image_size
    latent_dim = int(OmegaConf.select(cfg, 'model.latent_dim', default=cfg.training.model.latent_dim))
    beta = float(OmegaConf.select(cfg, 'model.beta', default=cfg.training.model.beta))
    vae = ModularVanillaVAE(
        input_dim=(C, H, W),
        latent_dim=latent_dim,
        encoder_architecture='mlp',
        decoder_architecture='mlp',
        beta=beta,
        device=device,
    )
    vae.train()
    opt = torch.optim.Adam(vae.parameters(), lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay)
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for batch in train_loader:
            x = batch[:, 0].to(device)  # t=0 only
            out = vae(x)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            n += 1
        avg = total / max(n, 1)
        wandb.log({'stageA/train_loss': avg, 'epoch': epoch})
        # quick val
        with torch.no_grad():
            total_val = 0.0
            nv = 0
            for vb in val_loader:
                x = vb[:, 0].to(device)
                out = vae(x)
                total_val += float(out.loss.detach().cpu())
                nv += 1
            wandb.log({'stageA/val_loss': total_val / max(nv, 1), 'epoch': epoch})
    # Save encoder/decoder
    stageA_dir = Path(cfg.checkpoint.stageA_dir)
    enc_path = stageA_dir / 'encoder.pt'
    dec_path = stageA_dir / 'decoder.pt'
    torch.save(vae.encoder_manager.state_dict(), enc_path)
    torch.save(vae.decoder_manager.state_dict(), dec_path)
    wandb.save(str(enc_path))
    wandb.save(str(dec_path))
    return {'encoder_path': str(enc_path), 'decoder_path': str(dec_path)}


def stage_a_train_rhvae(cfg: DictConfig, epochs: int, device: torch.device) -> dict:
    """Train RiemannianFlowVAE (RHVAE-like) on t=0 only using Lightning."""
    # Build runtime model config
    C = cfg.data.channels
    H, W = cfg.data.image_size
    model_cfg = {
        '_target_': 'rlvae.models.riemannian_flow_vae.RiemannianFlowVAE',
        'input_dim': [C, H, W],
        'latent_dim': int(cfg.model.latent_dim),
        'n_flows': 0,  # T=1
        'flow_hidden_size': int(cfg.model.get('flow_hidden_size', 64)),
        'flow_n_blocks': int(cfg.model.get('flow_n_blocks', 2)),
        'flow_n_hidden': int(cfg.model.get('flow_n_hidden', 1)),
        'epsilon': float(cfg.model.get('epsilon', 1e-6)),
        'beta': float(cfg.model.beta),
        'riemannian_beta': float(cfg.model.riemannian_beta),
        'posterior_type': 'riemannian_metric',
        'loop_mode': 'open',
        # KL / posterior controls
        'kl_use_metric_normalization': True,
        'kl_metric_norm_mode': 'geomean',
        'posterior_local_alpha': 0.5,
        'kl_amp_safe': True,
        'use_curvature_correction': True,
    }
    runtime_cfg = OmegaConf.create({'model': model_cfg, 'training': cfg.training, 'data': cfg.data, 'visualization': cfg.visualization, 'evaluation': {'enabled': False}})

    # DataModule
    dm = CyclicSpritesDataModule(cfg.data)
    dm.setup(stage='fit', training_config=cfg.training)
    # Wrap dataloaders to slice t=0 and keep T=1 dims
    class T0Dataset(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            x = self.base[idx]
            return x[:1]  # [1,C,H,W]
    train_loader = DataLoader(T0Dataset(dm.train_dataset), batch_size=cfg.training.data.batch_size, shuffle=True, num_workers=cfg.training.data.num_workers, pin_memory=cfg.training.data.pin_memory)
    val_loader = DataLoader(T0Dataset(dm.val_dataset), batch_size=cfg.training.data.batch_size, shuffle=False, num_workers=cfg.training.data.num_workers, pin_memory=cfg.training.data.pin_memory)

    # Late import to avoid heavy import chain unless needed
    from rlvae.training.lightning_trainer import LightningRlVAETrainer
    lit = LightningRlVAETrainer(runtime_cfg)
    trainer_args = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    trainer = L.Trainer(max_epochs=epochs, **trainer_args)
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save encoder/decoder from trained model
    stageA_dir = Path(cfg.checkpoint.stageA_dir)
    enc_path = stageA_dir / 'encoder.pt'
    dec_path = stageA_dir / 'decoder.pt'
    try:
        torch.save(lit.model.encoder.state_dict(), enc_path)
        torch.save(lit.model.decoder.state_dict(), dec_path)
    except Exception:
        # Fallback to manager state when available
        try:
            torch.save(lit.model.encoder_manager.state_dict(), enc_path)
            torch.save(lit.model.decoder_manager.state_dict(), dec_path)
        except Exception:
            pass
    wandb.save(str(enc_path))
    wandb.save(str(dec_path))
    return {'encoder_path': str(enc_path), 'decoder_path': str(dec_path), 'model': lit.model}


def stage_b_dump_metric_from_model(cfg: DictConfig, model) -> str | None:
    """Dump centroids/M from a trained RHVAE-like model if available; returns path or None."""
    try:
        centroids = getattr(model, 'centroids_tens', None)
        M = getattr(model, 'M_tens', None)
        temperature = getattr(model, 'temperature', torch.tensor(0.7)).item() if hasattr(model, 'temperature') else 0.7
        regularization = getattr(model, 'lbd', torch.tensor(0.01)).item() if hasattr(model, 'lbd') else 0.01
        if centroids is None or M is None:
            print("[Stage B] No centroids/M on model; falling back to KMeans extraction.")
            return None
        # Basic sanity
        if centroids.ndim != 2 or M.ndim != 3 or centroids.shape[0] != M.shape[0]:
            print("[Stage B] Invalid centroids/M shapes; falling back to KMeans extraction.")
            return None
        metric = {
            'centroids': centroids.detach().cpu(),
            'M_matrices': M.detach().cpu(),
            'temperature': float(temperature),
            'regularization': float(regularization),
        }
        metric_dir = Path(cfg.checkpoint.metric_dir)
        metric_path = metric_dir / 'metric_rhvae_t0.pt'
        torch.save(metric, metric_path)
        wandb.save(str(metric_path))
        wandb.log({'stageB/dumped_from_rhvae': 1, 'stageB/n_centroids': centroids.shape[0]})
        return str(metric_path)
    except Exception as e:
        print(f"[Stage B] Metric dump from RHVAE failed: {e}")
        return None

def stage_b_extract_metric(cfg: DictConfig, vae_enc_path: str, device: torch.device) -> str:
    """Extract metric at t=0: KMeans centroids + per-cluster precision matrices."""
    from sklearn.cluster import KMeans
    dm = CyclicSpritesDataModule(cfg.data)
    dm.setup(stage='fit', training_config=cfg.training)
    loader = dm.train_dataloader()
    # Rebuild encoder only
    C = cfg.data.channels
    H, W = cfg.data.image_size
    latent_dim = int(OmegaConf.select(cfg, 'model.latent_dim', default=cfg.training.model.latent_dim))
    tmp_vae = ModularVanillaVAE(input_dim=(C, H, W), latent_dim=latent_dim)
    tmp_vae.encoder_manager.load_pretrained(vae_enc_path)
    tmp_vae.to(device)
    tmp_vae.eval()
    # Collect mu at t=0
    mus = []
    with torch.no_grad():
        for batch in loader:
            x0 = batch[:, 0].to(device)
            enc_out = tmp_vae.encoder(x0)
            mu = enc_out.embedding.detach().cpu()
            mus.append(mu)
    mu_all = torch.cat(mus, dim=0)
    n_centroids = int(OmegaConf.select(cfg, 'experiment.stage_b.n_centroids', default=50))
    temperature = float(OmegaConf.select(cfg, 'experiment.stage_b.temperature', default=0.7))
    regularization = float(OmegaConf.select(cfg, 'experiment.stage_b.regularization', default=0.01))
    kmeans = KMeans(n_clusters=n_centroids, random_state=42, n_init=10)
    labels = kmeans.fit_predict(mu_all.numpy())
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    # Per-cluster precision (inverse covariance)
    D = mu_all.shape[1]
    M_mats = []
    for k in range(n_centroids):
        idx = (labels == k)
        pts = mu_all[idx]
        if pts.shape[0] < 2:
            M_mats.append(torch.eye(D))
            continue
        diffs = pts - pts.mean(dim=0, keepdim=True)
        cov = (diffs.T @ diffs) / max(1, (pts.shape[0] - 1))
        cov = cov + regularization * torch.eye(D)
        try:
            M = torch.linalg.inv(cov)
        except Exception:
            M = torch.eye(D)
        M_mats.append(M)
    M_mats = torch.stack(M_mats, dim=0)
    metric = {
        'centroids': centroids,
        'M_matrices': M_mats,
        'temperature': temperature,
        'regularization': regularization,
    }
    metric_dir = Path(cfg.checkpoint.metric_dir)
    metric_path = metric_dir / 'metric_t0.pt'
    torch.save(metric, metric_path)
    wandb.save(str(metric_path))
    wandb.log({'stageB/n_centroids': n_centroids})
    return str(metric_path)


def stage_c_train_modrlvae(cfg: DictConfig, encoder_path: str, decoder_path: str, metric_path: str, epochs: int, device: torch.device):
    # Build model config for ModRLVAE
    C = cfg.data.channels
    H, W = cfg.data.image_size
    seq_len = int(cfg.data.sequence_length)
    model_cfg = {
        '_target_': 'rlvae.models.modrlvae.ModRLVAE',
        'input_dim': [C, H, W],
        'latent_dim': int(OmegaConf.select(cfg, 'model.latent_dim', default=cfg.training.model.latent_dim)),
        'sequence_length': seq_len,
        'posterior_type': 'riemannian_metric',
        'beta': float(OmegaConf.select(cfg, 'model.beta', default=cfg.training.model.beta)),
        'riemannian_beta': float(OmegaConf.select(cfg, 'model.riemannian_beta', default=cfg.training.model.riemannian_beta)),
        'loop': {'mode': 'open', 'penalty': 0.0},
        'encoder': {'architecture': 'mlp'},
        'decoder': {'architecture': 'mlp'},
        'flow_hidden_size': int(OmegaConf.select(cfg, 'model.flow_hidden_size', default=64)),
        'flow_n_blocks': int(OmegaConf.select(cfg, 'model.flow_n_blocks', default=2)),
        'flow_n_hidden': int(OmegaConf.select(cfg, 'model.flow_n_hidden', default=1)),
        'metric': {
            'trainable': False,
            'temperature_override': 0.1,
            'regularization_override': 0.01,
        },
        'kl_use_metric_normalization': True,
        'kl_metric_norm_mode': 'geomean',
        'use_curvature_correction': True,
        'pretrained': {
            'encoder_path': encoder_path,
            'decoder_path': decoder_path,
            'metric_path': metric_path,
        },
        'sampling': {
            'method': OmegaConf.select(cfg, 'experiment.stage_c.sampling.method', default='enhanced')
        }
    }
    # Manual training loop to avoid heavy Lightning imports in this script
    dm = CyclicSpritesDataModule(cfg.data)
    dm.setup(stage='fit', training_config=cfg.training)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    # Instantiate model directly
    model = ModRLVAE(OmegaConf.create(model_cfg))
    compute_device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay)
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            x = batch.to(compute_device)
            out = model(x)
            loss = out['loss'] if 'loss' in out else out['total_loss']
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            n += 1
        wandb.log({'stageC/train_loss': total / max(n, 1), 'epoch': epoch})
        # quick val
        model.eval()
        with torch.no_grad():
            total_val, nv = 0.0, 0
            for vb in val_loader:
                x = vb.to(compute_device)
                out = model(x)
                vloss = out['loss'] if 'loss' in out else out['total_loss']
                total_val += float(vloss.detach().cpu())
                nv += 1
            wandb.log({'stageC/val_loss': total_val / max(nv, 1), 'epoch': epoch})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='conf/experiment/global_vanilla_modrlvae_pipeline.yaml')
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--epochs_a', type=int, default=10)
    parser.add_argument('--epochs_c', type=int, default=50)
    parser.add_argument('--stage_a_model', type=str, choices=['vanilla', 'vanilla_vae', 'rhvae'], default=None,
                        help='Override Stage A model without editing YAML: vanilla|rhvae')
    parser.add_argument('--training_conf', type=str, default=None,
                        help='Override training config YAML (e.g., conf/training/quick.yaml)')
    parser.add_argument('--stage_c_lightning', action='store_true', help='Use Lightning trainer for Stage C (visualizations & logs)')
    args = parser.parse_args()

    cfg = load_configs(args.experiment, training_yaml=args.training_conf)
    # Optional override of Stage A model from CLI
    if args.stage_a_model is not None:
        override = 'vanilla_vae' if args.stage_a_model in ('vanilla', 'vanilla_vae') else 'rhvae'
        try:
            if not hasattr(cfg, 'experiment'):
                cfg.experiment = OmegaConf.create({})
            if not hasattr(cfg.experiment, 'stage_a'):
                cfg.experiment.stage_a = OmegaConf.create({})
            cfg.experiment.stage_a.model = override
        except Exception:
            OmegaConf.update(cfg, 'experiment.stage_a.model', override, force_add=True)
    init_wandb(cfg, run_name=args.run_name, project=args.project)
    ensure_dirs(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Stage A
    stage_a_model = OmegaConf.select(cfg, 'experiment.stage_a.model', default='vanilla_vae')
    art = {}
    if stage_a_model == 'vanilla_vae':
        wandb.log({'stage': 'A', 'model': 'vanilla_vae'})
        art = stage_a_train_vanilla(cfg, epochs=args.epochs_a, device=device)
        # Stage B: KMeans+precision at t=0
        metric_path = stage_b_extract_metric(cfg, art.get('encoder_path'), device=device)
    else:
        wandb.log({'stage': 'A', 'model': 'rhvae_t0'})
        art = stage_a_train_rhvae(cfg, epochs=args.epochs_a, device=device)
        # Stage B: Try dumping metric directly from RHVAE; fallback to KMeans if unavailable
        metric_path = stage_b_dump_metric_from_model(cfg, art.get('model'))
        if metric_path is None:
            metric_path = stage_b_extract_metric(cfg, art.get('encoder_path'), device=device)

    # Stage C
    wandb.log({'stage': 'C', 'model': 'modrlvae'})
    if args.stage_c_lightning:
        # Build runtime cfg and use Lightning trainer for visuals + W&B
        from rlvae.training.lightning_trainer import LightningRlVAETrainer
        C = cfg.data.channels
        H, W = cfg.data.image_size
        seq_len = int(cfg.data.sequence_length)
        model_cfg = {
            '_target_': 'rlvae.models.modrlvae.ModRLVAE',
            'input_dim': [C, H, W],
            'latent_dim': int(OmegaConf.select(cfg, 'model.latent_dim', default=cfg.training.model.latent_dim)),
            'sequence_length': seq_len,
            'posterior_type': 'riemannian_metric',
            'beta': float(OmegaConf.select(cfg, 'model.beta', default=cfg.training.model.beta)),
            'riemannian_beta': float(OmegaConf.select(cfg, 'model.riemannian_beta', default=cfg.training.model.riemannian_beta)),
            'loop': {'mode': 'open', 'penalty': 0.0},
            'encoder': {'architecture': 'mlp'},
            'decoder': {'architecture': 'mlp'},
            'flow_hidden_size': int(OmegaConf.select(cfg, 'model.flow_hidden_size', default=64)),
            'flow_n_blocks': int(OmegaConf.select(cfg, 'model.flow_n_blocks', default=2)),
            'flow_n_hidden': int(OmegaConf.select(cfg, 'model.flow_n_hidden', default=1)),
            'metric': {
                'trainable': False,
                'temperature_override': 0.1,
                'regularization_override': 0.01,
            },
            'kl_use_metric_normalization': True,
            'kl_metric_norm_mode': 'geomean',
            'use_curvature_correction': True,
            'pretrained': {
                'encoder_path': art.get('encoder_path'),
                'decoder_path': art.get('decoder_path'),
                'metric_path': metric_path,
            },
            'sampling': {
                'method': OmegaConf.select(cfg, 'experiment.stage_c.sampling.method', default='enhanced')
            }
        }
        vis_cfg = cfg.visualization if hasattr(cfg, 'visualization') else OmegaConf.create({'level': 'standard', 'frequency': 10})
        runtime_cfg = OmegaConf.create({'model': model_cfg, 'training': cfg.training, 'data': cfg.data, 'visualization': vis_cfg, 'evaluation': {'enabled': True}})
        dm = CyclicSpritesDataModule(cfg.data)
        dm.setup(stage='fit', training_config=cfg.training)
        lit = LightningRlVAETrainer(runtime_cfg, data_module=dm)
        trainer_args = dict(OmegaConf.to_container(cfg.training.trainer, resolve=True))
        trainer_args['max_epochs'] = args.epochs_c
        trainer = L.Trainer(**trainer_args)
        trainer.fit(lit, datamodule=dm)
    else:
        stage_c_train_modrlvae(cfg, art.get('encoder_path'), art.get('decoder_path'), metric_path, epochs=args.epochs_c, device=device)

    wandb.finish()


if __name__ == '__main__':
    main()
