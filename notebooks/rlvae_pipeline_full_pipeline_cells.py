# %% [markdown]
# # RlVAE Full Pipeline (Copy-Paste Cells)
# This script is structured as Jupyter-friendly cells. Copy sections into your notebook as needed.
# It covers robust path setup, Hydra config, data diagnostics, model build, metric loading/diagnostics,
# sampling, latent-space visualization, forward/recon checks, optional short training, and generation/interpolation.

# %%
# Path setup (run first in the notebook)
import os, sys
from pathlib import Path

CWD = Path.cwd()
if (CWD / 'src').exists():
    ROOT = CWD
elif (CWD.parent / 'src').exists():
    ROOT = CWD.parent
else:
    ROOT = CWD
    for _ in range(6):
        if (ROOT / 'src').exists():
            break
        ROOT = ROOT.parent
SRC = ROOT / 'src'
LIB_SRC = SRC / 'lib' / 'src'
for p in [ROOT, SRC, LIB_SRC]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ['PYTHONPATH'] = os.pathsep.join([s for s in [os.environ.get('PYTHONPATH',''), str(ROOT), str(SRC), str(LIB_SRC)] if s])
print('Resolved ROOT:', ROOT)
print('sys.path[0:5]:', sys.path[:5])

# %% [markdown]
# ## 0. Environment & Imports

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from models.modular_rlvae import ModularRiemannianFlowVAE
from training.lightning_trainer import LightningRlVAETrainer
from data.cyclic_dataset import CyclicSpritesDataModule
from visualizations.manager import VisualizationManager, VisualizationLevel

print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# %% [markdown]
# ## 1. Hydra Configuration (robust absolute config dir)
# Compose the main config with safe absolute `conf/` resolution. Adjust overrides as needed.

# %%
import hydra
from hydra import compose, initialize_config_dir

CONF_DIR_ABS = str((ROOT / 'conf').resolve())
print('Using CONF_DIR:', CONF_DIR_ABS)

base_overrides = [
    'experiment=single_run',
    'training=quick',
    'visualization=standard',
]

with initialize_config_dir(version_base=None, config_dir=CONF_DIR_ABS):
    cfg = compose(config_name='config', overrides=base_overrides)

print(OmegaConf.to_yaml(cfg)[:800])

# %% [markdown]
# ## 2. Data Diagnostics
# Instantiate the data module; show example batches and shapes.

# %%
seed = cfg.get('seed', 42)
torch.manual_seed(seed)
np.random.seed(seed)

# Build data module
# Batch size is configured under training.data.batch_size (not in data config)
bs = int(getattr(getattr(cfg.training, 'data', {}), 'batch_size', 32))
print('Using batch size:', bs)

data_module = CyclicSpritesDataModule(cfg.data)
# Pass training_config so DataModule can pick up batch size/workers
data_module.setup(stage='fit', training_config=cfg.training)
train_loader = data_module.train_dataloader()
batch = next(iter(train_loader))

# Normalize batch to an images tensor [B,T,C,H,W]
if isinstance(batch, dict):
    images = batch.get('images', next(iter(batch.values())))
elif torch.is_tensor(batch):
    images = batch
else:
    raise TypeError(f'Unexpected batch type: {type(batch)}')

print('Images shape:', tuple(images.shape))

# Visualize first few frames of first sequence
imgs = images  # [B, T, C, H, W]
B, T, C, H, W = imgs.shape
n_show = min(8, B)
fig, axes = plt.subplots(n_show, min(T, 8), figsize=(min(T,8)*2, n_show*2))
axes = np.array(axes).reshape(n_show, -1)
for i in range(n_show):
    for t in range(axes.shape[1]):
        axes[i, t].imshow(imgs[i, t].permute(1,2,0).numpy(), cmap='gray')
        axes[i, t].axis('off')
plt.suptitle('Sample sequences (train)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Build Model
# Create `ModularRiemannianFlowVAE` and trainer wrapper.

# %%
# Sync model input dims from data if needed
if 'input_dim' in cfg.model and (cfg.model.input_dim is None or cfg.model.input_dim == 0):
    C = cfg.data.channels
    img_size = cfg.data.image_size
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        H, W = int(img_size[0]), int(img_size[1])
    else:
        H = W = int(img_size)
    cfg.model.input_dim = int(C) * int(H) * int(W)

cfg.model.sequence_length = getattr(cfg.data, 'sequence_length', getattr(cfg.model, 'sequence_length', 8))
if 'n_flows' in cfg.model and (cfg.model.n_flows is None or cfg.model.n_flows <= 0):
    cfg.model.n_flows = cfg.model.sequence_length - 1

model = ModularRiemannianFlowVAE(cfg.model)
trainer_wrapper = LightningRlVAETrainer(model, cfg.training)
model.eval()
model_device = next(model.parameters()).device
print('Model device:', model_device)

# %% [markdown]
# ## 4. Metric Tensor Diagnostics (auto-load if available)
# Attempts to load a fixed metric from `cfg.model.pretrained.metric_path` if the model metric indicates it’s not loaded.

# %%
from models.components.metric_tensor import MetricTensor

# Locate metric on model
metric = None
for name in ['modular_metric', 'metric', 'metric_tensor', 'G']:
    if hasattr(model, name):
        m = getattr(model, name)
        if all(hasattr(m, fn) for fn in ['compute_inverse_metric', 'compute_metric', 'compute_log_det_metric']):
            metric = m
            print(f'Using metric: model.{name}')
            break

# Load if needed
temp_metric = None
if metric is not None:
    is_loaded = getattr(metric, '_is_loaded', True)
    if not is_loaded and hasattr(metric, 'load_pretrained'):
        metric_path = None
        if 'pretrained' in cfg.model and 'metric_path' in cfg.model.pretrained:
            metric_path = cfg.model.pretrained.metric_path
        elif 'metric' in cfg.model and 'path' in cfg.model.metric:
            metric_path = str((ROOT / 'data' / 'pretrained' / cfg.model.metric.path))
        if metric_path is not None:
            from torch.serialization import add_safe_globals
            import numpy as _np
            add_safe_globals([_np.core.multiarray._reconstruct])
            try:
                state = torch.load(metric_path, map_location=model_device)
            except Exception:
                state = torch.load(metric_path, map_location=model_device, weights_only=False)
            centroids = state.get('centroids')
            matrices = state.get('metric_matrices') or state.get('M_matrices')
            temperature = state.get('temperature', None)
            regularization = state.get('regularization', None)
            if centroids is not None and matrices is not None:
                if centroids.shape[1] != getattr(metric, 'latent_dim', centroids.shape[1]):
                    print('Metric file latent_dim differs; using temporary metric for diagnostics.')
                    temp_metric = MetricTensor(latent_dim=centroids.shape[1], device=model_device)
                    temp_metric.load_pretrained(centroids, matrices, temperature, regularization)
                else:
                    metric.load_pretrained(centroids, matrices, temperature, regularization)
            else:
                print('Metric file missing centroids or metric_matrices; skipping load.')

metric_module = temp_metric if temp_metric is not None else metric
if metric_module is None:
    print('No metric found; skipping diagnostics.')
else:
    latent_dim = getattr(metric_module, 'latent_dim', cfg.model.latent_dim)
    z = torch.randn(16, latent_dim, device=model_device)
    G_inv = metric_module.compute_inverse_metric(z).detach().cpu()
    G = metric_module.compute_metric(z).detach().cpu()
    logdetG = metric_module.compute_log_det_metric(z).detach().cpu()
    print('G_inv:', tuple(G_inv.shape), 'G:', tuple(G.shape), 'log|G|:', tuple(logdetG.shape))
    # Eigenvalue summary
    vals = []
    for i in range(G.shape[0]):
        vals.append(np.linalg.eigvalsh(G[i].numpy()))
    vals = np.array(vals)
    print('Eigenvalues: min', vals.min(), 'max', vals.max())
    plt.figure(figsize=(5,3))
    plt.hist(logdetG.numpy(), bins=20)
    plt.title('Distribution of log|G| over random z')
    plt.show()

# %% [markdown]
# ## 5. Latent Space Sampling & Visualization
# Draw prior samples, encode sequences to latent, show 2D/3D projections.

# %%
from sklearn.decomposition import PCA

def visualize_latents(latents: torch.Tensor, title: str = 'Latent space (PCA)'):
    Z = latents.detach().cpu().numpy()
    if Z.shape[1] > 2:
        pca = PCA(n_components=2)
        Z2 = pca.fit_transform(Z)
    else:
        Z2 = Z
    plt.figure(figsize=(5,4))
    plt.scatter(Z2[:,0], Z2[:,1], s=8, alpha=0.7)
    plt.title(title)
    plt.xlabel('PC1' if Z.shape[1] > 2 else 'z1')
    plt.ylabel('PC2' if Z.shape[1] > 2 else 'z2')
    plt.tight_layout()
    plt.show()

# Prior samples
latent_dim = cfg.model.latent_dim
z_prior = torch.randn(512, latent_dim, device=model_device)
visualize_latents(z_prior, 'Prior samples (PCA)')

# Encode a batch of sequences
model.eval()
with torch.no_grad():
    batch = next(iter(train_loader))
    if isinstance(batch, dict):
        images = batch.get('images', next(iter(batch.values())))
    elif torch.is_tensor(batch):
        images = batch
    else:
        raise TypeError(f'Unexpected batch type: {type(batch)}')
    images = images.to(model_device)
    result = model(images)
    z_enc = result.get('latent_samples', None)
    if z_enc is None:
        # Fallback: try mean if provided
        z_enc = result.get('z', None)
    if isinstance(z_enc, torch.Tensor):
        Zflat = z_enc.reshape(-1, z_enc.shape[-1])
        visualize_latents(Zflat, 'Encoded latents (PCA)')
    else:
        print('No latent_samples found in model output.')

# %% [markdown]
# ## 6. Flow Diagnostics & Sampling
# Sample sequences from the latent flow (if available) and visualize trajectories in latent PCA.

# %%
flows = getattr(model, 'flow_manager', None)
if flows is None or not hasattr(flows, 'sample_sequence'):
    print('No flow_manager or sampling API not found; skipping flow sampling.')
else:
    with torch.no_grad():
        seq_len = cfg.model.sequence_length
        z0 = torch.randn(64, latent_dim, device=model_device)
        z_seq = flows.sample_sequence(z0, num_steps=seq_len-1)  # expect [B,T,d] or similar
        if isinstance(z_seq, torch.Tensor):
            Zflat = z_seq.reshape(-1, z_seq.shape[-1])
            visualize_latents(Zflat, 'Flow-sampled latent trajectories (PCA)')
        else:
            print('Unexpected flow sample type:', type(z_seq))

# %% [markdown]
# ## 7. RHMC Sampler Hooks (if enabled)
# If RHMC or Riemannian samplers are available, run brief posterior samples to assess stability.

# %%
samplers = getattr(model, 'samplers', None)
if samplers is None:
    print('No samplers attached to model; skipping RHMC checks.')
else:
    if hasattr(samplers, 'sample_posterior_latents'):
        with torch.no_grad():
            batch = next(iter(train_loader))
            if isinstance(batch, dict):
                images = batch.get('images', next(iter(batch.values())))
            elif torch.is_tensor(batch):
                images = batch
            else:
                raise TypeError(f'Unexpected batch type: {type(batch)}')
            images = images.to(model_device)
            post = samplers.sample_posterior_latents(images, num_samples=64)
            if isinstance(post, torch.Tensor):
                visualize_latents(post.reshape(-1, post.shape[-1]), 'Posterior latent samples (PCA)')
            else:
                print('Unexpected posterior sample type:', type(post))
    else:
        print('No sample_posterior_latents on samplers; skipping.')

# %% [markdown]
# ## 8. Forward Pass & Reconstruction Sanity
# Check that the model can forward and reconstruct a batch; visualize reconstructions.

# %%
model.eval()
with torch.no_grad():
    batch = next(iter(train_loader))
    if isinstance(batch, dict):
        images = batch.get('images', next(iter(batch.values())))
    elif torch.is_tensor(batch):
        images = batch
    else:
        raise TypeError(f'Unexpected batch type: {type(batch)}')
    images = images.to(model_device)
    result = model(images)
    recon = result.get('recon', result.get('reconstruction', None))
    loss = result.get('total_loss', result.get('loss', None))
    print('Forward keys:', list(result.keys()))
    print('Loss:', loss)

    if isinstance(recon, torch.Tensor):
        B, T, C, H, W = recon.shape
        n_show = min(4, B)
        fig, axes = plt.subplots(n_show, T, figsize=(T*2, n_show*2))
        axes = np.array(axes).reshape(n_show, T)
        for i in range(n_show):
            for t in range(T):
                axes[i, t].imshow(recon[i, t].detach().cpu().permute(1,2,0).numpy(), cmap='gray')
                axes[i, t].axis('off')
        plt.suptitle('Reconstructions')
        plt.tight_layout()
        plt.show()
    else:
        print('No recon tensor in model output; skipping visualization.')

# %% [markdown]
# ## 9. Optional Short Training (smoke test)
# Train for a few steps to verify losses decrease; keep this short.

# %%
short_cfg = cfg.copy()
short_cfg.training.max_epochs = 1
short_cfg.training.limit_train_batches = 5
short_cfg.training.limit_val_batches = 2
short_model = ModularRiemannianFlowVAE(short_cfg.model)
short_trainer = LightningRlVAETrainer(short_model, short_cfg.training)

print('Starting short training run (1 epoch, few batches)...')
try:
    short_trainer.fit(data_module)
    print('Short training completed.')
except Exception as e:
    print('Short training skipped or failed gracefully:', e)

# %% [markdown]
# ## 10. Generation & Interpolation
# Use the generator/evaluator if available; otherwise do a manual interpolation in latent space.

# %%
# Manual latent interpolation
with torch.no_grad():
    z_a = torch.randn(1, latent_dim, device=model_device)
    z_b = torch.randn(1, latent_dim, device=model_device)
    alphas = torch.linspace(0, 1, steps=10, device=model_device)
    z_interp = torch.stack([(1-a)*z_a + a*z_b for a in alphas], dim=1)  # [1, T, d]

    # If decoder supports decode_sequence, prefer that; else decode per frame
    decode = getattr(model, 'decode_sequence', None)
    if callable(decode):
        imgs_interp = model.decode_sequence(z_interp)  # [1, T, C, H, W]
    else:
        imgs_list = []
        for t in range(z_interp.shape[1]):
            imgs_list.append(model.decode(z_interp[:, t]))
        imgs_interp = torch.stack(imgs_list, dim=1)

B, T, C, H, W = imgs_interp.shape
fig, axes = plt.subplots(1, T, figsize=(T*2, 2))
for t in range(T):
    axes[t].imshow(imgs_interp[0, t].detach().cpu().permute(1,2,0).numpy(), cmap='gray')
    axes[t].axis('off')
plt.suptitle('Latent interpolation (manual)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Quick Exports (optional)
# Save a small set of generated samples and the current config to disk for quick inspection.

# %%
export_dir = ROOT / 'outputs' / 'diagnostics_quick'
export_dir.mkdir(parents=True, exist_ok=True)

# Save a grid from interpolation
plt.imsave(str(export_dir / 'interpolation_first_row.png'),
           np.concatenate([imgs_interp[0, t].detach().cpu().permute(1,2,0).numpy() for t in range(T)], axis=1))

# Save config snapshot
with open(export_dir / 'config_snapshot.yaml', 'w') as f:
    f.write(OmegaConf.to_yaml(cfg))

print('Saved quick diagnostics to:', export_dir)