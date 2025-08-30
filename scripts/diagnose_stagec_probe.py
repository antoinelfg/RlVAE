#!/usr/bin/env python3
import os
import sys
import argparse
import torch
from omegaconf import OmegaConf, DictConfig

# Ensure repo imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from src.models.modular_rlvae import ModularRiemannianFlowVAE


def tensor_stats(name, t):
    if not isinstance(t, torch.Tensor):
        print(f"[{name}] not a tensor: {type(t)}")
        return
    t_det = t.detach()
    nan = torch.isnan(t_det).any().item()
    inf = torch.isinf(t_det).any().item()
    print(f"[{name}] shape={tuple(t_det.shape)} min={t_det.min().item():.4f} max={t_det.max().item():.4f} mean={t_det.mean().item():.4f} std={t_det.std().item():.4f} nan={nan} inf={inf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_cfg", required=True, help="Path to model YAML (Hydra-style)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()

    # Strict load and verbose debug
    os.environ["RLVAE_STRICT_PRETRAIN"] = "1"
    os.environ["RLVAE_DEBUG"] = "1"

    cfg = OmegaConf.load(args.model_cfg)
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    # Safety: ensure required fields exist
    if not hasattr(cfg, "loop"):
        cfg.loop = OmegaConf.create({"mode": "open", "penalty": 0.0})
    if not hasattr(cfg, "sampling"):
        cfg.sampling = OmegaConf.create({"method": "standard", "use_riemannian": False})
    if not hasattr(cfg, "pretrained"):
        cfg.pretrained = OmegaConf.create({})

    # Instantiate model
    print("[DIAG] Building model from", args.model_cfg)
    model = ModularRiemannianFlowVAE(cfg).to(args.device)
    model.eval()

    # Dummy batch (close to data range)
    B, T, C, H, W = args.batch, args.seq_len, cfg.input_dim[0], cfg.input_dim[1], cfg.input_dim[2]
    x = torch.rand(B, T, C, H, W, device=args.device)

    with torch.no_grad():
        # Encode first frame
        x0 = x[:, 0]
        enc_out = model.encoder(x0)
        mu = enc_out.embedding
        log_var = enc_out.log_covariance
        tensor_stats("mu", mu)
        tensor_stats("log_var", log_var)

        # Safe sampling: clamp and z0=mu path
        log_var_c = torch.clamp(log_var, -10.0, 10.0)
        z0 = mu  # deterministic path to isolate decoder
        tensor_stats("z0(mu)", z0)

        # Decode
        dec_out = model.decoder(z0)
        recon = dec_out.reconstruction
        tensor_stats("recon(t0)", recon)

        # Full forward (to exercise loss paths)
        out = model.forward(x)
        print("[DIAG] forward() keys:", list(out.__dict__.keys()) if hasattr(out, "__dict__") else "ModelOutput")
        # Attempt to retrieve fields
        try:
            tensor_stats("fwd.recon_x", out.recon_x)
            tensor_stats("fwd.total_loss", out.total_loss if isinstance(out.total_loss, torch.Tensor) else torch.tensor(out.total_loss))
            tensor_stats("fwd.recon_loss", out.recon_loss if isinstance(out.recon_loss, torch.Tensor) else torch.tensor(out.recon_loss))
            tensor_stats("fwd.kl_loss", out.kl_loss if isinstance(out.kl_loss, torch.Tensor) else torch.tensor(out.kl_loss))
        except Exception as e:
            print("[DIAG] Could not introspect ModelOutput:", e)

    print("[DIAG] Done.")


if __name__ == "__main__":
    main()


