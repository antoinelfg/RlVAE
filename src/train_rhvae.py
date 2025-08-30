#!/usr/bin/env python3
"""
Main training script for RHVAE experiments using Hydra configuration.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from pathlib import Path
import os
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.rhvae_experiment import RHVAEExperiment

@hydra.main(version_base=None, config_path="../conf", config_name="experiment/rhvae_sprites")
def main(cfg: DictConfig):
    """Main training function."""
    
    print("🚀 Starting RHVAE Sprites Experiment")
    print("=" * 50)
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set device
    if cfg.hardware.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.hardware.device
        
    print(f"Using device: {device}")
    
    # Set deterministic behavior if requested
    if cfg.hardware.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Set seed
    torch.manual_seed(cfg.hardware.seed)
    
    # Initialize experiment
    experiment = RHVAEExperiment(
        input_dim=cfg.model.input_dim,
        latent_dim=cfg.model.latent_dim,
        n_lf=cfg.model.n_lf,
        eps_lf=cfg.model.eps_lf,
        beta_zero=cfg.model.beta_zero,
        temperature=cfg.model.temperature,
        regularization=cfg.model.regularization,
        encoder=cfg.model.encoder,
        decoder=cfg.model.decoder,
        device=device,
        seed=cfg.hardware.seed,
        max_centroids=cfg.model.get("max_centroids", None),
        centroid_subsample_method=cfg.model.get("centroid_subsample_method", "fps"),
        align_with_knn_cov=cfg.model.get("align_with_knn_cov", False),
        knn_k=cfg.model.get("knn_k", 300),
        alpha_align=cfg.model.get("alpha_align", 0.5),
        metric_normalization=cfg.model.get("metric_normalization", "none"),
        target_mean_eig=cfg.model.get("target_mean_eig", 1.0),
        weight_kernel=cfg.model.get("weight_kernel", "isotropic"),
        weight_metric_normalization=cfg.model.get("weight_metric_normalization", "trace"),
        normalize_weight_sum=cfg.model.get("normalize_weight_sum", False),
        topk_weights=cfg.model.get("topk_weights", None),
        reestimate_metric_from_decoder_jacobian=cfg.model.get("reestimate_metric_from_decoder_jacobian", False),
        jacobian_alpha=cfg.model.get("jacobian_alpha", 0.5),
        jacobian_h=cfg.model.get("jacobian_h", 1e-3),
        jacobian_stride=cfg.model.get("jacobian_stride", 4),
        metric_scale=cfg.model.get("metric_scale", 1.0),
        realign_centroids=cfg.model.get("realign_centroids", False),
        centroid_realign_method=cfg.model.get("centroid_realign_method", "kmeans"),
    )
    
    # Load data
    train_data, test_data = experiment.load_data(
        train_path=cfg.data.train_path,
        test_path=cfg.data.test_path,
        batch_size=cfg.data.batch_size
    )
    
    # Setup WandB configuration
    wandb_config = {
        "project": cfg.logging.wandb.project,
        "name": f"{cfg.experiment.name}_{cfg.hardware.seed}",
        "group": cfg.experiment.group,
        "tags": cfg.experiment.tags,
        "entity": cfg.logging.wandb.entity,
    }
    
    # Train the model
    experiment.train(
        epochs=cfg.training.epochs,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        log_every=cfg.logging.log_every,
        save_every=cfg.training.save_every,
        output_dir=cfg.output.dir,
        use_wandb=cfg.logging.wandb.project is not None,
        wandb_config=wandb_config,
    )
    
    # Optional: metric evolution logging (guarded if implemented)
    if cfg.logging.visualization.log_metric_evolution and hasattr(experiment, "log_metric_evolution"):
        experiment.log_metric_evolution()
    
    # Save final model
    if cfg.output.save_model:
        model_path = Path(cfg.output.dir) / "final_model.pt"
        experiment.save_model(str(model_path))
        
    # Generate and log samples
    if cfg.logging.visualization.log_reconstructions:
        print("🎨 Generating samples...")
        samples = experiment.sample(num_samples=cfg.evaluation.num_samples)
        
        # Log samples to WandB
        if wandb.run is not None:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            for i in range(min(16, len(samples))):
                row = i // 4
                col = i % 4
                
                img = samples[i].permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"Sample {i}")
                axes[row, col].axis('off')
            
            plt.tight_layout()
            wandb.log({"generated_samples": wandb.Image(fig)})
            plt.close()
    
    print("✅ Experiment completed successfully!")
    
    # Close WandB
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main() 