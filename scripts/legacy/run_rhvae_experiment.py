#!/usr/bin/env python3
"""
Simple script to run the RHVAE experiment.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.rhvae_experiment import RHVAEExperiment

def main():
    """Run the RHVAE experiment."""
    print("🚀 Starting RHVAE Sprites Experiment")
    print("=" * 50)
    
    # Initialize experiment with conservative parameters
    experiment = RHVAEExperiment(
        input_dim=[3, 64, 64],
        latent_dim=8,  # Smaller for stability
        n_lf=1,  # Fewer leapfrog steps
        eps_lf=0.0001,  # Smaller step size
        beta_zero=1.0,  # Higher beta
        temperature=0.5,  # Lower temperature
        regularization=0.1,  # Higher regularization
        device="auto",
        seed=42,
    )
    
    # Load data
    train_data, test_data = experiment.load_data(
        train_path="data/processed/Sprites_train_cyclic.pt",
        test_path="data/processed/Sprites_test_cyclic.pt",
        batch_size=16  # Smaller batch size for stability
    )
    
    # Train the model
    experiment.train(
        epochs=20,  # Fewer epochs for testing
        learning_rate=5e-5,  # Lower learning rate
        weight_decay=1e-5,
        log_every=5,
        save_every=5,
        output_dir="outputs/rhvae_sprites_test",
        use_wandb=True,
        wandb_config={
            "project": "rlvae_experiments",
            "name": "rhvae_sprites_test",
            "group": "rhvae_experiments",
            "tags": ["rhvae", "sprites", "test"],
        }
    )
    
    print("✅ Experiment completed!")

if __name__ == "__main__":
    main() 