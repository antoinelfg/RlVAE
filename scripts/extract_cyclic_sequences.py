#!/usr/bin/env python3
"""
Extract Cyclic Sequences (Synthetic Data Generator)
=================================================

Generates synthetic cyclic sprites data for testing the RlVAE evaluation system.
This creates data that matches the expected format: cyclic sequences of moving sprites.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List


def create_moving_sprite(
    seq_length: int = 8,
    image_size: Tuple[int, int] = (64, 64),
    sprite_size: int = 8,
    colors: List[Tuple[float, float, float]] = None
) -> torch.Tensor:
    """
    Create a single cyclic sequence of a moving colored sprite.
    
    Args:
        seq_length: Number of frames in sequence
        image_size: (H, W) size of each frame
        sprite_size: Size of the square sprite
        colors: List of RGB colors for the sprite
    
    Returns:
        Tensor of shape [T, C, H, W] representing the cyclic sequence
    """
    if colors is None:
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 
                 (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]
    
    H, W = image_size
    
    # Create circular trajectory for cyclic motion
    center_x, center_y = W // 2, H // 2
    radius = min(W, H) // 4
    
    sequence = torch.zeros(seq_length, 3, H, W)
    
    # Choose random color for this sprite
    color = colors[np.random.randint(len(colors))]
    
    for t in range(seq_length):
        # Calculate position on circular path (ensures cyclicity)
        angle = 2 * np.pi * t / seq_length
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        
        # Ensure sprite stays within bounds
        x = max(sprite_size//2, min(W - sprite_size//2, x))
        y = max(sprite_size//2, min(H - sprite_size//2, y))
        
        # Draw sprite
        x_start = max(0, x - sprite_size//2)
        x_end = min(W, x + sprite_size//2)
        y_start = max(0, y - sprite_size//2) 
        y_end = min(H, y + sprite_size//2)
        
        for c in range(3):
            sequence[t, c, y_start:y_end, x_start:x_end] = color[c]
    
    return sequence


def create_dataset(
    n_sequences: int,
    seq_length: int = 8,
    image_size: Tuple[int, int] = (64, 64),
    add_noise: bool = True,
    noise_level: float = 0.05
) -> torch.Tensor:
    """
    Create a dataset of cyclic sprite sequences.
    
    Args:
        n_sequences: Number of sequences to generate
        seq_length: Length of each sequence
        image_size: Size of each frame
        add_noise: Whether to add noise for realism
        noise_level: Standard deviation of Gaussian noise
    
    Returns:
        Tensor of shape [N, T, C, H, W]
    """
    print(f"🎨 Generating {n_sequences} cyclic sprite sequences...")
    
    # Different colors for variety
    colors = [
        (1.0, 0.2, 0.2),  # Red
        (0.2, 1.0, 0.2),  # Green
        (0.2, 0.2, 1.0),  # Blue
        (1.0, 1.0, 0.2),  # Yellow
        (1.0, 0.2, 1.0),  # Magenta
        (0.2, 1.0, 1.0),  # Cyan
        (1.0, 0.6, 0.2),  # Orange
        (0.6, 0.2, 1.0),  # Purple
    ]
    
    dataset = torch.zeros(n_sequences, seq_length, 3, *image_size)
    
    for i in range(n_sequences):
        if i % 100 == 0:
            print(f"   Generated {i}/{n_sequences} sequences")
        
        # Create sequence with random sprite properties
        sprite_size = np.random.randint(6, 12)
        sequence = create_moving_sprite(
            seq_length=seq_length,
            image_size=image_size,
            sprite_size=sprite_size,
            colors=colors
        )
        
        # Add some background variation
        if np.random.random() > 0.7:
            # Sometimes add a subtle background gradient
            bg_level = np.random.uniform(0.0, 0.1)
            for c in range(3):
                grad = torch.linspace(0, bg_level, image_size[0]).view(-1, 1)
                sequence[:, c] += grad
        
        dataset[i] = sequence
    
    # Add noise for realism
    if add_noise:
        noise = torch.randn_like(dataset) * noise_level
        dataset = dataset + noise
    
    # Normalize to [0, 1] range
    dataset = torch.clamp(dataset, 0.0, 1.0)
    
    print(f"✅ Generated dataset shape: {dataset.shape}")
    print(f"✅ Data range: [{dataset.min():.3f}, {dataset.max():.3f}]")
    
    return dataset


def verify_cyclicity(dataset: torch.Tensor, threshold: float = 0.01) -> None:
    """Verify that sequences are properly cyclic."""
    print(f"🔍 Verifying cyclicity...")
    
    n_sequences = len(dataset)
    cyclic_count = 0
    
    for i in range(min(10, n_sequences)):
        seq = dataset[i]
        mse = torch.mean((seq[0] - seq[-1]) ** 2).item()
        is_cyclic = mse < threshold
        
        if is_cyclic:
            cyclic_count += 1
        
        status = "✅" if is_cyclic else "❌"
        print(f"   Seq {i}: MSE = {mse:.2e} {status}")
    
    cyclicity_rate = cyclic_count / min(10, n_sequences)
    print(f"✅ Cyclicity rate: {cyclicity_rate:.1%}")


def save_sample_visualization(dataset: torch.Tensor, save_path: Path) -> None:
    """Save a visualization of the first sequence."""
    print(f"📊 Creating sample visualization...")
    
    seq = dataset[0]  # First sequence [T, C, H, W]
    seq_length = len(seq)
    
    fig, axes = plt.subplots(2, seq_length//2, figsize=(seq_length*2, 4))
    axes = axes.flatten()
    
    for t in range(seq_length):
        img = seq[t].permute(1, 2, 0)  # [H, W, C]
        axes[t].imshow(img)
        axes[t].set_title(f'Frame {t}')
        axes[t].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Sample visualization saved to: {save_path}")


def main():
    """Generate synthetic cyclic sprites dataset."""
    print("🚀 Generating Synthetic Cyclic Sprites Dataset")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training data
    print("\n📦 Generating training data...")
    train_data = create_dataset(
        n_sequences=1200,
        seq_length=8,
        image_size=(64, 64),
        add_noise=True,
        noise_level=0.03
    )
    
    # Generate test data
    print("\n📦 Generating test data...")
    test_data = create_dataset(
        n_sequences=300,
        seq_length=8, 
        image_size=(64, 64),
        add_noise=True,
        noise_level=0.03
    )
    
    # Verify cyclicity
    print("\n🔍 Verifying training data cyclicity...")
    verify_cyclicity(train_data)
    
    print("\n🔍 Verifying test data cyclicity...")
    verify_cyclicity(test_data)
    
    # Save datasets
    train_path = output_dir / "Sprites_train_cyclic.pt"
    test_path = output_dir / "Sprites_test_cyclic.pt"
    
    print(f"\n💾 Saving training data to: {train_path}")
    torch.save(train_data, train_path)
    
    print(f"💾 Saving test data to: {test_path}")
    torch.save(test_data, test_path)
    
    # Create metadata files (empty for now, but expected by the system)
    train_meta_path = output_dir / "Sprites_train_cyclic_metadata.pt"
    test_meta_path = output_dir / "Sprites_test_cyclic_metadata.pt"
    
    train_metadata = {
        'n_sequences': len(train_data),
        'sequence_length': train_data.shape[1],
        'image_shape': train_data.shape[2:],
        'data_range': (train_data.min().item(), train_data.max().item()),
        'generated': True,
        'description': 'Synthetic cyclic sprites with circular motion'
    }
    
    test_metadata = {
        'n_sequences': len(test_data),
        'sequence_length': test_data.shape[1],
        'image_shape': test_data.shape[2:],
        'data_range': (test_data.min().item(), test_data.max().item()),
        'generated': True,
        'description': 'Synthetic cyclic sprites with circular motion'
    }
    
    torch.save(train_metadata, train_meta_path)
    torch.save(test_metadata, test_meta_path)
    
    # Save sample visualization
    viz_path = output_dir / "sample_sequence.png"
    save_sample_visualization(train_data, viz_path)
    
    print("\n✅ Dataset generation complete!")
    print(f"   Training sequences: {len(train_data)}")
    print(f"   Test sequences: {len(test_data)}")
    print(f"   Sequence length: {train_data.shape[1]}")
    print(f"   Image size: {train_data.shape[2:]} ")
    print(f"   Files saved in: {output_dir}")


if __name__ == "__main__":
    main() 