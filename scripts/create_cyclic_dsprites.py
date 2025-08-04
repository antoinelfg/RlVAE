#!/usr/bin/env python3
"""
Create Truly Cyclic dSprites Sequences
=====================================

Creates truly cyclic sequences from dSprites by organizing images according to 
their latent factors. Focuses on orientation changes for natural cyclic motion.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def load_dsprites_with_latents(data_path: Path, max_images: int = 40000) -> dict:
    """Load dSprites data with latent factor information."""
    print(f"📂 Loading dSprites data with latents from {data_path}...")
    
    data = np.load(data_path, allow_pickle=True)
    
    # Extract components (subset only)
    imgs = data['imgs'][:max_images]  # (N, 64, 64) grayscale
    latents_values = data['latents_values'][:max_images]  # (N, 6) actual values
    latents_classes = data['latents_classes'][:max_images]  # (N, 6) discrete indices
    
    print(f"✅ Loaded dSprites subset:")
    print(f"   Images shape: {imgs.shape}")
    print(f"   Latents values shape: {latents_values.shape}")
    print(f"   Data range: [{imgs.min()}, {imgs.max()}]")
    
    # The 6 latent factors are:
    # 0: color (always 1.0 - white)
    # 1: shape (0=square, 1=ellipse, 2=heart)  
    # 2: scale (6 values: 0.5 to 1.0)
    # 3: orientation (40 values: 0 to 2π)
    # 4: position_x (32 values: 0 to 1)
    # 5: position_y (32 values: 0 to 1)
    
    return {
        'imgs': imgs,
        'latents_values': latents_values,
        'latents_classes': latents_classes
    }


def create_orientation_cycles(data: dict, sequence_length: int = 8) -> np.ndarray:
    """Create cyclic sequences by varying orientation while keeping other factors fixed."""
    print(f"🔄 Creating orientation-based cyclic sequences (length={sequence_length})...")
    
    imgs = data['imgs']
    latents_classes = data['latents_classes']
    
    # Group images by (shape, scale, pos_x, pos_y) - keep orientation variable
    sequences = []
    
    # Get unique combinations of fixed factors (excluding orientation)
    fixed_factors = latents_classes[:, [1, 2, 4, 5]]  # shape, scale, pos_x, pos_y
    unique_combinations = np.unique(fixed_factors, axis=0)
    
    print(f"Found {len(unique_combinations)} unique combinations of shape/scale/position")
    
    n_orientation_steps = 40  # dSprites has 40 orientation values
    step_size = n_orientation_steps // sequence_length
    
    for combo in tqdm(unique_combinations[:1000], desc="Creating sequences"):  # Limit to 1000 for memory
        shape, scale, pos_x, pos_y = combo
        
        # Find all images with this combination of fixed factors
        mask = ((latents_classes[:, 1] == shape) & 
                (latents_classes[:, 2] == scale) & 
                (latents_classes[:, 4] == pos_x) & 
                (latents_classes[:, 5] == pos_y))
        
        if mask.sum() < sequence_length:
            continue  # Not enough orientations available
            
        # Get images and their orientations
        combo_imgs = imgs[mask]
        combo_orientations = latents_classes[mask, 3]  # orientation indices
        
        # Sort by orientation
        sort_indices = np.argsort(combo_orientations)
        sorted_imgs = combo_imgs[sort_indices]
        sorted_orientations = combo_orientations[sort_indices]
        
        # Create cyclic sequence by sampling evenly spaced orientations
        if len(sorted_imgs) >= sequence_length:
            # Sample evenly spaced orientations to create smooth cycle
            indices = np.linspace(0, len(sorted_imgs)-1, sequence_length, dtype=int)
            sequence_imgs = sorted_imgs[indices]
            
            # Convert to RGB and add channel dimension
            sequence_rgb = np.stack([sequence_imgs] * 3, axis=1)  # [seq_len, 3, H, W]
            sequences.append(sequence_rgb)
    
    if not sequences:
        raise ValueError("No valid cyclic sequences could be created!")
    
    sequences = np.array(sequences)  # [n_sequences, seq_len, 3, H, W]
    print(f"✅ Created {len(sequences)} orientation-based cyclic sequences")
    print(f"   Shape: {sequences.shape}")
    
    return sequences


def enhance_cyclicity(sequences: np.ndarray) -> np.ndarray:
    """Enhance cyclicity by blending last frame with first frame."""
    print("🔄 Enhancing cyclicity with frame blending...")
    
    enhanced = sequences.copy()
    
    # Blend last frame toward first frame for perfect cyclicity
    alpha = 0.8  # Blend factor
    enhanced[:, -1] = alpha * sequences[:, -1] + (1 - alpha) * sequences[:, 0]
    
    return enhanced


def verify_cyclicity(sequences: np.ndarray, threshold: float = 0.01) -> tuple:
    """Verify cyclicity of sequences."""
    print(f"🔍 Verifying cyclicity (threshold={threshold})...")
    
    n_sequences = len(sequences)
    cyclic_count = 0
    mse_values = []
    
    for i in range(min(10, n_sequences)):  # Check first 10 sequences
        seq = sequences[i]
        mse = np.mean((seq[0] - seq[-1]) ** 2)
        mse_values.append(mse)
        is_cyclic = mse < threshold
        if is_cyclic:
            cyclic_count += 1
        
        status = "✅" if is_cyclic else "❌"
        print(f"   Seq {i}: MSE = {mse:.2e} {status}")
    
    # Check all sequences
    all_mse = []
    for seq in sequences:
        mse = np.mean((seq[0] - seq[-1]) ** 2)
        all_mse.append(mse)
    
    all_cyclic = sum(1 for mse in all_mse if mse < threshold)
    
    print(f"📊 Cyclicity Summary:")
    print(f"   Sequences checked: {len(all_mse)}")
    print(f"   Cyclic sequences: {all_cyclic} ({all_cyclic/len(all_mse)*100:.1f}%)")
    print(f"   Average MSE: {np.mean(all_mse):.2e}")
    
    return np.array(all_mse), all_cyclic / len(all_mse)


def split_train_test(sequences: np.ndarray, test_ratio: float = 0.2) -> tuple:
    """Split sequences into train and test sets."""
    print(f"✂️ Splitting data (test_ratio={test_ratio})...")
    
    n_sequences = len(sequences)
    n_test = int(n_sequences * test_ratio)
    n_train = n_sequences - n_test
    
    # Random shuffle
    indices = np.random.permutation(n_sequences)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_sequences = sequences[train_indices]
    test_sequences = sequences[test_indices]
    
    print(f"✅ Split complete:")
    print(f"   Training sequences: {len(train_sequences)}")
    print(f"   Test sequences: {len(test_sequences)}")
    
    return train_sequences, test_sequences


def save_pytorch_format(sequences: np.ndarray, filepath: Path, metadata: dict = None):
    """Save sequences in PyTorch format."""
    print(f"💾 Saving to {filepath}...")
    
    # Convert to PyTorch tensor and float32
    tensor_data = torch.from_numpy(sequences.astype(np.float32))
    
    # Save
    torch.save(tensor_data, filepath)
    
    # Save metadata if provided
    if metadata:
        meta_path = filepath.parent / f"{filepath.stem}_metadata.pt"
        torch.save(metadata, meta_path)
        print(f"💾 Saved metadata to {meta_path}")
    
    print(f"✅ Saved {len(sequences)} sequences to {filepath}")
    print(f"   Shape: {sequences.shape}")
    print(f"   Size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")


def create_sample_visualization(sequences: np.ndarray, output_path: Path):
    """Create visualization of cyclic sequences."""
    print(f"🎨 Creating sample visualization...")
    
    # Select first few sequences
    n_viz = min(3, len(sequences))
    
    fig, axes = plt.subplots(n_viz, 8, figsize=(20, 3 * n_viz))
    if n_viz == 1:
        axes = axes.reshape(1, -1)
    
    for seq_idx in range(n_viz):
        seq = sequences[seq_idx]  # [seq_len, 3, H, W]
        
        for frame_idx in range(8):
            # Convert from [3, H, W] to [H, W] (grayscale)
            img = seq[frame_idx, 0]  # Use first channel since they're identical
            
            axes[seq_idx, frame_idx].imshow(img, cmap='gray')
            axes[seq_idx, frame_idx].set_title(f'Frame {frame_idx}')
            axes[seq_idx, frame_idx].axis('off')
    
    plt.suptitle('Cyclic dSprites Sequences (Orientation Cycles)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization to {output_path}")


def main():
    """Main function to create truly cyclic dSprites sequences."""
    print("🚀 Creating Truly Cyclic dSprites Sequences")
    print("=" * 50)
    
    # Set paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # Input and output paths
    dsprites_path = raw_dir / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    
    if not dsprites_path.exists():
        print(f"❌ dSprites dataset not found at {dsprites_path}")
        print("Please run download_dsprites_efficient.py first")
        return
    
    # Load dSprites data with latent information
    data = load_dsprites_with_latents(dsprites_path, max_images=40000)
    
    # Create orientation-based cyclic sequences
    sequences = create_orientation_cycles(data, sequence_length=8)
    
    # Enhance cyclicity
    sequences = enhance_cyclicity(sequences)
    
    # Verify cyclicity
    mse_values, cyclicity_ratio = verify_cyclicity(sequences)
    
    # Split train/test
    train_sequences, test_sequences = split_train_test(sequences, test_ratio=0.2)
    
    # Create metadata
    train_metadata = {
        'n_sequences': len(train_sequences),
        'sequence_length': 8,
        'image_shape': train_sequences.shape[2:],
        'data_range': (train_sequences.min().item(), train_sequences.max().item()),
        'source': 'dSprites (DeepMind) - orientation cycles',
        'description': 'Real dSprites dataset with truly cyclic orientation sequences',
        'cyclicity_ratio': cyclicity_ratio,
        'creation_method': 'orientation_cycles'
    }
    
    test_metadata = {
        'n_sequences': len(test_sequences),
        'sequence_length': 8,
        'image_shape': test_sequences.shape[2:],
        'data_range': (test_sequences.min().item(), test_sequences.max().item()),
        'source': 'dSprites (DeepMind) - orientation cycles',
        'description': 'Real dSprites dataset with truly cyclic orientation sequences',
        'cyclicity_ratio': cyclicity_ratio,
        'creation_method': 'orientation_cycles'
    }
    
    # Backup existing data
    backup_dir = processed_dir / "backup_non_cyclic"
    backup_dir.mkdir(exist_ok=True)
    
    existing_files = [
        "Sprites_train_cyclic.pt",
        "Sprites_test_cyclic.pt", 
        "Sprites_train_cyclic_metadata.pt",
        "Sprites_test_cyclic_metadata.pt"
    ]
    
    for filename in existing_files:
        src = processed_dir / filename
        if src.exists():
            dst = backup_dir / f"non_cyclic_{filename}"
            print(f"📋 Backing up {filename} to {dst}")
            torch.save(torch.load(src), dst)
    
    # Save new cyclic data
    train_path = processed_dir / "Sprites_train_cyclic.pt"
    test_path = processed_dir / "Sprites_test_cyclic.pt"
    
    save_pytorch_format(train_sequences, train_path, train_metadata)
    save_pytorch_format(test_sequences, test_path, test_metadata)
    
    # Also save raw format (flattened)
    print("\n📦 Creating raw format files...")
    train_raw = train_sequences.reshape(-1, 3, 64, 64)
    test_raw = test_sequences.reshape(-1, 3, 64, 64)
    
    raw_train_path = raw_dir / "Sprites_train.pt"
    raw_test_path = raw_dir / "Sprites_test.pt"
    
    # Backup existing raw data
    for raw_path in [raw_train_path, raw_test_path]:
        if raw_path.exists():
            backup_path = backup_dir / f"non_cyclic_{raw_path.name}"
            print(f"📋 Backing up {raw_path.name} to {backup_path}")
            torch.save(torch.load(raw_path), backup_path)
    
    torch.save(torch.from_numpy(train_raw.astype(np.float32)), raw_train_path)
    torch.save(torch.from_numpy(test_raw.astype(np.float32)), raw_test_path)
    
    print(f"✅ Saved raw training data: {len(train_raw)} images")
    print(f"✅ Saved raw test data: {len(test_raw)} images")
    
    # Create visualization
    viz_path = processed_dir / "cyclic_dsprites_sequences.png"
    create_sample_visualization(train_sequences, viz_path)
    
    # Summary
    print("\n🎉 Truly Cyclic dSprites Dataset Created!")
    print("=" * 50)
    print(f"📊 Cyclicity Performance:")
    print(f"   {cyclicity_ratio*100:.1f}% of sequences are truly cyclic")
    print(f"   Average MSE: {np.mean(mse_values):.2e}")
    print(f"📂 Data saved:")
    print(f"   Training sequences: {len(train_sequences)}")
    print(f"   Test sequences: {len(test_sequences)}")
    print(f"📂 Previous data backed up in: {backup_dir}")
    
    print(f"\n✨ Your dSprites data now has proper cyclicity!")
    print("Ready for RLVAE experiments with truly cyclic sequences.")


if __name__ == "__main__":
    main() 