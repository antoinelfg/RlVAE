#!/usr/bin/env python3
"""
Download and Process dSprites Dataset
====================================

Downloads the original dSprites dataset from DeepMind and converts it to the format
expected by the RlVAE project. Creates both raw data and cyclic sequences.

The dSprites dataset contains:
- 737,280 images (64x64 grayscale)
- 6 latent factors: color, shape, scale, orientation, position_x, position_y
- 3 shapes: square, ellipse, heart
"""

import os
import sys
import urllib.request
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib


def download_file(url: str, filepath: Path) -> bool:
    """Download a file with progress bar."""
    try:
        print(f"📥 Downloading {filepath.name}...")
        
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size) / total_size * 100)
                bar_length = 50
                filled_length = int(bar_length * percent / 100)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'\r|{bar}| {percent:.1f}% ({block_num * block_size / 1024 / 1024:.1f}MB)', end='')
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\n✅ Downloaded {filepath.name}")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {filepath.name}: {e}")
        return False


def load_dsprites_data(data_path: Path) -> dict:
    """Load the dSprites .npz file."""
    print(f"📂 Loading dSprites data from {data_path}...")
    
    # Load the data (allow_pickle=True for metadata)
    data = np.load(data_path, allow_pickle=True)
    
    # Extract components
    imgs = data['imgs']  # (737280, 64, 64) - grayscale images
    latents_values = data['latents_values']  # (737280, 6) - actual latent values
    latents_classes = data['latents_classes']  # (737280, 6) - discrete class indices
    
    # Try to load metadata, but don't fail if it doesn't work
    try:
        metadata = data['metadata'][()]  # metadata dict
        print(f"✅ Loaded metadata successfully")
    except (UnicodeDecodeError, ValueError) as e:
        print(f"⚠️ Could not load metadata (encoding issue): {e}")
        # Create a minimal metadata dict
        metadata = {
            'latents_names': ['color', 'shape', 'scale', 'orientation', 'posX', 'posY'],
            'latents_sizes': [1, 3, 6, 40, 32, 32]
        }
    
    print(f"✅ Loaded dSprites dataset:")
    print(f"   Images shape: {imgs.shape}")
    print(f"   Latents values shape: {latents_values.shape}")
    print(f"   Latents classes shape: {latents_classes.shape}")
    print(f"   Data range: [{imgs.min()}, {imgs.max()}]")
    
    return {
        'imgs': imgs,
        'latents_values': latents_values,
        'latents_classes': latents_classes,
        'metadata': metadata
    }


def convert_to_rgb_sequences(imgs: np.ndarray, sequence_length: int = 8) -> np.ndarray:
    """Convert grayscale images to RGB and organize into cyclic sequences."""
    print(f"🎨 Converting to RGB cyclic sequences (length={sequence_length})...")
    
    n_total = len(imgs)
    n_sequences = n_total // sequence_length
    
    # Trim to exact multiple of sequence_length
    n_used = n_sequences * sequence_length
    imgs_trimmed = imgs[:n_used]
    
    # Reshape to sequences: [n_sequences, seq_len, H, W]
    sequences = imgs_trimmed.reshape(n_sequences, sequence_length, 64, 64)
    
    # Convert to RGB by replicating grayscale across 3 channels
    # Shape: [n_sequences, seq_len, 3, H, W]
    rgb_sequences = np.stack([sequences] * 3, axis=2)
    
    # Convert to float32 and normalize to [0, 1]
    rgb_sequences = rgb_sequences.astype(np.float32)
    
    print(f"✅ Created {n_sequences} RGB sequences of length {sequence_length}")
    print(f"   Final shape: {rgb_sequences.shape}")
    print(f"   Data range: [{rgb_sequences.min():.3f}, {rgb_sequences.max():.3f}]")
    
    return rgb_sequences


def create_cyclic_sequences(sequences: np.ndarray) -> np.ndarray:
    """Make sequences more cyclic by blending first and last frames."""
    print("🔄 Enhancing cyclicity...")
    
    cyclic_sequences = sequences.copy()
    
    # Blend last frame with first frame for better cyclicity
    alpha = 0.7  # Weight for last frame
    cyclic_sequences[:, -1] = alpha * sequences[:, -1] + (1 - alpha) * sequences[:, 0]
    
    # Verify cyclicity
    print("🔍 Verifying cyclicity of first 5 sequences:")
    for i in range(min(5, len(cyclic_sequences))):
        seq = cyclic_sequences[i]
        mse = np.mean((seq[0] - seq[-1]) ** 2)
        is_cyclic = mse < 0.01
        status = "✅" if is_cyclic else "❌"
        print(f"   Seq {i}: MSE = {mse:.2e} {status}")
    
    return cyclic_sequences


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
    
    # Convert to PyTorch tensor
    tensor_data = torch.from_numpy(sequences)
    
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
    """Create a sample visualization of sequences."""
    print(f"🎨 Creating sample visualization...")
    
    # Select first sequence
    sample_seq = sequences[0]  # [seq_len, 3, H, W]
    seq_len = len(sample_seq)
    
    # Create plot
    fig, axes = plt.subplots(2, seq_len//2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(seq_len):
        # Convert from [3, H, W] to [H, W, 3] for matplotlib
        img = sample_seq[i].transpose(1, 2, 0)
        
        # Since it's grayscale replicated, just use one channel
        img_gray = img[:, :, 0]
        
        axes[i].imshow(img_gray, cmap='gray')
        axes[i].set_title(f'Frame {i}')
        axes[i].axis('off')
    
    plt.suptitle('Sample dSprites Sequence (Cyclic)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved sample visualization to {output_path}")


def main():
    """Main function to download and process dSprites dataset."""
    print("🚀 dSprites Dataset Downloader and Processor")
    print("=" * 50)
    
    # Set paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dSprites dataset
    dsprites_url = "https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    dsprites_path = raw_dir / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    
    if not dsprites_path.exists():
        success = download_file(dsprites_url, dsprites_path)
        if not success:
            print("❌ Failed to download dSprites dataset")
            return
    else:
        print(f"✅ dSprites dataset already exists at {dsprites_path}")
    
    # Load and process data
    data = load_dsprites_data(dsprites_path)
    
    # Convert to RGB sequences
    sequence_length = 8
    rgb_sequences = convert_to_rgb_sequences(data['imgs'], sequence_length)
    
    # Make sequences more cyclic
    cyclic_sequences = create_cyclic_sequences(rgb_sequences)
    
    # Split train/test
    train_sequences, test_sequences = split_train_test(cyclic_sequences, test_ratio=0.2)
    
    # Create metadata
    train_metadata = {
        'n_sequences': len(train_sequences),
        'sequence_length': sequence_length,
        'image_shape': train_sequences.shape[2:],
        'data_range': (train_sequences.min().item(), train_sequences.max().item()),
        'source': 'dSprites (DeepMind)',
        'description': 'Real dSprites dataset converted to cyclic RGB sequences',
        'url': dsprites_url
    }
    
    test_metadata = {
        'n_sequences': len(test_sequences),
        'sequence_length': sequence_length,
        'image_shape': test_sequences.shape[2:],
        'data_range': (test_sequences.min().item(), test_sequences.max().item()),
        'source': 'dSprites (DeepMind)',
        'description': 'Real dSprites dataset converted to cyclic RGB sequences',
        'url': dsprites_url
    }
    
    # Save processed data
    train_path = processed_dir / "Sprites_train_cyclic.pt"
    test_path = processed_dir / "Sprites_test_cyclic.pt"
    
    save_pytorch_format(train_sequences, train_path, train_metadata)
    save_pytorch_format(test_sequences, test_path, test_metadata)
    
    # Also save raw format (flattened for traditional VAE use)
    print("\n📦 Creating raw format files...")
    
    # Flatten sequences for raw format: [n_sequences * seq_len, 3, H, W]
    train_raw = train_sequences.reshape(-1, 3, 64, 64)
    test_raw = test_sequences.reshape(-1, 3, 64, 64)
    
    raw_train_path = raw_dir / "Sprites_train.pt"
    raw_test_path = raw_dir / "Sprites_test.pt"
    
    torch.save(torch.from_numpy(train_raw), raw_train_path)
    torch.save(torch.from_numpy(test_raw), raw_test_path)
    
    print(f"✅ Saved raw training data: {len(train_raw)} images to {raw_train_path}")
    print(f"✅ Saved raw test data: {len(test_raw)} images to {raw_test_path}")
    
    # Create sample visualization
    viz_path = processed_dir / "sample_dsprites_sequence.png"
    create_sample_visualization(train_sequences, viz_path)
    
    # Summary
    print("\n🎉 dSprites Dataset Processing Complete!")
    print("=" * 50)
    print(f"📂 Raw data saved in: {raw_dir}")
    print(f"   - Sprites_train.pt: {len(train_raw)} images")
    print(f"   - Sprites_test.pt: {len(test_raw)} images")
    print(f"📂 Processed data saved in: {processed_dir}")
    print(f"   - Sprites_train_cyclic.pt: {len(train_sequences)} sequences")
    print(f"   - Sprites_test_cyclic.pt: {len(test_sequences)} sequences")
    print(f"   - Metadata and visualization included")
    
    print("\n✨ Your real dSprites data is now ready!")
    print("You can now run your experiments with the authentic dSprites dataset.")


if __name__ == "__main__":
    main() 