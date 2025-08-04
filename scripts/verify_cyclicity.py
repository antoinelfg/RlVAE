#!/usr/bin/env python3
"""
Verify Cyclicity of Sprites Data
===============================

Verifies the cyclicity of the current sprites data files and provides
detailed analysis and visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple


def load_and_verify_cyclicity(data_path: Path, threshold: float = 0.01) -> Tuple[float, float, int]:
    """Load data and verify cyclicity."""
    print(f"📂 Loading data from {data_path}...")
    
    # Load data
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    print(f"✅ Data shape: {data.shape}")
    print(f"✅ Data range: [{data.min().item():.3f}, {data.max().item():.3f}]")
    print(f"✅ Data type: {data.dtype}")
    
    # Check if it's sequence data [N, T, C, H, W]
    if len(data.shape) != 5:
        print(f"❌ Expected 5D data [N, T, C, H, W], got {len(data.shape)}D")
        return 0.0, 0.0, 0
    
    n_sequences, seq_len, channels, height, width = data.shape
    print(f"📊 Data structure:")
    print(f"   Sequences: {n_sequences}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Channels: {channels}")
    print(f"   Image size: {height}x{width}")
    
    # Verify cyclicity
    print(f"\n🔍 Verifying cyclicity (threshold={threshold})...")
    
    mse_values = []
    cyclic_count = 0
    
    # Check all sequences
    for i in range(n_sequences):
        seq = data[i]  # [T, C, H, W]
        first_frame = seq[0]  # [C, H, W]
        last_frame = seq[-1]  # [C, H, W]
        
        # Calculate MSE between first and last frame
        mse = torch.mean((first_frame - last_frame) ** 2).item()
        mse_values.append(mse)
        
        if mse < threshold:
            cyclic_count += 1
        
        # Print first 10 for detail
        if i < 10:
            status = "✅" if mse < threshold else "❌"
            print(f"   Seq {i}: MSE = {mse:.2e} {status}")
    
    # Summary statistics
    mse_array = np.array(mse_values)
    cyclicity_ratio = cyclic_count / n_sequences
    
    print(f"\n📊 Cyclicity Summary:")
    print(f"   Total sequences: {n_sequences}")
    print(f"   Cyclic sequences: {cyclic_count} ({cyclicity_ratio*100:.1f}%)")
    print(f"   Average MSE: {np.mean(mse_array):.2e}")
    print(f"   Median MSE: {np.median(mse_array):.2e}")
    print(f"   Max MSE: {np.max(mse_array):.2e}")
    print(f"   Min MSE: {np.min(mse_array):.2e}")
    
    return np.mean(mse_array), cyclicity_ratio, n_sequences


def load_metadata(metadata_path: Path) -> dict:
    """Load metadata if available."""
    try:
        metadata = torch.load(metadata_path, map_location='cpu', weights_only=False)
        print(f"📋 Metadata loaded from {metadata_path}:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        return metadata
    except Exception as e:
        print(f"⚠️ Could not load metadata: {e}")
        return {}


def create_cyclicity_visualization(data_path: Path, output_path: Path, n_examples: int = 3):
    """Create visualization showing cyclicity."""
    print(f"🎨 Creating cyclicity visualization...")
    
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    n_sequences = min(n_examples, len(data))
    
    fig, axes = plt.subplots(n_sequences, 8, figsize=(20, 3 * n_sequences))
    if n_sequences == 1:
        axes = axes.reshape(1, -1)
    
    for seq_idx in range(n_sequences):
        seq = data[seq_idx]  # [T, C, H, W]
        
        # Calculate MSE for this sequence
        mse = torch.mean((seq[0] - seq[-1]) ** 2).item()
        
        for frame_idx in range(8):
            # Convert to grayscale for visualization
            if len(seq.shape) == 4 and seq.shape[1] == 3:  # [T, C, H, W] with RGB
                img = seq[frame_idx, 0].numpy()  # Use first channel
            elif len(seq.shape) == 3:  # [T, H, W] grayscale
                img = seq[frame_idx].numpy()
            else:
                img = seq[frame_idx, 0].numpy()  # Default to first channel
            
            axes[seq_idx, frame_idx].imshow(img, cmap='gray')
            axes[seq_idx, frame_idx].set_title(f'Frame {frame_idx}')
            axes[seq_idx, frame_idx].axis('off')
        
        # Add sequence info
        status = "✅ Cyclic" if mse < 0.01 else "❌ Non-cyclic"
        axes[seq_idx, 0].text(0, -5, f'Seq {seq_idx}: MSE={mse:.2e} {status}', 
                              transform=axes[seq_idx, 0].transData, fontsize=10)
    
    plt.suptitle(f'Cyclicity Verification - {data_path.name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization to {output_path}")


def main():
    """Main function to verify cyclicity of current sprites data."""
    print("🔍 Verifying Cyclicity of Current Sprites Data")
    print("=" * 55)
    
    # Set paths
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    
    train_path = processed_dir / "Sprites_train_cyclic.pt"
    test_path = processed_dir / "Sprites_test_cyclic.pt"
    train_meta_path = processed_dir / "Sprites_train_cyclic_metadata.pt"
    test_meta_path = processed_dir / "Sprites_test_cyclic_metadata.pt"
    
    # Check if files exist
    if not train_path.exists():
        print(f"❌ Training data not found: {train_path}")
        return
    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        return
    
    print("🗂️ Current Files:")
    print(f"   Training: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"   Test: {test_path} ({test_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Load and check metadata
    print(f"\n📋 Training Data Metadata:")
    train_metadata = load_metadata(train_meta_path)
    
    print(f"\n📋 Test Data Metadata:")
    test_metadata = load_metadata(test_meta_path)
    
    # Verify training data cyclicity
    print(f"\n" + "="*55)
    print("🔍 TRAINING DATA CYCLICITY VERIFICATION")
    print("="*55)
    train_avg_mse, train_cyclicity, train_count = load_and_verify_cyclicity(train_path)
    
    # Verify test data cyclicity
    print(f"\n" + "="*55)
    print("🔍 TEST DATA CYCLICITY VERIFICATION")
    print("="*55)
    test_avg_mse, test_cyclicity, test_count = load_and_verify_cyclicity(test_path)
    
    # Create visualizations
    viz_dir = processed_dir
    train_viz_path = viz_dir / "cyclicity_verification_train.png"
    test_viz_path = viz_dir / "cyclicity_verification_test.png"
    
    create_cyclicity_visualization(train_path, train_viz_path)
    create_cyclicity_visualization(test_path, test_viz_path)
    
    # Overall summary
    print(f"\n" + "="*55)
    print("📊 OVERALL SUMMARY")
    print("="*55)
    print(f"Training Data:")
    print(f"   ✅ {train_count} sequences")
    print(f"   ✅ {train_cyclicity*100:.1f}% cyclic")
    print(f"   ✅ Avg MSE: {train_avg_mse:.2e}")
    
    print(f"Test Data:")
    print(f"   ✅ {test_count} sequences")
    print(f"   ✅ {test_cyclicity*100:.1f}% cyclic")
    print(f"   ✅ Avg MSE: {test_avg_mse:.2e}")
    
    # Quality assessment
    if train_cyclicity >= 0.95 and test_cyclicity >= 0.95:
        print(f"\n🎉 EXCELLENT! Your data has outstanding cyclicity!")
        print(f"   Ready for RLVAE experiments! 🚀")
    elif train_cyclicity >= 0.8 and test_cyclicity >= 0.8:
        print(f"\n✅ GOOD! Your data has good cyclicity.")
        print(f"   Should work well for RLVAE experiments.")
    else:
        print(f"\n⚠️ WARNING! Cyclicity is below recommended levels.")
        print(f"   Consider regenerating sequences for better results.")
    
    print(f"\n📁 Visualizations saved:")
    print(f"   Training: {train_viz_path}")
    print(f"   Test: {test_viz_path}")


if __name__ == "__main__":
    main() 