#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import argparse

def analyze_and_extract_cyclic_sequences(
    data_path, 
    output_path, 
    similarity_threshold=1e-4,
    visualize_examples=True,
    max_examples=20
):
    """
    Extract only the cyclic sequences from a sprites dataset.
    
    Args:
        data_path: Path to the original sprites data
        output_path: Path to save the cyclic sequences
        similarity_threshold: MSE threshold for considering sequences cyclic
        visualize_examples: Whether to create visualization of extracted sequences
        max_examples: Maximum examples to show in visualization
    
    Returns:
        Dictionary with extraction results
    """
    print(f"🔍 Analyzing and extracting cyclic sequences from: {data_path}")
    print(f"📊 Using similarity threshold: {similarity_threshold}")
    
    # Load original data
    full_data = torch.load(data_path, map_location='cpu')
    if isinstance(full_data, dict) and 'data' in full_data:
        data = full_data['data']
    else:
        data = full_data
    
    print(f"📋 Original data shape: {data.shape}")
    print(f"📋 Original data range: [{data.min():.3f}, {data.max():.3f}]")
    
    # Normalize to [0, 1] if needed
    if data.max() > 1.0:
        data = data.float() / 255.0
    else:
        data = data.float()
    
    # Convert from [N, T, H, W, C] to [N, T, C, H, W] if needed
    if data.shape[-1] == 3:  # If last dim is channels
        data = data.permute(0, 1, 4, 2, 3)  # [N, T, H, W, C] -> [N, T, C, H, W]
        print(f"✅ Converted data format to: {data.shape}")
    
    # Analyze each sequence for cyclicity
    cyclic_indices = []
    cyclic_mse_values = []
    sequence_stats = []
    
    print(f"\n🔬 Analyzing {len(data)} sequences for cyclicity...")
    
    for i in tqdm(range(len(data)), desc="Analyzing sequences"):
        sequence = data[i]  # [T, C, H, W]
        n_timesteps = sequence.shape[0]
        
        if n_timesteps < 2:
            continue
        
        # Get first and last frames
        first_frame = sequence[0]      # [C, H, W]
        last_frame = sequence[-1]      # [C, H, W]
        
        # Compute MSE difference
        mse_diff = torch.mean((first_frame - last_frame) ** 2).item()
        
        # Check if cyclic based on threshold
        is_cyclic = mse_diff < similarity_threshold
        
        if is_cyclic:
            cyclic_indices.append(i)
            cyclic_mse_values.append(mse_diff)
        
        # Store stats for all sequences
        pixel_diff = torch.abs(first_frame - last_frame).mean().item()
        max_pixel_diff = torch.abs(first_frame - last_frame).max().item()
        correlation = torch.corrcoef(torch.stack([first_frame.flatten(), last_frame.flatten()]))[0, 1].item()
        
        sequence_stats.append({
            'sequence_idx': i,
            'n_timesteps': n_timesteps,
            'mse_diff': mse_diff,
            'pixel_diff': pixel_diff,
            'max_pixel_diff': max_pixel_diff,
            'correlation': correlation,
            'is_cyclic': is_cyclic
        })
    
    # Extract cyclic sequences
    cyclic_data = data[cyclic_indices]
    
    print(f"\n📈 EXTRACTION RESULTS:")
    print(f"=" * 50)
    print(f"Total original sequences: {len(data)}")
    print(f"Cyclic sequences found: {len(cyclic_indices)} ({100*len(cyclic_indices)/len(data):.1f}%)")
    print(f"Extracted data shape: {cyclic_data.shape}")
    print(f"MSE statistics for cyclic sequences:")
    print(f"  Mean: {np.mean(cyclic_mse_values):.2e}")
    print(f"  Max:  {np.max(cyclic_mse_values):.2e}")
    print(f"  Min:  {np.min(cyclic_mse_values):.2e}")
    
    # Save cyclic sequences
    torch.save(cyclic_data, output_path)
    print(f"💾 Saved {len(cyclic_indices)} cyclic sequences to: {output_path}")
    
    # Save metadata
    metadata_path = output_path.replace('.pt', '_metadata.pt')
    metadata = {
        'original_indices': cyclic_indices,
        'mse_values': cyclic_mse_values,
        'similarity_threshold': similarity_threshold,
        'original_data_shape': data.shape,
        'cyclic_data_shape': cyclic_data.shape,
        'extraction_stats': {
            'total_original': len(data),
            'total_cyclic': len(cyclic_indices),
            'cyclic_percentage': 100*len(cyclic_indices)/len(data)
        }
    }
    torch.save(metadata, metadata_path)
    print(f"💾 Saved metadata to: {metadata_path}")
    
    # Create visualization if requested
    if visualize_examples and len(cyclic_indices) > 0:
        create_cyclic_examples_visualization(
            cyclic_data, 
            cyclic_indices, 
            cyclic_mse_values, 
            max_examples, 
            output_path
        )
    
    return {
        'cyclic_data': cyclic_data,
        'cyclic_indices': cyclic_indices,
        'mse_values': cyclic_mse_values,
        'sequence_stats': sequence_stats,
        'metadata': metadata
    }

def create_cyclic_examples_visualization(cyclic_data, cyclic_indices, mse_values, max_examples, output_path):
    """Create visualization of extracted cyclic sequences."""
    
    print(f"🎨 Creating visualization of cyclic sequences...")
    
    n_examples = min(max_examples, len(cyclic_data))
    n_cols = 5  # first, middle, last, difference, sequence info
    
    fig, axes = plt.subplots(n_examples, n_cols, figsize=(20, 4*n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    elif n_examples == 0:
        print("⚠️ No cyclic sequences to visualize")
        return
    
    fig.suptitle(f'Extracted Cyclic Sequences (Total: {len(cyclic_data)})', fontsize=16, fontweight='bold')
    
    # Sort by MSE to show best examples first
    sorted_indices = np.argsort(mse_values)
    
    for i in range(n_examples):
        idx = sorted_indices[i]
        sequence = cyclic_data[idx]  # [T, C, H, W]
        original_idx = cyclic_indices[idx]
        mse_val = mse_values[idx]
        
        n_timesteps = sequence.shape[0]
        
        # First frame
        first_frame = sequence[0].permute(1, 2, 0).numpy()
        first_frame = np.clip(first_frame, 0, 1)
        axes[i, 0].imshow(first_frame)
        axes[i, 0].set_title(f'Original Seq {original_idx}\nFirst Frame (t=0)')
        axes[i, 0].axis('off')
        
        # Middle frame
        mid_idx = n_timesteps // 2
        mid_frame = sequence[mid_idx].permute(1, 2, 0).numpy()
        mid_frame = np.clip(mid_frame, 0, 1)
        axes[i, 1].imshow(mid_frame)
        axes[i, 1].set_title(f'Middle Frame\n(t={mid_idx})')
        axes[i, 1].axis('off')
        
        # Last frame
        last_frame = sequence[-1].permute(1, 2, 0).numpy()
        last_frame = np.clip(last_frame, 0, 1)
        axes[i, 2].imshow(last_frame)
        axes[i, 2].set_title(f'Last Frame\n(t={n_timesteps-1})')
        axes[i, 2].axis('off')
        
        # Difference between first and last
        diff_frame = np.abs(first_frame - last_frame)
        im = axes[i, 3].imshow(diff_frame, cmap='hot')
        axes[i, 3].set_title(f'|First - Last|\nMSE: {mse_val:.2e}')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], shrink=0.6)
        
        # Sequence information
        axes[i, 4].axis('off')
        info_text = f"""SEQUENCE INFO
{"="*15}

Original Index: {original_idx}
Timesteps: {n_timesteps}
MSE (first vs last): {mse_val:.2e}

PIXEL STATISTICS:
Max difference: {diff_frame.max():.3f}
Mean difference: {diff_frame.mean():.3f}

CYCLICITY:
✅ CYCLIC SEQUENCE
(MSE < 1e-4)

SHAPE: {sequence.shape}
"""
        
        axes[i, 4].text(0.05, 0.95, info_text, transform=axes[i, 4].transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = output_path.replace('.pt', '_examples.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved cyclic examples visualization to: {viz_path}")
    plt.close()

def create_comparison_statistics(train_results, test_results):
    """Create comparison statistics between train and test cyclic extraction."""
    
    print(f"\n📊 TRAIN vs TEST COMPARISON:")
    print(f"=" * 60)
    
    train_stats = train_results['metadata']['extraction_stats']
    test_stats = test_results['metadata']['extraction_stats']
    
    print(f"{'Metric':<25} {'Train':<15} {'Test':<15} {'Difference':<15}")
    print(f"-" * 70)
    print(f"{'Total sequences':<25} {train_stats['total_original']:<15} {test_stats['total_original']:<15} {test_stats['total_original'] - train_stats['total_original']:<15}")
    print(f"{'Cyclic sequences':<25} {train_stats['total_cyclic']:<15} {test_stats['total_cyclic']:<15} {test_stats['total_cyclic'] - train_stats['total_cyclic']:<15}")
    print(f"{'Cyclic percentage':<25} {train_stats['cyclic_percentage']:.1f}%{'':<10} {test_stats['cyclic_percentage']:.1f}%{'':<10} {test_stats['cyclic_percentage'] - train_stats['cyclic_percentage']:+.1f}%{'':<10}")
    
    # MSE comparison
    train_mse = train_results['mse_values']
    test_mse = test_results['mse_values']
    
    print(f"\n📈 MSE STATISTICS COMPARISON:")
    print(f"{'Metric':<20} {'Train':<15} {'Test':<15}")
    print(f"-" * 50)
    print(f"{'Mean MSE':<20} {np.mean(train_mse):.2e:<15} {np.mean(test_mse):.2e}")
    print(f"{'Max MSE':<20} {np.max(train_mse):.2e:<15} {np.max(test_mse):.2e}")
    print(f"{'Min MSE':<20} {np.min(train_mse):.2e:<15} {np.min(test_mse):.2e}")
    print(f"{'Std MSE':<20} {np.std(train_mse):.2e:<15} {np.std(test_mse):.2e}")

def main():
    """Main extraction function."""
    parser = argparse.ArgumentParser(description='Extract cyclic sequences from Sprites dataset')
    parser.add_argument('--train_path', default='src/datasprites/Sprites_train.pt', 
                       help='Path to training sprites data')
    parser.add_argument('--test_path', default='src/datasprites/Sprites_test.pt',
                       help='Path to test sprites data')
    parser.add_argument('--output_dir', default='cyclic_sprites_data',
                       help='Output directory for cyclic sequences')
    parser.add_argument('--threshold', type=float, default=1e-4,
                       help='MSE threshold for considering sequences cyclic')
    parser.add_argument('--max_examples', type=int, default=15,
                       help='Maximum examples to show in visualization')
    
    args = parser.parse_args()
    
    print("🚀 Starting Cyclic Sequence Extraction")
    print(f"🎯 Threshold: {args.threshold}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract from training data
    print(f"\n{'='*60}")
    print(f"🏋️ PROCESSING TRAINING DATA")
    print(f"{'='*60}")
    
    if os.path.exists(args.train_path):
        train_output = os.path.join(args.output_dir, 'Sprites_train_cyclic.pt')
        train_results = analyze_and_extract_cyclic_sequences(
            data_path=args.train_path,
            output_path=train_output,
            similarity_threshold=args.threshold,
            visualize_examples=True,
            max_examples=args.max_examples
        )
    else:
        print(f"❌ Training data not found: {args.train_path}")
        train_results = None
    
    # Extract from test data
    print(f"\n{'='*60}")
    print(f"🧪 PROCESSING TEST DATA") 
    print(f"{'='*60}")
    
    if os.path.exists(args.test_path):
        test_output = os.path.join(args.output_dir, 'Sprites_test_cyclic.pt')
        test_results = analyze_and_extract_cyclic_sequences(
            data_path=args.test_path,
            output_path=test_output,
            similarity_threshold=args.threshold,
            visualize_examples=True,
            max_examples=args.max_examples
        )
    else:
        print(f"❌ Test data not found: {args.test_path}")
        test_results = None
    
    # Create comparison if both datasets processed
    if train_results and test_results:
        create_comparison_statistics(train_results, test_results)
    
    # Summary
    print(f"\n🎉 EXTRACTION COMPLETE!")
    print(f"=" * 60)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📋 Files created:")
    
    if train_results:
        print(f"   ✅ {os.path.join(args.output_dir, 'Sprites_train_cyclic.pt')}")
        print(f"   ✅ {os.path.join(args.output_dir, 'Sprites_train_cyclic_metadata.pt')}")
        print(f"   ✅ {os.path.join(args.output_dir, 'Sprites_train_cyclic_examples.png')}")
    
    if test_results:
        print(f"   ✅ {os.path.join(args.output_dir, 'Sprites_test_cyclic.pt')}")
        print(f"   ✅ {os.path.join(args.output_dir, 'Sprites_test_cyclic_metadata.pt')}")
        print(f"   ✅ {os.path.join(args.output_dir, 'Sprites_test_cyclic_examples.png')}")
    
    print(f"\n💡 Next steps:")
    print(f"   • Use the cyclic datasets for training experiments")
    print(f"   • Compare loop_mode='open' vs loop_mode='closed' performance")
    print(f"   • Analyze how cyclicity affects model learning")
    
    return train_results, test_results

if __name__ == "__main__":
    train_results, test_results = main() 