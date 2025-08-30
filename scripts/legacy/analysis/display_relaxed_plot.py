#!/usr/bin/env python3
"""
Display Relaxed Manifold Sampling Plot
======================================
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def display_relaxed_plot():
    """Display the relaxed manifold sampling analysis plot."""
    print("🎨 Displaying Relaxed Manifold Sampling Analysis")
    print("=" * 50)
    
    try:
        # Load and display the image
        img = mpimg.imread('relaxed_manifold_sampling_analysis.png')
        
        plt.figure(figsize=(16, 12))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Relaxed Manifold-Guided Sampling Analysis\n(More Flexible Conditions)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add text overlay with key differences
        plt.figtext(0.02, 0.02, 
                   "Key Relaxations Applied:\n" +
                   "• Larger step sizes (0.3 vs 0.1)\n" +
                   "• Looser bounds (-5 to 5 vs -4 to 4)\n" +
                   "• More randomness (60% vs 70% metric-guided)\n" +
                   "• Less aggressive metric scaling\n" +
                   "• More frequent direction changes",
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Plot displayed successfully!")
        print("📊 This shows the relaxed sampling with:")
        print("   - More spread-out sampling around centroids")
        print("   - Greater exploration between regions")
        print("   - Less tight clustering while respecting G⁻¹ metric")
        
    except Exception as e:
        print(f"❌ Error displaying plot: {e}")
        print("   Make sure 'relaxed_manifold_sampling_analysis.png' exists")

if __name__ == "__main__":
    display_relaxed_plot() 