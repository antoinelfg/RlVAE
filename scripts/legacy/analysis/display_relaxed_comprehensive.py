#!/usr/bin/env python3
"""
Display Relaxed Comprehensive Sampling Plot
==========================================
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def display_relaxed_comprehensive_plot():
    """Display the relaxed comprehensive sampling analysis plot."""
    print("🎨 Displaying Relaxed Comprehensive G⁻¹ Sampling Analysis")
    print("=" * 60)
    
    try:
        # Load and display the image
        img = mpimg.imread('relaxed_comprehensive_sampling_analysis.png')
        
        plt.figure(figsize=(16, 12))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Relaxed Comprehensive G⁻¹ Sampling Analysis\n(Same Data as Comprehensive Analysis, Relaxed Conditions)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add text overlay with key differences
        plt.figtext(0.02, 0.02, 
                   "Key Features:\n" +
                   "• Uses EXACT same data as comprehensive_g_inverse_analysis.py\n" +
                   "• Same 5000 data points, 50 centroids, metric matrices\n" +
                   "• Same pretrained encoder/decoder/metric components\n" +
                   "• Relaxed conditions: larger steps, more randomness\n" +
                   "• Wider bounds: [-5, 5] vs [-4, 4]\n" +
                   "• More exploration while respecting G⁻¹ metric",
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Plot displayed successfully!")
        print("📊 This shows relaxed sampling with the SAME data as comprehensive analysis:")
        print("   - Same 5000 latent data points")
        print("   - Same 50 centroids from k-means")
        print("   - Same metric matrices and pretrained components")
        print("   - But with relaxed conditions for more flexibility")
        
    except Exception as e:
        print(f"❌ Error displaying plot: {e}")
        print("   Make sure 'relaxed_comprehensive_sampling_analysis.png' exists")

if __name__ == "__main__":
    display_relaxed_comprehensive_plot() 