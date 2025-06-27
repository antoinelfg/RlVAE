"""
Base Visualization Class
=======================

Provides common functionality for all visualization modules.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from pathlib import Path


class BaseVisualization:
    """Base class for all visualization modules."""
    
    def __init__(self, model, device, config, should_log_to_wandb=True):
        self.model = model
        self.device = device
        self.config = config
        self._should_log_to_wandb = should_log_to_wandb
        
    def should_log_to_wandb(self):
        """Check if WandB logging is enabled."""
        return self._should_log_to_wandb and wandb.run is not None
        
    def _get_output_path(self, filename, subfolder="visualizations"):
        """Get output path for files - organized in wandb folder structure."""
        output_dir = f"wandb/{subfolder}"
        os.makedirs(output_dir, exist_ok=True)
        return f"{output_dir}/{filename}"
    
    def _safe_save_plt_figure(self, filename, **kwargs):
        """Safely save matplotlib figure to organized wandb folder."""
        output_path = self._get_output_path(filename, "plots")
        
        try:
            plt.savefig(output_path, **kwargs)
            print(f"💾 Saved matplotlib figure: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Failed to save matplotlib figure {output_path}: {e}")
            return None
    
    def _safe_write_image(self, fig, filename, **kwargs):
        """Safely write Plotly figure to organized wandb folder."""
        output_path = self._get_output_path(filename, "interactive")
            
        try:
            # Check if figure has frames (animated)
            if hasattr(fig, 'frames') and fig.frames:
                # Save as HTML instead for animated figures
                html_filename = output_path.replace('.png', '.html')
                fig.write_html(html_filename)
                print(f"💾 Saved animated figure as HTML: {html_filename}")
                return html_filename
            else:
                # Regular static figure - safe to export as PNG
                fig.write_image(output_path, **kwargs)
                print(f"💾 Saved static figure as PNG: {output_path}")
                return output_path
        except Exception as e:
            print(f"⚠️ Image export failed for {output_path}: {e}")
            # Fallback: try to save as HTML
            try:
                html_filename = output_path.replace('.png', '.html')
                fig.write_html(html_filename)
                print(f"💾 Fallback: Saved as HTML: {html_filename}")
                return html_filename
            except Exception as e2:
                print(f"❌ Both PNG and HTML export failed: {e2}")
                return None
                
    def model_forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
        
    def _prepare_pca_data(self, z_seq, n_components=3):
        """Prepare PCA projection of latent sequences."""
        from sklearn.decomposition import PCA
        
        batch_size, n_obs, latent_dim = z_seq.shape
        z_flat = z_seq.reshape(-1, latent_dim).cpu().numpy()
        
        pca = PCA(n_components=n_components)
        z_pca = pca.fit_transform(z_flat)
        z_pca_seq = z_pca.reshape(batch_size, n_obs, n_components)
        
        return z_pca_seq, pca
        
    def _get_viz_count(self):
        """Get number of sequences to visualize."""
        return getattr(self.config, 'sequence_viz_count', 8) 