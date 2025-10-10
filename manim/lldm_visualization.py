import os
import sys
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from lightning.pytorch import LightningModule
import pandas as pd
import wandb
from sklearn.decomposition import PCA
import plotly.express as px
from PIL import Image
import base64
from io import BytesIO
from dash import Dash, dcc, html, Input, Output, no_update
from plotly.subplots import make_subplots

class LLDMLightning(LightningModule):
    def __init__(self, lldm_model, training_config):
        """
        Args:
            lldm_model: your LLDM instance (nn.Module)
            training_config: instance of BaseTrainerConfig with training hyperparameters
        """
        super().__init__()
        self.lldm = lldm_model
        self.training_config = training_config
        # Initialize validation outputs as a list
        self._validation_outputs = []
        # Add a counter for validation steps
        self._val_step_count = 0
        # Add a flag to track if we're in validation
        self._in_validation = False

    def forward(self, x):
        # Call the LLDM's reconstruct method
        return self.lldm.reconstruct({"data": x}, vi_index=5)

    def training_step(self, batch, batch_idx):
        # Extract the tensor from the batch
        x = batch["data"] if isinstance(batch, dict) else batch
        output = self.lldm.reconstruct({"data": x}, vi_index=5)
        
        # Handle different output formats
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            _, x_rec = output[:2]
        else:
            x_rec = output
        if isinstance(x_rec, dict) and "reconstruction" in x_rec:
            x_rec = x_rec["reconstruction"]

        # Calculate loss
        loss = 0.5 * F.mse_loss(
            x_rec.reshape(x.shape[0] * self.lldm.n_obs, -1),
            x.reshape(x.shape[0] * self.lldm.n_obs, -1),
            reduction="mean"
        )
        
        # Log training loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        print(f"[Training Step] Batch {batch_idx}: loss = {loss.item():.4f}")
        return loss

    def on_validation_epoch_start(self):
        """Called at the start of validation"""
        print(f"Starting validation epoch {self.current_epoch}")
        self._validation_outputs = []
        self._val_step_count = 0
        self._in_validation = True

    def validation_step(self, batch, batch_idx):
        if not self._in_validation:
            print("Warning: validation_step called outside of validation epoch")
            return None

        # Extract the tensor from the batch
        x = batch["data"] if isinstance(batch, dict) else batch
        output = self.lldm.reconstruct({"data": x}, vi_index=5)
        
        # Handle different output formats
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            _, x_rec = output[:2]
        else:
            x_rec = output
        if isinstance(x_rec, dict) and "reconstruction" in x_rec:
            x_rec = x_rec["reconstruction"]

        # Calculate validation loss
        loss = 0.5 * F.mse_loss(
            x_rec.reshape(x.shape[0] * self.lldm.n_obs, -1),
            x.reshape(x.shape[0] * self.lldm.n_obs, -1),
            reduction="mean"
        )
        
        # Log validation loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        
        # Store batch index for debugging
        self._val_step_count += 1
        print(f"[Validation Step] Batch {batch_idx} (Total: {self._val_step_count}): loss = {loss.item():.4f}")
        
        # Return validation outputs (move to CPU to save memory)
        outputs = {
            "val_loss": loss,
            "x_orig": x.detach().cpu(),
            "x_rec": x_rec.detach().cpu(),
            "batch_idx": batch_idx
        }
        
        # Add outputs to collection
        self._validation_outputs.append(outputs)
        print(f"[Validation Step] Added outputs for batch {batch_idx}")
        return outputs

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch"""
        print(f"Ending validation epoch {self.current_epoch}")
        print(f"Number of collected validation outputs: {len(self._validation_outputs)}")
        
        if not self._validation_outputs:
            print("Warning: No validation outputs were collected this epoch")
            self._in_validation = False
            return

        # Sort outputs by batch_idx to ensure consistent ordering
        sorted_outputs = sorted(self._validation_outputs, key=lambda x: x["batch_idx"])
        first_output = sorted_outputs[0]
        
        # Log the last frame for each epoch first
        if "x_orig" in first_output and "x_rec" in first_output:
            x_orig = first_output["x_orig"]
            x_rec = first_output["x_rec"]
            
            if len(x_orig.shape) == 5:  # [B, n_obs, C, H, W]
                sample_orig = x_orig[0]
                n_obs = sample_orig.shape[0]
                sample_rec = x_rec.reshape(n_obs, -1, 64, 64)
                
                last_frame_orig = sample_orig[-1]
                last_frame_rec = sample_rec[-1]
                
                # Convert to numpy and ensure proper shape for wandb
                orig_np = last_frame_orig.cpu().numpy()
                rec_np = last_frame_rec.cpu().numpy()
                
                # Print shapes for debugging
                print(f"Original shape: {orig_np.shape}")
                print(f"Reconstructed shape: {rec_np.shape}")
                
                # Handle the case where we have too many channels (before normalization)
                if len(orig_np.shape) == 3 and orig_np.shape[0] > 3:  # If channels-first and more than 3 channels
                    print(f"Reducing channels from {orig_np.shape[0]} to 3")
                    orig_np = orig_np[:3]  # Take first 3 channels
                    rec_np = rec_np[:3]  # Take first 3 channels
                
                # Create a more sophisticated visualization
                fig = go.Figure()

                # Create subplots for original and reconstructed images
                fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Original Last Frame', 'Reconstructed Last Frame'),
                    horizontal_spacing=0.1)

                # Add original image
                fig.add_trace(
                    go.Image(
                        z=orig_np.transpose(1, 2, 0),
                        name='Original',
                        hoverongaps=False,
                        hovertemplate="Original Image<br>Value: %{z}<extra></extra>"
                    ),
                    row=1, col=1
                )

                # Add reconstructed image
                fig.add_trace(
                    go.Image(
                        z=rec_np.transpose(1, 2, 0),
                        name='Reconstructed',
                        hoverongaps=False,
                        hovertemplate="Reconstructed Image<br>Value: %{z}<extra></extra>"
                    ),
                    row=1, col=2
                )

                # Update layout with seaborn-like styling
                fig.update_layout(
                    title=dict(
                        text=f'Last Frame Comparison (Epoch {self.current_epoch})',
                        font=dict(size=24, family='Arial, sans-serif', color='#37474F'),
                        x=0.5,
                        xanchor='center',
                        y=0.95
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    width=1000,
                    height=500,
                    showlegend=False,
                    margin=dict(l=40, r=40, t=100, b=40),
                )

                # Update axes
                fig.update_xaxes(showticklabels=False, showgrid=False)
                fig.update_yaxes(showticklabels=False, showgrid=False)

                # Add color scale annotation
                fig.add_annotation(
                    text="RGB Color Channels",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.1,
                    showarrow=False,
                    font=dict(size=12, family='Arial, sans-serif', color='#37474F'),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='#37474F',
                    borderwidth=1,
                    borderpad=4
                )

                # Log to wandb
                self.logger.experiment.log({
                    "Last Frame Comparison": wandb.Plotly(fig),
                    "epoch": self.current_epoch
                })

                # Also log individual images for compatibility
                self.logger.experiment.log({
                    "Last Frame Original": wandb.Image(orig_np.transpose(1, 2, 0), caption=f"Original Last Frame"),
                    "Last Frame Reconstructed": wandb.Image(rec_np.transpose(1, 2, 0), caption=f"Reconstructed Last Frame"),
                    "epoch": self.current_epoch
                })
        
        # Now handle PCA visualization
        latents = []
        images = []
        
        for output in sorted_outputs:
            if "x_orig" in output and "x_rec" in output:
                x_orig = output["x_orig"]  # Shape: [B, n_obs, C, H, W]
                latent = self.lldm.encode(x_orig)
                latents.append(latent.cpu().numpy())
                # Take the last frame of each sequence
                last_frames = x_orig[:, -1].cpu().numpy()  # Shape: [B, C, H, W]
                images.append(last_frames)
        
        latents = np.concatenate(latents, axis=0)
        latents = latents.reshape(latents.shape[0], -1)
        
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'PCA1': latents_2d[:, 0],
            'PCA2': latents_2d[:, 1],
            'Distance': np.sqrt(latents_2d[:, 0]**2 + latents_2d[:, 1]**2),  # Distance from origin
            'epoch': [self.current_epoch] * len(latents_2d)
        })

        # Create the scatter plot with seaborn-like styling
        fig = go.Figure()

        # Add main scatter points with continuous color based on distance from origin
        fig.add_trace(go.Scatter(
            x=df['PCA1'],
            y=df['PCA2'],
            mode='markers',
            marker=dict(
                size=10,
                color=df['Distance'],
                colorscale='viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text='Distance from Origin',
                        font=dict(size=12, family='Arial, sans-serif')
                    ),
                    thickness=15,
                    len=0.7,
                    tickfont=dict(size=10)
                ),
                line=dict(
                    color='white',
                    width=1
                )
            ),
            hovertemplate=(
                "<b>Point Information</b><br>" +
                "PCA1: %{x:.3f}<br>" +
                "PCA2: %{y:.3f}<br>" +
                "Distance from Origin: %{marker.color:.3f}<br>" +
                "<extra></extra>"
            ),
            name='Latent Points'
        ))

        # Add zero lines with custom styling
        fig.add_hline(y=0, line=dict(color='rgba(128, 128, 128, 0.2)', width=1, dash='dash'))
        fig.add_vline(x=0, line=dict(color='rgba(128, 128, 128, 0.2)', width=1, dash='dash'))

        # Update layout with seaborn-like styling
        fig.update_layout(
            title=dict(
                text=f'Latent Space PCA Visualization (Epoch {self.current_epoch})',
                font=dict(size=24, family='Arial, sans-serif', color='#37474F'),
                x=0.5,
                xanchor='center',
                y=0.95
            ),
            xaxis=dict(
                title=dict(
                    text='First Principal Component',
                    font=dict(size=14, family='Arial, sans-serif', color='#37474F')
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.1)',
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='#37474F',
                mirror=True,
                tickfont=dict(size=12, family='Arial, sans-serif', color='#37474F')
            ),
            yaxis=dict(
                title=dict(
                    text='Second Principal Component',
                    font=dict(size=14, family='Arial, sans-serif', color='#37474F')
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.1)',
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='#37474F',
                mirror=True,
                tickfont=dict(size=12, family='Arial, sans-serif', color='#37474F')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1000,
            height=800,
            showlegend=False,
            margin=dict(l=80, r=80, t=100, b=80),
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Arial, sans-serif',
                bordercolor='#37474F'
            ),
            hovermode='closest'
        )

        # Add annotations for explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        fig.add_annotation(
            text=f'Explained Variance:<br>PC1: {explained_variance_ratio[0]:.1%}<br>PC2: {explained_variance_ratio[1]:.1%}',
            xref='paper',
            yref='paper',
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=12, family='Arial, sans-serif', color='#37474F'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#37474F',
            borderwidth=1,
            borderpad=4
        )

        # Log to wandb using Plotly
        self.logger.experiment.log({
            "Latent Space PCA": wandb.Plotly(fig),
            "epoch": self.current_epoch
        })
        
        print(f"Logged PCA visualization for epoch {self.current_epoch}")
        
        # Clear validation outputs for next epoch
        self._validation_outputs = []
        self._in_validation = False

    def configure_optimizers(self):
        """Configure the optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.training_config.learning_rate)
        return optimizer 
        return optimizer 