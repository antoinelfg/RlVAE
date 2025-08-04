"""
Enhanced Visualization Components
================================

Comprehensive visualization components for the RlVAE Streamlit app including
training progress, model comparisons, latent space analysis, and interactive plots.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import torch
from datetime import datetime
import json


class TrainingProgressVisualizer:
    """Visualize training progress with real-time updates."""
    
    def __init__(self):
        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'riemannian_kl': [],
            'learning_rate': []
        }
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics history."""
        for key in self.metrics_history:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
    
    def plot_training_curves(self, title: str = "Training Progress"):
        """Plot training curves with Plotly."""
        
        if not self.metrics_history['epoch']:
            st.info("No training data available yet.")
            return
        
        # Create subplots
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Loss', 'Reconstruction Loss', 'KL Loss', 'Learning Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Total loss
        fig.add_trace(
            go.Scatter(
                x=self.metrics_history['epoch'],
                y=self.metrics_history['train_loss'],
                mode='lines+markers',
                name='Train Loss',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.metrics_history['epoch'],
                y=self.metrics_history['val_loss'],
                mode='lines+markers',
                name='Val Loss',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Reconstruction loss
        fig.add_trace(
            go.Scatter(
                x=self.metrics_history['epoch'],
                y=self.metrics_history['reconstruction_loss'],
                mode='lines+markers',
                name='Reconstruction Loss',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # KL loss
        fig.add_trace(
            go.Scatter(
                x=self.metrics_history['epoch'],
                y=self.metrics_history['kl_loss'],
                mode='lines+markers',
                name='KL Loss',
                line=dict(color='orange', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # Riemannian KL loss (if available)
        if self.metrics_history['riemannian_kl'] and any(v > 0 for v in self.metrics_history['riemannian_kl']):
            fig.add_trace(
                go.Scatter(
                    x=self.metrics_history['epoch'],
                    y=self.metrics_history['riemannian_kl'],
                    mode='lines+markers',
                    name='Riemannian KL',
                    line=dict(color='purple', width=2),
                    marker=dict(size=6)
                ),
                row=2, col=1
            )
        
        # Learning rate
        fig.add_trace(
            go.Scatter(
                x=self.metrics_history['epoch'],
                y=self.metrics_history['learning_rate'],
                mode='lines+markers',
                name='Learning Rate',
                line=dict(color='brown', width=2),
                marker=dict(size=6)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=2, col=1)
        fig.update_yaxes(title_text="LR", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_loss_breakdown(self):
        """Plot detailed loss breakdown."""
        
        if not self.metrics_history['epoch']:
            return
        
        # Create loss breakdown
        fig = go.Figure()
        
        # Stacked area chart for loss components
        fig.add_trace(go.Scatter(
            x=self.metrics_history['epoch'],
            y=self.metrics_history['reconstruction_loss'],
            mode='lines',
            name='Reconstruction Loss',
            stackgroup='one',
            fillcolor='rgba(0, 255, 0, 0.3)',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.metrics_history['epoch'],
            y=self.metrics_history['kl_loss'],
            mode='lines',
            name='KL Loss',
            stackgroup='one',
            fillcolor='rgba(255, 165, 0, 0.3)',
            line=dict(color='orange')
        ))
        
        if self.metrics_history['riemannian_kl'] and any(v > 0 for v in self.metrics_history['riemannian_kl']):
            fig.add_trace(go.Scatter(
                x=self.metrics_history['epoch'],
                y=self.metrics_history['riemannian_kl'],
                mode='lines',
                name='Riemannian KL',
                stackgroup='one',
                fillcolor='rgba(128, 0, 128, 0.3)',
                line=dict(color='purple')
            ))
        
        fig.update_layout(
            title="Loss Component Breakdown",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)


class ModelComparisonVisualizer:
    """Visualize model comparisons and analysis."""
    
    def plot_model_comparison(self, comparison_data: Dict[str, Any]):
        """Plot comprehensive model comparison."""
        
        if not comparison_data or 'models' not in comparison_data:
            st.warning("No comparison data available.")
            return
        
        models = comparison_data['models']
        metrics_comparison = comparison_data.get('metrics_comparison', {})
        
        # Create comparison dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            self._plot_performance_comparison(models, metrics_comparison)
        
        with col2:
            self._plot_architecture_comparison(models)
        
        # Detailed metrics comparison
        if metrics_comparison:
            self._plot_metrics_heatmap(metrics_comparison)
        
        # Configuration comparison
        config_comparison = comparison_data.get('config_comparison', {})
        if config_comparison:
            self._plot_config_comparison(config_comparison)
    
    def _plot_performance_comparison(self, models: Dict[str, Any], metrics: Dict[str, Any]):
        """Plot performance comparison."""
        
        # Extract performance metrics
        model_names = list(models.keys())
        final_losses = [models[mid]['final_loss'] for mid in model_names]
        best_val_losses = [models[mid]['best_val_loss'] for mid in model_names]
        
        fig = go.Figure()
        
        # Final loss comparison
        fig.add_trace(go.Bar(
            x=model_names,
            y=final_losses,
            name='Final Loss',
            marker_color='lightblue',
            text=[f'{loss:.4f}' for loss in final_losses],
            textposition='auto'
        ))
        
        # Best validation loss comparison
        fig.add_trace(go.Bar(
            x=model_names,
            y=best_val_losses,
            name='Best Val Loss',
            marker_color='lightcoral',
            text=[f'{loss:.4f}' for loss in best_val_losses],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Loss",
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_architecture_comparison(self, models: Dict[str, Any]):
        """Plot architecture comparison."""
        
        model_names = list(models.keys())
        latent_dims = [models[mid]['latent_dim'] for mid in model_names]
        n_flows = [models[mid]['n_flows'] for mid in model_names]
        model_types = [models[mid]['type'] for mid in model_names]
        
        fig = go.Figure()
        
        # Scatter plot of latent dim vs n_flows
        fig.add_trace(go.Scatter(
            x=latent_dims,
            y=n_flows,
            mode='markers+text',
            text=model_names,
            textposition="top center",
            marker=dict(
                size=15,
                color=[hash(t) % 20 for t in model_types],  # Color by type
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Model Type")
            ),
            hovertemplate="<b>%{text}</b><br>" +
                         "Latent Dim: %{x}<br>" +
                         "N Flows: %{y}<br>" +
                         "<extra></extra>"
        ))
        
        fig.update_layout(
            title="Architecture Comparison",
            xaxis_title="Latent Dimension",
            yaxis_title="Number of Flows",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_metrics_heatmap(self, metrics_comparison: Dict[str, Any]):
        """Plot metrics comparison heatmap."""
        
        if not metrics_comparison:
            return
        
        # Prepare data for heatmap
        metric_names = list(metrics_comparison.keys())
        model_names = list(next(iter(metrics_comparison.values())).keys())
        
        # Create matrix
        matrix = []
        for metric in metric_names:
            row = []
            for model in model_names:
                value = metrics_comparison[metric].get(model, np.nan)
                row.append(value)
            matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=model_names,
            y=metric_names,
            colorscale='Viridis',
            text=[[f'{val:.4f}' if not np.isnan(val) else 'N/A' for val in row] for row in matrix],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Metrics Comparison Heatmap",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_config_comparison(self, config_comparison: Dict[str, Any]):
        """Plot configuration comparison."""
        
        if not config_comparison:
            return
        
        # Select key configuration parameters
        key_params = ['latent_dim', 'n_flows', 'beta', 'riemannian_beta', 'learning_rate', 'batch_size']
        
        fig = go.Figure()
        
        for i, param in enumerate(key_params):
            if param in config_comparison:
                values = list(config_comparison[param].values())
                model_names = list(config_comparison[param].keys())
                
                # Convert to numeric if possible
                numeric_values = []
                for val in values:
                    try:
                        numeric_values.append(float(val))
                    except (ValueError, TypeError):
                        numeric_values.append(i)  # Use index for non-numeric
                
                fig.add_trace(go.Bar(
                    x=model_names,
                    y=numeric_values,
                    name=param,
                    text=[str(v) for v in values],
                    textposition='auto'
                ))
        
        fig.update_layout(
            title="Configuration Comparison",
            xaxis_title="Models",
            yaxis_title="Parameter Values",
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)


class LatentSpaceVisualizer:
    """Visualize latent space analysis."""
    
    def __init__(self):
        self.latent_data = None
        self.labels = None
    
    def set_latent_data(self, latent_vectors: np.ndarray, labels: Optional[np.ndarray] = None):
        """Set latent space data for visualization."""
        self.latent_data = latent_vectors
        self.labels = labels
    
    def plot_latent_space_2d(self, method: str = 'pca', title: str = "Latent Space Visualization"):
        """Plot 2D latent space visualization."""
        
        if self.latent_data is None:
            st.warning("No latent data available. Please encode some data first.")
            return
        
        # Dimensionality reduction
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                st.error("UMAP not available. Please install with: pip install umap-learn")
                return
        else:
            st.error(f"Unknown reduction method: {method}")
            return
        
        # Reduce dimensions
        with st.spinner(f"Computing {method.upper()}..."):
            latent_2d = reducer.fit_transform(self.latent_data)
        
        # Create scatter plot
        fig = go.Figure()
        
        if self.labels is not None:
            # Color by labels
            unique_labels = np.unique(self.labels)
            colors = px.colors.qualitative.Set3[:len(unique_labels)]
            
            for i, label in enumerate(unique_labels):
                mask = self.labels == label
                fig.add_trace(go.Scatter(
                    x=latent_2d[mask, 0],
                    y=latent_2d[mask, 1],
                    mode='markers',
                    name=f'Class {label}',
                    marker=dict(
                        size=8,
                        color=colors[i],
                        opacity=0.7
                    ),
                    hovertemplate=f'Class {label}<br>' +
                                 'X: %{x:.3f}<br>' +
                                 'Y: %{y:.3f}<br>' +
                                 '<extra></extra>'
                ))
        else:
            # No labels, use single color
            fig.add_trace(go.Scatter(
                x=latent_2d[:, 0],
                y=latent_2d[:, 1],
                mode='markers',
                name='Latent Points',
                marker=dict(
                    size=8,
                    color='blue',
                    opacity=0.7
                ),
                hovertemplate='X: %{x:.3f}<br>' +
                             'Y: %{y:.3f}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"{title} ({method.upper()})",
            xaxis_title=f"{method.upper()} Component 1",
            yaxis_title=f"{method.upper()} Component 2",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_latent_distribution(self):
        """Plot latent space distribution."""
        
        if self.latent_data is None:
            st.warning("No latent data available.")
            return
        
        # Create subplots for each latent dimension
        n_dims = min(8, self.latent_data.shape[1])  # Show max 8 dimensions
        
        fig = sp.make_subplots(
            rows=2, cols=4,
            subplot_titles=[f'Dim {i+1}' for i in range(n_dims)]
        )
        
        for i in range(n_dims):
            row = (i // 4) + 1
            col = (i % 4) + 1
            
            fig.add_trace(
                go.Histogram(
                    x=self.latent_data[:, i],
                    nbinsx=30,
                    name=f'Dim {i+1}',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Latent Space Distribution",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_interpolation_path(self, z1: np.ndarray, z2: np.ndarray, 
                               interpolated: np.ndarray, title: str = "Latent Interpolation"):
        """Plot interpolation path in latent space."""
        
        # Reduce to 2D for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        
        # Combine all points for PCA
        all_points = np.vstack([z1, z2, interpolated])
        all_points_2d = pca.fit_transform(all_points)
        
        # Extract transformed points
        z1_2d = all_points_2d[0:1]
        z2_2d = all_points_2d[1:2]
        interpolated_2d = all_points_2d[2:]
        
        fig = go.Figure()
        
        # Plot interpolation path
        fig.add_trace(go.Scatter(
            x=interpolated_2d[:, 0],
            y=interpolated_2d[:, 1],
            mode='lines+markers',
            name='Interpolation Path',
            line=dict(color='blue', width=3),
            marker=dict(size=8, color='blue')
        ))
        
        # Plot start and end points
        fig.add_trace(go.Scatter(
            x=[z1_2d[0, 0], z2_2d[0, 0]],
            y=[z1_2d[0, 1], z2_2d[0, 1]],
            mode='markers',
            name='Endpoints',
            marker=dict(size=15, color=['green', 'red'], symbol='diamond')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


class SystemMonitorVisualizer:
    """Visualize system monitoring information."""
    
    def plot_system_metrics(self, device_info: Dict[str, Any]):
        """Plot system monitoring metrics."""
        
        # Create system overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Device",
                device_info.get('device', 'Unknown'),
                delta=None
            )
        
        with col2:
            if device_info.get('cuda_available'):
                st.metric(
                    "GPU Memory",
                    f"{device_info.get('gpu_memory_used_gb', 0):.1f} GB",
                    delta=None
                )
            else:
                st.metric(
                    "System Memory",
                    f"{device_info.get('available_memory_gb', 0):.1f} GB",
                    delta=None
                )
        
        with col3:
            if device_info.get('cuda_available'):
                st.metric(
                    "GPU Utilization",
                    f"{device_info.get('gpu_utilization', 0):.1f}%",
                    delta=None
                )
            else:
                st.metric(
                    "CPU Cores",
                    device_info.get('num_cpus', 0),
                    delta=None
                )
        
        with col4:
            st.metric(
                "PyTorch Version",
                device_info.get('torch_version', 'Unknown'),
                delta=None
            )
        
        # Detailed system information
        with st.expander("🔧 Detailed System Information"):
            st.json(device_info)


class ExperimentHistoryVisualizer:
    """Visualize experiment history and statistics."""
    
    def plot_experiment_timeline(self, experiments: List[Dict[str, Any]]):
        """Plot experiment timeline."""
        
        if not experiments:
            st.info("No experiments in history.")
            return
        
        # Prepare timeline data
        timeline_data = []
        for exp in experiments:
            timeline_data.append({
                'name': exp.get('name', 'Unknown'),
                'start_time': exp.get('start_time', datetime.now()),
                'end_time': exp.get('end_time', datetime.now()),
                'status': exp.get('status', 'unknown'),
                'model_type': exp.get('model_type', 'unknown'),
                'final_loss': exp.get('final_loss', 0.0)
            })
        
        # Create timeline
        fig = go.Figure()
        
        colors = {'completed': 'green', 'failed': 'red', 'running': 'blue', 'stopped': 'orange'}
        
        for exp in timeline_data:
            fig.add_trace(go.Scatter(
                x=[exp['start_time'], exp['end_time']],
                y=[exp['name'], exp['name']],
                mode='lines+markers',
                name=exp['model_type'],
                line=dict(color=colors.get(exp['status'], 'gray'), width=3),
                marker=dict(size=10),
                hovertemplate=f"<b>{exp['name']}</b><br>" +
                             f"Status: {exp['status']}<br>" +
                             f"Model: {exp['model_type']}<br>" +
                             f"Final Loss: {exp['final_loss']:.4f}<br>" +
                             "<extra></extra>"
            ))
        
        fig.update_layout(
            title="Experiment Timeline",
            xaxis_title="Time",
            yaxis_title="Experiments",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_experiment_statistics(self, experiments: List[Dict[str, Any]]):
        """Plot experiment statistics."""
        
        if not experiments:
            return
        
        # Extract statistics
        model_types = [exp.get('model_type', 'unknown') for exp in experiments]
        final_losses = [exp.get('final_loss', 0.0) for exp in experiments]
        durations = []
        
        for exp in experiments:
            start = exp.get('start_time', datetime.now())
            end = exp.get('end_time', datetime.now())
            if isinstance(start, str):
                start = datetime.fromisoformat(start)
            if isinstance(end, str):
                end = datetime.fromisoformat(end)
            duration = (end - start).total_seconds() / 3600  # hours
            durations.append(duration)
        
        # Create statistics plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Model type distribution
            fig = px.pie(
                values=[model_types.count(t) for t in set(model_types)],
                names=list(set(model_types)),
                title="Model Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Loss distribution
            fig = px.histogram(
                x=final_losses,
                title="Final Loss Distribution",
                nbins=20
            )
            fig.update_layout(xaxis_title="Final Loss", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        # Duration vs Loss scatter plot
        fig = px.scatter(
            x=durations,
            y=final_losses,
            color=model_types,
            title="Training Duration vs Final Loss",
            labels={'x': 'Duration (hours)', 'y': 'Final Loss'}
        )
        st.plotly_chart(fig, use_container_width=True)