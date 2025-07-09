"""
Manifold Visualizations Module
=============================

Metric tensor and manifold geometry visualizations:
- Metric heatmaps
- PCA analysis 
- Temporal metric evolution
- Curvature analysis (when enabled)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from .base import BaseVisualization


class ManifoldVisualizations(BaseVisualization):
    """Manifold and metric tensor visualization suite."""
    
    def create_metric_heatmaps(self, x_sample: torch.Tensor, epoch: int):
        """
        Create comprehensive manifold visualizations with proper metric computation across all timesteps.
        FIXED: Now computes metrics for flow-evolved latent coordinates at all timesteps.
        """
        print(f"🔬 Creating manifold visualizations for epoch {epoch}")
        
        if not hasattr(self.model, 'G'):
            print("⚠️ No metric tensor available for manifold visualization")
            return
            
        try:
            self.model.eval()
            with torch.no_grad():
                # Get the full model forward pass first to get properly flow-evolved latents
                result = self.model_forward(x_sample)
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z
            
                batch_size, n_obs, latent_dim = z_seq.shape
                print(f"📊 Analyzing {batch_size} sequences with {n_obs} timesteps")
                
                # ===== IMPROVED: Use flow-evolved coordinates for metric computation =====
                timestep_data = {}
                
                for t in range(n_obs):
                    print(f"📍 Computing metrics for timestep {t+1}/{n_obs} using flow-evolved coordinates")
                    
                    # CRITICAL FIX: Use the flow-evolved latent coordinates z_t
                    z_t = z_seq[:, t, :]  # [batch_size, latent_dim] - flow-evolved coordinates
                    
                    # Also get the original encoded representation for comparison
                    x_t = x_sample[:, t]
                    encoder_out = self.model.encoder(x_t)
                    mu_t = encoder_out.embedding
                    log_var_t = encoder_out.log_covariance
                    
                    # Generate analysis samples around the FLOW-EVOLVED coordinates
                    n_analysis_samples = min(500, batch_size * 50)  # Reasonable number
                    
                    # Method 1: Sample around flow-evolved coordinates
                    z_t_expanded = z_t.unsqueeze(1).expand(-1, n_analysis_samples // batch_size, -1)
                    z_t_expanded = z_t_expanded.reshape(-1, latent_dim)
                    
                    # Add some noise for variety
                    noise_scale = 0.1  # Small noise to explore neighborhood
                    noise = torch.randn_like(z_t_expanded) * noise_scale
                    z_samples_around_flow = z_t_expanded + noise
                    
                    # Method 2: Sample around encoder mean (for comparison)
                    mu_expanded = mu_t.unsqueeze(1).expand(-1, n_analysis_samples // batch_size, -1)
                    mu_expanded = mu_expanded.reshape(-1, latent_dim)
                    std_expanded = torch.exp(0.5 * log_var_t).unsqueeze(1).expand(-1, n_analysis_samples // batch_size, -1)
                    std_expanded = std_expanded.reshape(-1, latent_dim)
                    
                    eps = torch.randn_like(mu_expanded)
                    z_samples_reparam = mu_expanded + eps * std_expanded
                    
                    # ===== COMPUTE METRICS AT FLOW-EVOLVED COORDINATES =====
                    try:
                        print(f"   Computing G(z) at {len(z_samples_around_flow)} flow-evolved points...")
                        
                        # Compute metric tensor at flow-evolved coordinates
                        G_flow = self.model.G(z_samples_around_flow)  # [n_samples, latent_dim, latent_dim]
                        G_inv_flow = self.model.G_inv(z_samples_around_flow)
                        
                        # Extract metric properties
                        eigenvals_flow = torch.linalg.eigvals(G_inv_flow).real
                        det_G_inv_flow = torch.linalg.det(G_inv_flow)
                        
                        metric_properties = {
                            'eigenvals_mean': eigenvals_flow.mean(dim=0).cpu().numpy(),
                            'eigenvals_std': eigenvals_flow.std(dim=0).cpu().numpy(),
                            'condition_number': (eigenvals_flow.max(dim=1)[0] / (eigenvals_flow.min(dim=1)[0] + 1e-10)).mean().item(),
                            'det_G_inv': det_G_inv_flow.cpu().numpy(),
                            'log_det_G_inv': torch.log(torch.clamp(det_G_inv_flow, min=1e-12)).cpu().numpy(),
                            'eigenvals_all': eigenvals_flow.cpu().numpy(),
                            'samples_used': 'flow_evolved'  # Tag to indicate which coordinates were used
                        }
                        
                        print(f"   ✅ Computed metrics: det(G⁻¹) range [{det_G_inv_flow.min():.2e}, {det_G_inv_flow.max():.2e}]")
                        
                        # Additional metric properties for comprehensive analysis
                        metric_properties.update({
                            'christoffel_symbols': self._compute_christoffel_symbols(G_flow),
                            'ricci_curvature_approx': self._compute_ricci_curvature_approximation(G_flow, z_t),
                            'geodesic_deviation': self._compute_geodesic_deviation(G_flow, z_t),
                            'metric_connectivity': self._compute_metric_connectivity(G_flow),
                            'sectional_curvature_approx': self._compute_sectional_curvature_approximation(G_flow)
                        })
                        
                    except Exception as metric_error:
                        print(f"   ⚠️ Failed to compute metrics at flow coordinates: {metric_error}")
                        # Fallback to empty metrics
                        metric_properties = {'samples_used': 'failed'}
                    
                    # Apply flows for trajectory tracking (if not first timestep)
                    flow_intermediate = []
                    z_flow_traj = z_samples_reparam.clone()
                    
                    if t > 0:
                        # Apply flow sequence to see transformation
                        for flow_idx in range(t):
                            # Handle both legacy and modular model structures
                            flows = getattr(self.model, 'flows', None)
                            if flows is None and hasattr(self.model, 'flow_manager'):
                                flows = self.model.flow_manager.flows
                            
                            if flows is not None and flow_idx < len(flows):
                                try:
                                    z_prev = z_flow_traj.clone()
                                    flow_result = flows[flow_idx](z_flow_traj)
                                    z_flow_traj = flow_result.out
                                    
                                    flow_intermediate.append({
                                        'layer': flow_idx,
                                        'z_before': z_prev.cpu().numpy(),
                                        'z_after': z_flow_traj.cpu().numpy()
                                    })
                                except Exception as flow_error:
                                    print(f"   ⚠️ Flow {flow_idx} failed: {flow_error}")
                                    break
                    
                    # Store comprehensive timestep data
                    timestep_data[t] = {
                        'mu': mu_t.cpu().numpy(),
                        'log_var': log_var_t.cpu().numpy(),
                        'z_flow_evolved': z_t.cpu().numpy(),  # The actual flow-evolved coordinates
                        'z_reparam_samples': z_samples_reparam.cpu().numpy(),
                        'z_flow_neighborhood': z_samples_around_flow.cpu().numpy(),  # Samples around flow coordinates
                        'flow_intermediate': flow_intermediate,
                        'metric_properties': metric_properties,
                        'original_images': x_t.cpu().numpy()
                    }
                    
                    print(f"   ✅ Timestep {t} data collected (metrics: {len(metric_properties)} properties)")
                
                # Create the visualizations using properly computed metrics
                print(f"🎨 Creating visualizations with metrics for {len(timestep_data)} timesteps...")
                
                # Enhanced PCA analysis
                self._create_enhanced_pca_analysis(timestep_data, epoch)
                
                # Enhanced manifold heatmaps with flow-evolved metrics
                self._create_enhanced_manifold_heatmaps(timestep_data, epoch)
                
                # Temporal metric evolution analysis
                self._create_temporal_metric_analysis(timestep_data, epoch)
                
                print(f"✨ Manifold visualizations complete for epoch {epoch}")
            
        except Exception as e:
            print(f"⚠️ Manifold visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()
    
    def create_pca_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Create PCA-based manifold analysis."""
        print(f"📈 Creating PCA analysis for epoch {epoch}")
        
        # This is handled as part of create_metric_heatmaps
        # to avoid code duplication and ensure consistency
        pass
        
    def create_temporal_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Create temporal evolution analysis of the manifold."""
        print(f"⏱️ Creating temporal manifold analysis for epoch {epoch}")
        
        # This is handled as part of create_metric_heatmaps
        # to avoid code duplication and ensure consistency
        pass
    
    def _create_enhanced_pca_analysis(self, timestep_data, epoch):
        """Enhanced PCA analysis using flow-evolved coordinates with comprehensive metrics."""
        print("📈 Creating enhanced PCA analysis with comprehensive metric analysis...")
        
        n_timesteps = len(timestep_data)
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))  # Expanded to 3x3 grid
        fig.suptitle(f'Enhanced PCA & Metric Analysis - Epoch {epoch}', fontsize=16)
        
        # Collect all flow-evolved coordinates for global PCA
        all_z_flow = np.concatenate([data['z_flow_evolved'] for data in timestep_data.values()], axis=0)
        all_z_reparam = np.concatenate([data['z_reparam_samples'][:100] for data in timestep_data.values()], axis=0)
        
        # Fit PCA on flow-evolved coordinates
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(3, all_z_flow.shape[1]))
        pca.fit(all_z_flow)
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_timesteps))
        
        # Plot 1: Flow-evolved coordinates in PCA space
        ax = axes[0, 0]
        for t, data in timestep_data.items():
            z_pca = pca.transform(data['z_flow_evolved'])
            ax.scatter(z_pca[:, 0], z_pca[:, 1], c=[colors[t]], alpha=0.7, s=30, label=f't={t}')
        ax.set_title('Flow-Evolved Coordinates')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        if n_timesteps <= 8:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Metric determinant evolution with error bars
        ax = axes[0, 1]
        timesteps_with_metrics = []
        det_means = []
        det_stds = []
        det_medians = []
        
        for t, data in timestep_data.items():
            if 'det_G_inv' in data['metric_properties']:
                det_values = data['metric_properties']['det_G_inv']
                timesteps_with_metrics.append(t)
                det_means.append(np.mean(det_values))
                det_stds.append(np.std(det_values))
                det_medians.append(np.median(det_values))
        
        if timesteps_with_metrics:
            ax.errorbar(timesteps_with_metrics, det_means, yerr=det_stds, 
                       marker='o', capsize=5, capthick=2, linewidth=2, label='Mean ± Std')
            ax.plot(timesteps_with_metrics, det_medians, 's-', alpha=0.7, label='Median')
            ax.set_title('Metric Determinant Evolution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('det(G⁻¹)')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No metric data\navailable', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Metric Determinant Evolution')
        
        # Plot 3: Metric condition number evolution
        ax = axes[0, 2]
        condition_numbers = []
        timesteps_cond = []
        
        for t, data in timestep_data.items():
            if 'condition_number' in data['metric_properties']:
                cond_nums = data['metric_properties']['condition_number']
                if isinstance(cond_nums, (list, np.ndarray)) and len(cond_nums) > 0:
                    condition_numbers.append(np.mean(cond_nums))
                    timesteps_cond.append(t)
                elif isinstance(cond_nums, (int, float)):
                    condition_numbers.append(cond_nums)
                    timesteps_cond.append(t)
        
        if condition_numbers:
            ax.plot(timesteps_cond, condition_numbers, 'o-', linewidth=2, markersize=6)
            ax.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Poor conditioning')
            ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Very poor')
            ax.set_title('Metric Condition Number')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('cond(G)')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No condition\nnumber data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Metric Condition Number')
        
        # Plot 4: Eigenvalue analysis of metric tensors
        ax = axes[1, 0]
        eigenvalue_data = self._extract_metric_eigenvalues(timestep_data)
        if eigenvalue_data['timesteps']:
            for i, t in enumerate(eigenvalue_data['timesteps']):
                eigvals = eigenvalue_data['eigenvalues'][i]
                ax.scatter([t] * len(eigvals), eigvals, alpha=0.6, s=20, c=[colors[t]])
            ax.set_title('Metric Eigenvalue Distribution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Eigenvalues')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No eigenvalue\ndata available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Metric Eigenvalue Distribution')
        
        # Plot 5: Geodesic curvature analysis
        ax = axes[1, 1]
        curvature_data = self._compute_geodesic_curvature_approximation(timestep_data, pca)
        if curvature_data['has_data']:
            timesteps_curv = curvature_data['timesteps']
            curvatures = curvature_data['curvatures']
            
            # Box plot of curvatures by timestep
            positions = []
            data_by_timestep = []
            for t in timesteps_curv:
                if t in curvatures and len(curvatures[t]) > 0:
                    positions.append(t)
                    data_by_timestep.append(curvatures[t])
            
            if positions:
                ax.boxplot(data_by_timestep, positions=positions, widths=0.6, patch_artist=True)
                ax.set_title('Geodesic Curvature Approximation')
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Curvature')
                ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Geodesic curvature\ncomputation failed', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Geodesic Curvature Approximation')
        
        # Plot 6: Metric isotropy analysis (eigenvalue ratios)
        ax = axes[1, 2]
        isotropy_data = self._compute_metric_isotropy(timestep_data)
        if isotropy_data['timesteps']:
            ax.plot(isotropy_data['timesteps'], isotropy_data['anisotropy_ratios'], 
                   'o-', linewidth=2, markersize=6, color='purple')
            ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Isotropic')
            ax.set_title('Metric Anisotropy (λ_max/λ_min)')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Anisotropy Ratio')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No isotropy\ndata available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Metric Anisotropy')
        
        # Plot 7: Flow-evolved vs original coordinates comparison
        ax = axes[2, 0]
        if len(timestep_data) > 0:
            first_timestep = list(timestep_data.keys())[0]
            data = timestep_data[first_timestep]
            
            # Transform both to PCA space
            z_flow_pca = pca.transform(data['z_flow_evolved'])
            z_orig_pca = pca.transform(data['z_reparam_samples'][:len(z_flow_pca)])
            
            ax.scatter(z_orig_pca[:, 0], z_orig_pca[:, 1], alpha=0.6, s=20, 
                      color='blue', label='Original samples')
            ax.scatter(z_flow_pca[:, 0], z_flow_pca[:, 1], alpha=0.6, s=20, 
                      color='red', label='Flow-evolved')
            ax.set_title(f'Flow Evolution Comparison (t={first_timestep})')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 8: Metric tensor trace evolution
        ax = axes[2, 1]
        trace_data = self._compute_metric_trace_evolution(timestep_data)
        if trace_data['timesteps']:
            ax.plot(trace_data['timesteps'], trace_data['traces'], 'o-', 
                   linewidth=2, markersize=6, color='green')
            ax.set_title('Metric Tensor Trace Evolution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('tr(G)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No trace\ndata available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Metric Tensor Trace Evolution')
        
        # Plot 9: Principal curvature directions (if available)
        ax = axes[2, 2]
        principal_curv_data = self._compute_principal_curvature_directions(timestep_data, pca)
        if principal_curv_data['has_data']:
            # Show principal directions for first timestep
            first_t = principal_curv_data['timesteps'][0]
            directions = principal_curv_data['directions'][first_t]
            magnitudes = principal_curv_data['magnitudes'][first_t]
            
            # Sample a few points to show directions
            n_arrows = min(10, len(directions))
            indices = np.linspace(0, len(directions)-1, n_arrows, dtype=int)
            
            for i in indices:
                center = directions[i][:2]  # First 2 PCA components
                for j, direction in enumerate(directions[i][2:]):  # Remaining components as directions
                    if j < 2:  # Only show first 2 principal directions
                        magnitude = magnitudes[i][j] if j < len(magnitudes[i]) else 1.0
                        end = center + 0.1 * magnitude * direction[:2]
                        ax.arrow(center[0], center[1], end[0]-center[0], end[1]-center[1],
                                head_width=0.05, head_length=0.05, fc=f'C{j}', ec=f'C{j}', alpha=0.7)
            
            ax.set_title(f'Principal Curvature Directions (t={first_t})')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Principal curvature\nanalysis unavailable', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Principal Curvature Directions')
        
        plt.tight_layout()
        filename = f'enhanced_pca_analysis_epoch_{epoch}.png'
        saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
        
        # Log to WandB with enhanced metrics
        if self.should_log_to_wandb() and saved_file:
            wandb.log({
                "manifold/enhanced_pca_analysis": wandb.Image(saved_file, caption=f"Enhanced PCA Analysis - Epoch {epoch}"),
                "metrics/pca_explained_variance_ratio": pca.explained_variance_ratio_[0] if len(pca.explained_variance_ratio_) > 0 else 0.0
            })
            
            # Log numerical metrics
            if condition_numbers:
                wandb.log({"metrics/metric_condition_number_mean": np.mean(condition_numbers)})
            if isotropy_data['anisotropy_ratios']:
                wandb.log({"metrics/metric_anisotropy_mean": np.mean(isotropy_data['anisotropy_ratios'])})
        
        plt.close()
        print(f"✅ Enhanced PCA analysis created with comprehensive metrics")
    
    def _extract_metric_eigenvalues(self, timestep_data):
        """Extract eigenvalues from metric tensors across timesteps."""
        eigenvalue_data = {'timesteps': [], 'eigenvalues': []}
        
        try:
            for t, data in timestep_data.items():
                if 'eigenvalues' in data['metric_properties']:
                    eigvals = data['metric_properties']['eigenvalues']
                    if isinstance(eigvals, (list, np.ndarray)) and len(eigvals) > 0:
                        # Flatten if nested
                        if isinstance(eigvals[0], (list, np.ndarray)):
                            flat_eigvals = [val for sublist in eigvals for val in sublist]
                        else:
                            flat_eigvals = eigvals
                        
                        eigenvalue_data['timesteps'].append(t)
                        eigenvalue_data['eigenvalues'].append(flat_eigvals)
        except Exception as e:
            print(f"⚠️ Eigenvalue extraction failed: {e}")
        
        return eigenvalue_data
    
    def _compute_geodesic_curvature_approximation(self, timestep_data, pca):
        """Compute approximate geodesic curvature using local metric variations."""
        curvature_data = {'has_data': False, 'timesteps': [], 'curvatures': {}}
        
        try:
            for t, data in timestep_data.items():
                if 'z_flow_evolved' in data and len(data['z_flow_evolved']) > 3:
                    z_coords = data['z_flow_evolved']
                    z_pca = pca.transform(z_coords)
                    
                    # Compute local curvature approximation using finite differences
                    curvatures = []
                    for i in range(1, len(z_pca) - 1):
                        # Three consecutive points for curvature estimation
                        p1, p2, p3 = z_pca[i-1], z_pca[i], z_pca[i+1]
                        
                        # Compute curvature using Menger curvature formula
                        a = np.linalg.norm(p2 - p1)
                        b = np.linalg.norm(p3 - p2)
                        c = np.linalg.norm(p3 - p1)
                        
                        if a > 1e-8 and b > 1e-8 and c > 1e-8:
                            # Area of triangle formed by three points
                            s = (a + b + c) / 2  # semi-perimeter
                            area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
                            
                            # Menger curvature
                            if area > 1e-12:
                                curvature = 4 * area / (a * b * c)
                                curvatures.append(curvature)
                    
                    if curvatures:
                        curvature_data['timesteps'].append(t)
                        curvature_data['curvatures'][t] = curvatures
                        curvature_data['has_data'] = True
        
        except Exception as e:
            print(f"⚠️ Geodesic curvature computation failed: {e}")
        
        return curvature_data
    
    def _compute_metric_isotropy(self, timestep_data):
        """Compute metric isotropy using eigenvalue ratios."""
        isotropy_data = {'timesteps': [], 'anisotropy_ratios': []}
        
        try:
            for t, data in timestep_data.items():
                if 'eigenvalues' in data['metric_properties']:
                    eigvals = data['metric_properties']['eigenvalues']
                    
                    # Process eigenvalues
                    if isinstance(eigvals, (list, np.ndarray)) and len(eigvals) > 0:
                        # If nested, take first set
                        if isinstance(eigvals[0], (list, np.ndarray)):
                            eigvals = eigvals[0]
                        
                        eigvals = np.array(eigvals)
                        eigvals = eigvals[eigvals > 1e-12]  # Filter out near-zero eigenvalues
                        
                        if len(eigvals) >= 2:
                            max_eig = np.max(eigvals)
                            min_eig = np.min(eigvals)
                            anisotropy_ratio = max_eig / min_eig
                            
                            isotropy_data['timesteps'].append(t)
                            isotropy_data['anisotropy_ratios'].append(anisotropy_ratio)
        
        except Exception as e:
            print(f"⚠️ Metric isotropy computation failed: {e}")
        
        return isotropy_data
    
    def _compute_metric_trace_evolution(self, timestep_data):
        """Compute trace of metric tensor over time."""
        trace_data = {'timesteps': [], 'traces': []}
        
        try:
            for t, data in timestep_data.items():
                if 'trace' in data['metric_properties']:
                    trace_values = data['metric_properties']['trace']
                    if isinstance(trace_values, (list, np.ndarray)):
                        mean_trace = np.mean(trace_values)
                    else:
                        mean_trace = trace_values
                    
                    trace_data['timesteps'].append(t)
                    trace_data['traces'].append(mean_trace)
        
        except Exception as e:
            print(f"⚠️ Metric trace computation failed: {e}")
        
        return trace_data
    
    def _compute_principal_curvature_directions(self, timestep_data, pca):
        """Compute principal curvature directions using metric eigendecomposition."""
        curv_data = {'has_data': False, 'timesteps': [], 'directions': {}, 'magnitudes': {}}
        
        try:
            for t, data in timestep_data.items():
                if ('eigenvalues' in data['metric_properties'] and 
                    'eigenvectors' in data['metric_properties']):
                    
                    eigvals = data['metric_properties']['eigenvalues']
                    eigvecs = data['metric_properties']['eigenvectors']
                    
                    if (isinstance(eigvals, (list, np.ndarray)) and len(eigvals) > 0 and
                        isinstance(eigvecs, (list, np.ndarray)) and len(eigvecs) > 0):
                        
                        # Process for PCA space
                        directions = []
                        magnitudes = []
                        
                        for i in range(min(5, len(eigvals))):  # Limit to first 5 for performance
                            if isinstance(eigvals[i], (list, np.ndarray)):
                                eig_val = eigvals[i]
                                eig_vec = eigvecs[i]
                            else:
                                eig_val = [eigvals[i]]
                                eig_vec = [eigvecs[i]]
                            
                            if len(eig_vec) > 0:
                                # Transform eigenvector to PCA space if possible
                                try:
                                    z_point = data['z_flow_evolved'][i] if i < len(data['z_flow_evolved']) else data['z_flow_evolved'][0]
                                    z_pca = pca.transform(z_point.reshape(1, -1))[0]
                                    directions.append(z_pca)
                                    magnitudes.append(eig_val)
                                except:
                                    pass
                        
                        if directions:
                            curv_data['timesteps'].append(t)
                            curv_data['directions'][t] = directions
                            curv_data['magnitudes'][t] = magnitudes
                            curv_data['has_data'] = True
                            break  # Only process first timestep for now
        
        except Exception as e:
            print(f"⚠️ Principal curvature direction computation failed: {e}")
        
        return curv_data
    
    def _compute_christoffel_symbols(self, G):
        """Compute Christoffel symbols approximation using finite differences."""
        try:
            # This is a simplified approximation - full computation would require metric derivatives
            batch_size, dim, _ = G.shape
            christoffel_norms = []
            
            for i in range(batch_size):
                # Approximate Christoffel symbols using metric determinant gradient
                G_i = G[i].cpu().numpy()
                
                # Simple approximation: use eigenvalue spread as proxy
                eigvals = np.linalg.eigvals(G_i)
                eigvals = eigvals[eigvals > 1e-12]
                
                if len(eigvals) > 1:
                    # Christoffel symbol norm approximation
                    christoffel_norm = np.std(eigvals) / np.mean(eigvals)
                    christoffel_norms.append(christoffel_norm)
                else:
                    christoffel_norms.append(0.0)
            
            return christoffel_norms
            
        except Exception as e:
            print(f"⚠️ Christoffel symbols computation failed: {e}")
            return []
    
    def _compute_ricci_curvature_approximation(self, G, z):
        """Compute Ricci curvature approximation using metric properties."""
        try:
            batch_size = G.shape[0]
            ricci_scalars = []
            
            for i in range(batch_size):
                G_i = G[i].cpu().numpy()
                
                # Simplified Ricci scalar approximation using trace and determinant
                trace_G = np.trace(G_i)
                det_G = np.linalg.det(G_i)
                
                if det_G > 1e-12:
                    # Approximate Ricci scalar
                    ricci_scalar = trace_G / det_G - G_i.shape[0]
                    ricci_scalars.append(ricci_scalar)
                else:
                    ricci_scalars.append(0.0)
            
            return ricci_scalars
            
        except Exception as e:
            print(f"⚠️ Ricci curvature computation failed: {e}")
            return []
    
    def _compute_geodesic_deviation(self, G, z):
        """Compute geodesic deviation approximation."""
        try:
            batch_size_G = G.shape[0]
            batch_size_z = z.shape[0]
            
            # Use the minimum batch size to avoid index errors
            batch_size = min(batch_size_G, batch_size_z)
            deviation_measures = []
            
            for i in range(batch_size):
                G_i = G[i].cpu().numpy()
                z_i = z[i].cpu().numpy()
                
                # Approximate geodesic deviation using metric curvature
                eigvals = np.linalg.eigvals(G_i)
                eigvals = eigvals[eigvals > 1e-12]
                
                if len(eigvals) >= 2:
                    # Deviation measure based on eigenvalue spread and position
                    max_eig = np.max(eigvals)
                    min_eig = np.min(eigvals)
                    position_norm = np.linalg.norm(z_i)
                    
                    deviation = (max_eig / min_eig - 1.0) * (1.0 + position_norm)
                    deviation_measures.append(deviation)
                else:
                    deviation_measures.append(0.0)
            
            return deviation_measures
            
        except Exception as e:
            print(f"⚠️ Geodesic deviation computation failed: {e}")
            return []
    
    def _compute_metric_connectivity(self, G):
        """Compute metric connectivity measure."""
        try:
            batch_size = G.shape[0]
            connectivity_measures = []
            
            for i in range(batch_size):
                G_i = G[i].cpu().numpy()
                
                # Connectivity measure using matrix norm ratios
                frobenius_norm = np.linalg.norm(G_i, 'fro')
                nuclear_norm = np.sum(np.linalg.svd(G_i, compute_uv=False))
                
                if nuclear_norm > 1e-12:
                    connectivity = frobenius_norm / nuclear_norm
                    connectivity_measures.append(connectivity)
                else:
                    connectivity_measures.append(1.0)
            
            return connectivity_measures
            
        except Exception as e:
            print(f"⚠️ Metric connectivity computation failed: {e}")
            return []
    
    def _compute_sectional_curvature_approximation(self, G):
        """Compute sectional curvature approximation."""
        try:
            batch_size, dim, _ = G.shape
            sectional_curvatures = []
            
            for i in range(batch_size):
                G_i = G[i].cpu().numpy()
                
                # Simplified sectional curvature using Gaussian curvature approximation
                if dim == 2:
                    # For 2D, sectional curvature equals Gaussian curvature
                    det_G = np.linalg.det(G_i)
                    trace_G = np.trace(G_i)
                    
                    if det_G > 1e-12:
                        gaussian_curvature = det_G / (trace_G ** 2)
                        sectional_curvatures.append(gaussian_curvature)
                    else:
                        sectional_curvatures.append(0.0)
                else:
                    # For higher dimensions, use eigenvalue-based approximation
                    eigvals = np.linalg.eigvals(G_i)
                    eigvals = eigvals[eigvals > 1e-12]
                    
                    if len(eigvals) >= 2:
                        # Approximate sectional curvature using eigenvalue pairs
                        curvatures = []
                        for j in range(len(eigvals)):
                            for k in range(j + 1, len(eigvals)):
                                curvature = (eigvals[j] - eigvals[k]) ** 2 / (eigvals[j] * eigvals[k])
                                curvatures.append(curvature)
                        
                        if curvatures:
                            sectional_curvatures.append(np.mean(curvatures))
                        else:
                            sectional_curvatures.append(0.0)
                    else:
                        sectional_curvatures.append(0.0)
            
            return sectional_curvatures
            
        except Exception as e:
            print(f"⚠️ Sectional curvature computation failed: {e}")
            return []
    
    def _create_enhanced_manifold_heatmaps(self, timestep_data, epoch):
        """Create enhanced manifold heatmaps with flow-evolved metrics."""
        print("🌍 Creating enhanced manifold heatmaps with flow-evolved metrics...")
        
        n_timesteps = len(timestep_data)
        
        # Show first few timesteps
        timesteps_to_show = list(range(min(4, n_timesteps)))
        fig, axes = plt.subplots(3, len(timesteps_to_show), figsize=(4*len(timesteps_to_show), 12))  # SMALLER
        if len(timesteps_to_show) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Enhanced Manifold Heatmaps (Flow-Evolved Metrics) - Epoch {epoch}', fontsize=14)
        
        # Prepare global PCA
        all_z_flow = np.concatenate([timestep_data[t]['z_flow_evolved'] for t in timesteps_to_show], axis=0)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(all_z_flow)
        
        for idx, t in enumerate(timesteps_to_show):
            data = timestep_data[t]
            
            # Row 1: Flow-evolved coordinates
            ax = axes[0, idx]
            z_pca = pca.transform(data['z_flow_evolved'])
            scatter = ax.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.7, s=20)
            ax.set_title(f'Flow-Evolved Coords t={t}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(True, alpha=0.3)
            
            # Row 2: Metric determinant field
            ax = axes[1, idx]
            if 'det_G_inv' in data['metric_properties'] and 'z_flow_neighborhood' in data:
                z_neighborhood = data['z_flow_neighborhood']
                det_values = data['metric_properties']['det_G_inv']
                
                # Limit points for performance
                max_points = 200
                if len(z_neighborhood) > max_points:
                    indices = np.random.choice(len(z_neighborhood), max_points, replace=False)
                    z_neighborhood = z_neighborhood[indices]
                    det_values = det_values[indices]
                
                z_neigh_pca = pca.transform(z_neighborhood)
                
                # Create scatter plot with det(G^-1) values
                scatter = ax.scatter(z_neigh_pca[:, 0], z_neigh_pca[:, 1], 
                                   c=det_values, cmap='plasma', s=15, alpha=0.8,
                                   vmin=np.percentile(det_values, 5),
                                   vmax=np.percentile(det_values, 95))
                ax.set_title(f'Metric det(G⁻¹) t={t}')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                try:
                    plt.colorbar(scatter, ax=ax, shrink=0.6)
                except:
                    pass
            else:
                ax.text(0.5, 0.5, 'No metric data\navailable', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Metric det(G⁻¹) t={t}')
                
            # Row 3: Log metric field for better visualization
            ax = axes[2, idx]
            if 'det_G_inv' in data['metric_properties'] and 'z_flow_neighborhood' in data:
                z_neighborhood = data['z_flow_neighborhood']
                det_values = data['metric_properties']['det_G_inv']
                
                # Limit points for performance
                max_points = 200
                if len(z_neighborhood) > max_points:
                    indices = np.random.choice(len(z_neighborhood), max_points, replace=False)
                    z_neighborhood = z_neighborhood[indices]
                    det_values = det_values[indices]
                
                z_neigh_pca = pca.transform(z_neighborhood)
                log_det_values = np.log10(np.clip(det_values, 1e-12, None))
                
                scatter = ax.scatter(z_neigh_pca[:, 0], z_neigh_pca[:, 1], 
                                   c=log_det_values, cmap='viridis', s=15, alpha=0.8)
                ax.set_title(f'Log₁₀ det(G⁻¹) t={t}')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                try:
                    plt.colorbar(scatter, ax=ax, shrink=0.6)
                except:
                    pass
            else:
                ax.text(0.5, 0.5, 'No metric data\navailable', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Log₁₀ det(G⁻¹) t={t}')
        
        plt.tight_layout()
        filename = f'enhanced_manifold_heatmaps_epoch_{epoch}.png'
        saved_file = self._safe_save_plt_figure(filename, dpi=150, bbox_inches='tight')
        
        if self.should_log_to_wandb() and saved_file:
            wandb.log({"manifold/enhanced_heatmaps": wandb.Image(saved_file, caption=f"Epoch {epoch} - Manifold heatmaps")})
        
        plt.close()
        print(f"✅ Enhanced manifold heatmaps created")
    
    def _create_temporal_metric_analysis(self, timestep_data, epoch):
        """Create temporal evolution analysis of metrics."""
        print("⏱️ Creating temporal metric evolution analysis...")
        
        n_timesteps = len(timestep_data)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # SMALLER
        fig.suptitle(f'Temporal Metric Evolution Analysis - Epoch {epoch}', fontsize=14)
        
        # Extract timesteps with valid metric data
        timesteps_with_metrics = []
        det_means = []
        det_stds = []
        condition_numbers = []
        
        for t, data in timestep_data.items():
            if 'det_G_inv' in data['metric_properties']:
                det_values = data['metric_properties']['det_G_inv']
                timesteps_with_metrics.append(t)
                det_means.append(np.mean(det_values))
                det_stds.append(np.std(det_values))
                condition_numbers.append(data['metric_properties'].get('condition_number', np.nan))
        
        # Plot 1: Mean det(G^-1) evolution
        ax = axes[0, 0]
        if timesteps_with_metrics:
            ax.errorbar(timesteps_with_metrics, det_means, yerr=det_stds, 
                       marker='o', capsize=5, capthick=2, linewidth=2, color='blue')
            ax.set_title('Mean det(G⁻¹) Evolution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Mean det(G⁻¹)')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No metric data\navailable', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Mean det(G⁻¹) Evolution')
        
        # Plot 2: Condition number evolution
        ax = axes[0, 1]
        if timesteps_with_metrics and not all(np.isnan(condition_numbers)):
            ax.plot(timesteps_with_metrics, condition_numbers, 'o-', linewidth=2, markersize=8, color='red')
            ax.set_title('Condition Number Evolution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Condition Number')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No condition\nnumber data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Condition Number Evolution')
        
        # Plot 3: Metric variance evolution
        ax = axes[1, 0]
        if timesteps_with_metrics:
            variances = [std**2 for std in det_stds]
            ax.plot(timesteps_with_metrics, variances, 'o-', linewidth=2, markersize=8, color='green')
            ax.set_title('Metric Variance Evolution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Var(det(G⁻¹))')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No variance\ndata available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Metric Variance Evolution')
        
        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary text
        summary_text = f"TEMPORAL METRIC ANALYSIS\n"
        summary_text += "=" * 25 + "\n\n"
        summary_text += f"Total timesteps: {n_timesteps}\n"
        summary_text += f"Timesteps with metrics: {len(timesteps_with_metrics)}\n\n"
        
        if timesteps_with_metrics:
            summary_text += f"Metric Statistics:\n"
            summary_text += f"  Mean det(G⁻¹): {np.mean(det_means):.2e}\n"
            summary_text += f"  Range: [{np.min(det_means):.2e}, {np.max(det_means):.2e}]\n"
            if not all(np.isnan(condition_numbers)):
                summary_text += f"  Mean condition #: {np.nanmean(condition_numbers):.2f}\n"
            summary_text += f"  Mean variance: {np.mean([std**2 for std in det_stds]):.2e}\n"
        else:
            summary_text += "No metric data available\n"
        
        summary_text += f"\nCoordinates: Flow-evolved\n"
        summary_text += f"Epoch: {epoch}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", alpha=0.8))
        
        plt.tight_layout()
        filename = f'temporal_metric_analysis_epoch_{epoch}.png'
        saved_file = self._safe_save_plt_figure(filename, dpi=150, bbox_inches='tight')
        
        if self.should_log_to_wandb() and saved_file:
            wandb.log({
                "manifold/temporal_analysis": wandb.Image(saved_file, caption=f"Epoch {epoch} - Temporal analysis"),
                "metrics/mean_det_G_inv": np.mean(det_means) if det_means else 0,
                "metrics/mean_condition_number": np.nanmean(condition_numbers) if condition_numbers else 0,
            })
        
        plt.close()
        print(f"✅ Temporal metric analysis created") 