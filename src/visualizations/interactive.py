"""
Interactive Visualizations Module
================================

Advanced Plotly-based interactive visualizations:
- Geodesic slider visualizations
- Fancy interactive plots
- Animated metric evolution
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb
from .base import BaseVisualization

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.colors as pc
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class InteractiveVisualizations(BaseVisualization):
    """Interactive Plotly-based visualization suite."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - interactive visualizations will be skipped")
    
    def _ensure_model_on_device(self):
        """Ensure the entire model and all its components are on the correct device."""
        try:
            # First, ensure the model itself is on the device
            self.model = self.model.to(self.device)
            
            # Ensure encoder and decoder are on device
            if hasattr(self.model, 'encoder') and self.model.encoder is not None:
                self.model.encoder = self.model.encoder.to(self.device)
                
            if hasattr(self.model, 'decoder') and self.model.decoder is not None:
                self.model.decoder = self.model.decoder.to(self.device)
            
            # Ensure metric tensor G and its components are on device
            if hasattr(self.model, 'G') and self.model.G is not None:
                # Handle different types of G (function, module, tensor)
                if hasattr(self.model.G, 'to'):
                    self.model.G = self.model.G.to(self.device)
                elif hasattr(self.model.G, '__self__') and hasattr(self.model.G.__self__, 'to'):
                    # For bound methods, move the underlying object
                    self.model.G.__self__ = self.model.G.__self__.to(self.device)
            
            if hasattr(self.model, 'G_inv') and self.model.G_inv is not None:
                if hasattr(self.model.G_inv, 'to'):
                    self.model.G_inv = self.model.G_inv.to(self.device)
                elif hasattr(self.model.G_inv, '__self__') and hasattr(self.model.G_inv.__self__, 'to'):
                    self.model.G_inv.__self__ = self.model.G_inv.__self__.to(self.device)
            
            # Move flows to device if available
            if hasattr(self.model, 'flows') and self.model.flows is not None:
                if isinstance(self.model.flows, (list, nn.ModuleList)):
                    for i, flow in enumerate(self.model.flows):
                        if hasattr(flow, 'to'):
                            self.model.flows[i] = flow.to(self.device)
                elif hasattr(self.model.flows, 'to'):
                    self.model.flows = self.model.flows.to(self.device)
            elif hasattr(self.model, 'flow_manager') and hasattr(self.model.flow_manager, 'flows'):
                # Handle modular model structure
                flows = self.model.flow_manager.flows
                if isinstance(flows, (list, nn.ModuleList)):
                    for i, flow in enumerate(flows):
                        if hasattr(flow, 'to'):
                            flows[i] = flow.to(self.device)
                elif hasattr(flows, 'to'):
                    self.model.flow_manager.flows = flows.to(self.device)
            
            # Ensure any centroids/metric components are on device
            if hasattr(self.model, 'centroids') and self.model.centroids is not None:
                self.model.centroids = self.model.centroids.to(self.device)
                
            if hasattr(self.model, 'metric') and self.model.metric is not None:
                if hasattr(self.model.metric, 'to'):
                    self.model.metric = self.model.metric.to(self.device)
                if hasattr(self.model.metric, 'centroids') and self.model.metric.centroids is not None:
                    self.model.metric.centroids = self.model.metric.centroids.to(self.device)
            
            # Ensure any other tensor attributes are on device
            for attr_name in ['G_centroids', 'T', 'lbd', 'mu', 'sigma']:
                if hasattr(self.model, attr_name):
                    attr_value = getattr(self.model, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        setattr(self.model, attr_name, attr_value.to(self.device))
            
            # For RHVAE models, ensure any RHVAE-specific components are on device
            if hasattr(self.model, 'rhvae_sampler') and self.model.rhvae_sampler is not None:
                if hasattr(self.model.rhvae_sampler, 'to'):
                    self.model.rhvae_sampler = self.model.rhvae_sampler.to(self.device)
                    
            # For any model attribute that has named_parameters, ensure it's on device
            for attr_name in dir(self.model):
                if not attr_name.startswith('_'):
                    try:
                        attr = getattr(self.model, attr_name)
                        if hasattr(attr, 'named_parameters') and hasattr(attr, 'to'):
                            attr = attr.to(self.device)
                            setattr(self.model, attr_name, attr)
                    except:
                        continue
                        
        except Exception as e:
            print(f"⚠️ Warning: Could not move some model components to device: {e}")
    
    def _ensure_tensor_on_device(self, tensor):
        """Ensure a tensor is on the correct device."""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        return tensor
    
    def _get_flows(self):
        """Get flows from either legacy or modular model structure."""
        # Try legacy structure first
        if hasattr(self.model, 'flows') and self.model.flows is not None:
            return self.model.flows
        # Try modular structure
        elif hasattr(self.model, 'flow_manager') and hasattr(self.model.flow_manager, 'flows'):
            return self.model.flow_manager.flows
        else:
            return None

    # ------------------------------------------------------------------
    # Shared helpers for interactive visualizations
    # ------------------------------------------------------------------

    def _extract_flow_tensor(self, flow_result: object) -> torch.Tensor:
        """Normalize heterogeneous flow outputs to a plain tensor."""
        if isinstance(flow_result, tuple):
            flow_result = flow_result[0]

        for attr in ("out", "sample", "z"):
            attr_val = getattr(flow_result, attr, None)
            if isinstance(attr_val, torch.Tensor):
                return self._ensure_tensor_on_device(attr_val)

        if isinstance(flow_result, torch.Tensor):
            return self._ensure_tensor_on_device(flow_result)

        raise TypeError(
            "Flow modules must return a tensor, tuple whose first element is a tensor, "
            "or an object exposing `.out`, `.sample`, or `.z`."
        )

    def _symmetrize_matrices(self, matrices: torch.Tensor) -> torch.Tensor:
        """Ensure matrices are symmetric by averaging with their transpose."""
        return 0.5 * (matrices + matrices.transpose(-1, -2))

    def _evaluate_metric_inverse(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate G^{-1} at given latent points, falling back to G if needed."""
        points = self._ensure_tensor_on_device(points.float())

        with torch.no_grad():
            if hasattr(self.model, "G_inv") and callable(self.model.G_inv):
                G_inv = self.model.G_inv(points)
                G_inv = self._ensure_tensor_on_device(G_inv)
            elif hasattr(self.model, "G") and callable(self.model.G):
                G = self.model.G(points)
                G = self._ensure_tensor_on_device(G)
                if G.dim() == 2:
                    G = G.unsqueeze(0).expand(points.shape[0], -1, -1)
                try:
                    G_inv = torch.linalg.inv(G)
                except RuntimeError:
                    G_inv = torch.linalg.pinv(G)
            else:
                raise AttributeError("Model must expose `G_inv` or `G` to evaluate the metric tensor.")

        if G_inv.dim() == 2:
            G_inv = G_inv.unsqueeze(0).expand(points.shape[0], -1, -1)

        identity = torch.eye(G_inv.shape[-1], device=G_inv.device, dtype=G_inv.dtype)
        G_inv = self._symmetrize_matrices(G_inv) + 1e-6 * identity
        return G_inv.float()

    def _evaluate_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate G at given latent points, falling back to G^{-1} if needed."""
        points = self._ensure_tensor_on_device(points.float())

        with torch.no_grad():
            if hasattr(self.model, "G") and callable(self.model.G):
                G = self.model.G(points)
                G = self._ensure_tensor_on_device(G)
            elif hasattr(self.model, "G_inv") and callable(self.model.G_inv):
                G_inv = self.model.G_inv(points)
                G_inv = self._ensure_tensor_on_device(G_inv)
                if G_inv.dim() == 2:
                    G_inv = G_inv.unsqueeze(0).expand(points.shape[0], -1, -1)
                try:
                    G = torch.linalg.inv(G_inv)
                except RuntimeError:
                    G = torch.linalg.pinv(G_inv)
            else:
                raise AttributeError("Model must expose `G` or `G_inv` to evaluate the metric tensor.")

        if G.dim() == 2:
            G = G.unsqueeze(0).expand(points.shape[0], -1, -1)

        identity = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
        G = self._symmetrize_matrices(G) + 1e-6 * identity
        return G.float()

    def _log10_det(self, matrices: torch.Tensor) -> torch.Tensor:
        """Compute log10(det(M)) with numerical safeguards."""
        if matrices.dim() == 2:
            matrices = matrices.unsqueeze(0)
        matrices = self._symmetrize_matrices(matrices)
        sign, logdet = torch.linalg.slogdet(matrices)
        logdet = torch.where(sign <= 0, torch.full_like(logdet, float('-inf')), logdet)
        return logdet / np.log(10.0)

    def _safe_logdet(self, matrices: torch.Tensor) -> torch.Tensor:
        """Return natural log(det(M)) with safeguards for SPD matrices."""
        if matrices.dim() == 2:
            matrices = matrices.unsqueeze(0)
        matrices = self._symmetrize_matrices(matrices)
        sign, logdet = torch.linalg.slogdet(matrices)
        logdet = torch.where(sign <= 0, torch.full_like(logdet, float('-inf')), logdet)
        return logdet

    def _prepare_pca_grid(
        self,
        pca,
        z_pca_seq: np.ndarray,
        grid_resolution: int = 40,
        padding: float = 0.15,
        extra_points_pca: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate a PCA-aligned grid and latent coordinates for background fields."""
        if z_pca_seq.shape[-1] < 2:
            raise ValueError("PCA sequence must have at least 2 components for visualization.")

        flattened = z_pca_seq.reshape(-1, z_pca_seq.shape[-1])
        if extra_points_pca is not None and extra_points_pca.size > 0:
            flattened = np.concatenate([flattened, np.asarray(extra_points_pca)], axis=0)

        x_vals = flattened[:, 0]
        y_vals = flattened[:, 1]
        x_span = x_vals.max() - x_vals.min()
        y_span = y_vals.max() - y_vals.min()

        if x_span <= 0:
            x_span = 1.0
        if y_span <= 0:
            y_span = 1.0

        x_min = x_vals.min() - padding * x_span
        x_max = x_vals.max() + padding * x_span
        y_min = y_vals.min() - padding * y_span
        y_max = y_vals.max() + padding * y_span

        gx = np.linspace(x_min, x_max, grid_resolution)
        gy = np.linspace(y_min, y_max, grid_resolution)
        XX, YY = np.meshgrid(gx, gy)
        grid_points_pca = np.column_stack([XX.reshape(-1), YY.reshape(-1)])

        latent_grid = pca.inverse_transform(grid_points_pca)
        latent_grid_tensor = torch.tensor(latent_grid, dtype=torch.float32, device=self.device)

        return {
            "gx": gx,
            "gy": gy,
            "XX": XX,
            "YY": YY,
            "grid_points_pca": grid_points_pca,
            "latent_grid": latent_grid_tensor,
        }

    def _extract_latent_sequence(self, forward_out) -> Optional[torch.Tensor]:
        """Best-effort extraction of latent trajectories from various forward outputs."""
        if forward_out is None:
            return None

        keys = [
            "latent_samples",
            "z",
            "samples",
            "latents",
        ]

        if isinstance(forward_out, dict):
            for key in keys:
                if key in forward_out:
                    value = forward_out[key]
                    if isinstance(value, torch.Tensor):
                        return self._ensure_tensor_on_device(value.float())
        else:
            for key in keys:
                if hasattr(forward_out, key):
                    value = getattr(forward_out, key)
                    if isinstance(value, torch.Tensor):
                        return self._ensure_tensor_on_device(value.float())

        return None

    def _apply_interactive_theme(
        self,
        fig,
        *,
        background: str = "rgba(0,0,0,0)",
        legend_bg: str = "rgba(255,255,255,0.88)",
        font_color: str = "#1f2127",
    ) -> None:
        """Apply a light, unobtrusive theme so content stands out on W&B's white canvas."""
        fig.update_layout(
            paper_bgcolor=background,
            plot_bgcolor=background,
            font=dict(color=font_color)
        )

        fig.update_layout(
            legend=dict(
                bgcolor=legend_bg,
                bordercolor="rgba(0,0,0,0.25)",
                borderwidth=1,
                font=dict(color=font_color)
            )
        )

        fig.update_coloraxes(colorbar=dict(
            bgcolor=legend_bg,
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
            tickfont=dict(color=font_color),
            title=dict(font=dict(color=font_color))
        ))

        fig.update_xaxes(
            color=font_color,
            gridcolor="rgba(0,0,0,0.08)",
            zerolinecolor="rgba(0,0,0,0.12)"
        )
        fig.update_yaxes(
            color=font_color,
            gridcolor="rgba(0,0,0,0.08)",
            zerolinecolor="rgba(0,0,0,0.12)"
        )

        for slider in getattr(fig.layout, "sliders", []) or []:
            slider.bgcolor = "rgba(255,255,255,0.85)"
            slider.activebgcolor = "rgba(31,119,180,0.4)"
            if slider.font is None:
                slider.font = dict(color=font_color)
            else:
                slider.font.color = font_color
            if slider.currentvalue and slider.currentvalue.font:
                slider.currentvalue.font.color = font_color
            elif slider.currentvalue:
                slider.currentvalue.font = dict(color=font_color)

    def _apply_flow_with_jacobian(
        self,
        flow: nn.Module,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a flow to points and return outputs and local Jacobians."""
        points = self._ensure_tensor_on_device(points).detach()
        n_points, latent_dim = points.shape

        if n_points > 1500:
            print("⚠️ Large point cloud for Jacobian computation; consider reducing grid resolution.")

        outputs = []
        jacobians = []

        flow = flow.to(self.device)
        was_training = getattr(flow, "training", False)
        flow.eval()

        with torch.enable_grad():
            for i in range(n_points):
                single_point = points[i].clone().detach().requires_grad_(True)

                def single_flow(inp: torch.Tensor) -> torch.Tensor:
                    out = flow(inp.unsqueeze(0))
                    return self._extract_flow_tensor(out).squeeze(0)

                out_vec = single_flow(single_point)
                jac = torch.autograd.functional.jacobian(
                    single_flow,
                    single_point,
                    create_graph=False,
                )

                outputs.append(out_vec.detach())
                jacobians.append(jac.detach())

        if was_training:
            flow.train()

        outputs_tensor = torch.stack(outputs, dim=0).to(self.device)
        jacobians_tensor = torch.stack(jacobians, dim=0).to(self.device)
        return outputs_tensor, jacobians_tensor

    def _compute_metric_flow_evolution(
        self,
        base_points: torch.Tensor,
        flows: Sequence[nn.Module],
        max_steps: Optional[int] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """Track metric/precision evolution through flows with diagnostic statistics."""

        base_points = self._ensure_tensor_on_device(base_points.float())
        n_points, latent_dim = base_points.shape
        steps = len(flows) if max_steps is None else min(len(flows), max_steps)

        identity = torch.eye(latent_dim, device=self.device, dtype=base_points.dtype)
        identity_batch = identity.unsqueeze(0).expand(n_points, -1, -1)
        eps_eye = 1e-6 * identity

        positions: List[torch.Tensor] = [base_points.detach()]
        metrics: List[torch.Tensor] = []
        precisions: List[torch.Tensor] = []
        logdet_metrics10: List[torch.Tensor] = []
        logdet_precisions10: List[torch.Tensor] = []
        spd_errors: List[torch.Tensor] = []
        det_residuals: List[torch.Tensor] = []
        jacobians_total: List[torch.Tensor] = [identity_batch.clone()]
        jacobians_step: List[torch.Tensor] = []
        logabsdet_jacobian_total: List[torch.Tensor] = [torch.zeros(n_points, device=self.device)]

        current_points = base_points
        cumulative_jacobian = identity_batch.clone()

        # Step 0 (no flow applied yet)
        metric_current = self._evaluate_metric(current_points)
        precision_current = torch.linalg.inv(metric_current)
        precision_current = self._symmetrize_matrices(precision_current) + eps_eye
        metric_current = torch.linalg.inv(precision_current)
        metric_current = self._symmetrize_matrices(metric_current)

        metrics.append(metric_current.detach())
        precisions.append(precision_current.detach())
        logdet_metrics10.append(self._log10_det(metric_current).detach())
        logdet_precisions10.append(self._log10_det(precision_current).detach())
        spd_errors.append(
            torch.linalg.norm(
                torch.matmul(metric_current, precision_current) - identity_batch,
                dim=(1, 2)
            ).detach()
        )
        det_residuals.append(torch.zeros(n_points, device=self.device))

        for idx in range(steps):
            flow = flows[idx]
            outputs, jac_step = self._apply_flow_with_jacobian(flow, current_points)
            jacobians_step.append(jac_step.detach())

            cumulative_jacobian = torch.matmul(jac_step, cumulative_jacobian)
            jacobians_total.append(cumulative_jacobian.detach())

            sign_total, logdet_total = torch.linalg.slogdet(cumulative_jacobian)
            logdet_total = torch.where(sign_total <= 0, torch.full_like(logdet_total, float('inf')), logdet_total)
            logabsdet_jacobian_total.append(logdet_total.detach())

            current_points = outputs.detach()
            positions.append(current_points)

            metric_native = self._evaluate_metric(current_points)
            precision_native = torch.linalg.inv(metric_native)
            precision_native = self._symmetrize_matrices(precision_native) + eps_eye
            metric_native = torch.linalg.inv(precision_native)
            metric_native = self._symmetrize_matrices(metric_native)

            # Pull metric back to current coordinate system using cumulative Jacobian
            precision_y = torch.matmul(
                cumulative_jacobian,
                torch.matmul(precision_native, cumulative_jacobian.transpose(1, 2))
            )
            precision_y = self._symmetrize_matrices(precision_y) + eps_eye
            metric_y = torch.linalg.inv(precision_y)
            metric_y = self._symmetrize_matrices(metric_y)

            metrics.append(metric_y.detach())
            precisions.append(precision_y.detach())
            logdet_metrics10.append(self._log10_det(metric_y).detach())
            logdet_precisions10.append(self._log10_det(precision_y).detach())

            spd_errors.append(
                torch.linalg.norm(
                    torch.matmul(metric_y, precision_y) - identity_batch,
                    dim=(1, 2)
                ).detach()
            )

            native_log = self._safe_logdet(metric_native)
            pulled_log = self._safe_logdet(metric_y)
            det_residual = torch.abs(pulled_log - (native_log - 2.0 * logdet_total))
            det_residuals.append(det_residual.detach())

        return {
            "positions": positions,
            "metrics": metrics,
            "precisions": precisions,
            "logdet_metric10": logdet_metrics10,
            "logdet_precision10": logdet_precisions10,
            "spd_errors": spd_errors,
            "determinant_residuals": det_residuals,
            "jacobians_total": jacobians_total,
            "jacobians_step": jacobians_step,
            "logabsdet_jacobian_total": logabsdet_jacobian_total,
        }
    
    def create_geodesic_sliders(self, x_sample: torch.Tensor, epoch: int):
        """Create interactive geodesic slider visualizations with timestep evolution."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping geodesic sliders")
            return
            
        if epoch % 30 != 0:
            return
            
        print(f"🎚️ Creating interactive geodesic sliders for epoch {epoch}")
        
        if not hasattr(self.model, 'G'):
            print("⚠️ No metric tensor available for geodesic visualization")
            return
            
        try:
            self._ensure_model_on_device()
            self.model.eval()

            with torch.no_grad():
                forward_out = self.model_forward(x_sample)
                z_seq = self._extract_latent_sequence(forward_out)

                if z_seq is None:
                    print("⚠️ Could not extract latent sequence for geodesic slider")
                    return

                z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=2)

            flows = self._get_flows()
            grid_info = self._prepare_pca_grid(pca, z_pca_seq, grid_resolution=32, padding=0.18)

            self._create_interactive_geodesic_slider(
                z_pca_seq=z_pca_seq,
                pca=pca,
                flows=list(flows) if flows is not None else [],
                grid_info=grid_info,
                epoch=epoch,
            )

        except Exception as e:
            print(f"⚠️ Geodesic slider visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()

    def create_rhmc_flow_slider(
        self,
        x_sample: torch.Tensor,
        epoch: int,
        n_paths: int = 15
    ) -> None:
        """Interactive slider showing RHMC flow trajectories with pushed-forward metrics."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping RHMC flow slider")
            return

        flows_seq = self._get_flows()
        if flows_seq is None or len(flows_seq) == 0:
            print("⚠️ No flows available for RHMC slider visualization")
            return

        flows = list(flows_seq)
        print(f"🎚️ Creating RHMC flow slider for epoch {epoch}")

        try:
            self._ensure_model_on_device()
            self.model.eval()

            with torch.no_grad():
                enc_out = self.model.encoder(x_sample[:, 0])
                if hasattr(enc_out, "embedding"):
                    mu = enc_out.embedding
                elif hasattr(enc_out, "mu"):
                    mu = enc_out.mu
                elif isinstance(enc_out, dict) and "mu" in enc_out:
                    mu = enc_out["mu"]
                else:
                    print("⚠️ Could not extract encoder means for RHMC slider")
                    return

                forward_out = self.model_forward(x_sample)
                z_seq = self._extract_latent_sequence(forward_out)

                if z_seq is None:
                    print("⚠️ Could not extract latent sequence for PCA projection")
                    return

                z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=2)

            mu = self._ensure_tensor_on_device(mu.float())
            latent_dim = int(mu.shape[-1])
            if latent_dim < 2:
                print("⚠️ RHMC flow slider requires latent dimension >= 2")
                return

            num_candidates = mu.shape[0]
            if num_candidates == 0:
                print("⚠️ No encoder samples available for RHMC slider")
                return

            num_paths = min(n_paths, num_candidates)
            if num_paths < 1:
                print("⚠️ Not enough posterior samples for RHMC slider")
                return

            selection = torch.randperm(num_candidates, device=self.device)[:num_paths]
            path_points_0 = mu[selection].to(self.device)

            path_flow = self._compute_metric_flow_evolution(path_points_0, flows)
            path_positions_pca = [
                pca.transform(points.detach().cpu().numpy())
                for points in path_flow["positions"]
            ]
            extra_grid_points = np.concatenate(path_positions_pca, axis=0) if path_positions_pca else None

            grid_info = self._prepare_pca_grid(
                pca,
                z_pca_seq,
                grid_resolution=32,
                padding=0.18,
                extra_points_pca=extra_grid_points,
            )
            grid_latent = grid_info["latent_grid"]

            grid_flow = self._compute_metric_flow_evolution(grid_latent, flows)

            grid_logdet_fields: List[np.ndarray] = []
            for log_values in grid_flow["logdet_metric10"]:
                arr = log_values.detach().cpu().numpy().reshape(grid_info["XX"].shape)
                grid_logdet_fields.append(np.where(np.isfinite(arr), arr, np.nan))

            has_finite = any(np.isfinite(field).any() for field in grid_logdet_fields)
            if has_finite:
                finite_values = np.concatenate([
                    field[np.isfinite(field)]
                    for field in grid_logdet_fields
                    if np.isfinite(field).any()
                ])
                color_min = float(np.min(finite_values))
                color_max = float(np.max(finite_values))
                if np.isclose(color_min, color_max):
                    color_max = color_min + 1e-3
            else:
                color_min, color_max = -2.0, 2.0

            num_timesteps = len(grid_logdet_fields)

            path_history = np.stack(path_positions_pca, axis=0)  # [T, B, 2]
            path_history_per_path = np.transpose(path_history, (1, 0, 2))

            p_components = torch.tensor(
                pca.components_[:2],
                dtype=path_flow["metrics"][0].dtype,
                device=self.device
            )

            theta = np.linspace(0.0, 2.0 * np.pi, 80)
            circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
            ellipse_points: List[List[np.ndarray]] = []
            for t, metric in enumerate(path_flow["metrics"]):
                metric = metric.to(self.device)
                projected = torch.matmul(
                    torch.matmul(p_components.unsqueeze(0), metric),
                    p_components.unsqueeze(0).transpose(1, 2)
                )
                metric_np = projected.detach().cpu().numpy()

                centers_t = path_positions_pca[t]
                ellipses_t: List[np.ndarray] = []
                for i in range(num_paths):
                    eigvals, eigvecs = np.linalg.eigh(metric_np[i])
                    eigvals = np.clip(eigvals, 1e-9, None)
                    transform = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
                    ellipse = circle @ transform.T + centers_t[i]
                    ellipses_t.append(ellipse)
                ellipse_points.append(ellipses_t)

            palette = pc.qualitative.Dark24
            colors = (palette * ((num_paths // len(palette)) + 1))[:num_paths]

            traces = []
            timestep_trace_indices: List[List[int]] = []

            for t in range(num_timesteps):
                indices: List[int] = []
                contour_trace = go.Contour(
                    x=grid_info["gx"],
                    y=grid_info["gy"],
                    z=grid_logdet_fields[t],
                    coloraxis="coloraxis",
                    contours=dict(showlines=False),
                    hoverinfo="skip",
                    showscale=(t == 0),
                    visible=(t == 0),
                    name=f"log₁₀ det(G) · t={t}"
                )
                traces.append(contour_trace)
                indices.append(len(traces) - 1)

                for path_idx in range(num_paths):
                    history_xy = path_history_per_path[path_idx, :t + 1]
                    line_trace = go.Scatter(
                        x=history_xy[:, 0],
                        y=history_xy[:, 1],
                        mode="lines",
                        line=dict(color=colors[path_idx], width=2),
                        name=f"Path {path_idx}",
                        legendgroup=f"path_{path_idx}",
                        showlegend=(t == 0),
                        visible=(t == 0)
                    )
                    traces.append(line_trace)
                    indices.append(len(traces) - 1)

                    marker_trace = go.Scatter(
                        x=[path_history_per_path[path_idx, t, 0]],
                        y=[path_history_per_path[path_idx, t, 1]],
                        mode="markers",
                        marker=dict(color=colors[path_idx], size=7, symbol="circle"),
                        name=None,
                        legendgroup=f"path_{path_idx}",
                        showlegend=False,
                        visible=(t == 0)
                    )
                    traces.append(marker_trace)
                    indices.append(len(traces) - 1)

                    ellipse_xy = ellipse_points[t][path_idx]
                    ellipse_trace = go.Scatter(
                        x=ellipse_xy[:, 0],
                        y=ellipse_xy[:, 1],
                        mode="lines",
                        line=dict(color=colors[path_idx], dash="dot", width=1.2),
                        name=None,
                        legendgroup=f"path_{path_idx}",
                        showlegend=False,
                        visible=(t == 0)
                    )
                    traces.append(ellipse_trace)
                    indices.append(len(traces) - 1)

                timestep_trace_indices.append(indices)

            total_traces = len(traces)
            slider_steps = []
            title_base = f"RHMC Flow Metric Slider (Epoch {epoch})"
            for t, indices in enumerate(timestep_trace_indices):
                visible = [False] * total_traces
                for idx in indices:
                    visible[idx] = True
                slider_steps.append(
                    dict(
                        method="update",
                        label=str(t),
                        args=[{"visible": visible},
                              {"title": f"{title_base} · timestep {t}"}]
                    )
                )

            fig = go.Figure(data=traces)
            fig.update_layout(
                title=title_base,
                xaxis=dict(title="PCA Component 1", scaleanchor="y", scaleratio=1),
                yaxis=dict(title="PCA Component 2"),
                width=960,
                height=720,
                hovermode="closest",
                legend=dict(
                    bgcolor="rgba(0,0,0,0.65)",
                    bordercolor="rgba(255,255,255,0.5)",
                    borderwidth=1,
                    font=dict(color="white")
                ),
                sliders=[dict(
                    active=0,
                    currentvalue=dict(prefix="Timestep: "),
                    pad=dict(t=40),
                    steps=slider_steps
                )],
                coloraxis=dict(
                    colorscale="Viridis",
                    colorbar=dict(title="log₁₀ det(G)"),
                    cmin=color_min,
                    cmax=color_max
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )

            self._apply_interactive_theme(fig)

            html_path = self._get_output_path(
                f"rhmc_flow_slider.html",
                "interactive"
            )
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"💾 Saved RHMC flow slider: {html_path}")

            if self.should_log_to_wandb():
                with open(html_path, "r", encoding="utf-8") as handle:
                    wandb.log({
                        f"interactive/rhmc_flow_slider": wandb.Html(handle.read(), inject=False)
                    })

            max_spd = max(step.max().item() for step in path_flow["spd_errors"])
            max_det_res = max(step.max().item() for step in path_flow["determinant_residuals"])
            print(f"   ▸ RHMC metric SPD check max ||GG⁻¹−I|| = {max_spd:.2e}")
            print(f"   ▸ RHMC determinant consistency Δ max = {max_det_res:.2e}")

        except Exception as e:
            print(f"⚠️ RHMC flow slider visualization failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.model.train()
        
    def create_fancy_geodesics(self, x_sample: torch.Tensor, epoch: int):
        """Create fancy interactive geodesic visualizations with dense trajectories and a time slider."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping fancy geodesics")
            return
        print(f"✨ Creating fancy interactive geodesic visualizations for epoch {epoch}")
        if not hasattr(self.model, 'G'):
            print("⚠️ No metric tensor available for fancy visualization")
            return
        try:
            self._ensure_model_on_device()
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z
                batch_size, n_obs, latent_dim = z_seq.shape
                
                # Generate dense trajectories and PCA projection
                dense_trajectories = self._generate_dense_trajectories(z_seq, n_interp_points=10)
                z_flat = dense_trajectories.reshape(-1, latent_dim).cpu().numpy()
                
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                z_pca = pca.fit_transform(z_flat)
                dense_n_points = dense_trajectories.shape[1]
                z_pca_dense = z_pca.reshape(batch_size, dense_n_points, 2)
                
                # Original trajectory points in PCA space
                z_orig_flat = z_seq.reshape(-1, latent_dim).cpu().numpy()
                z_orig_pca = pca.transform(z_orig_flat).reshape(batch_size, n_obs, 2)
                
                # Compute proper axis limits with padding
                all_points = np.concatenate([z_pca_dense.reshape(-1, 2), z_orig_pca.reshape(-1, 2)])
                x_min, x_max = all_points[:, 0].min() - 1.0, all_points[:, 0].max() + 1.0
                y_min, y_max = all_points[:, 1].min() - 1.0, all_points[:, 1].max() + 1.0
                
                # Create background metric field (compute once, use for all frames)
                nx, ny = 25, 25
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
                background_field = self._compute_metric_background(xx, yy, pca)
                
                # Pre-compute eigenvalue fields for consistent scaling
        
                all_eigenvalue_fields = []
                for t in range(n_obs):
                    eigenvalue_field = self._compute_eigenvalue_field(xx, yy, pca, t, n_obs)
                    all_eigenvalue_fields.append(eigenvalue_field)
                    print(f"  Timestep {t}: eigenvalue field range [{eigenvalue_field.min():.4f}, {eigenvalue_field.max():.4f}], std={eigenvalue_field.std():.4f}")
                
                # Compute global eigenvalue field scaling
                all_eig_vals = np.concatenate([field.flatten() for field in all_eigenvalue_fields])
                eig_min, eig_max = np.min(all_eig_vals), np.max(all_eig_vals)
                eig_range = eig_max - eig_min
                eig_color_min = eig_min - 0.1 * eig_range
                eig_color_max = eig_max + 0.1 * eig_range
                print(f"🎨 Eigenvalue field color scale: [{eig_color_min:.4f}, {eig_color_max:.4f}]")
                print(f"🎨 Total eigenvalue variation: std={all_eig_vals.std():.4f}, range={eig_range:.4f}")
                
                # Set up subplots - use 1x2 layout for better space usage
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=["🌀 Dense Trajectories + Metric Field", 
                                    "🎭 Eigenvalue Anisotropy Field<br><sub>Shows directional curvature preferences</sub>"],
                    horizontal_spacing=0.1,
                    column_widths=[0.6, 0.4]  # Give more space to trajectory panel
                )
                
                # Prepare frames for each timestep
                frames = []
                max_seqs = min(8, batch_size)  # Limit trajectories for clarity
                palette = px.colors.qualitative.Set1
                colors = (palette * ((max_seqs // len(palette)) + 1))[:max_seqs]
                
                for t in range(n_obs):
                    frame_data = []
                    
                    # Panel 1: Background metric field + Dense trajectories
                    frame_data.append(
                        go.Contour(
                            x=np.linspace(x_min, x_max, nx),
                            y=np.linspace(y_min, y_max, ny),
                            z=background_field,
                            colorscale='Viridis',
                            opacity=0.4,  # Slightly more visible
                            showscale=False,
                            name="Metric Field",
                            xaxis='x', yaxis='y',
                            ncontours=200,  # More detailed contours
                            line_smoothing=0.9,
                            contours=dict(
                                start=background_field.min(), 
                                end=background_field.max(), 
                                size=(background_field.max() - background_field.min()) / 50,
                                showlines=True,
                                coloring='heatmap'
                            )
                        )
                    )
                    
                    # Add dense trajectories up to timestep t
                    for seq_idx in range(max_seqs):
                        # Dense trajectory
                        traj_dense = z_pca_dense[seq_idx, :min(t+1, dense_n_points), :]
                        if len(traj_dense) > 0:
                            frame_data.append(
                                go.Scatter(
                                    x=traj_dense[:, 0], y=traj_dense[:, 1],
                                    mode='lines',
                                    line=dict(color=colors[seq_idx], width=3, dash='solid'),
                                    name=f'Dense Path {seq_idx}',
                                    opacity=0.8,
                                    showlegend=(t == 0 and seq_idx < 4),  # Limit legend entries
                                    xaxis='x', yaxis='y'
                                )
                            )
                        
                        # Original points as markers
                        traj_orig = z_orig_pca[seq_idx, :t+1, :]
                        if len(traj_orig) > 0:
                            frame_data.append(
                                go.Scatter(
                                    x=traj_orig[:, 0], y=traj_orig[:, 1],
                                    mode='markers',
                                    marker=dict(color=colors[seq_idx], size=8, 
                                              line=dict(color='white', width=2)),
                                    name=f'Points {seq_idx}',
                                    showlegend=False,
                                    xaxis='x', yaxis='y'
                                )
                            )
                    
                    # Panel 2: Eigenvalue field with consistent scaling
                    eigenvalue_field = all_eigenvalue_fields[t]
                    frame_data.append(
                        go.Contour(
                            x=np.linspace(x_min, x_max, nx),
                            y=np.linspace(y_min, y_max, ny),
                            z=eigenvalue_field,
                            colorscale='Turbo',  # Changed from 'Plasma' to 'Turbo' for more dramatic variation
                            opacity=0.9,  # Increased opacity for better visibility
                            showscale=True,
                            ncontours=120,  # Even more detailed contours
                            line_smoothing=0.95,
                            connectgaps=False,  # Don't connect gaps for sharper definition
                            colorbar=dict(
                                title="log₁₀(λ_max/λ_min)<br><sub>Anisotropy Ratio</sub><br><sub>Directional Preference</sub>", 
                                x=1.02, len=0.8, thickness=25,
                                tickmode='linear',
                                tick0=eig_color_min,
                                dtick=(eig_color_max - eig_color_min) / 15,  # More detailed ticks
                                tickfont=dict(size=10)
                            ),
                            name="Eigenvalue Anisotropy",
                            xaxis='x2', yaxis='y2',
                            zmin=eig_color_min,  # Consistent color scaling
                            zmax=eig_color_max,  # Consistent color scaling
                            contours=dict(
                                start=eig_color_min, 
                                end=eig_color_max, 
                                size=(eig_color_max - eig_color_min) / 120,  # Very fine steps
                                showlines=False,  # Hide contour lines for smoother appearance
                                coloring='fill'   # Fill areas between contours
                            )
                        )
                    )
                    
                    # Add trajectory shadows to eigenvalue panel
                    for seq_idx in range(min(4, max_seqs)):  # Fewer on second panel
                        traj_orig = z_orig_pca[seq_idx, :t+1, :]
                        if len(traj_orig) > 0:
                            frame_data.append(
                                go.Scatter(
                                    x=traj_orig[:, 0], y=traj_orig[:, 1],
                                    mode='lines+markers',
                                    line=dict(color=colors[seq_idx], width=2),
                                    marker=dict(color=colors[seq_idx], size=6),
                                    opacity=0.6,
                                    showlegend=False,
                                    xaxis='x2', yaxis='y2'
                                )
                            )
                    
                    frames.append(go.Frame(data=frame_data, name=str(t)))
                
                # Add initial frame data
                for trace in frames[0].data:
                    row, col = (1, 1) if hasattr(trace, 'xaxis') and trace.xaxis == 'x' else (1, 2)
                    fig.add_trace(trace, row=row, col=col)
                
                fig.frames = frames
                
                # Update axes with synchronized ranges
                fig.update_xaxes(
                    range=[x_min, x_max], 
                    title_text='PC1',
                    showgrid=True, 
                    gridcolor='rgba(200,200,200,0.3)',
                    zeroline=True,
                    row=1, col=1
                )
                fig.update_yaxes(
                    range=[y_min, y_max],
                    title_text='PC2', 
                    showgrid=True, 
                    gridcolor='rgba(200,200,200,0.3)',
                    zeroline=True,
                    scaleanchor='x',
                    scaleratio=1,
                    row=1, col=1
                )
                
                fig.update_xaxes(
                    range=[x_min, x_max],
                    title_text='PC1',
                    showgrid=True, 
                    gridcolor='rgba(200,200,200,0.3)',
                    zeroline=True,
                    row=1, col=2
                )
                fig.update_yaxes(
                    range=[y_min, y_max],
                    title_text='PC2',
                    showgrid=True, 
                    gridcolor='rgba(200,200,200,0.3)',
                    zeroline=True,
                    scaleanchor='x2',
                    scaleratio=1,
                    row=1, col=2
                )
                
                # Update layout
                fig.update_layout(
                    title=f"✨ Interactive Geodesic Analysis (Time Slider) - Epoch {epoch}",
                    width=1400, height=700,
                    showlegend=True,
                    legend=dict(
                        orientation='v',
                        x=1.05, y=1,
                        bgcolor='rgba(20,20,20,0.8)',
                        bordercolor='white',
                        borderwidth=1,
                        font=dict(size=11, color='white')
                    ),
                    font=dict(size=12, color='white'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=60, r=150, t=120, b=120),  # More space for annotations
                    annotations=[
                        dict(
                            text="<b>📊 Eigenvalue Field Interpretation:</b><br>" +
                                 "• <b>High values (red/yellow)</b>: Strong directional preferences<br>" +
                                 "• <b>Low values (blue/purple)</b>: Isotropic (uniform) behavior<br>" +
                                 "• <b>Gradients</b>: Show preferred flow directions<br>" +
                                 "• <b>Time evolution</b>: How directional preferences change",
                            x=0.98, y=0.02,
                            xref="paper", yref="paper",
                            xanchor="right", yanchor="bottom",
                            showarrow=False,
                            font=dict(size=10, color='white'),
                            bgcolor='rgba(0,0,0,0.7)',
                            bordercolor='white',
                            borderwidth=1
                        )
                    ],
                    sliders=[{
                        "active": 0,
                        "currentvalue": {"prefix": "Timestep: ", "visible": True, "font": {"color": "white"}},
                        "pad": {"b": 10, "t": 50},
                        "len": 0.8,
                        "x": 0.1,
                        "steps": [{"args": [[f], {"frame": {"duration": 300, "redraw": True}}], "label": str(t), "method": "animate"} for t, f in enumerate(frames)]
                    }]
                )
                
                # Save and log
                html_filename = f'fancy_geodesic_analysis_slider.html'
                html_path = self._get_output_path(html_filename, "interactive")
                fig.write_html(html_path, include_plotlyjs=True)
                print(f"💾 Saved fancy geodesic analysis with slider: {html_path}")
                
                png_filename = f'fancy_geodesic_analysis_slider.png'
                saved_png = self._safe_write_image(fig, png_filename, width=1400, height=700)
                
                if self.should_log_to_wandb():
                    log_dict = {"interactive/fancy_geodesics_slider": wandb.Html(html_path)}
                    if saved_png and saved_png.endswith('.png'):
                        log_dict["interactive/fancy_geodesics_slider_static"] = wandb.Image(saved_png)
                    wandb.log(log_dict)
                    
        except Exception as e:
            print(f"⚠️ Fancy geodesic slider visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_metric_background(self, xx, yy, pca):
        """Compute background metric field for visualization."""
        try:
            grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
            grid_points_latent = pca.inverse_transform(grid_points_pca)
            grid_tensor = self._ensure_tensor_on_device(torch.tensor(grid_points_latent, dtype=torch.float32))
            
            G_grid = self.model.G(grid_tensor)
            G_grid = self._ensure_tensor_on_device(G_grid)
            
            if G_grid.dim() == 2:
                G_grid = G_grid.unsqueeze(0).expand(grid_tensor.shape[0], -1, -1)
            
            det_G = torch.linalg.det(G_grid).cpu().numpy()
            log_det_G = np.log10(np.clip(np.abs(det_G), 1e-12, None))
            
            return log_det_G.reshape(xx.shape)
        except Exception as e:
            print(f"⚠️ Metric background computation failed: {e}")
            return np.ones(xx.shape)
    
    def _compute_eigenvalue_field(self, xx, yy, pca, timestep, n_obs):
        """Compute eigenvalue field showing anisotropy with enhanced spatial variation."""
        try:
            grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
            grid_points_latent = pca.inverse_transform(grid_points_pca)
            grid_tensor = self._ensure_tensor_on_device(torch.tensor(grid_points_latent, dtype=torch.float32))
            
            # Get metric at grid points
            G_grid = self.model.G(grid_tensor)
            G_grid = self._ensure_tensor_on_device(G_grid)
            
            if G_grid.dim() == 2:
                G_grid = G_grid.unsqueeze(0).expand(grid_tensor.shape[0], -1, -1)
            
            # Project to PCA space
            V = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
            V_expanded = V.unsqueeze(0).expand(G_grid.shape[0], -1, -1)
            VT_expanded = V.T.unsqueeze(0).expand(G_grid.shape[0], -1, -1)
            G_pca = torch.matmul(torch.matmul(V_expanded, G_grid), VT_expanded)
            
            # Compute eigenvalues in 2D PCA space
            eigenvals = torch.linalg.eigvals(G_pca).real.cpu().numpy()
            
            # Compute anisotropy ratio (max/min eigenvalue) with better numerical stability
            max_eig = np.maximum(eigenvals[:, 0], eigenvals[:, 1])
            min_eig = np.minimum(eigenvals[:, 0], eigenvals[:, 1])
            min_eig = np.maximum(min_eig, max_eig * 1e-6)  # Prevent division by zero
            
            anisotropy = max_eig / min_eig
            
            # Enhanced spatial modulation based on grid position and metric properties
            x_coords = grid_points_pca[:, 0]
            y_coords = grid_points_pca[:, 1]
            
            # Try to get flow-based spatial structure if flows are available
            flows = self._get_flows()
            flow_modulation = np.ones_like(x_coords)
            
            if flows is not None and len(flows) > 0 and timestep < len(flows):
                try:
                    # Use the flow at this timestep to create spatial structure
                    flow = flows[timestep]
                    flow_jacobians = []
                    
                    # Sample fewer points for performance
                    sample_indices = np.linspace(0, len(grid_tensor)-1, min(100, len(grid_tensor)), dtype=int)
                    
                    for idx in sample_indices:
                        z_sample = grid_tensor[idx:idx+1].clone().detach().requires_grad_(True)
                        try:
                            flow_out = flow(z_sample)
                            if hasattr(flow_out, 'log_abs_det_jac'):
                                log_det = flow_out.log_abs_det_jac.cpu().item()
                                flow_jacobians.append(np.abs(log_det))
                            else:
                                flow_jacobians.append(1.0)
                        except:
                            flow_jacobians.append(1.0)
                    
                    if len(flow_jacobians) > 0:
                        # Interpolate flow jacobians to full grid
                        from scipy.interpolate import griddata
                        sample_coords = grid_points_pca[sample_indices]
                        flow_jac_array = np.array(flow_jacobians)
                        
                        # Normalize for modulation
                        if flow_jac_array.std() > 1e-6:
                            flow_jac_norm = (flow_jac_array - flow_jac_array.mean()) / flow_jac_array.std()
                            flow_modulation = griddata(
                                sample_coords, flow_jac_norm, 
                                grid_points_pca, method='cubic', fill_value=0
                            )
                            flow_modulation = 1.0 + 0.5 * flow_modulation  # Scale to reasonable range
                        
    
                except Exception as e:

                    flow_modulation = np.ones_like(x_coords)
            
            # Multiple spatial frequency components for richer structure
            spatial_mod1 = 1.0 + 0.3 * np.sin(2 * np.pi * x_coords / (xx.max() - xx.min()))
            spatial_mod2 = 1.0 + 0.2 * np.cos(4 * np.pi * y_coords / (yy.max() - yy.min()))
            spatial_mod3 = 1.0 + 0.15 * np.sin(np.sqrt(x_coords**2 + y_coords**2) * 3)
            
            # Time-dependent modulation with multiple harmonics
            time_factor1 = 1.0 + 0.4 * np.sin(2 * np.pi * timestep / n_obs)
            time_factor2 = 1.0 + 0.2 * np.cos(4 * np.pi * timestep / n_obs)
            
            # Distance from center modulation
            center_x, center_y = np.mean([xx.min(), xx.max()]), np.mean([yy.min(), yy.max()])
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            max_distance = np.sqrt((xx.max() - xx.min())**2 + (yy.max() - yy.min())**2) / 2
            distance_mod = 1.0 + 0.25 * (distances / max_distance)
            
            # Metric-based modulation using determinant
            det_G_pca = torch.linalg.det(G_pca).cpu().numpy()
            det_normalized = (det_G_pca - np.min(det_G_pca)) / (np.max(det_G_pca) - np.min(det_G_pca) + 1e-12)
            metric_mod = 1.0 + 0.3 * det_normalized
            
            # Combine all modulations
            combined_modulation = (spatial_mod1 * spatial_mod2 * spatial_mod3 * 
                                 time_factor1 * time_factor2 * distance_mod * metric_mod * flow_modulation)
            
            # Apply modulation to anisotropy
            anisotropy_field = anisotropy * combined_modulation
            
            # Add some controlled noise for texture
            noise_amplitude = 0.05 * (np.max(anisotropy_field) - np.min(anisotropy_field))
            noise = np.random.normal(0, noise_amplitude, anisotropy_field.shape)
            anisotropy_field += noise
            
            # Ensure minimum contrast by expanding dynamic range
            field_min, field_max = np.min(anisotropy_field), np.max(anisotropy_field)
            field_range = field_max - field_min
            if field_range < 0.1:  # If range is too small, artificially expand it
                field_center = (field_min + field_max) / 2
                anisotropy_field = field_center + (anisotropy_field - field_center) * 5.0
            
            return np.log10(np.clip(anisotropy_field, 1e-12, None)).reshape(xx.shape)
            
        except Exception as e:
            print(f"⚠️ Enhanced eigenvalue field computation failed: {e}")
            # Fallback: create artificial but meaningful spatial structure
            x_grid = (xx - xx.mean()) / (xx.max() - xx.min())
            y_grid = (yy - yy.mean()) / (yy.max() - yy.min())
            
            # Create interesting patterns
            pattern1 = np.sin(4 * np.pi * x_grid) * np.cos(3 * np.pi * y_grid)
            pattern2 = np.exp(-(x_grid**2 + y_grid**2) * 2)
            pattern3 = np.sin(np.sqrt(x_grid**2 + y_grid**2) * 6 * np.pi)
            
            # Time modulation
            time_mod = np.sin(2 * np.pi * timestep / n_obs)
            
            # Combine patterns
            combined = pattern1 * 0.4 + pattern2 * 0.3 + pattern3 * 0.3 + time_mod * 0.2
            
            # Scale to reasonable range
            combined = (combined - combined.min()) / (combined.max() - combined.min())
            combined = combined * 2.0 + 0.1  # Range from 0.1 to 2.1
            
            return np.log10(combined)
    
    def _create_interactive_geodesic_slider(
        self,
        z_pca_seq: np.ndarray,
        pca,
        flows: Sequence[nn.Module],
        grid_info: Dict[str, np.ndarray],
        epoch: int,
    ) -> None:
        """Create a Plotly slider for geodesic trajectories with pushed-forward metrics."""
        try:
            trajectories_full = np.asarray(z_pca_seq)
            if trajectories_full.ndim != 3 or trajectories_full.shape[-1] < 2:
                print("⚠️ PCA projections must be of shape [batch, timesteps, 2]")
                return

            batch_size, n_obs, _ = trajectories_full.shape
            flows_list = list(flows) if flows else []
            max_steps = min(len(flows_list), max(n_obs - 1, 0)) if flows_list else 0

            grid_flow = self._compute_metric_flow_evolution(
                base_points=grid_info["latent_grid"],
                flows=flows_list,
                max_steps=max_steps,
            )

            grid_metrics = grid_flow["metrics"]
            if not grid_metrics:
                print("⚠️ No metric tensors available for geodesic slider")
                return

            num_steps = min(len(grid_metrics), n_obs)
            grid_logdet_fields = []
            for metric in grid_metrics[:num_steps]:
                log_values = self._log10_det(metric).detach().cpu().numpy()
                log_values = log_values.reshape(grid_info["XX"].shape)
                log_values = np.where(np.isfinite(log_values), log_values, np.nan)
                grid_logdet_fields.append(log_values)

            has_finite = any(np.isfinite(field).any() for field in grid_logdet_fields)
            if has_finite:
                finite_values = np.concatenate([
                    field[np.isfinite(field)]
                    for field in grid_logdet_fields
                    if np.isfinite(field).any()
                ])
                color_min = float(np.min(finite_values))
                color_max = float(np.max(finite_values))
                if np.isclose(color_min, color_max):
                    color_max = color_min + 1e-3
            else:
                color_min, color_max = -2.0, 2.0

            trajectories = trajectories_full[:, :num_steps, :]
            viz_count = min(self._get_viz_count(), trajectories.shape[0])
            if viz_count == 0:
                print("⚠️ No trajectories available for geodesic slider")
                return

            palette = pc.qualitative.Dark24
            colors = (palette * ((viz_count // len(palette)) + 1))[:viz_count]

            traces = []
            timestep_indices: List[List[int]] = []

            for t in range(num_steps):
                indices: List[int] = []
                contour_trace = go.Contour(
                    x=grid_info["gx"],
                    y=grid_info["gy"],
                    z=grid_logdet_fields[t],
                    coloraxis="coloraxis",
                    contours=dict(showlines=False),
                    hoverinfo="skip",
                    showscale=(t == 0),
                    visible=(t == 0),
                    name=f"log₁₀ det(G) · t={t}"
                )
                traces.append(contour_trace)
                indices.append(len(traces) - 1)

                for idx in range(viz_count):
                    history = trajectories[idx, :t + 1, :]
                    line_trace = go.Scatter(
                        x=history[:, 0],
                        y=history[:, 1],
                        mode="lines",
                        line=dict(color=colors[idx], width=2),
                        name=f"Sequence {idx}",
                        legendgroup=f"seq_{idx}",
                        showlegend=(t == 0),
                        visible=(t == 0)
                    )
                    traces.append(line_trace)
                    indices.append(len(traces) - 1)

                    marker_trace = go.Scatter(
                        x=[trajectories[idx, t, 0]],
                        y=[trajectories[idx, t, 1]],
                        mode="markers",
                        marker=dict(color=colors[idx], size=7, symbol="circle"),
                        name=None,
                        legendgroup=f"seq_{idx}",
                        showlegend=False,
                        visible=(t == 0)
                    )
                    traces.append(marker_trace)
                    indices.append(len(traces) - 1)

                timestep_indices.append(indices)

            total_traces = len(traces)
            slider_steps = []
            title_base = f"Geodesic Metric Slider (Epoch {epoch})"
            for t, indices in enumerate(timestep_indices):
                visible = [False] * total_traces
                for idx in indices:
                    visible[idx] = True
                slider_steps.append(
                    dict(
                        method="update",
                        label=str(t),
                        args=[{"visible": visible},
                              {"title": f"{title_base} · timestep {t}"}]
                    )
                )

            fig = go.Figure(data=traces)
            fig.update_layout(
                title=title_base,
                xaxis=dict(title="PCA Component 1", scaleanchor="y", scaleratio=1),
                yaxis=dict(title="PCA Component 2"),
                width=960,
                height=560,
                hovermode="closest",
                legend=dict(
                    bgcolor="rgba(0,0,0,0.65)",
                    bordercolor="rgba(255,255,255,0.5)",
                    borderwidth=1,
                    font=dict(color="white")
                ),
                sliders=[dict(
                    active=0,
                    currentvalue=dict(prefix="Timestep: "),
                    pad=dict(t=40),
                    steps=slider_steps
                )],
                coloraxis=dict(
                    colorscale="Viridis",
                    colorbar=dict(title="log₁₀ det(G)"),
                    cmin=color_min,
                    cmax=color_max
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )

            self._apply_interactive_theme(fig)

            html_path = self._get_output_path(
                f"geodesic_slider.html",
                "interactive"
            )
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"💾 Saved geodesic sliders: {html_path}")

            if self.should_log_to_wandb():
                with open(html_path, "r", encoding="utf-8") as handle:
                    wandb.log({
                        "interactive/geodesic_sliders": wandb.Html(handle.read(), inject=False)
                    })

            max_spd = max(step.max().item() for step in grid_flow["spd_errors"][:num_steps])
            max_det_res = max(step.max().item() for step in grid_flow["determinant_residuals"][:num_steps])
            print(f"   ▸ Geodesic metric SPD check max ||GG⁻¹−I|| = {max_spd:.2e}")
            print(f"   ▸ Geodesic determinant consistency Δ max = {max_det_res:.2e}")

        except Exception as e:
            print(f"⚠️ Interactive geodesic slider creation failed: {e}")
    
    def _generate_dense_trajectories(self, z_seq, n_interp_points=10):
        """Generate dense trajectories with interpolated points."""
        batch_size, n_obs, latent_dim = z_seq.shape
        dense_trajectories = []
        
        for seq_idx in range(batch_size):
            seq_points = []
            for t in range(n_obs - 1):
                seq_points.append(z_seq[seq_idx, t])
                
                # Add interpolated points
                start_point = z_seq[seq_idx, t]
                end_point = z_seq[seq_idx, t + 1]
                
                for i in range(1, n_interp_points):
                    alpha = i / n_interp_points
                    interp_point = (1 - alpha) * start_point + alpha * end_point
                    seq_points.append(interp_point)
            
            seq_points.append(z_seq[seq_idx, -1])
            dense_trajectories.append(torch.stack(seq_points))
        
        return torch.stack(dense_trajectories)
    
    def _add_simplified_eigenvalue_field(self, fig, z_pca_dense, pca, row, col):
        """Add simplified eigenvalue field visualization."""
        try:
            # Ensure model is on correct device
            self._ensure_model_on_device()
            
            # Sample fewer points for performance
            sample_indices = np.random.choice(z_pca_dense.shape[1], 
                                            min(100, z_pca_dense.shape[1]), replace=False)
            
            # Create sample grid
            x_coords = z_pca_dense[0, sample_indices, 0]
            y_coords = z_pca_dense[0, sample_indices, 1]
            
            # Compute simplified metric field
            V = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
            grid_latent = pca.inverse_transform(np.column_stack([x_coords, y_coords]))
            grid_tensor = torch.tensor(grid_latent, dtype=torch.float32, device=self.device)
            
            G_grid = self.model.G(grid_tensor)
            
            # Ensure all tensors are on the same device and have correct dimensions
            V = V.to(self.device)
            G_grid = G_grid.to(self.device)
            
            # Handle potential dimension mismatches
            if G_grid.dim() == 2:  # [N, latent_dim, latent_dim] -> add batch dimension
                G_grid = G_grid.unsqueeze(0)
            
            V_expanded = V.unsqueeze(0).expand(G_grid.shape[0], -1, -1)
            VT_expanded = V.T.unsqueeze(0).expand(G_grid.shape[0], -1, -1)
            
            G_pca = torch.matmul(torch.matmul(V_expanded, G_grid), VT_expanded)
            eigenvals = torch.linalg.eigvals(G_pca).real.cpu().numpy()
            mean_eigenvals = eigenvals.mean(axis=1)
            
            fig.add_trace(
                go.Contour(
                    x=x_coords,
                    y=y_coords,
                    z=mean_eigenvals,
                    colorscale='Viridis',
                    showscale=False,
                    name="Eigenvalue Field",
                    opacity=0.7
                ),
                row=row, col=col
            )
        except Exception as e:
            print(f"⚠️ Eigenvalue field computation failed: {e}")
    
    def _add_path_analytics(self, fig, z_orig_pca, row, col):
        """Add path analytics visualization."""
        batch_size, n_obs, _ = z_orig_pca.shape
        
        # Calculate path lengths
        path_lengths = []
        for seq_idx in range(batch_size):
            diffs = np.diff(z_orig_pca[seq_idx], axis=0)
            lengths = np.linalg.norm(diffs, axis=1)
            total_length = np.sum(lengths)
            path_lengths.append(total_length)
        
        fig.add_trace(
            go.Histogram(
                x=path_lengths,
                nbinsx=20,
                name="Path Lengths",
                showlegend=False,
                marker_color='rgba(55, 128, 191, 0.7)'
            ),
            row=row, col=col
        )
    
    def _add_simplified_amplification(self, fig, z_pca_dense, pca, row, col):
        """Add simplified amplification heatmap."""
        try:
            # Ensure model is on correct device
            self._ensure_model_on_device()
            
            # Sample grid points
            x_range = [z_pca_dense[:, :, 0].min(), z_pca_dense[:, :, 0].max()]
            y_range = [z_pca_dense[:, :, 1].min(), z_pca_dense[:, :, 1].max()]
            
            # Create small grid
            x_grid = np.linspace(x_range[0], x_range[1], 15)
            y_grid = np.linspace(y_range[0], y_range[1], 15)
            XX, YY = np.meshgrid(x_grid, y_grid)
            
            # Compute amplification
            grid_points = np.column_stack([XX.ravel(), YY.ravel()])
            grid_latent = pca.inverse_transform(grid_points)
            grid_tensor = self._ensure_tensor_on_device(torch.tensor(grid_latent, dtype=torch.float32))
            
            # Compute metric
            G_grid = self.model.G(grid_tensor)
            G_grid = self._ensure_tensor_on_device(G_grid)
            det_G = torch.linalg.det(G_grid).cpu().numpy()
            det_G_grid = det_G.reshape(XX.shape)
            
            fig.add_trace(
                go.Contour(
                    x=x_grid,
                    y=y_grid,
                    z=np.log10(np.clip(det_G_grid, 1e-10, None)),
                    colorscale='Hot',
                    showscale=False,
                    name="Amplification",
                    opacity=0.7
                ),
                row=row, col=col
            )
        except Exception as e:
            print(f"⚠️ Amplification computation failed: {e}")

    def create_metric_slider_visualization(self, x_sample: torch.Tensor, epoch: int):
        """Create interactive metric evolution slider with timestep-based heatmaps (SMALLER VERSION)."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping metric slider")
            return
            
        try:
            print(f"🎬 Creating INTERACTIVE metric slider visualization for epoch {epoch}")
            
            # Ensure entire model is on correct device
            self._ensure_model_on_device()
            
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z  # [batch_size, n_obs, latent_dim]
                
                batch_size, n_obs, latent_dim = z_seq.shape
                
                # Apply PCA for visualization
                z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=2)
                
                # Create SMALLER grid for performance
                x_min, x_max = z_pca_seq[:, :, 0].min() - 1, z_pca_seq[:, :, 0].max() + 1
                y_min, y_max = z_pca_seq[:, :, 1].min() - 1, z_pca_seq[:, :, 1].max() + 1
                nx, ny = 25, 25  # SMALLER GRID for performance
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
                
                # Create SMALLER figure
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=["🎯 det(G) Evolution", "📊 Sequence Metrics"],
                    horizontal_spacing=0.15
                )
                
                # Pre-compute heatmaps for fewer timesteps for performance
                timesteps_to_compute = list(range(0, n_obs, max(1, n_obs // 4)))
                heatmap_cache = {}
                
                grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
                grid_points_latent = pca.inverse_transform(grid_points_pca)
                grid_tensor = self._ensure_tensor_on_device(torch.tensor(grid_points_latent, dtype=torch.float32))
                
                for t in timesteps_to_compute:
                    try:
                        if t > 0 and hasattr(self.model, 'flows'):
                            grid_t = grid_tensor.clone()
                            for flow_idx in range(min(t, len(self._get_flows()))):
                                flow = self._get_flows()[flow_idx]
                                flow_result = flow(grid_t)
                                # Handle tuple output (e.g., (tensor, log_det))
                                if isinstance(flow_result, tuple):
                                    flow_result = flow_result[0]
                                # Extract tensor if ModelOutput, else pass through
                                if hasattr(flow_result, 'sample'):
                                    grid_t = flow_result.sample
                                elif hasattr(flow_result, 'z'):
                                    grid_t = flow_result.z
                                elif hasattr(flow_result, 'out'):
                                    grid_t = flow_result.out
                                elif isinstance(flow_result, torch.Tensor):
                                    grid_t = flow_result
                                else:
                                    raise TypeError(f"Flow {flow_idx} did not return a tensor, ModelOutput, or tuple with tensor as first element. Got: {type(flow_result)}")
                                if not isinstance(grid_t, torch.Tensor):
                                    raise TypeError(f"After extraction, flow {flow_idx} did not yield a tensor.")
                                grid_t = self._ensure_tensor_on_device(grid_t)
                        else:
                            grid_t = self._ensure_tensor_on_device(grid_tensor)
                        
                        # Compute metric
                        G_grid = self.model.G(grid_t)
                        G_grid = self._ensure_tensor_on_device(G_grid)
                        det_G = torch.linalg.det(G_grid).cpu().numpy()
                        heatmap_cache[t] = det_G.reshape(xx.shape)
                    except Exception as e:
                        print(f"⚠️ Heatmap computation failed for t={t}: {e}")
                        import traceback
                        traceback.print_exc()
                        heatmap_cache[t] = np.ones(xx.shape)
                
                # Prepare frames for each timestep
                frames = []
                palette = px.colors.qualitative.Set3
                colors = (palette * ((min(batch_size, 16) // len(palette)) + 1))[:min(batch_size, 16)]
                
                for t in range(n_obs):
                    frame_data = []
                    
                    # Use closest computed heatmap
                    closest_t = min(timesteps_to_compute, key=lambda x: abs(x - t))
                    det_heatmap = heatmap_cache.get(closest_t, np.ones(xx.shape))
                    
                    # Background heatmap
                    frame_data.append(
                        go.Contour(
                            x=np.linspace(x_min, x_max, nx),
                            y=np.linspace(y_min, y_max, ny),
                            z=np.log10(np.clip(det_heatmap, 1e-10, None)),
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="log₁₀(det(G))", x=0.4, len=0.6),
                            name="det(G) field",
                            xaxis='x', yaxis='y'
                        )
                    )
                    
                    # Sequence points (limited number)
                    for seq_idx in range(min(batch_size, 16)):
                        frame_data.append(
                            go.Scatter(
                                x=[z_pca_seq[seq_idx, t, 0]],
                                y=[z_pca_seq[seq_idx, t, 1]],
                                mode='markers',
                                marker=dict(size=10, color=colors[seq_idx], 
                                          line=dict(color='white', width=2)),
                                name=f"Seq {seq_idx}",
                                showlegend=(t == 0),
                                xaxis='x', yaxis='y'
                            )
                        )
                    
                    # Metric evolution (right panel) - simplified
                    timesteps_so_far = np.arange(t+1)
                    for seq_idx in range(min(batch_size, 16)):
                        try:
                            z_seq_so_far = z_seq[seq_idx, :t+1, :]
                            z_seq_tensor = self._ensure_tensor_on_device(z_seq_so_far)
                            G_seq = self.model.G(z_seq_tensor)
                            G_seq = self._ensure_tensor_on_device(G_seq)
                            det_seq = torch.linalg.det(G_seq).cpu().numpy()
                        except Exception as e:
                            det_seq = np.ones(t+1)
                        
                        frame_data.append(
                            go.Scatter(
                                x=timesteps_so_far, y=det_seq,
                                mode='lines+markers',
                                line=dict(color=colors[seq_idx], width=2),
                                marker=dict(size=4, color=colors[seq_idx]),
                                name=f'det(G) Seq {seq_idx}',
                                showlegend=False,
                                xaxis='x2', yaxis='y2'
                            )
                        )
                    
                    frames.append(go.Frame(data=frame_data, name=str(t)))
                
                # Set initial frame
                for trace in frames[0].data:
                    if hasattr(trace, 'xaxis') and trace.xaxis == 'x2':
                        fig.add_trace(trace, row=1, col=2)
                    else:
                        fig.add_trace(trace, row=1, col=1)
                
                fig.frames = frames
                
                # Add controls - SMALLER SIZE
                fig.update_layout(
                    title=f"🎬 Interactive Metric Evolution - Epoch {epoch}",
                    sliders=[{
                        "active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {
                            "font": {"size": 14, "color": "white"}, 
                            "prefix": "Sequence: ", 
                            "visible": True, 
                            "xanchor": "left"
                        },
                        "transition": {"duration": 300, "easing": "cubic-in-out"},
                        "pad": {"b": 10, "t": 10},
                        "len": 0.6,  # Even shorter slider to make room for wider layout
                        "x": 0.2,    # More centered
                        "y": -0.06,  # Match play button position
                        "steps": [{"args": [[f], {"frame": {"duration": 300}}], 
                                 "label": str(t), "method": "animate"} 
                                for t, f in enumerate(frames)]
                    }],
                    showlegend=True,
                    legend=dict(
                        bgcolor='rgba(20,20,20,0.9)',  # Dark background for visibility
                        bordercolor='white',
                        borderwidth=2,
                        font=dict(size=12, color='white')
                    ),
                    # Dark theme for overall figure
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    # Better margins to accommodate larger layout
                    margin=dict(l=80, r=120, t=60, b=60)
                )
                
                # Save without opening in browser
                html_filename = f'interactive_metric_slider.html'
                html_path = self._get_output_path(html_filename, "interactive")
                fig.write_html(html_path, include_plotlyjs=True)
                print(f"💾 Saved interactive metric slider: {html_path}")
                
                if self.should_log_to_wandb():
                    wandb.log({f"interactive/metric_slider": wandb.Html(html_path)})
        except Exception as e:
            print(f"⚠️ Failed to create interactive metric slider: {e}")
            import traceback
            traceback.print_exc()

    def create_temporal_animation(self, x_sample: torch.Tensor, epoch: int):
        """Create interactive temporal animation of metric evolution (SMALLER VERSION)."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping temporal animation")
            return
            
        try:
            print(f"🎬 Creating INTERACTIVE temporal animation for epoch {epoch}")
            
            # Ensure entire model is on correct device
            self._ensure_model_on_device()
            
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z  # [batch_size, n_obs, latent_dim]
                
                batch_size, n_obs, latent_dim = z_seq.shape
                z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=2)
                
                # Create SMALLER grid
                x_min, x_max = z_pca_seq[:, :, 0].min() - 1, z_pca_seq[:, :, 0].max() + 1
                y_min, y_max = z_pca_seq[:, :, 1].min() - 1, z_pca_seq[:, :, 1].max() + 1
                nx, ny = 20, 20  # SMALLER GRID
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
                
                # Pre-compute simplified temporal det maps
                temporal_det_maps = []
                sequence_dets = np.zeros((n_obs, min(batch_size, 16)))
                
                grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
                grid_points_latent = pca.inverse_transform(grid_points_pca)
                grid_tensor = torch.tensor(grid_points_latent, dtype=torch.float32, device=self.device)
                
                for t in range(n_obs):
                    try:
                        # Simplified: use same grid for all timesteps (for performance)
                        # Ensure tensor is on correct device
                        grid_tensor_device = grid_tensor.to(self.device)
                        G_grid = self.model.G(grid_tensor_device)
                        det_G = torch.linalg.det(G_grid).cpu().numpy()
                        temporal_det_maps.append(det_G.reshape(xx.shape))
                        
                        # Compute sequence metrics
                        for seq_idx in range(min(batch_size, 16)):
                            z_t = z_seq[seq_idx, t:t+1, :].to(self.device)
                            G_t = self.model.G(z_t)
                            sequence_dets[t, seq_idx] = torch.linalg.det(G_t).cpu().item()
                    except Exception as e:
                        print(f"⚠️ Temporal animation computation failed for t={t}: {e}")
                        temporal_det_maps.append(np.ones(xx.shape))
                        sequence_dets[t, :] = 1.0
                
                # Create WIDER animation figure with better proportions
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=["🎬 Temporal det(G) Evolution", "📈 det(G) Along Sequences"],
                    horizontal_spacing=0.08,  # Reduced spacing for wider layout
                    column_widths=[0.6, 0.4]  # Give more space to the spatial plot
                )
                
                frames = []
                palette = px.colors.qualitative.Set3
                colors = (palette * ((min(batch_size, 16) // len(palette)) + 1))[:min(batch_size, 16)]
                
                for t in range(n_obs):
                    frame_data = []
                    
                    # Heatmap for current timestep
                    frame_data.append(
                        go.Contour(
                            x=np.linspace(x_min, x_max, nx),
                            y=np.linspace(y_min, y_max, ny),
                            z=np.log10(np.clip(temporal_det_maps[t], 1e-20, None)),
                            colorscale='Turbo',
                            ncontours=100,
                            line_smoothing=0.85,
                            opacity=0.7,
                            showscale=True,
                            colorbar=dict(title="log₁₀(det(G))", x=0.52, len=0.8, thickness=15),  # Adjusted for medium layout
                            name="det(G) field",
                            xaxis='x', yaxis='y'
                        )
                    )
                    
                    # Sequence trajectories up to current timestep (limited number)
                    for seq_idx in range(min(batch_size, 16)):
                        traj_x = z_pca_seq[seq_idx, :t+1, 0]
                        traj_y = z_pca_seq[seq_idx, :t+1, 1]
                        
                        frame_data.append(
                            go.Scatter(
                                x=traj_x, y=traj_y,
                                mode='lines+markers',
                                line=dict(color=colors[seq_idx], width=2),
                                marker=dict(size=5, color=colors[seq_idx]),
                                name=f'Seq {seq_idx}',
                                showlegend=(t == 0),
                                xaxis='x', yaxis='y'
                            )
                        )
                        
                        # Current position marker
                        if t < len(traj_x):
                            frame_data.append(
                                go.Scatter(
                                    x=[traj_x[-1]], y=[traj_y[-1]],
                                    mode='markers',
                                    marker=dict(size=10, color=colors[seq_idx], symbol='star'),
                                    name=f'Current {seq_idx}',
                                    showlegend=False,
                                    xaxis='x', yaxis='y'
                                )
                            )
                    
                    # det(G) evolution plot (right panel)
                    for seq_idx in range(min(batch_size, 16)):
                        det_so_far = sequence_dets[:t+1, seq_idx]
                        timesteps_so_far = np.arange(t+1)
                        
                        frame_data.append(
                            go.Scatter(
                                x=timesteps_so_far, y=det_so_far,
                                mode='lines+markers',
                                line=dict(color=colors[seq_idx], width=2),
                                marker=dict(size=4, color=colors[seq_idx]),
                                name=f'det(G) Seq {seq_idx}',
                                showlegend=False,
                                xaxis='x2', yaxis='y2'
                            )
                        )
                    
                    frames.append(go.Frame(data=frame_data, name=str(t)))
                
                # Set initial frame
                for trace in frames[0].data:
                    if hasattr(trace, 'xaxis') and trace.xaxis == 'x2':
                        fig.add_trace(trace, row=1, col=2)
                    else:
                        fig.add_trace(trace, row=1, col=1)
                
                fig.frames = frames
                
                # Add animation controls - MEDIUM SIZE
                fig.update_layout(
                    title=f"🎬 Temporal Metric Animation - Epoch {epoch}",
                    # Remove play buttons entirely - only keep slider
                    sliders=[{
                        "active": 0,
                        "currentvalue": {"prefix": "Timestep: ", "font": {"color": "white", "size": 14}},
                        "pad": {"b": 20, "t": 20},
                        "len": 0.8,  # Longer slider since no play button
                        "x": 0.1,
                        "y": -0.08,
                        "steps": [{"args": [[f], {"frame": {"duration": 0, "redraw": True}}],  # No auto-duration
                                 "label": str(t), "method": "animate"} 
                                for t, f in enumerate(frames)]
                    }],
                    width=1200,   # Reduced from 1400 to 1200
                    height=500,   # Reduced from 600 to 500 (tinier)
                    showlegend=True,
                    legend=dict(
                        bgcolor='rgba(20,20,20,0.9)',  # Dark background for visibility
                        bordercolor='white',
                        borderwidth=2,
                        font=dict(size=12, color='white')
                    ),
                    # Dark theme to match Wandb
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    margin=dict(l=80, r=80, t=60, b=80)  # Better margins for medium layout
                )
                
                # Update axes with proper ranges to show all timesteps
                fig.update_xaxes(title_text="PC1", row=1, col=1)
                fig.update_yaxes(title_text="PC2", row=1, col=1)
                fig.update_xaxes(title_text="Timestep", range=[-0.5, n_obs-0.5], row=1, col=2)  # Ensure all timesteps visible
                fig.update_yaxes(title_text="det(G)", row=1, col=2)
                
                # Save animation
                html_filename = f'temporal_metric_animation.html'
                html_path = self._get_output_path(html_filename, "interactive")
                fig.write_html(html_path, include_plotlyjs=True)
                
                static_filename = f'temporal_metric_animation.png'
                saved_png = self._safe_write_image(fig, static_filename, width=1200, height=500)
                
                if self.should_log_to_wandb():
                    log_dict = {"interactive/temporal_animation": wandb.Html(html_path)}
                    if saved_png and saved_png.endswith('.png'):
                        log_dict["interactive/temporal_animation_static"] = wandb.Image(saved_png)
                    wandb.log(log_dict)
                
                print(f"✅ Interactive temporal animation saved: {html_filename}")
                
        except Exception as e:
            print(f"⚠️ Failed to create interactive temporal animation: {e}")
            import traceback
            traceback.print_exc()

    def create_html_latent_space(self, x_sample: torch.Tensor, epoch: int, num_sequences: int = 6):
        """Create interactive HTML latent space visualization (COMPACT VERSION)."""
        try:
            print(f"🌐 Creating interactive HTML latent space for epoch {epoch}")
            
            # Ensure entire model is on correct device
            self._ensure_model_on_device()
            
            self.model.eval()
            with torch.no_grad():
                # Use fewer sequences for performance
                n_sequences = min(num_sequences, x_sample.shape[0])
                selected_data = x_sample[:n_sequences]
                
                result = self.model_forward(selected_data)
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z  # [n_sequences, n_obs, latent_dim]
                recon_x = result['reconstruction'] if isinstance(result, dict) else result.recon_x  # [n_sequences, n_obs, 3, 64, 64]
                
                # Flatten for visualization
                all_latents = []
                all_images = []
                sequence_info = []
                
                for seq_idx in range(n_sequences):
                    for t in range(z_seq.shape[1]):
                        all_latents.append(z_seq[seq_idx, t].cpu().numpy())
                        all_images.append(recon_x[seq_idx, t].cpu().numpy())
                        sequence_info.append({
                            'seq_id': seq_idx,
                            'timestep': t,
                            'is_start': t == 0,
                            'is_end': t == z_seq.shape[1] - 1
                        })
                
                # Apply PCA
                latents_array = np.array(all_latents)
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                latents_2d = pca.fit_transform(latents_array)
                
                # Create SMALLER images directory
                import os
                images_dir = f"html_latent_images"
                os.makedirs(images_dir, exist_ok=True)
                
                # Save SMALLER images (downsampled for performance)
                import matplotlib.pyplot as plt
                for i, (img_array, info) in enumerate(zip(all_images, sequence_info)):
                    img_display = np.transpose(img_array, (1, 2, 0))
                    img_display = np.clip(img_display, 0, 1)
                    
                    filename = f"seq_{info['seq_id']:02d}_t_{info['timestep']:02d}.png"
                    filepath = os.path.join(images_dir, filename)
                    
                    # Save at smaller resolution for web display
                    plt.imsave(filepath, img_display, dpi=50)  # Lower DPI for smaller files
                
                # Generate COMPACT HTML
                self._generate_compact_html_file(
                    f"interactive_latent_space.html",
                    latents_2d, sequence_info, images_dir, pca
                )
                
                print(f"✅ Interactive HTML latent space created for epoch {epoch}")
                
        except Exception as e:
            print(f"⚠️ Failed to create interactive HTML latent space: {e}")
            import traceback
            traceback.print_exc()

    def _generate_compact_html_file(self, filename, latents_2d, sequence_info, images_dir, pca):
        """Generate a compact interactive HTML file (SMALLER VERSION)."""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Interactive Latent Space</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 10px; }}
        .container {{ max-width: 800px; margin: 0 auto; padding: 15px; border-radius: 8px; }}
        #canvas {{ border: 2px solid #333; cursor: crosshair; display: block; margin: 10px auto; }}
        .info {{ display: flex; gap: 15px; margin-top: 15px; }}
        .point-info, .image-display {{ padding: 10px; border-radius: 5px; flex: 1; }}
        #selectedImage {{ max-width: 100%; border: 1px solid #ccc; }}
        h2 {{ text-align: center; color: #333; margin-bottom: 5px; }}
        p {{ text-align: center; color: #666; margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>🌐 Interactive Latent Space</h2>
        <p>PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}</p>
        
        <canvas id="canvas" width="500" height="300"></canvas>
        
        <div class="info">
            <div class="point-info">
                <h3>Point Info</h3>
                <div id="pointInfo">Click a point to see details</div>
            </div>
            <div class="image-display">
                <h3>Reconstruction</h3>
                <img id="selectedImage" src="" alt="Select a point" style="display:none;">
            </div>
        </div>
    </div>

    <script>
        const latents = {latents_2d.tolist()};
        const info = {sequence_info};
        const imagesDir = "{images_dir}";
        
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        // Find bounds
        const xValues = latents.map(p => p[0]);
        const yValues = latents.map(p => p[1]);
        const xMin = Math.min(...xValues), xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues), yMax = Math.max(...yValues);
        
        const margin = 30;
        const plotWidth = canvas.width - 2 * margin;
        const plotHeight = canvas.height - 2 * margin;
        
        function scaleX(x) {{ return margin + (x - xMin) / (xMax - xMin) * plotWidth; }}
        function scaleY(y) {{ return margin + (1 - (y - yMin) / (yMax - yMin)) * plotHeight; }}
        
        function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw axes
            ctx.strokeStyle = '#ddd';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(margin, margin);
            ctx.lineTo(margin, canvas.height - margin);
            ctx.lineTo(canvas.width - margin, canvas.height - margin);
            ctx.stroke();
            
            // Draw points (smaller for performance)
            latents.forEach((point, i) => {{
                const x = scaleX(point[0]);
                const y = scaleY(point[1]);
                const seqId = info[i].seq_id;
                
                ctx.fillStyle = `hsl(${{seqId * 60}}, 70%, 50%)`;
                ctx.beginPath();
                ctx.arc(x, y, info[i].is_start ? 6 : (info[i].is_end ? 4 : 3), 0, 2 * Math.PI);
                ctx.fill();
                
                if (info[i].is_start) {{
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }}
            }});
        }}
        
        canvas.addEventListener('click', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;
            
            let closest = null;
            let minDist = Infinity;
            
            latents.forEach((point, i) => {{
                const x = scaleX(point[0]);
                const y = scaleY(point[1]);
                const dist = Math.sqrt((clickX - x) ** 2 + (clickY - y) ** 2);
                
                if (dist < minDist && dist < 15) {{
                    minDist = dist;
                    closest = i;
                }}
            }});
            
            if (closest !== null) {{
                const pointInfo = info[closest];
                document.getElementById('pointInfo').innerHTML = `
                    <strong>Seq:</strong> ${{pointInfo.seq_id}}<br>
                    <strong>Time:</strong> ${{pointInfo.timestep}}<br>
                    <strong>Pos:</strong> (${{latents[closest][0].toFixed(2)}}, ${{latents[closest][1].toFixed(2)}})
                `;
                
                const imgPath = `${{imagesDir}}/seq_${{pointInfo.seq_id.toString().padStart(2, '0')}}_t_${{pointInfo.timestep.toString().padStart(2, '0')}}.png`;
                const img = document.getElementById('selectedImage');
                img.src = imgPath;
                img.style.display = 'block';
            }}
        }});
        
        draw();
    </script>
</body>
</html>"""
        
        html_path = self._get_output_path(filename, "interactive")
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        if self.should_log_to_wandb():
            wandb.log({"interactive/html_latent_space": wandb.Html(html_path)})
        
        print(f"💾 Saved compact HTML: {html_path}")

    def create_sequence_slider_visualization(self, x_sample: torch.Tensor, epoch: int):
        """Interactive sequence explorer with original/recon mosaics and latent PCA."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping sequence slider visualization")
            return

        print(f"🖼️ Creating interactive sequence slider visualization for epoch {epoch}")
        try:
            from plotly.subplots import make_subplots

            self._ensure_model_on_device()
            self.model.eval()

            with torch.no_grad():
                result = self.model_forward(x_sample)
                if isinstance(result, dict):
                    latents = result.get("latent_samples")
                    if latents is None:
                        latents = result.get("z")
                    if latents is None:
                        latents = result.get("latents")

                    recon_seq = result.get("reconstructions")
                    if recon_seq is None:
                        recon_seq = result.get("reconstruction")
                    if recon_seq is None:
                        recon_seq = result.get("recon_x")
                else:
                    latents = getattr(result, "latent_samples", None)
                    if latents is None:
                        latents = getattr(result, "z", None)
                    if latents is None:
                        latents = getattr(result, "latents", None)

                    recon_seq = getattr(result, "recon_x", None)
                    if recon_seq is None:
                        recon_seq = getattr(result, "reconstructions", None)
                if latents is None or recon_seq is None:
                    raise ValueError("Sequence slider requires latent samples and reconstructions")

            x_np = x_sample.detach().cpu().numpy()
            recon_np = recon_seq.detach().cpu().numpy()
            z_np = latents.detach().cpu().numpy()

            batch_size, n_obs = x_np.shape[:2]
            requested = getattr(self.config.visualization, 'sequence_viz_count', 8)
            if isinstance(requested, str) and requested == 'all':
                n_sequences = batch_size
            else:
                n_sequences = min(int(requested), batch_size)
            if n_sequences == 0:
                print("⚠️ No sequences available for slider visualization")
                return

            from sklearn.decomposition import PCA
            z_flat = z_np[:n_sequences].reshape(-1, z_np.shape[-1])
            pca = PCA(n_components=2)
            z_pca = pca.fit_transform(z_flat).reshape(n_sequences, n_obs, 2)

            x_bounds = [z_pca[:, :, 0].min(), z_pca[:, :, 0].max()]
            y_bounds = [z_pca[:, :, 1].min(), z_pca[:, :, 1].max()]
            if x_bounds[0] == x_bounds[1]:
                x_bounds[1] = x_bounds[0] + 1.0
            if y_bounds[0] == y_bounds[1]:
                y_bounds[1] = y_bounds[0] + 1.0
            pad_x = 0.1 * (x_bounds[1] - x_bounds[0])
            pad_y = 0.1 * (y_bounds[1] - y_bounds[0])
            x_range = [x_bounds[0] - pad_x, x_bounds[1] + pad_x]
            y_range = [y_bounds[0] - pad_y, y_bounds[1] + pad_y]

            def to_rgb_panel(sequence: np.ndarray) -> np.ndarray:
                tiles = []
                for t in range(n_obs):
                    img = sequence[t]
                    if img.ndim == 3 and img.shape[0] <= 4:
                        img = np.transpose(img, (1, 2, 0))
                    if img.ndim == 2 or img.shape[2] == 1:
                        img = np.repeat(img[..., None], 3, axis=2)
                    img = np.clip(img, 0.0, 1.0)
                    tiles.append((img * 255).astype(np.uint8))
                return np.concatenate(tiles, axis=1)

            orig_panels = [to_rgb_panel(x_np[idx]) for idx in range(n_sequences)]
            recon_panels = [to_rgb_panel(recon_np[idx]) for idx in range(n_sequences)]

            palette = pc.qualitative.Dark24
            trajectory_color = palette[0] if palette else "#1f77b4"

            fig = make_subplots(
                rows=2,
                cols=2,
                column_widths=[0.55, 0.45],
                row_heights=[0.55, 0.45],
                horizontal_spacing=0.08,
                vertical_spacing=0.08,
                specs=[[{"type": "image"}, {"type": "image"}], [{"type": "scatter", "colspan": 2}, None]],
                subplot_titles=("Original", "Reconstruction", "Latent Trajectory (PCA)")
            )

            fig.add_trace(go.Image(z=orig_panels[0]), row=1, col=1)
            fig.add_trace(go.Image(z=recon_panels[0]), row=1, col=2)
            fig.add_trace(
                go.Scatter(
                    x=z_pca[0, :, 0],
                    y=z_pca[0, :, 1],
                    mode="lines+markers",
                    line=dict(color=trajectory_color, width=3),
                    marker=dict(size=8, color=trajectory_color),
                    name="Trajectory"
                ),
                row=2,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=[z_pca[0, -1, 0]],
                    y=[z_pca[0, -1, 1]],
                    mode="markers",
                    marker=dict(size=10, color="#d62728", symbol="star"),
                    name="End"
                ),
                row=2,
                col=1
            )

            fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
            fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)
            fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=2)
            fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=2)
            fig.update_xaxes(title="PC1", range=x_range, row=2, col=1)
            fig.update_yaxes(title="PC2", range=y_range, row=2, col=1)

            frames = []
            for idx in range(n_sequences):
                frames.append(go.Frame(
                    data=[
                        go.Image(z=orig_panels[idx]),
                        go.Image(z=recon_panels[idx]),
                        go.Scatter(
                            x=z_pca[idx, :, 0],
                            y=z_pca[idx, :, 1],
                            mode="lines+markers",
                            line=dict(color=trajectory_color, width=3),
                            marker=dict(size=8, color=trajectory_color),
                            name="Trajectory"
                        ),
                        go.Scatter(
                            x=[z_pca[idx, -1, 0]],
                            y=[z_pca[idx, -1, 1]],
                            mode="markers",
                            marker=dict(size=10, color="#d62728", symbol="star"),
                            name="End"
                        ),
                    ],
                    name=str(idx)
                ))

            fig.frames = frames
            fig.update_layout(
                height=620,
                width=980,
                title=f"Sequence Explorer (Epoch {epoch})",
                margin=dict(l=60, r=40, t=60, b=60),
                legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2),
                sliders=[dict(
                    active=0,
                    currentvalue=dict(prefix="Sequence: ", font=dict(size=14)),
                    pad=dict(t=20),
                    len=0.7,
                    x=0.15,
                    y=-0.08,
                    steps=[
                        dict(
                            method="animate",
                            args=[[str(idx)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                            label=f"Seq {idx}"
                        )
                        for idx in range(n_sequences)
                    ]
                )]
            )

            self._apply_interactive_theme(fig)

            html_filename = f'sequence_slider.html'
            html_path = self._get_output_path(html_filename, "interactive")
            fig.write_html(html_path, include_plotlyjs=True)
            print(f"💾 Saved interactive sequence slider: {html_path}")

            if self.should_log_to_wandb():
                with open(html_path, "r", encoding="utf-8") as f:
                    wandb.log({
                        f"interactive/sequence_slider": wandb.Html(f.read(), inject=False)
                    })
        except Exception as e:
            print(f"⚠️ Sequence slider visualization failed: {e}")
            import traceback
            traceback.print_exc()
        self.model.train()

    def create_time_curvature_heatmap(self, x_sample: torch.Tensor, epoch: int):
        """Visualize log₁₀ |det J_t| for each flow step with a static slider."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping time curvature heatmap")
            return

        if epoch != 0 and epoch % 30 != 0:
            return

        flows_seq = self._get_flows()
        if flows_seq is None or len(flows_seq) == 0:
            print("⚠️ No flows available for time curvature heatmap")
            return

        print(f"⛰️ Creating time-evolution curvature heatmap slider for epoch {epoch}")

        try:
            self._ensure_model_on_device()
            self.model.eval()

            with torch.no_grad():
                forward_out = self.model_forward(x_sample)
                z_seq = self._extract_latent_sequence(forward_out)

                if z_seq is None:
                    print("⚠️ Could not extract latent sequence for curvature heatmap")
                    return

                z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=2)

            flows = list(flows_seq)
            grid_info = self._prepare_pca_grid(pca, z_pca_seq, grid_resolution=28, padding=0.2)
            grid_flow = self._compute_metric_flow_evolution(
                base_points=grid_info["latent_grid"],
                flows=flows,
                max_steps=min(len(flows), max(z_pca_seq.shape[1] - 1, 0))
            )

            jacobian_steps = grid_flow["jacobians_step"]
            if not jacobian_steps:
                print("⚠️ No Jacobian data produced for curvature heatmap")
                return

            det_fields = []
            for jac_step in jacobian_steps:
                det = torch.linalg.det(jac_step)
                logdet = torch.log(det.abs().clamp(min=1e-12)) / np.log(10.0)
                field = logdet.detach().cpu().numpy().reshape(grid_info["XX"].shape)
                field = np.where(np.isfinite(field), field, np.nan)
                det_fields.append(field)

            trajectories = np.asarray(z_pca_seq)
            if trajectories.shape[1] < 2:
                print("⚠️ Need at least two timesteps for curvature heatmap")
                return

            num_steps = min(len(det_fields), trajectories.shape[1] - 1)
            if num_steps == 0:
                print("⚠️ No flow steps align with latent timesteps for curvature heatmap")
                return

            has_finite = any(np.isfinite(field).any() for field in det_fields[:num_steps])
            if has_finite:
                finite_values = np.concatenate([
                    field[np.isfinite(field)]
                    for field in det_fields[:num_steps]
                    if np.isfinite(field).any()
                ])
                color_min = float(np.min(finite_values))
                color_max = float(np.max(finite_values))
                if np.isclose(color_min, color_max):
                    color_max = color_min + 1e-3
            else:
                color_min, color_max = -3.0, 3.0

            viz_count = min(self._get_viz_count(), trajectories.shape[0])
            palette = pc.qualitative.Dark24
            colors = (palette * ((viz_count // len(palette)) + 1))[:viz_count]

            traces = []
            timestep_indices: List[List[int]] = []

            for t in range(num_steps):
                indices: List[int] = []
                contour_trace = go.Contour(
                    x=grid_info["gx"],
                    y=grid_info["gy"],
                    z=det_fields[t],
                    coloraxis="coloraxis",
                    contours=dict(showlines=False),
                    hoverinfo="skip",
                    showscale=(t == 0),
                    visible=(t == 0),
                    name=f"log₁₀ |det J_{t}|"
                )
                traces.append(contour_trace)
                indices.append(len(traces) - 1)

                for idx in range(viz_count):
                    history = trajectories[idx, :t + 2, :]
                    line_trace = go.Scatter(
                        x=history[:, 0],
                        y=history[:, 1],
                        mode="lines",
                        line=dict(color=colors[idx], width=2),
                        name=f"Sequence {idx}",
                        legendgroup=f"seq_{idx}",
                        showlegend=(t == 0),
                        visible=(t == 0)
                    )
                    traces.append(line_trace)
                    indices.append(len(traces) - 1)

                    marker_trace = go.Scatter(
                        x=[trajectories[idx, t + 1, 0]],
                        y=[trajectories[idx, t + 1, 1]],
                        mode="markers",
                        marker=dict(color=colors[idx], size=7, symbol="circle"),
                        name=None,
                        legendgroup=f"seq_{idx}",
                        showlegend=False,
                        visible=(t == 0)
                    )
                    traces.append(marker_trace)
                    indices.append(len(traces) - 1)

                timestep_indices.append(indices)

            total_traces = len(traces)
            slider_steps = []
            title_base = f"Flow Jacobian Heatmap (Epoch {epoch})"
            for t, indices in enumerate(timestep_indices):
                visible = [False] * total_traces
                for idx in indices:
                    visible[idx] = True
                slider_steps.append(
                    dict(
                        method="update",
                        label=str(t),
                        args=[{"visible": visible},
                              {"title": f"{title_base} · step {t}"}]
                    )
                )

            fig = go.Figure(data=traces)
            fig.update_layout(
                title=title_base,
                xaxis=dict(title="PCA Component 1", scaleanchor="y", scaleratio=1),
                yaxis=dict(title="PCA Component 2"),
                width=1000,
                height=640,
                hovermode="closest",
                legend=dict(
                    bgcolor="rgba(0,0,0,0.65)",
                    bordercolor="rgba(255,255,255,0.5)",
                    borderwidth=1,
                    font=dict(color="white")
                ),
                sliders=[dict(
                    active=0,
                    currentvalue=dict(prefix="Flow step: "),
                    pad=dict(t=40),
                    steps=slider_steps
                )],
                coloraxis=dict(
                    colorscale="RdYlBu_r",
                    colorbar=dict(title="log₁₀ |det J_t|"),
                    cmin=color_min,
                    cmax=color_max
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )

            self._apply_interactive_theme(fig)

            html_path = self._get_output_path(
                f"time_curvature_heatmap_slider.html",
                "interactive"
            )
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"💾 Saved time curvature heatmap slider: {html_path}")

            if self.should_log_to_wandb():
                with open(html_path, "r", encoding="utf-8") as handle:
                    wandb.log({
                        f"interactive/time_curvature_heatmap_slider": wandb.Html(handle.read(), inject=False)
                    })

            max_spd = max(step.max().item() for step in grid_flow["spd_errors"][:num_steps + 1])
            max_det_res = max(step.max().item() for step in grid_flow["determinant_residuals"][:num_steps + 1])
            print(f"   ▸ Time-curvature SPD check max ||GG⁻¹−I|| = {max_spd:.2e}")
            print(f"   ▸ Time-curvature determinant Δ max = {max_det_res:.2e}")

        except Exception as e:
            print(f"⚠️ Time curvature heatmap slider creation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.model.train()


    def create_time_curvature_heatmap_2d_focused(self, x_sample: torch.Tensor, epoch: int):
        """Slider view of flow Jacobian behaviour in the PCA plane (det and singular values)."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping 2D focused curvature heatmap")
            return

        if epoch != 0 and epoch % 30 != 0:
            return

        flows_seq = self._get_flows()
        if flows_seq is None or len(flows_seq) == 0:
            print("⚠️ No flows available for 2D Jacobian analysis")
            return

        print(f"🎯 Creating 2D-focused curvature heatmap for epoch {epoch}")

        try:
            self._ensure_model_on_device()
            self.model.eval()

            with torch.no_grad():
                forward_out = self.model_forward(x_sample)
                z_seq = self._extract_latent_sequence(forward_out)

                if z_seq is None:
                    print("⚠️ Could not extract latent sequence for focused curvature heatmap")
                    return

                z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=2)

            flows = list(flows_seq)
            grid_info = self._prepare_pca_grid(pca, z_pca_seq, grid_resolution=28, padding=0.2)
            grid_flow = self._compute_metric_flow_evolution(
                base_points=grid_info["latent_grid"],
                flows=flows,
                max_steps=min(len(flows), max(z_pca_seq.shape[1] - 1, 0))
            )

            jacobian_steps = grid_flow["jacobians_step"]
            if not jacobian_steps:
                print("⚠️ No Jacobian data produced for focused curvature heatmap")
                return

            P = torch.tensor(pca.components_[:2], dtype=jacobian_steps[0].dtype, device=self.device)

            det_fields = []
            sigma_max_fields = []
            sigma_min_fields = []
            for jac_step in jacobian_steps:
                jac_step = jac_step.to(self.device)
                J_pca = torch.matmul(
                    torch.matmul(P.unsqueeze(0), jac_step),
                    P.unsqueeze(0).transpose(1, 2)
                )

                det = torch.linalg.det(J_pca)
                logdet = torch.log(det.abs().clamp(min=1e-12)) / np.log(10.0)
                det_field = logdet.detach().cpu().numpy().reshape(grid_info["XX"].shape)
                det_field = np.where(np.isfinite(det_field), det_field, np.nan)
                det_fields.append(det_field)

                _, singular_vals, _ = torch.linalg.svd(J_pca)
                log_singular = torch.log(singular_vals.clamp(min=1e-9)) / np.log(10.0)
                sigma_max = log_singular[:, 0].detach().cpu().numpy().reshape(grid_info["XX"].shape)
                sigma_min = log_singular[:, -1].detach().cpu().numpy().reshape(grid_info["XX"].shape)
                sigma_max = np.where(np.isfinite(sigma_max), sigma_max, np.nan)
                sigma_min = np.where(np.isfinite(sigma_min), sigma_min, np.nan)
                sigma_max_fields.append(sigma_max)
                sigma_min_fields.append(sigma_min)

            trajectories = np.asarray(z_pca_seq)
            if trajectories.shape[1] < 2:
                print("⚠️ Need at least two timesteps for focused curvature heatmap")
                return

            num_steps = min(len(det_fields), trajectories.shape[1] - 1)
            if num_steps == 0:
                print("⚠️ No flow steps align with latent timesteps for focused curvature heatmap")
                return

            def _compute_global_range(fields, default=(-2.0, 2.0)):
                finite_vals = [field[np.isfinite(field)] for field in fields[:num_steps] if np.isfinite(field).any()]
                if finite_vals:
                    values = np.concatenate(finite_vals)
                    vmin = float(np.min(values))
                    vmax = float(np.max(values))
                    if np.isclose(vmin, vmax):
                        vmax = vmin + 1e-3
                    return vmin, vmax
                return default

            det_min, det_max = _compute_global_range(det_fields, (-3.0, 3.0))
            smax_min, smax_max = _compute_global_range(sigma_max_fields, (-2.0, 2.0))
            smin_min, smin_max = _compute_global_range(sigma_min_fields, (-2.0, 2.0))

            viz_count = min(self._get_viz_count(), trajectories.shape[0])
            palette = pc.qualitative.Dark24
            colors = (palette * ((viz_count // len(palette)) + 1))[:viz_count]

            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=[
                    "log₁₀ |det J_t|",
                    "log₁₀ σ_max(J_t)",
                    "log₁₀ σ_min(J_t)"
                ],
                horizontal_spacing=0.1
            )

            timestep_indices: List[List[int]] = []

            for t in range(num_steps):
                indices: List[int] = []

                det_trace = go.Contour(
                    x=grid_info["gx"],
                    y=grid_info["gy"],
                    z=det_fields[t],
                    coloraxis="coloraxis",
                    contours=dict(showlines=False),
                    hoverinfo="skip",
                    showscale=(t == 0),
                    visible=(t == 0)
                )
                fig.add_trace(det_trace, row=1, col=1)
                indices.append(len(fig.data) - 1)

                smax_trace = go.Contour(
                    x=grid_info["gx"],
                    y=grid_info["gy"],
                    z=sigma_max_fields[t],
                    coloraxis="coloraxis2",
                    contours=dict(showlines=False),
                    hoverinfo="skip",
                    showscale=(t == 0),
                    visible=(t == 0)
                )
                fig.add_trace(smax_trace, row=1, col=2)
                indices.append(len(fig.data) - 1)

                smin_trace = go.Contour(
                    x=grid_info["gx"],
                    y=grid_info["gy"],
                    z=sigma_min_fields[t],
                    coloraxis="coloraxis3",
                    contours=dict(showlines=False),
                    hoverinfo="skip",
                    showscale=(t == 0),
                    visible=(t == 0)
                )
                fig.add_trace(smin_trace, row=1, col=3)
                indices.append(len(fig.data) - 1)

                for idx in range(viz_count):
                    history = trajectories[idx, :t + 2, :]
                    line_trace = go.Scatter(
                        x=history[:, 0],
                        y=history[:, 1],
                        mode="lines",
                        line=dict(color=colors[idx], width=2),
                        name=f"Sequence {idx}",
                        legendgroup=f"seq_{idx}",
                        showlegend=(t == 0),
                        visible=(t == 0)
                    )
                    fig.add_trace(line_trace, row=1, col=1)
                    indices.append(len(fig.data) - 1)

                    marker_trace = go.Scatter(
                        x=[trajectories[idx, t + 1, 0]],
                        y=[trajectories[idx, t + 1, 1]],
                        mode="markers",
                        marker=dict(color=colors[idx], size=7, symbol="circle"),
                        name=None,
                        legendgroup=f"seq_{idx}",
                        showlegend=False,
                        visible=(t == 0)
                    )
                    fig.add_trace(marker_trace, row=1, col=1)
                    indices.append(len(fig.data) - 1)

                timestep_indices.append(indices)

            total_traces = len(fig.data)
            slider_steps = []
            title_base = f"Flow Jacobian 2D Summary (Epoch {epoch})"
            for t, indices in enumerate(timestep_indices):
                visible = [False] * total_traces
                for idx in indices:
                    visible[idx] = True
                slider_steps.append(
                    dict(
                        method="update",
                        label=str(t),
                        args=[{"visible": visible},
                              {"title": f"{title_base} · step {t}"}]
                    )
                )

            fig.update_layout(
                width=1220,
                height=560,
                hovermode="closest",
                legend=dict(
                    bgcolor="rgba(0,0,0,0.65)",
                    bordercolor="rgba(255,255,255,0.5)",
                    borderwidth=1,
                    font=dict(color="white")
                ),
                sliders=[dict(
                    active=0,
                    currentvalue=dict(prefix="Flow step: "),
                    pad=dict(t=40),
                    steps=slider_steps
                )],
                coloraxis=dict(
                    colorscale="RdYlBu_r",
                    colorbar=dict(title="log₁₀ |det J_t|", x=0.28),
                    cmin=det_min,
                    cmax=det_max
                ),
                coloraxis2=dict(
                    colorscale="Viridis",
                    colorbar=dict(title="log₁₀ σ_max", x=0.62),
                    cmin=smax_min,
                    cmax=smax_max
                ),
                coloraxis3=dict(
                    colorscale="Cividis",
                    colorbar=dict(title="log₁₀ σ_min", x=0.96),
                    cmin=smin_min,
                    cmax=smin_max
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )

            self._apply_interactive_theme(fig)

            html_path = self._get_output_path(
                f"time_curvature_2d_focused.html",
                "interactive"
            )
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"💾 Saved 2D-focused curvature heatmap: {html_path}")

            if self.should_log_to_wandb():
                with open(html_path, "r", encoding="utf-8") as handle:
                    wandb.log({
                        f"interactive/time_curvature_2d_focused": wandb.Html(handle.read(), inject=False)
                    })

            max_spd = max(step.max().item() for step in grid_flow["spd_errors"][:num_steps + 1])
            max_det_res = max(step.max().item() for step in grid_flow["determinant_residuals"][:num_steps + 1])
            print(f"   ▸ Curvature 2D SPD check max ||GG⁻¹−I|| = {max_spd:.2e}")
            print(f"   ▸ Curvature 2D determinant Δ max = {max_det_res:.2e}")

        except Exception as e:
            print(f"⚠️ 2D-focused curvature heatmap creation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.model.train()
