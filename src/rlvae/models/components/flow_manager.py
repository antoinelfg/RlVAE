import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from pythae.models.normalizing_flows.iaf import IAF, IAFConfig


class FlowManager(nn.Module):
    """
    FlowManager: Modular Normalizing Flow Handler

    - Builds a stack of IAF flows.
    - Applies flows sequentially to produce a latent sequence z_0 -> z_1 -> ... -> z_T.
    - Returns per-sample log|det J| for each flow (shape [B]), with *correct direction*.
    - Avoids hard clamping of log-dets; only NaN-guards and optional output clipping.
    - Optional tiny autograd verifier (2D only) to detect whether the underlying layer
      reports forward or inverse log-dets.

    Args:
        latent_dim:      Dimensionality of the latent space.
        n_flows:         Number of IAF flows (for a sequence of length T, you typically use T-1).
        flow_hidden_size: Hidden width used by IAF.
        flow_n_blocks:   Number of blocks per IAF.
        flow_n_hidden:   Number of hidden layers per block.
        device:          Torch device.
        output_clip:     Optional hard clamp on flow outputs (z_t) for safety; None disables.
        logdet_clip:     (Deprecated) Hard clamp for log-dets; kept for compatibility but unused.
        logdet_direction: "forward" if flow returns log|det ∂z'/∂z|,
                          "inverse" if flow returns log|det ∂z/∂z'| (common for IAF); we flip sign.
        enable_logdet_verify: If True and latent_dim==2, runs a tiny autograd check at first step.
    """

    def __init__(
        self,
        latent_dim: int,
        n_flows: int = 8,
        flow_hidden_size: int = 256,
        flow_n_blocks: int = 2,
        flow_n_hidden: int = 1,
        device: Optional[torch.device] = None,
        # ↓↓↓ REMOVE hard clamps; keep for outputs only if you must
        output_clip: Optional[float] = None,   # was 50.0
        logdet_clip: Optional[float] = None,   # was 20.0  (DISABLE)
        # NEW: tell us what the flow returns for logdet
        # 'forward'  -> log|det ∂z'/∂z|
        # 'inverse'  -> log|det ∂z/∂z'| (common for IAF); we flip the sign here
        logdet_direction: str = "inverse",
        enable_logdet_verify: bool = False,    # tiny autograd check on 2D
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_flows = n_flows
        self.flow_hidden_size = flow_hidden_size
        self.flow_n_blocks = flow_n_blocks
        self.flow_n_hidden = flow_n_hidden
        self.device = device or torch.device('cpu')

        # Build flow stack
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            config = IAFConfig(
                input_dim=(latent_dim,),
                hidden_size=flow_hidden_size,
                n_blocks=flow_n_blocks,
                n_hidden=flow_n_hidden,
            )
            self.flows.append(IAF(config))
        self.to(self.device)

        # Store settings
        self.flow_output_clip = float(output_clip) if output_clip else None
        self.flow_logdet_clip = None  # (intentionally disabled)
        if logdet_direction not in {"forward", "inverse"}:
            raise ValueError("logdet_direction must be 'forward' or 'inverse'")
        self.logdet_direction = logdet_direction
        self.enable_logdet_verify = bool(enable_logdet_verify)

        self._debug_flow_counter = 0

    @torch.no_grad()
    def _verify_logdet_sign(self, flow: nn.Module, z: torch.Tensor) -> Optional[str]:
        """
        Quick sanity check (2D only, tiny batch):
        Computes the forward Jacobian via autograd and compares with the flow's reported logdet.
        Returns "forward" or "inverse" if we can infer the direction, else None.
        """
        if z.ndim != 2 or z.shape[1] != 2:
            return None
        z_small = z[: min(4, z.shape[0])].detach().clone().requires_grad_(True)
        out_struct = flow(z_small)
        z_out = out_struct.out  # [B, 2]

        # Build full Jacobian per sample by reverse-mode
        J = torch.zeros(z_small.shape[0], 2, 2, device=z_small.device, dtype=z_small.dtype)
        for k in range(2):
            grads = torch.autograd.grad(
                z_out[:, k].sum(),
                z_small,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            J[:, k, :] = grads

        sign, logabsdet_forward = torch.slogdet(J.float())  # [B]
        reported = out_struct.log_abs_det_jac
        if not isinstance(reported, torch.Tensor):
            return None
        if reported.dim() == 0:
            # scalar is not usable for per-sample comparison
            return None
        reported = reported[: logabsdet_forward.shape[0]].detach().to(logabsdet_forward.dtype)

        diff_forward = (reported - logabsdet_forward).abs().mean().item()
        diff_inverse = (reported + logabsdet_forward).abs().mean().item()
        return "inverse" if diff_inverse < diff_forward else "forward"

    def apply_flows(self, z_sequence, n_obs: Optional[int] = None):
        """
        Apply the flow stack to produce a latent sequence and per-step log|det J|.

        Args:
            z_sequence: torch.Tensor [B, D] for z_0, or list whose first item is z_0.
            n_obs:      Total number of timesteps in the sequence (including z_0). If None,
                        use len(z_sequence) if list, else assumes at least 2.
        Returns:
            (z_sequence, log_det_jacobians)
              - z_sequence: list of length n_obs with tensors [B, D]
              - log_det_jacobians: list of length n_obs-1 with tensors [B]
        """
        if isinstance(z_sequence, list):
            if n_obs is None:
                n_obs = len(z_sequence)
        else:
            # z_sequence is z_0 tensor
            if n_obs is None:
                n_obs = 2  # minimal 2: z_0 -> z_1
            z_sequence = [z_sequence] + [None] * (n_obs - 1)

        # Ensure list has correct length
        while len(z_sequence) < n_obs:
            z_sequence.append(None)

        z0 = z_sequence[0]
        if not isinstance(z0, torch.Tensor):
            raise ValueError("z_sequence[0] must be a tensor (z_0).")
        batch_size = z0.shape[0]
        log_det_jacobians: List[torch.Tensor] = []

        for t in range(1, n_obs):
            flow_idx = (t - 1) % len(self.flows)
            flow = self.flows[flow_idx]

            z_prev = z_sequence[t - 1]
            out_struct = flow(z_prev)  # expects [B, D]
            z_t = out_struct.out

            # Guard NaNs/Infs in outputs; optional hard clamp if configured
            z_t = torch.nan_to_num(z_t, nan=0.0, posinf=0.0, neginf=0.0)
            if self.flow_output_clip is not None and self.flow_output_clip > 0:
                z_t = torch.clamp(z_t, -self.flow_output_clip, self.flow_output_clip)

            # Log-det (per-sample). No hard clamp; just NaN-guard and direction fix.
            log_det = out_struct.log_abs_det_jac
            if not isinstance(log_det, torch.Tensor):
                raise RuntimeError("Flow returned non-tensor log_abs_det_jac.")

            log_det = torch.nan_to_num(log_det, nan=0.0, posinf=0.0, neginf=0.0)

            # Expect shape [B]; try to fix if possible
            if log_det.dim() == 0:
                print(f"[FLOW WARN] scalar logdet at step {t}; replacing with zeros of shape [B].")
                log_det = torch.zeros(batch_size, device=z_t.device, dtype=z_t.dtype)
            elif log_det.shape[0] != batch_size:
                print(f"[FLOW WARN] logdet shape {tuple(log_det.shape)} != (B,)={batch_size}; attempting reshape.")
                if log_det.numel() == batch_size:
                    log_det = log_det.reshape(batch_size)
                else:
                    log_det = torch.zeros(batch_size, device=z_t.device, dtype=z_t.dtype)

            # Convert inverse -> forward if needed
            if self.logdet_direction == "inverse":
                log_det = -log_det

            # Save
            z_sequence[t] = z_t
            log_det_jacobians.append(log_det)

            # Optional one-time verifier (2D, first step)
            if (
                self.enable_logdet_verify
                and t == 1
                and self.latent_dim == 2
                and z_prev.requires_grad is not False  # need grads for autograd Jacobian
            ):
                inferred = self._verify_logdet_sign(flow, z_prev)
                if inferred and inferred != self.logdet_direction:
                    print(f"[FLOW VERIFY] Detected '{inferred}' logdet; switching manager to '{inferred}'.")
                    self.logdet_direction = inferred  # adjust for subsequent steps

            # Periodic debug
            self._debug_flow_counter += 1
            if self._debug_flow_counter % 100 == 0:
                with torch.no_grad():
                    print(f"🔍 FLOW DEBUG (step {t}):")
                    print(f"   Input z range: [{z_prev.min():.3f}, {z_prev.max():.3f}]")
                    print(f"   Output z range: [{z_t.min():.3f}, {z_t.max():.3f}]")
                    print(f"   Log det Jacobian (mean): {log_det.mean():.3f}")
                    print(f"   Jacobian range: [{log_det.min():.3f}, {log_det.max():.3f}]")

        return z_sequence, log_det_jacobians

    def invert_flows(self, z_seq: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        IAF is not easily invertible in closed form; placeholder for future invertible flows.
        """
        raise NotImplementedError("Invert flows is not implemented for IAF.")

    def get_log_det_jacobians(self, z_seq: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Re-compute per-step log|det J| for a given latent sequence (non-destructive).
        """
        # copy to avoid mutating caller's list
        z_copy = [zi.clone() if isinstance(zi, torch.Tensor) else zi for zi in z_seq]
        _, log_det_jacobians = self.apply_flows(z_copy, n_obs=len(z_copy))
        return log_det_jacobians

    def get_flow_params(self) -> Dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "n_flows": self.n_flows,
            "flow_hidden_size": self.flow_hidden_size,
            "flow_n_blocks": self.flow_n_blocks,
            "flow_n_hidden": self.flow_n_hidden,
        }

    def diagnose_flows(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "total_params": total_params,
            "n_flows": self.n_flows,
        }