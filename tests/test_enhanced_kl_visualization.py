import torch
from types import SimpleNamespace

from src.visualizations.basic import BasicVisualizations
from src.rlvae.models.components.riemannian_rhmc_posterior import (
    RiemannianRHMCPosterior,
)


class TinyRhmcModel(torch.nn.Module):
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = torch.device("cpu")
        self.n_flows = 0
        self.training = True
        self.rhmc_alpha = 1.0
        self.posterior_config = {"rhmc_alpha": 1.0, "rhmc_eps_reg": 1e-4}
        self.config = SimpleNamespace(rhmc_alpha=1.0)
        self._current_epoch = 0
        self.centroids_tens = torch.zeros(4, latent_dim)
        # Minimal fields used by RHVAEVolumeElementHMCSampler
        self.M_tens = torch.stack([torch.eye(latent_dim) for _ in range(4)], dim=0)
        self.temperature = torch.tensor(0.1)
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1)))

        self.sampler_manager = SimpleNamespace()
        self.sampler_manager.riemannian_rhmc_posterior = RiemannianRHMCPosterior(
            self, {"rhmc_steps": 0, "rhmc_alpha": 1.0, "rhmc_eps_reg": 1e-4}
        )

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        z = torch.zeros(batch, self.latent_dim)
        recon = x.clone()
        return SimpleNamespace(
            z=z,
            recon_x=recon,
            loss=torch.tensor(0.0),
            reconstruction_loss=torch.tensor(0.0),
            reg_loss=torch.tensor(0.0),
        )

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def G(self, z: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        return eye.unsqueeze(0).expand(z.shape[0], -1, -1)

    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        return self.G(z)


def test_enhanced_kl_visualization_smoke(tmp_path):
    model = TinyRhmcModel(latent_dim=2)
    viz = BasicVisualizations(model=model, device=torch.device("cpu"), config={}, should_log_to_wandb=False)

    def fake_get_output_path(filename, subfolder="visualizations"):
        folder = tmp_path / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder / filename)

    viz._get_output_path = fake_get_output_path

    x_sample = torch.zeros(2, 1, 1, 4, 4)
    viz.create_enhanced_kl_visualization(x_sample, epoch=0)

    saved_path = tmp_path / "plots" / "enhanced_kl_visualization.png"
    assert saved_path.exists()
