import torch
from omegaconf import OmegaConf
from src.models.modular_rlvae import ModularRiemannianFlowVAE
from src.generation.generator import create_generator
from src.inference.inference_pipeline import create_inference_pipeline

# Complete dummy model config (adjust as needed for your model)
model_config = OmegaConf.create({
    "input_dim": [3, 64, 64],
    "latent_dim": 8,
    "n_flows": 2,
    "flow_hidden_size": 32,
    "flow_n_blocks": 2,
    "flow_n_hidden": 1,
    "epsilon": 1e-6,
    "beta": 1.0,
    "riemannian_beta": 8.0,
    "encoder": {"architecture": "mlp"},
    "decoder": {"architecture": "mlp"},
    "posterior": {"type": "riemannian_metric"},
    "sampling": {"method": "geodesic", "use_riemannian": True},
    "loop": {"mode": "open", "penalty": 5.0},
    "metric": {"temperature_override": 3.0},
    "pretrained": {
        "encoder_path": None,
        "decoder_path": None,
        "metric_path": None,
    },
})
model = ModularRiemannianFlowVAE(model_config)

# Load Hydra config for generation
gen_cfg = OmegaConf.load("conf/generation/default.yaml")
generator = create_generator(model, config=gen_cfg)

# Test generation
result = generator.generate_from_prior(gen_cfg)
print("Generated images shape:", result['images'].shape)

# Load Hydra config for inference
inf_cfg = OmegaConf.load("conf/inference/default.yaml")
inference_pipeline = create_inference_pipeline(model, config=inf_cfg)

# Dummy images for inference (replace with real data for a true test)
dummy_images = torch.rand(4, 3, 64, 64)
encoding = inference_pipeline.encode_images(dummy_images, config=inf_cfg)
print("Latents shape:", encoding['latents'].shape) 