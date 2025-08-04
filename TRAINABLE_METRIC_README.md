# Trainable Metric Tensor for RlVAE: Recap & Usage Guide

## Overview
This document summarizes the new modular, trainable metric tensor feature for the RlVAE framework. It covers the motivation, architecture options, initialization from a fixed metric, optimizer and regularization updates, and how to configure and run experiments with these features.

---

## 1. Motivation
- **Why?**
  - The original two-stage RlVAE pipeline used a fixed metric tensor learned from a vanilla VAE (Stage 1).
  - After Stage 2 training, the latent space evolves, making the fixed metric less accurate for the new data distribution.
  - A **trainable metric tensor** allows the geometry to adapt during RlVAE training, improving the accuracy of Riemannian priors, KL, and geometric computations.

---

## 2. Modular Trainable Metric Tensor
- **Architectures Supported:**
  - `mlp` (Multi-Layer Perceptron)
  - `resnet` (ResNet-style MLP)
  - `transformer` (Transformer-based network)
- **Configurable via YAML:**
  - Select architecture and hyperparameters in your config file.

**Example:**
```yaml
metric:
  trainable: true
  architecture: "mlp"  # Options: "mlp", "resnet", "transformer"
  arch_kwargs: {}
```

---

## 3. Initialization from Fixed Metric (Vanilla VAE)
- **Why?**
  - For stability and leveraging prior knowledge, you can initialize the trainable metric to match the fixed metric from Stage 1.
- **How?**
  - Set `init_from_fixed: true` and provide the path to the fixed metric file.
  - The metric network is pretrained (using MSE loss) to match the fixed metric before RlVAE training begins.

**Example:**
```yaml
metric:
  trainable: true
  architecture: "mlp"
  arch_kwargs: {}
  init_from_fixed: true
  fixed_metric_path: "data/pretrained/vanilla_metric.pt"
```

---

## 4. Optimizer Integration
- The optimizer **automatically includes the metric network’s parameters** if the metric is trainable.
- No manual changes needed—handled in the Lightning trainer.

---

## 5. Metric Regularization Options
- **Why?**
  - To ensure the metric remains well-behaved (e.g., not degenerate, ill-conditioned, or overly rough).
- **Types Supported:**
  - `determinant`: Encourage the metric determinant to be near a target value (e.g., 0 for unit volume).
  - `condition`: Encourage the metric to be well-conditioned (target condition number).
  - `smoothness`: Penalize large changes in the metric for nearby points.
- **Configurable via YAML:**

**Example:**
```yaml
metric:
  metric_reg_weight: 1.0
  metric_reg_type: 'determinant'  # or 'condition', 'smoothness'
  metric_reg_target: 0.0          # e.g., 0 for logdet, 10 for condition number
```

---

## 6. Example Full Config Snippet
```yaml
metric:
  trainable: true
  architecture: "resnet"
  arch_kwargs:
    hidden_dim: 128
    n_blocks: 4
  init_from_fixed: true
  fixed_metric_path: "data/pretrained/vanilla_metric.pt"
  metric_reg_weight: 1.0
  metric_reg_type: 'condition'
  metric_reg_target: 10.0
```

---

## 7. Commands to Run Training

**Standard Lightning Training:**
```bash
python run_experiment.py model=mlp_rlvae training=default
```

**With custom config overrides:**
```bash
python run_experiment.py model=mlp_rlvae metric.trainable=true metric.architecture=resnet metric.init_from_fixed=true metric.fixed_metric_path=data/pretrained/vanilla_metric.pt metric.metric_reg_weight=1.0 metric.metric_reg_type=condition metric.metric_reg_target=10.0
```

**For SLURM cluster runs:**
```bash
sbatch scripts/slurm/run_experiment_rlvae.sbatch model=mlp_rlvae metric.trainable=true ...
```

---

## 8. Tips & Best Practices
- Start with `init_from_fixed: true` for stability, then experiment with regularization.
- Monitor metric determinant, condition number, and smoothness during training.
- Use the modular config to easily switch architectures and regularization strategies.

---

## 9. Extending Further
- You can add new architectures by extending the metric network factory in `metric_tensor.py`.
- Regularization logic is modular—add new types as needed in `LossManager`.

---

**This modular, trainable metric system gives you full flexibility and control over the geometry of your RlVAE latent space!** 

---

## 10. Standard Vanilla VAE Checkpoints for RlVAE Experiments

For all RlVAE experiments with an evolving (trainable) metric, use the following fixed vanilla VAE checkpoint files:

- **VAE:**      `data/pretrained/vae_diverse_mlp_ld16_20250717_134643.pt`
- **Encoder:**  `data/pretrained/encoder_diverse_mlp_ld16_20250717_134643.pt`
- **Decoder:**  `data/pretrained/decoder_diverse_mlp_ld16_20250717_134643.pt`
- **Metric:**   `data/pretrained/metric_diverse_mlp_ld16_20250717_134647.pt`

**Example config snippet:**
```yaml
pretrained:
  encoder_path: "data/pretrained/encoder_diverse_mlp_ld16_20250717_134643.pt"
  decoder_path: "data/pretrained/decoder_diverse_mlp_ld16_20250717_134643.pt"
  metric_path: "data/pretrained/metric_diverse_mlp_ld16_20250717_134647.pt"

metric:
  trainable: true
  architecture: "resnet"
  arch_kwargs:
    hidden_dim: 128
    n_blocks: 4
  init_from_fixed: true
  fixed_metric_path: "data/pretrained/metric_diverse_mlp_ld16_20250717_134647.pt"
  metric_reg_weight: 1.0
  metric_reg_type: 'condition'
  metric_reg_target: 10.0
```

**Full command to launch RlVAE with evolving metric:**
```bash
python run_experiment.py \
  model=mlp_rlvae \
  pretrained.encoder_path=data/pretrained/encoder_diverse_mlp_ld16_20250717_134643.pt \
  pretrained.decoder_path=data/pretrained/decoder_diverse_mlp_ld16_20250717_134643.pt \
  pretrained.metric_path=data/pretrained/metric_diverse_mlp_ld16_20250717_134647.pt \
  metric.trainable=true \
  metric.architecture=resnet \
  metric.arch_kwargs.hidden_dim=128 \
  metric.arch_kwargs.n_blocks=4 \
  metric.init_from_fixed=true \
  metric.fixed_metric_path=data/pretrained/metric_diverse_mlp_ld16_20250717_134647.pt \
  metric.metric_reg_weight=1.0 \
  metric.metric_reg_type=condition \
  metric.metric_reg_target=10.0 \
  training.optimizer.lr=1e-3 \
  training.optimizer.weight_decay=1e-5 \
  training.epochs=100
```

--- 