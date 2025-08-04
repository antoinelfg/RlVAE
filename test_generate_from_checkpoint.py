import argparse
import torch
import numpy as np
from pathlib import Path
from torchvision.utils import save_image, make_grid
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

from src.models.modular_rlvae import ModularRiemannianFlowVAE
from src.generation.generator import create_generator, GenerationConfig
from src.inference.inference_pipeline import create_inference_pipeline, InferenceConfig


def print_tensor_stats(name, tensor):
    print(f"{name}: shape={tuple(tensor.shape)}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")

def save_qualitative_inpainting_grid(imgs_orig, imgs_masked, imgs_inpaint, missing_idx, save_path, method, sample_idx):
    """
    Create a qualitative grid with 4 rows: original, masked, inpainted, abs diff.
    Highlight the inpainted/masked frame in red, and the given frames in green (except original row).
    Borders are drawn using FancyBboxPatch for robust visibility.
    """
    imgs_orig = imgs_orig.cpu().clamp(0, 1)
    imgs_masked = imgs_masked.cpu().clamp(0, 1)
    imgs_inpaint = imgs_inpaint.cpu().clamp(0, 1)
    diff = (imgs_inpaint - imgs_orig).abs()
    T = imgs_orig.shape[0]
    fig, axes = plt.subplots(4, T, figsize=(T * 2, 8))
    row_titles = ['Original', 'Masked', 'Inpainted', 'Abs Diff']
    for row, imgs in enumerate([imgs_orig, imgs_masked, imgs_inpaint, diff]):
        for t in range(T):
            ax = axes[row, t]
            img = imgs[t].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
            if t == 0:
                ax.set_ylabel(row_titles[row], fontsize=12)
            if row in [1, 2, 3]:
                border_offset = 0.5
                border_width = img.shape[1] - 1
                border_height = img.shape[0] - 1
                color = 'red' if t == missing_idx else 'green'
                # Draw a thick white background box for contrast
                ax.add_patch(
                    FancyBboxPatch(
                        (border_offset, border_offset), border_width, border_height,
                        boxstyle="square,pad=0",
                        linewidth=12, edgecolor='white', facecolor='none', zorder=20
                    )
                )
                # Draw the colored border on top
                ax.add_patch(
                    FancyBboxPatch(
                        (border_offset, border_offset), border_width, border_height,
                        boxstyle="square,pad=0",
                        linewidth=8, edgecolor=color, facecolor='none', zorder=21
                    )
                )
            ax.set_xlim([0, img.shape[1]])
            ax.set_ylim([img.shape[0], 0])
            ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Test generation and inference from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt or .pt)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to generate")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Extract config
    config = checkpoint.get('config', checkpoint.get('hyper_parameters', {}).get('model', None))
    if config is None:
        raise ValueError("No config found in checkpoint!")

    # Load model
    model = ModularRiemannianFlowVAE(config)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    # Remove 'model.' prefix if present
    if any(k.startswith('model.') for k in state_dict):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("[INFO] Model loaded and set to eval mode.")

    # Print centroids/metric stats if available
    if hasattr(model, 'centroids_tens'):
        print_tensor_stats("centroids_tens", model.centroids_tens)
    if hasattr(model, 'M_tens'):
        print_tensor_stats("M_tens", model.M_tens)
    if hasattr(model, 'modular_metric') and hasattr(model.modular_metric, 'centroids'):
        print_tensor_stats("modular_metric.centroids", model.modular_metric.centroids)

    # Create generator
    generator = create_generator(model, device=device)
    print("[INFO] Generator created.")
    print(f"[INFO] Available samplers: {list(generator.samplers.keys())}")
    # Only use geodesic method
    methods = ["geodesic"]
    sequence_length = 8  # Use sequence length 8 as requested
    os.makedirs("test_outputs", exist_ok=True)
    # Check that generator uses model's flow_manager and modular_metric
    if not hasattr(generator.model, 'flow_manager') or generator.model.flow_manager is None:
        print("[WARNING] Model does not have a flow_manager! Sequence generation may not use correct flows.")
    else:
        print("[INFO] Using model's flow_manager for sequence generation.")
    if not hasattr(generator.model, 'modular_metric') or generator.model.modular_metric is None:
        print("[WARNING] Model does not have a modular_metric! Metric-aware sampling may not use correct metric.")
    else:
        print("[INFO] Using model's modular_metric for metric-aware sampling.")
    # Note: If you want to use per-timestep metrics, extend the generator/model to support this explicitly.
    missing_idx = 3  # Position to inpaint (can be changed)
    for method in methods:
        try:
            print(f"\n[TEST] Generating samples with method: {method}")
            config = GenerationConfig(num_samples=args.num_samples, sampling_method=method, sampler_type="working", batch_size=args.num_samples, sequence_length=sequence_length)
            result = generator.generate_from_prior(config)
            latents = result['latents']
            images = result['images']  # [B, T, C, H, W]
            print_tensor_stats(f"{method} latents", latents)
            print_tensor_stats(f"{method} images", images)
            # Per-frame stats
            for t in range(images.shape[1]):
                frame = images[:, t]
                print_tensor_stats(f"{method} images (t={t})", frame)
            # Geodesic inpainting/interpolation for each sample
            for i in range(images.shape[0]):
                z_seq = latents[i]  # [T, latent_dim]
                # Get neighbors
                if missing_idx == 0 or missing_idx == sequence_length - 1:
                    print(f"[WARNING] Cannot inpaint at edge position {missing_idx}.")
                    continue
                z_before = z_seq[missing_idx - 1]
                z_after = z_seq[missing_idx + 1]
                # Geodesic interpolation between neighbors
                interp_result = generator.interpolate(z_before, z_after, num_steps=3, method='geodesic')
                z_interp = interp_result['latents'][1]  # Middle step
                print(f"[DEBUG] z_interp shape: {z_interp.shape}")
                z_inpaint = z_seq.clone()
                z_inpaint[missing_idx] = z_interp
                print(f"[DEBUG] z_inpaint shape before decode: {z_inpaint.shape}")
                # Always decode inpainted sequence per-frame to avoid shape errors
                imgs_inpaint_list = []
                for t in range(z_inpaint.shape[0]):
                    z_t = z_inpaint[t].unsqueeze(0)  # [1, latent_dim]
                    img_t = generator.model.decode(z_t)
                    if hasattr(img_t, 'reconstruction'):
                        img_t = img_t.reconstruction
                    elif isinstance(img_t, dict) and 'reconstruction' in img_t:
                        img_t = img_t['reconstruction']
                    if img_t.dim() == 4 and img_t.shape[0] == 1:
                        img_t = img_t[0]  # [C, H, W]
                    imgs_inpaint_list.append(img_t)
                imgs_inpaint = torch.stack(imgs_inpaint_list, dim=0)  # [T, C, H, W]
                print(f"[DEBUG] imgs_inpaint shape after decode: {imgs_inpaint.shape}")

                # Masked sequence (blank at missing_idx)
                imgs_orig = images[i]  # [T, C, H, W]
                imgs_masked = imgs_orig.clone()
                imgs_masked[missing_idx] = 0.0
                # Debug prints and assertions for grid shapes
                print(f"[DEBUG] imgs_orig shape for grid: {imgs_orig.shape}, device: {imgs_orig.device}")
                print(f"[DEBUG] imgs_masked shape for grid: {imgs_masked.shape}, device: {imgs_masked.device}")
                print(f"[DEBUG] imgs_inpaint shape for grid: {imgs_inpaint.shape}, device: {imgs_inpaint.device}")

                try:
                    imgs_orig_cpu = imgs_orig.cpu()
                    imgs_masked_cpu = imgs_masked.cpu()
                    imgs_inpaint_cpu = imgs_inpaint.cpu()
                    grid = torch.cat([
                        make_grid(imgs_orig_cpu, nrow=sequence_length, pad_value=1.0),
                        make_grid(imgs_masked_cpu, nrow=sequence_length, pad_value=1.0),
                        make_grid(imgs_inpaint_cpu, nrow=sequence_length, pad_value=1.0)
                    ], dim=1)  # Stack vertically
                    print(f"[DEBUG] grid shape before save_image: {grid.shape}")
                    save_path = f"test_outputs/{method}_sample{i}_inpainting.png"
                    save_image(grid, save_path)
                    print(f"[INFO] Saved inpainting grid for {method} sample {i} to {save_path}")

                    # Save qualitative matplotlib grid
                    qual_save_path = f"test_outputs/{method}_sample{i}_inpainting_qualitative.png"
                    save_qualitative_inpainting_grid(imgs_orig, imgs_masked, imgs_inpaint, missing_idx, qual_save_path, method, i)
                    print(f"[INFO] Saved qualitative inpainting grid for {method} sample {i} to {qual_save_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to create/save inpainting grid for {method} sample {i}: {e}")
                    print(f"imgs_orig shape: {imgs_orig.shape}, imgs_masked shape: {imgs_masked.shape}, imgs_inpaint shape: {imgs_inpaint.shape}")

            # Check for NaNs in latents
            if torch.isnan(latents).any():
                print(f"[WARNING] NaNs detected in {method} latents!")
                nan_indices = torch.isnan(latents).nonzero()
                print(f"First 5 NaN indices: {nan_indices[:5]}")
                print(f"First 5 latent values with NaN: {latents.flatten()[torch.isnan(latents.flatten())][:5]}")
            # Save summary grid for first frame
            if not torch.isnan(images).any():
                if images.dim() == 5:
                    images_to_save = images[:, 0]  # [B, C, H, W]
                elif images.dim() == 4:
                    images_to_save = images
                else:
                    print(f"[ERROR] Unexpected image shape: {images.shape}")
                    continue
                images_to_save = torch.clamp(images_to_save, 0, 1)
                save_path = f"test_outputs/generated_{method}.png"
                save_image(images_to_save[:16], save_path, nrow=4)
                print(f"[INFO] Saved image grid for {method} (first frame) to {save_path}")
            else:
                print(f"[WARNING] Not saving images for {method} due to NaNs.")
            # Save per-sample sequence grids
            if not torch.isnan(images).any():
                for i in range(images.shape[0]):
                    sample_seq = images[i]  # [T, C, H, W]
                    sample_seq = torch.clamp(sample_seq, 0, 1)
                    save_path = f"test_outputs/{method}_sample{i}_sequence.png"
                    save_image(sample_seq, save_path, nrow=sequence_length)
                print(f"[INFO] Saved per-sample sequence grids for {method}.")
                # Save per-timestep grids
                for t in range(images.shape[1]):
                    timestep_grid = images[:, t]  # [B, C, H, W]
                    timestep_grid = torch.clamp(timestep_grid, 0, 1)
                    save_path = f"test_outputs/{method}_t{t}_grid.png"
                    save_image(timestep_grid, save_path, nrow=4)
                print(f"[INFO] Saved per-timestep grids for {method}.")
        except Exception as e:
            print(f"[ERROR] Generation failed for method {method}: {e}")

    # Create inference pipeline
    inference_pipeline = create_inference_pipeline(model, device=device)
    print("\n[INFO] Inference pipeline created.")
    # Test inference on random images
    dummy_images = torch.randn(args.num_samples, *model.input_dim, device=device)
    inf_config = InferenceConfig(batch_size=args.num_samples, use_mean=True)
    try:
        print("[TEST] Running inference (encode_images) on random images...")
        encoding_result = inference_pipeline.encode_images(dummy_images, inf_config)
        latents = encoding_result['latents']
        print_tensor_stats("inference latents", latents)
        if 'reconstructions' in encoding_result:
            print_tensor_stats("reconstructions", encoding_result['reconstructions'])
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")

    print("\n[INFO] Test complete.")

if __name__ == "__main__":
    main() 