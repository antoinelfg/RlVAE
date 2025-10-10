import os
from pathlib import Path
import argparse

import torch
from torchvision.utils import save_image, make_grid
import imageio.v2 as imageio

from src.data.ellipse_sequences import EllipseSequenceDataset


def save_sequence_grid(seq: torch.Tensor, out_path: Path, nrow: int = 8) -> None:
    """
    Save a grid showing the T frames of a sequence horizontally.

    seq: (T, 1, H, W)
    """
    # Make a grid across time dimension
    frames = [seq[t] for t in range(seq.shape[0])]
    grid = make_grid(torch.stack(frames, dim=0), nrow=nrow, pad_value=0.0)
    save_image(grid, str(out_path))


def save_sequence_gif(seq: torch.Tensor, out_path: Path, fps: int = 4) -> None:
    """Save an animated GIF of the sequence.

    seq: (T, 1, H, W)
    """
    imgs = []
    T = seq.shape[0]
    for t in range(T):
        frame = seq[t]
        # map to uint8
        img = (frame.clamp(0, 1) * 255).byte().squeeze(0).cpu().numpy()
        imgs.append(img)
    imageio.mimsave(out_path, imgs, duration=1.0 / max(1, fps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/ellipse_preview")
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--image_size", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--seed", type=int, default=42)
    # DoF controls
    parser.add_argument("--fix_center", action="store_true")
    parser.add_argument("--fix_theta", action="store_true")
    parser.add_argument("--fix_intensity", action="store_true")
    parser.add_argument("--keep_major_axis_constant", action="store_true")
    parser.add_argument("--keep_area_constant", action="store_true")
    # Rendering controls
    parser.add_argument("--outline_only", action="store_true")
    parser.add_argument("--outline_width", type=int, default=2)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = EllipseSequenceDataset(
        num_sequences=max(args.num_examples, 8),
        seq_len=args.seq_len,
        image_size=tuple(args.image_size),
        seed=args.seed,
        fix_center=args.fix_center,
        fix_theta=args.fix_theta,
        fix_intensity=args.fix_intensity,
        keep_major_axis_constant=args.keep_major_axis_constant,
        keep_area_constant=args.keep_area_constant,
        outline_only=args.outline_only,
        outline_width=args.outline_width,
    )

    for i in range(args.num_examples):
        seq, _ = ds[i]
        grid_path = outdir / f"ellipse_seq_{i:03d}.png"
        gif_path = outdir / f"ellipse_seq_{i:03d}.gif"
        save_sequence_grid(seq, grid_path, nrow=args.seq_len)
        save_sequence_gif(seq, gif_path, fps=4)
        print(f"Saved {grid_path} and {gif_path}")


if __name__ == "__main__":
    main()


