#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--download_dir", type=str, default="data/raw", help="Torchvision download root")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError("torchvision is required to prepare MNIST. Please install torchvision.") from e

    transform = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.MNIST(root=args.download_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=args.download_dir, train=False, download=True, transform=transform)

    def stack(ds):
        xs = torch.stack([ds[i][0] for i in range(len(ds))], dim=0).contiguous()  # [N,1,28,28]
        return xs.float().clamp(0.0, 1.0)

    train_x = stack(train_ds)
    test_x = stack(test_ds)

    train_path = out_dir / "MNIST_train.pt"
    test_path = out_dir / "MNIST_test.pt"
    torch.save(train_x, train_path)
    torch.save(test_x, test_path)

    print(f"Saved: {train_path} {tuple(train_x.shape)}")
    print(f"Saved: {test_path} {tuple(test_x.shape)}")

if __name__ == "__main__":
    main()












