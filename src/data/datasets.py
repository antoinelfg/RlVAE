"""
Datasets module for RlVAE
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .ellipse_sequences import EllipseSequenceDataset


class MockDataset(Dataset):
    """Mock dataset for testing purposes."""
    
    def __init__(self, num_samples: int = 1000, data_shape: Tuple[int, ...] = (1, 64, 64)):
        self.num_samples = num_samples
        self.data_shape = data_shape
        self.data = torch.randn(num_samples, *data_shape)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], 0  # Return data and dummy label

def get_dataloader(dataset_name: str, 
                  batch_size: int = 32,
                  shuffle: bool = True,
                  num_workers: int = 4,
                  **kwargs) -> DataLoader:
    """
    Get a dataloader for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        **kwargs: Additional arguments
        
    Returns:
        DataLoader instance
    """
    if dataset_name in ("dsprites", "sprites"):
        dataset = MockDataset(num_samples=1000, data_shape=(1, 64, 64))
    elif dataset_name in ("ellipse_sequences", "ellipses"):
        # Expect shape: (T, 1, H, W). Return as-is; downstream code using sequences
        # should collate appropriately or flatten per needs.
        dataset = EllipseSequenceDataset(
            num_sequences=int(kwargs.pop("num_sequences", 1000)),
            seq_len=int(kwargs.pop("seq_len", 8)),
            image_size=tuple(kwargs.pop("image_size", (64, 64))),
            min_eccentricity=float(kwargs.pop("min_eccentricity", 0.0)),
            max_eccentricity=float(kwargs.pop("max_eccentricity", 0.9)),
            min_radius=int(kwargs.pop("min_radius", 8)),
            max_radius=int(kwargs.pop("max_radius", 20)),
            center_jitter=int(kwargs.pop("center_jitter", 4)),
            antialias=bool(kwargs.pop("antialias", True)),
            seed=kwargs.pop("seed", 42),
        )
    else:
        # Default to mock dataset
        dataset = MockDataset(num_samples=1000, data_shape=(1, 64, 64))
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    ) 