"""
Datasets module for RlVAE
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

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
    if dataset_name == "dsprites" or dataset_name == "sprites":
        # For now, return a mock dataset
        # In a real implementation, this would load the actual dSprites dataset
        dataset = MockDataset(num_samples=1000, data_shape=(1, 64, 64))
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