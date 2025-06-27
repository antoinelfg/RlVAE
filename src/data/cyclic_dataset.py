"""
Cyclic Sprites Data Module
=========================

Lightning data module for cyclic sprites with Hydra configuration support.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from pathlib import Path
from typing import Optional, Tuple
from omegaconf import DictConfig


class CyclicSpritesDataset(Dataset):
    """Cyclic Sprites dataset with enhanced validation."""
    
    def __init__(
        self, 
        data_path: str, 
        subset_size: Optional[int] = None, 
        split: str = 'train',
        verify_cyclicity: bool = True,
        cyclicity_threshold: float = 0.01
    ):
        print(f"Loading cyclic sprites data from {data_path}...")
        
        # Load cyclic data
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        
        print(f"Cyclic sprites data shape: {data.shape}")
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        
        # Data should already be in [N, T, C, H, W] format and normalized
        self.data = data.float()
        
        if subset_size is not None:
            self.data = self.data[:subset_size]
        
        self.split = split
        self.verify_cyclicity = verify_cyclicity
        self.cyclicity_threshold = cyclicity_threshold
        
        print(f"✅ {split} cyclic dataset size: {len(self.data)}")
        print(f"✅ Final data shape: {self.data.shape}")
        print(f"✅ Final data range: [{self.data.min():.3f}, {self.data.max():.3f}]")
        
        # Verify cyclicity if requested
        if verify_cyclicity:
            self._verify_cyclicity()
    
    def _verify_cyclicity(self):
        """Verify that sequences are properly cyclic."""
        print(f"🔍 Verifying cyclicity of first 5 sequences:")
        non_cyclic_count = 0
        
        for i in range(min(5, len(self.data))):
            seq = self.data[i]
            mse = torch.mean((seq[0] - seq[-1]) ** 2).item()
            is_cyclic = mse < self.cyclicity_threshold
            
            if not is_cyclic:
                non_cyclic_count += 1
            
            status = "✅" if is_cyclic else "❌"
            print(f"   Seq {i}: MSE = {mse:.2e} {status}")
        
        if non_cyclic_count > 0:
            print(f"⚠️ Warning: {non_cyclic_count}/5 sequences may not be properly cyclic")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]  # [T, C, H, W]
    
    def get_sequence_info(self, idx: int) -> dict:
        """Get information about a specific sequence."""
        seq = self.data[idx]
        cyclicity_mse = torch.mean((seq[0] - seq[-1]) ** 2).item()
        
        return {
            'index': idx,
            'sequence_length': len(seq),
            'image_shape': seq[0].shape,
            'cyclicity_mse': cyclicity_mse,
            'is_cyclic': cyclicity_mse < self.cyclicity_threshold,
            'data_range': (seq.min().item(), seq.max().item())
        }
    
    def get_dataset_stats(self) -> dict:
        """Get comprehensive dataset statistics."""
        cyclicity_errors = []
        
        for i in range(len(self.data)):
            seq = self.data[i]
            mse = torch.mean((seq[0] - seq[-1]) ** 2).item()
            cyclicity_errors.append(mse)
        
        cyclicity_errors = torch.tensor(cyclicity_errors)
        
        return {
            'num_sequences': len(self.data),
            'sequence_length': self.data.shape[1],
            'image_shape': self.data.shape[2:],
            'data_range': (self.data.min().item(), self.data.max().item()),
            'cyclicity_stats': {
                'mean_error': cyclicity_errors.mean().item(),
                'std_error': cyclicity_errors.std().item(),
                'max_error': cyclicity_errors.max().item(),
                'cyclic_sequences': (cyclicity_errors < self.cyclicity_threshold).sum().item(),
                'cyclicity_rate': (cyclicity_errors < self.cyclicity_threshold).float().mean().item()
            }
        }


class CyclicSpritesDataModule(L.LightningDataModule):
    """Lightning data module for cyclic sprites."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        
        # Store parameters
        self.train_path = config.train_path
        self.test_path = config.test_path
        self.batch_size = None  # Will be set by training config
        self.num_workers = config.get('num_workers', 4)
        self.pin_memory = config.get('pin_memory', True)
        self.persistent_workers = config.get('persistent_workers', True)
        
        # Data validation
        self.verify_cyclicity = config.get('verify_cyclicity', True)
        self.cyclicity_threshold = config.get('cyclicity_threshold', 0.01)
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Data splits (will be set by training config)
        self.n_train_samples = None
        self.n_val_samples = None
        
        print(f"📊 CyclicSpritesDataModule initialized")
        print(f"   Train path: {self.train_path}")
        print(f"   Test path: {self.test_path}")
        print(f"   Verify cyclicity: {self.verify_cyclicity}")
        print(f"   Cyclicity threshold: {self.cyclicity_threshold}")
    
    def setup(self, stage: str = None, training_config: DictConfig = None):
        """Setup data splits."""
        
        # Get training config for data splits
        if training_config:
            # Handle new config structure
            if hasattr(training_config, 'data'):
                self.batch_size = training_config.data.batch_size
                self.num_workers = training_config.data.num_workers
                self.pin_memory = training_config.data.pin_memory
            else:
                # Fallback to old structure
                self.batch_size = training_config.batch_size
                self.num_workers = training_config.get('num_workers', 4)
                self.pin_memory = training_config.get('pin_memory', True)
            
            # Handle data splits
            if hasattr(training_config, 'data_splits'):
                # Calculate splits based on percentages
                total_samples = 1000  # Default, should be configurable
                self.n_train_samples = int(total_samples * training_config.data_splits.train)
                self.n_val_samples = int(total_samples * training_config.data_splits.val)
            else:
                # Fallback to old structure
                self.n_train_samples = training_config.get('n_train_samples', 1000)
                self.n_val_samples = training_config.get('n_val_samples', 600)
        
        if stage == "fit" or stage is None:
            # Load training data
            self.train_dataset = CyclicSpritesDataset(
                data_path=self.train_path,
                subset_size=self.n_train_samples,
                split='train',
                verify_cyclicity=self.verify_cyclicity,
                cyclicity_threshold=self.cyclicity_threshold
            )
            
            # Create validation split from test data
            self.val_dataset = CyclicSpritesDataset(
                data_path=self.test_path,
                subset_size=self.n_val_samples,
                split='val',
                verify_cyclicity=self.verify_cyclicity,
                cyclicity_threshold=self.cyclicity_threshold
            )
        
        if stage == "test" or stage is None:
            # Load test data
            self.test_dataset = CyclicSpritesDataset(
                data_path=self.test_path,
                subset_size=None,  # Use full test set
                split='test',
                verify_cyclicity=self.verify_cyclicity,
                cyclicity_threshold=self.cyclicity_threshold
            )
    
    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Setup must be called before creating dataloaders")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0  # Enable if we have workers
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Setup must be called before creating dataloaders")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0  # Enable if we have workers
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("Setup must be called before creating dataloaders")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0  # Enable if we have workers
        )
    
    def get_sample_batch(self, split: str = 'train', batch_size: int = 8) -> torch.Tensor:
        """Get a sample batch for visualization."""
        if split == 'train' and self.train_dataset:
            dataset = self.train_dataset
        elif split == 'val' and self.val_dataset:
            dataset = self.val_dataset
        elif split == 'test' and self.test_dataset:
            dataset = self.test_dataset
        else:
            raise ValueError(f"Dataset for split '{split}' not available")
        
        # Get random samples
        indices = torch.randperm(len(dataset))[:batch_size]
        return torch.stack([dataset[i] for i in indices])
    
    def get_data_stats(self) -> dict:
        """Get comprehensive statistics about the data."""
        stats = {}
        
        if self.train_dataset:
            stats['train'] = self.train_dataset.get_dataset_stats()
        
        if self.val_dataset:
            stats['val'] = self.val_dataset.get_dataset_stats()
        
        if self.test_dataset:
            stats['test'] = self.test_dataset.get_dataset_stats()
        
        return stats 