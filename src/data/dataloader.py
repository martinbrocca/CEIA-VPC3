"""
DataLoader utilities
Creates train/val/test dataloaders with appropriate settings
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np

from .dataset import IngredientsDataset, create_class_mapping
from .transforms import get_train_transforms, get_val_transforms, get_test_transforms


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    augmentation_level: str = "medium",
    use_weighted_sampler: bool = False,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and optionally test dataloaders
    
    Args:
        data_dir: Path to data directory (should contain Train/, val/, test/)
        batch_size: Batch size
        num_workers: Number of workers for data loading
        img_size: Image size
        augmentation_level: Augmentation level for training
        use_weighted_sampler: Whether to use weighted sampling for imbalanced classes
        pin_memory: Whether to pin memory (faster for GPU)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
               test_loader is None if test/ doesn't exist
    """
    
    data_path = Path(data_dir)
    
    # Paths to splits
    train_dir = data_path / "Train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"
    
    # Create consistent class mapping from training data
    class_to_idx = create_class_mapping(str(train_dir))
    
    # Get transforms
    train_transform = get_train_transforms(img_size, augmentation_level)
    val_transform = get_val_transforms(img_size)
    test_transform = get_test_transforms(img_size)
    
    # Create datasets
    train_dataset = IngredientsDataset(
        data_dir=str(train_dir),
        transform=train_transform,
        class_to_idx=class_to_idx
    )
    
    val_dataset = IngredientsDataset(
        data_dir=str(val_dir),
        transform=val_transform,
        class_to_idx=class_to_idx
    )
    
    # Test dataset (optional)
    test_dataset = None
    if test_dir.exists():
        test_dataset = IngredientsDataset(
            data_dir=str(test_dir),
            transform=test_transform,
            class_to_idx=class_to_idx
        )
    
    # Create sampler for training if needed
    train_sampler = None
    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(train_dataset)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader, test_loader


def create_weighted_sampler(dataset: IngredientsDataset) -> WeightedRandomSampler:
    """
    Create weighted random sampler for imbalanced datasets
    
    Args:
        dataset: IngredientsDataset instance
        
    Returns:
        WeightedRandomSampler
    """
    # Get class distribution
    class_counts = np.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    # Calculate weights (inverse frequency)
    class_weights = 1.0 / class_counts
    
    # Assign weight to each sample based on its class
    sample_weights = np.array([class_weights[label] for _, label in dataset.samples])
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def get_class_weights(dataset: IngredientsDataset) -> torch.Tensor:
    """
    Calculate class weights for weighted loss function
    
    Args:
        dataset: IngredientsDataset instance
        
    Returns:
        Tensor of class weights
    """
    # Get class distribution
    class_counts = np.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    # Calculate weights (inverse frequency)
    total_samples = len(dataset)
    class_weights = total_samples / (len(dataset.classes) * class_counts)
    
    return torch.FloatTensor(class_weights)


def get_dataloader_info(loader: DataLoader) -> Dict:
    """
    Get information about a dataloader
    
    Args:
        loader: DataLoader instance
        
    Returns:
        dict with dataloader info
    """
    dataset = loader.dataset
    
    return {
        'num_samples': len(dataset),
        'num_classes': len(dataset.classes),
        'batch_size': loader.batch_size,
        'num_batches': len(loader),
        'num_workers': loader.num_workers,
        'classes': dataset.classes
    }


if __name__ == "__main__":
    """Test dataloaders"""
    
    # Example usage
    data_dir = "data/raw/huggingface"
    
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4,
        img_size=224,
        augmentation_level="medium",
        use_weighted_sampler=False
    )
    
    print("\nDataloader Information:")
    print("=" * 60)
    
    for name, loader in [("Train", train_loader), ("Val", val_loader)]:
        if loader is None:
            continue
        info = get_dataloader_info(loader)
        print(f"\n{name}:")
        print(f"  Samples: {info['num_samples']}")
        print(f"  Classes: {info['num_classes']}")
        print(f"  Batches: {info['num_batches']}")
        print(f"  Batch size: {info['batch_size']}")
    
    # Test loading a batch
    print("\nTesting batch loading...")
    images, labels = next(iter(train_loader))
    print(f"  Image batch shape: {images.shape}")
    print(f"  Label batch shape: {labels.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
