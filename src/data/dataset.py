"""
Custom Dataset for Food Ingredients Classification
Handles loading images and labels from directory structure
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Optional, Callable, List, Tuple, Dict
import pandas as pd


class IngredientsDataset(Dataset):
    """
    Food Ingredients Dataset
    
    Expected directory structure:
        data_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                ...
    
    Args:
        data_dir: Path to directory containing class folders
        transform: Optional transform to apply to images
        class_to_idx: Optional mapping of class names to indices
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find all image files and their classes
        self.samples = []
        self.classes = []
        
        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
        
        # Collect all class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        class_dirs = sorted(class_dirs, key=lambda x: x.name)
        
        # Build class to index mapping
        if class_to_idx is None:
            self.class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())
        
        # Collect all image paths and labels
        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            # Find all images in this class directory
            for img_path in class_dir.iterdir():
                if img_path.suffix in valid_extensions:
                    self.samples.append((img_path, class_idx))
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label at index
        
        Args:
            idx: Index
            
        Returns:
            tuple: (image, label) where image is a tensor and label is an int
        """
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of samples per class
        
        Returns:
            dict: {class_name: count}
        """
        distribution = {cls: 0 for cls in self.classes}
        
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1
        
        return distribution
    
    def get_sample_path(self, idx: int) -> Path:
        """Get the file path for a sample"""
        return self.samples[idx][0]
    
    def get_class_name(self, idx: int) -> str:
        """Get class name for a label index"""
        return self.idx_to_class[idx]


def create_class_mapping(train_dir: str) -> Dict[str, int]:
    """
    Create consistent class to index mapping from training directory
    
    Args:
        train_dir: Path to training directory
        
    Returns:
        dict: {class_name: index}
    """
    train_path = Path(train_dir)
    class_dirs = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    return {cls: idx for idx, cls in enumerate(class_dirs)}


def get_dataset_stats(dataset: IngredientsDataset) -> pd.DataFrame:
    """
    Get statistics about the dataset
    
    Args:
        dataset: IngredientsDataset instance
        
    Returns:
        DataFrame with class statistics
    """
    distribution = dataset.get_class_distribution()
    
    stats = []
    for class_name, count in distribution.items():
        stats.append({
            'class': class_name,
            'count': count,
            'percentage': (count / len(dataset)) * 100
        })
    
    df = pd.DataFrame(stats)
    df = df.sort_values('count', ascending=False).reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    """Test the dataset"""
    from torchvision import transforms
    
    # Example usage
    train_dir = "data/raw/huggingface/Train"
    
    # Simple transform for testing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Create dataset
    dataset = IngredientsDataset(
        data_dir=train_dir,
        transform=transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"\nFirst 10 classes: {dataset.classes[:10]}")
    
    # Test loading an image
    img, label = dataset[0]
    print(f"\nSample image shape: {img.shape}")
    print(f"Sample label: {label} ({dataset.get_class_name(label)})")
    
    # Show distribution
    print("\nClass distribution:")
    stats = get_dataset_stats(dataset)
    print(stats.head(10))
