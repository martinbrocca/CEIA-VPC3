"""
Data Augmentation and Transforms
Defines different augmentation strategies for training and validation
"""

from torchvision import transforms
from typing import Tuple


# ImageNet statistics (standard for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    img_size: int = 224,
    augmentation_level: str = "medium"
) -> transforms.Compose:
    """
    Get training transforms with different augmentation levels
    
    Args:
        img_size: Target image size
        augmentation_level: One of ["none", "light", "medium", "heavy"]
        
    Returns:
        Composed transforms
    """
    
    if augmentation_level == "none":
        # Minimal augmentation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    elif augmentation_level == "light":
        # Light augmentation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    elif augmentation_level == "medium":
        # Medium augmentation (recommended baseline)
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    elif augmentation_level == "heavy":
        # Heavy augmentation
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.15
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation)
    
    Args:
        img_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_test_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get test transforms (same as validation)
    
    Args:
        img_size: Target image size
        
    Returns:
        Composed transforms
    """
    return get_val_transforms(img_size)


def denormalize_image(
    tensor: 'torch.Tensor',
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> 'torch.Tensor':
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized tensor
    """
    import torch
    
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    return tensor * std + mean


if __name__ == "__main__":
    """Test transforms"""
    from PIL import Image
    import numpy as np
    
    # Create dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    
    print("Testing different augmentation levels:")
    print("=" * 50)
    
    for aug_level in ["none", "light", "medium", "heavy"]:
        transform = get_train_transforms(img_size=224, augmentation_level=aug_level)
        transformed = transform(dummy_img)
        print(f"{aug_level:10s}: shape={transformed.shape}, "
              f"min={transformed.min():.3f}, max={transformed.max():.3f}")
    
    print("\nValidation transform:")
    val_transform = get_val_transforms(img_size=224)
    val_transformed = val_transform(dummy_img)
    print(f"shape={val_transformed.shape}, "
          f"min={val_transformed.min():.3f}, max={val_transformed.max():.3f}")
