"""
Model Factory
Creates different Vision Transformer architectures
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, List


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.1,
    drop_path_rate: float = 0.0,
    freeze_backbone: bool = False,
    freeze_layers: Optional[List[str]] = None
) -> nn.Module:
    """
    Create a Vision Transformer model
    
    Args:
        model_name: Name of the model architecture
                   Options: 'deit_tiny', 'deit_small', 'deit_base',
                           'mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        drop_path_rate: DropPath rate (stochastic depth)
        freeze_backbone: Whether to freeze the entire backbone
        freeze_layers: List of specific layer names to freeze
        
    Returns:
        PyTorch model
    """
    
    # Map model names to timm model names
    model_map = {
        'deit_tiny': 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base': 'deit_base_patch16_224',
        'mobilevit_s': 'mobilevit_s',
        'mobilevit_xs': 'mobilevit_xs',
        'mobilevit_xxs': 'mobilevit_xxs'
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available models: {list(model_map.keys())}")
    
    timm_model_name = model_map[model_name]
    
    # Create model
    model = timm.create_model(
        timm_model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
        drop_path_rate=drop_path_rate
    )
    
    # Freeze backbone if requested
    if freeze_backbone:
        freeze_model_backbone(model, model_name)
    
    # Freeze specific layers if requested
    if freeze_layers:
        freeze_specific_layers(model, freeze_layers)
    
    return model


def freeze_model_backbone(model: nn.Module, model_name: str):
    """
    Freeze the backbone of the model (all except classifier head)
    
    Args:
        model: PyTorch model
        model_name: Name of the model architecture
    """
    # For DeiT models
    if 'deit' in model_name:
        # Freeze patch embedding and transformer blocks
        for name, param in model.named_parameters():
            if 'head' not in name:  # Don't freeze the head
                param.requires_grad = False
    
    # For MobileViT models
    elif 'mobilevit' in model_name:
        # Freeze everything except classifier
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False


def freeze_specific_layers(model: nn.Module, layer_names: List[str]):
    """
    Freeze specific layers by name
    
    Args:
        model: PyTorch model
        layer_names: List of layer names (or partial names) to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                break


def unfreeze_model(model: nn.Module):
    """
    Unfreeze all parameters in the model
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about the model
    
    Args:
        model: PyTorch model
        
    Returns:
        dict with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_percentage': (trainable_params / total_params) * 100
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """Test model creation"""
    
    print("Testing model creation...")
    print("=" * 60)
    
    models_to_test = [
        ('deit_tiny', 51),
        ('mobilevit_xxs', 51),
    ]
    
    for model_name, num_classes in models_to_test:
        print(f"\nModel: {model_name}")
        print("-" * 40)
        
        # Create model
        model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            dropout=0.1,
            freeze_backbone=False
        )
        
        # Get info
        info = get_model_info(model)
        print(f"Total parameters: {info['total_params']:,}")
        print(f"Trainable parameters: {info['trainable_params']:,}")
        print(f"Trainable percentage: {info['trainable_percentage']:.2f}%")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # Test with frozen backbone
        print("\nWith frozen backbone:")
        model_frozen = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=True
        )
        info_frozen = get_model_info(model_frozen)
        print(f"Trainable parameters: {info_frozen['trainable_params']:,}")
        print(f"Trainable percentage: {info_frozen['trainable_percentage']:.2f}%")
