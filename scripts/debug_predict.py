"""
Debug Inference Script
Check predictions and visualize preprocessing
"""

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_factory import create_model
from src.data.transforms import get_val_transforms
import yaml


def visualize_preprocessing(image_path: str, transform, save_path: str = 'debug_preprocess.png'):
    """Visualize original vs preprocessed image"""
    # Load original
    image = Image.open(image_path).convert('RGB')
    
    # Apply transform
    image_tensor = transform(image)
    
    # Convert back for visualization (denormalize)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_denorm = image_tensor * std + mean
    image_denorm = torch.clamp(image_denorm, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(image_denorm.permute(1, 2, 0).numpy())
    axes[1].set_title('After Preprocessing')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved preprocessing visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/config.yaml')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = config['data']['classes']
    
    print(f"Image path: {args.image}")
    print(f"True label from filename: {Path(args.image).stem.split('_')[0]}")
    
    # Load model
    model_name = config['model']['architecture'].replace('_patch16_224', '')
    model = create_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False,
        dropout=config['model']['dropout']
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nCheckpoint info:")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            print(f"  Val Acc: {checkpoint['metrics'].get('val_acc', 'unknown'):.2f}%")
    
    model.eval()
    
    # Get transform
    transform = get_val_transforms(img_size=config['data']['img_size'])
    
    # Visualize preprocessing
    visualize_preprocessing(args.image, transform)
    
    # Load and predict
    image = Image.open(args.image).convert('RGB')
    print(f"\nOriginal image size: {image.size}")
    
    image_tensor = transform(image).unsqueeze(0)
    print(f"Tensor shape: {image_tensor.shape}")
    print(f"Tensor mean: {image_tensor.mean():.3f}, std: {image_tensor.std():.3f}")
    print(f"Tensor min: {image_tensor.min():.3f}, max: {image_tensor.max():.3f}")
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    # Top predictions
    top_probs, top_indices = torch.topk(probabilities, 10)
    
    print(f"\nTop 10 Predictions:")
    print("=" * 50)
    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), 1):
        class_name = class_names[idx.item()]
        print(f"{i:2d}. {class_name:20s} - {prob.item()*100:6.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    main()
