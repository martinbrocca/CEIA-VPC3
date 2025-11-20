"""
Fixed Inference Script
Uses class order from actual dataset, not config
"""

import torch
import torch.nn.functional as F
from PIL import Image
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_factory import create_model
from src.data.transforms import get_val_transforms
from src.data.dataset import IngredientsDataset
import yaml


def get_class_names_from_dataset(data_dir: str):
    """Get class names in the same order as training dataset"""
    train_dir = Path(data_dir) / 'Train'
    # Classes are sorted alphabetically by ImageFolder
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    return classes


def load_model(checkpoint_path: str, config_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get class names from actual dataset (in same order as training!)
    data_dir = config['paths']['data_dir'] + '/raw/huggingface'
    class_names = get_class_names_from_dataset(data_dir)
    num_classes = len(class_names)
    
    print(f"Loading classes from dataset: {num_classes} classes")
    print(f"First 10 classes: {class_names[:10]}")
    
    # Create model
    model_name = config['model']['architecture'].replace('_patch16_224', '')
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        dropout=config['model']['dropout']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            print(f"Validation accuracy: {checkpoint['metrics'].get('val_acc', 0):.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Get transforms
    transform = get_val_transforms(img_size=config['data']['img_size'])
    
    return model, class_names, transform


def predict_image(image_path: str, model, class_names, transform, device: str = 'cuda', top_k: int = 5):
    """Predict class for a single image"""
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, min(top_k, len(class_names)))
    
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        predictions.append((class_names[idx.item()], prob.item()))
    
    return predictions


def predict_batch(image_dir: str, model, class_names, transform, device: str = 'cuda', top_k: int = 3):
    """Predict classes for all images in a directory"""
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    results = {}
    
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() in image_extensions:
            predictions = predict_image(
                str(image_path), model, class_names, transform, device, top_k
            )
            results[image_path.name] = predictions
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Make predictions on images')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Path to directory of images')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        parser.error("Must provide either --image or --image-dir")
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
    
    print(f"Loading model from {args.checkpoint}...")
    model, class_names, transform = load_model(args.checkpoint, args.config, device)
    print(f"Model loaded successfully!")
    
    # Single image prediction
    if args.image:
        print(f"\nPredicting for image: {args.image}")
        predictions = predict_image(args.image, model, class_names, transform, device, args.top_k)
        
        print(f"\nTop {args.top_k} Predictions:")
        print("=" * 50)
        for i, (class_name, prob) in enumerate(predictions, 1):
            print(f"{i}. {class_name:20s} - {prob*100:.2f}%")
        print("=" * 50)
    
    # Batch prediction
    if args.image_dir:
        print(f"\nPredicting for all images in: {args.image_dir}")
        results = predict_batch(args.image_dir, model, class_names, transform, device, args.top_k)
        
        print(f"\nProcessed {len(results)} images")
        print("=" * 70)
        
        for image_name, predictions in results.items():
            print(f"\n{image_name}:")
            for i, (class_name, prob) in enumerate(predictions, 1):
                print(f"  {i}. {class_name:20s} - {prob*100:.2f}%")
        
        print("=" * 70)


if __name__ == '__main__':
    main()
