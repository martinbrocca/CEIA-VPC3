"""
Evaluation Script
Evaluate trained models on validation/test sets
"""

import torch
import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataloader import create_dataloaders
from src.models.model_factory import create_model
from src.evaluation.metrics import (
    evaluate_model,
    print_metrics_summary,
    get_per_class_metrics_df
)
from src.evaluation.visualizations import create_evaluation_report
from src.utils.logging_utils import setup_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--split', type=str, default='val',
                       choices=['val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                       help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: str):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model instance
        device: Device to load to
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            print(f"Checkpoint metrics: {checkpoint['metrics']}")
    else:
        # Assume checkpoint is just state dict
        model.load_state_dict(checkpoint)
        print(f"Loaded model state dict")
    
    return model


def main():
    """Main evaluation function"""
    
    # Parse arguments
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(
        name='evaluation',
        log_file=f'{args.output_dir}/evaluation.log',
        console=True
    )
    
    logger.info("Starting evaluation...")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Using CPU.")
    
    logger.info(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    data_dir = config['paths']['data_dir'] + '/raw/huggingface'
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        img_size=config['data']['img_size'],
        augmentation_level='none',  # No augmentation for evaluation
        use_weighted_sampler=False,
        pin_memory=(device == 'cuda')
    )
    
    # Select loader based on split
    if args.split == 'val':
        eval_loader = val_loader
    else:
        if test_loader is None:
            logger.error("Test split not available. Using validation split.")
            eval_loader = val_loader
        else:
            eval_loader = test_loader
    
    logger.info(f"Evaluation samples: {len(eval_loader.dataset)}")
    
    # Get class names
    class_names = train_loader.dataset.classes
    num_classes = len(class_names)
    logger.info(f"Number of classes: {num_classes}")
    
    # Create model
    logger.info(f"Creating model: {config['model']['architecture']}")
    model = create_model(
        model_name=config['model']['architecture'].replace('_patch16_224', '').replace('deit_', 'deit_').replace('mobilevit_', 'mobilevit_'),
        num_classes=num_classes,
        pretrained=False,  # Will load from checkpoint
        dropout=config['model']['dropout'],
        drop_path_rate=config['model']['drop_path_rate']
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, model, device)
    model = model.to(device)
    
    # Evaluate
    logger.info("\nEvaluating model...")
    metrics = evaluate_model(
        model=model,
        dataloader=eval_loader,
        device=device,
        class_names=class_names
    )
    
    # Print metrics summary
    print_metrics_summary(metrics, class_names)
    
    # Save detailed results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Per-class metrics
    logger.info("\nSaving per-class metrics...")
    metrics_df = get_per_class_metrics_df(metrics, class_names)
    metrics_df.to_csv(output_dir / 'per_class_metrics.csv', index=False)
    logger.info(f"Saved to {output_dir / 'per_class_metrics.csv'}")
    
    # Classification report
    import json
    with open(output_dir / 'classification_report.json', 'w') as f:
        json.dump(metrics['classification_report'], f, indent=2)
    logger.info(f"Saved to {output_dir / 'classification_report.json'}")
    
    # Overall metrics
    summary_metrics = {
        'accuracy': float(metrics['accuracy']),
        'balanced_accuracy': float(metrics['balanced_accuracy']),
        'top3_accuracy': float(metrics['top3_accuracy']),
        'top5_accuracy': float(metrics['top5_accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'precision_weighted': float(metrics['precision_weighted']),
        'recall_weighted': float(metrics['recall_weighted']),
        'f1_weighted': float(metrics['f1_weighted'])
    }
    
    with open(output_dir / 'summary_metrics.json', 'w') as f:
        json.dump(summary_metrics, f, indent=2)
    logger.info(f"Saved to {output_dir / 'summary_metrics.json'}")
    
    # Generate plots
    if not args.no_plots:
        logger.info("\nGenerating visualizations...")
        
        # Load training history if available
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        history = checkpoint.get('history', None)
        
        create_evaluation_report(
            metrics=metrics,
            class_names=class_names,
            history=history,
            output_dir=str(output_dir / 'figures')
        )
        
        logger.info(f"Plots saved to {output_dir / 'figures'}")
    
    logger.info("\nEvaluation completed!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
