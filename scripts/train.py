"""
Main Training Script
Train Vision Transformer models with MLflow tracking
"""

import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataloader import create_dataloaders, get_class_weights
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.training.losses import get_criterion
from src.training.optimizers import create_optimizer_and_scheduler
from src.utils.logging_utils import setup_logger, log_system_info, log_config
from src.utils.mlflow_utils import (
    log_params_from_config,
    log_model_info,
    log_dataset_info,
    save_config_as_artifact
)
from config.mlflow_config import setup_mlflow
import mlflow


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Vision Transformer for Ingredients Classification')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--run-name', type=str, default=None,
                       help='MLflow run name')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow logging')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    
    # Override config parameters
    parser.add_argument('--model', type=str, default=None,
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--augmentation', type=str, default=None,
                       choices=['none', 'light', 'medium', 'heavy'],
                       help='Augmentation level')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.model:
        config['model']['architecture'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.augmentation:
        config['augmentation']['level'] = args.augmentation
    
    # Setup logger
    logger = setup_logger(
        name='training',
        log_file='outputs/logs/training.log',
        console=True
    )
    
    logger.info("Starting training...")
    log_system_info(logger)
    log_config(logger, config)
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Using CPU.")
    
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    data_dir = config['paths']['data_dir'] + '/raw/huggingface'
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        img_size=config['data']['img_size'],
        augmentation_level=config.get('augmentation', {}).get('level', 'medium'),
        use_weighted_sampler=False,
        pin_memory=(device == 'cuda')
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    if test_loader:
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Get class names
    class_names = train_loader.dataset.classes
    num_classes = len(class_names)
    logger.info(f"Number of classes: {num_classes}")
    
    # Create model
    logger.info(f"Creating model: {config['model']['architecture']}")
    model = create_model(
        model_name=config['model']['architecture'].replace('_patch16_224', '').replace('deit_', 'deit_').replace('mobilevit_', 'mobilevit_'),
        num_classes=num_classes,
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout'],
        drop_path_rate=config['model']['drop_path_rate'],
        freeze_backbone=config['model'].get('freeze_backbone', False)
    )
    
    from src.models.model_factory import get_model_info
    model_info = get_model_info(model)
    logger.info(f"Model parameters: {model_info['total_params']:,}")
    logger.info(f"Trainable parameters: {model_info['trainable_params']:,}")
    
    # Create loss function
    criterion = get_criterion(
        criterion_name=config['training'].get('criterion', 'ce'),
        num_classes=num_classes,
        label_smoothing=config['training'].get('label_smoothing', 0.0),
        class_weights=None
    )
    
    logger.info(f"Loss function: {criterion.__class__.__name__}")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        config=config['training']
    )
    
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Scheduler: {scheduler.__class__.__name__ if scheduler else 'None'}")
    
    # Setup MLflow
    use_mlflow = not args.no_mlflow
    
    if use_mlflow:
        logger.info("Setting up MLflow...")
        setup_mlflow()
        
        # Create run name
        run_name = args.run_name
        if run_name is None:
            model_short = config['model']['architecture'].replace('_patch16_224', '')
            aug_level = config.get('augmentation', {}).get('level', 'medium')
            lr = config['training']['learning_rate']
            run_name = f"{model_short}_lr{lr}_aug{aug_level}"
        
        logger.info(f"MLflow run name: {run_name}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            
            # Log all parameters
            log_params_from_config(config)
            log_model_info(model, config['model']['architecture'])
            log_dataset_info(train_loader, val_loader, test_loader)
            
            # Save config as artifact
            save_config_as_artifact(config)
            
            # Set tags
            mlflow.set_tag('model_family', 'vision_transformer')
            mlflow.set_tag('dataset', 'food_ingredients_51')
            
            # Create trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                output_dir=config['paths']['models_dir'],
                use_mlflow=True
            )
            
            # Train
            logger.info(f"\nStarting training for {config['training']['epochs']} epochs...")
            trainer.fit(
                epochs=config['training']['epochs'],
                run_name=None,  # Already in MLflow context
                log_interval=1
            )
            
            logger.info("\nTraining completed!")
            logger.info(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    
    else:
        # Train without MLflow
        logger.info("Training without MLflow logging...")
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=config['paths']['models_dir'],
            use_mlflow=False
        )
        
        trainer.fit(
            epochs=config['training']['epochs'],
            run_name=None,
            log_interval=1
        )
        
        logger.info("\nTraining completed!")
        logger.info(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
