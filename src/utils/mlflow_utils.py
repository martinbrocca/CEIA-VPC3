"""
MLflow utilities
Helper functions for MLflow logging
"""

import mlflow
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def log_params_from_config(config: Dict[str, Any], prefix: str = ''):
    """
    Log parameters from nested config dict to MLflow
    
    Args:
        config: Configuration dictionary
        prefix: Prefix for parameter names
    """
    for key, value in config.items():
        param_name = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recursively log nested dicts
            log_params_from_config(value, prefix=f"{param_name}_")
        else:
            # Log primitive types
            try:
                mlflow.log_param(param_name, value)
            except Exception as e:
                print(f"Warning: Could not log parameter {param_name}: {e}")


def log_model_info(model: torch.nn.Module, model_name: str):
    """
    Log model information to MLflow
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    from src.models.model_factory import get_model_info
    
    info = get_model_info(model)
    
    mlflow.log_param('model_name', model_name)
    mlflow.log_param('total_params', info['total_params'])
    mlflow.log_param('trainable_params', info['trainable_params'])
    mlflow.log_param('frozen_params', info['frozen_params'])
    mlflow.log_param('trainable_percentage', f"{info['trainable_percentage']:.2f}%")


def log_dataset_info(train_loader, val_loader, test_loader=None):
    """
    Log dataset information to MLflow
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader (optional)
    """
    mlflow.log_param('train_samples', len(train_loader.dataset))
    mlflow.log_param('val_samples', len(val_loader.dataset))
    
    if test_loader:
        mlflow.log_param('test_samples', len(test_loader.dataset))
    
    mlflow.log_param('batch_size', train_loader.batch_size)
    mlflow.log_param('num_workers', train_loader.num_workers)
    mlflow.log_param('num_classes', len(train_loader.dataset.classes))


def log_metrics_dict(metrics: Dict[str, float], step: Optional[int] = None):
    """
    Log multiple metrics at once
    
    Args:
        metrics: Dictionary of metric names to values
        step: Optional step number
    """
    mlflow.log_metrics(metrics, step=step)


def log_confusion_matrix_as_artifact(
    confusion_matrix,
    class_names,
    save_path: str = 'outputs/figures/confusion_matrix.png'
):
    """
    Log confusion matrix as artifact
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
    """
    from src.evaluation.visualizations import plot_confusion_matrix
    
    plot_confusion_matrix(
        confusion_matrix,
        class_names,
        save_path=save_path,
        normalize=True
    )
    
    mlflow.log_artifact(save_path)


def log_training_curves_as_artifact(
    history: Dict,
    save_path: str = 'outputs/figures/training_history.png'
):
    """
    Log training curves as artifact
    
    Args:
        history: Training history dict
        save_path: Path to save figure
    """
    from src.evaluation.visualizations import plot_training_history
    
    plot_training_history(history, save_path=save_path)
    mlflow.log_artifact(save_path)


def save_config_as_artifact(config: Dict, save_path: str = 'outputs/config.yaml'):
    """
    Save and log config as artifact
    
    Args:
        config: Configuration dict
        save_path: Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    mlflow.log_artifact(str(save_path))


def log_pytorch_model(
    model: torch.nn.Module,
    artifact_path: str = 'model',
    registered_model_name: Optional[str] = None
):
    """
    Log PyTorch model to MLflow
    
    Args:
        model: PyTorch model
        artifact_path: Artifact path in MLflow
        registered_model_name: Name for model registry (optional)
    """
    mlflow.pytorch.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name
    )


def log_classification_report(
    report: Dict,
    save_path: str = 'outputs/classification_report.txt'
):
    """
    Log classification report as artifact
    
    Args:
        report: Classification report dict from sklearn
        save_path: Path to save report
    """
    from sklearn.metrics import classification_report
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dict to formatted text
    with open(save_path, 'w') as f:
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"\n{class_name}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
            else:
                f.write(f"{class_name}: {metrics:.4f}\n")
    
    mlflow.log_artifact(str(save_path))


def set_tags(tags: Dict[str, str]):
    """
    Set multiple tags at once
    
    Args:
        tags: Dictionary of tag names to values
    """
    for key, value in tags.items():
        mlflow.set_tag(key, value)


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get or create MLflow experiment
    
    Args:
        experiment_name: Name of experiment
        
    Returns:
        Experiment ID
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id


if __name__ == "__main__":
    """Test MLflow utilities"""
    
    print("Testing MLflow utilities...")
    
    # Test config logging
    config = {
        'model': {
            'name': 'deit_tiny',
            'num_classes': 51,
            'dropout': 0.1
        },
        'training': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 5e-5,
            'optimizer': 'adamw'
        }
    }
    
    print("Config to log:")
    print(config)
    
    # Note: Actual MLflow logging would happen in a run context:
    # with mlflow.start_run():
    #     log_params_from_config(config)
    #     ...
    
    print("\nMLflow utilities ready to use!")
