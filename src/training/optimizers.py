"""
Optimizers and Learning Rate Schedulers
Factory functions for creating optimizers and schedulers
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    MultiStepLR,
    _LRScheduler
)
from typing import Optional, Dict, Any


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    **kwargs
) -> Optimizer:
    """
    Get optimizer
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        momentum: Momentum (for SGD)
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Optimizer
    """
    
    params = model.parameters()
    
    if optimizer_name.lower() == 'adam':
        return Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_name.lower() == 'adamw':
        return AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_name.lower() == 'sgd':
        return SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: str,
    num_epochs: int,  # Changed from 'epochs' to 'num_epochs' to avoid conflict
    **kwargs
) -> Optional[_LRScheduler]:
    """
    Get learning rate scheduler
    
    Args:
        optimizer: Optimizer
        scheduler_name: Name of scheduler 
                       ('cosine', 'step', 'multistep', 'plateau', 'none')
        num_epochs: Total number of epochs (for cosine annealing)
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Scheduler or None
    """
    
    if scheduler_name.lower() == 'none':
        return None
    
    elif scheduler_name.lower() == 'cosine':
        T_max = kwargs.get('T_max', num_epochs)
        eta_min = kwargs.get('eta_min', 0)
        return CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
    
    elif scheduler_name.lower() == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    
    elif scheduler_name.lower() == 'multistep':
        milestones = kwargs.get('milestones', [30, 60, 90])
        gamma = kwargs.get('gamma', 0.1)
        return MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
    
    elif scheduler_name.lower() == 'plateau':
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 10)
        return ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=True
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: Dict[str, Any]
) -> tuple[Optimizer, Optional[_LRScheduler]]:
    """
    Create optimizer and scheduler from config dict
    
    Args:
        model: PyTorch model
        config: Configuration dict with keys:
                - optimizer: str
                - learning_rate: float
                - weight_decay: float
                - scheduler: str
                - epochs: int
                - (other scheduler params)
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    
    optimizer = get_optimizer(
        model=model,
        optimizer_name=config.get('optimizer', 'adamw'),
        learning_rate=config.get('learning_rate', 5e-5),
        weight_decay=config.get('weight_decay', 1e-4),
        momentum=config.get('momentum', 0.9)
    )
    
    # Extract scheduler-specific params, excluding 'epochs'
    scheduler_kwargs = {k: v for k, v in config.items() 
                       if k not in ['optimizer', 'learning_rate', 'weight_decay', 
                                   'momentum', 'scheduler', 'epochs']}
    
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=config.get('scheduler', 'cosine'),
        num_epochs=config.get('epochs', 50),
        **scheduler_kwargs
    )
    
    return optimizer, scheduler


if __name__ == "__main__":
    """Test optimizers and schedulers"""
    import torch.nn as nn
    
    # Create dummy model
    model = nn.Linear(10, 5)
    
    print("Testing optimizers and schedulers...")
    print("=" * 60)
    
    # Test optimizers
    print("\nOptimizers:")
    for opt_name in ['adam', 'adamw', 'sgd']:
        optimizer = get_optimizer(
            model=model,
            optimizer_name=opt_name,
            learning_rate=1e-3,
            weight_decay=1e-4
        )
        print(f"  {opt_name:10s}: {optimizer.__class__.__name__}")
    
    # Test schedulers
    print("\nSchedulers:")
    optimizer = get_optimizer(model, 'adamw', 1e-3)
    
    for sched_name in ['cosine', 'step', 'multistep', 'plateau', 'none']:
        scheduler = get_scheduler(
            optimizer=optimizer,
            scheduler_name=sched_name,
            num_epochs=50
        )
        sched_type = scheduler.__class__.__name__ if scheduler else 'None'
        print(f"  {sched_name:10s}: {sched_type}")
    
    # Test with config
    print("\nFrom config dict:")
    config = {
        'optimizer': 'adamw',
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'epochs': 50
    }
    
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    print(f"  Optimizer: {optimizer.__class__.__name__}")
    print(f"  Scheduler: {scheduler.__class__.__name__}")
    print(f"  Initial LR: {optimizer.param_groups[0]['lr']}")