"""
Loss Functions
Different loss functions for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing
    Helps prevent overconfidence and improves generalization
    
    Args:
        smoothing: Label smoothing factor (0.0 to 1.0)
        weight: Class weights (optional)
    """
    
    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (batch_size, num_classes)
            target: Ground truth labels (batch_size,)
            
        Returns:
            Loss value
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        if self.weight is not None:
            true_dist = true_dist * self.weight.unsqueeze(0).to(pred.device)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses on hard examples by down-weighting easy examples
    
    Args:
        alpha: Balancing factor (optional)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (batch_size, num_classes)
            target: Ground truth labels (batch_size,)
            
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[target].to(pred.device)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_criterion(
    criterion_name: str,
    num_classes: int,
    label_smoothing: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0
) -> nn.Module:
    """
    Get loss function
    
    Args:
        criterion_name: Name of criterion ('ce', 'label_smoothing', 'focal', 'weighted_ce')
        num_classes: Number of classes
        label_smoothing: Label smoothing factor
        class_weights: Class weights for weighted loss
        focal_gamma: Gamma for focal loss
        
    Returns:
        Loss function
    """
    
    if criterion_name == 'ce':
        return nn.CrossEntropyLoss()
    
    elif criterion_name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    
    elif criterion_name == 'focal':
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    
    elif criterion_name == 'weighted_ce':
        if class_weights is None:
            raise ValueError("class_weights must be provided for weighted_ce")
        return nn.CrossEntropyLoss(weight=class_weights)
    
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")


if __name__ == "__main__":
    """Test loss functions"""
    
    # Create dummy data
    batch_size = 32
    num_classes = 51
    
    pred = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    
    print("Testing loss functions...")
    print("=" * 60)
    
    # Test different losses
    losses = {
        'CrossEntropy': nn.CrossEntropyLoss(),
        'LabelSmoothing': LabelSmoothingCrossEntropy(smoothing=0.1),
        'FocalLoss': FocalLoss(gamma=2.0)
    }
    
    for name, criterion in losses.items():
        loss = criterion(pred, target)
        print(f"{name:20s}: {loss.item():.4f}")
    
    # Test with class weights
    print("\nWith class weights:")
    class_weights = torch.ones(num_classes)
    class_weights[0] = 2.0  # Give more weight to class 0
    
    weighted_ce = nn.CrossEntropyLoss(weight=class_weights)
    loss = weighted_ce(pred, target)
    print(f"{'Weighted CE':20s}: {loss.item():.4f}")
