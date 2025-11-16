"""
Evaluation Metrics
Comprehensive metrics for classification
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    top_k_accuracy_score
)
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    class_names: List[str] = None
) -> Dict:
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to use
        class_names: List of class names
        
    Returns:
        dict with all metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    
    # Balanced accuracy (important for imbalanced datasets)
    metrics['balanced_accuracy'] = balanced_accuracy_score(all_labels, all_preds)
    
    # Top-k accuracy
    metrics['top3_accuracy'] = top_k_accuracy_score(all_labels, all_probs, k=3)
    metrics['top5_accuracy'] = top_k_accuracy_score(all_labels, all_probs, k=5)
    
    # Precision, Recall, F1 (macro and weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    metrics['precision_macro'] = precision_macro
    metrics['recall_macro'] = recall_macro
    metrics['f1_macro'] = f1_macro
    metrics['precision_weighted'] = precision_weighted
    metrics['recall_weighted'] = recall_weighted
    metrics['f1_weighted'] = f1_weighted
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    metrics['per_class'] = {
        'precision': precision_per_class,
        'recall': recall_per_class,
        'f1': f1_per_class,
        'support': support
    }
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(all_labels)))]
    
    metrics['classification_report'] = classification_report(
        all_labels, all_preds, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Store predictions and probabilities for further analysis
    metrics['predictions'] = all_preds
    metrics['labels'] = all_labels
    metrics['probabilities'] = all_probs
    
    return metrics


def get_per_class_metrics_df(
    metrics: Dict,
    class_names: List[str]
) -> pd.DataFrame:
    """
    Get per-class metrics as DataFrame
    
    Args:
        metrics: Metrics dict from evaluate_model
        class_names: List of class names
        
    Returns:
        DataFrame with per-class metrics
    """
    per_class = metrics['per_class']
    
    df = pd.DataFrame({
        'class': class_names,
        'precision': per_class['precision'],
        'recall': per_class['recall'],
        'f1_score': per_class['f1'],
        'support': per_class['support']
    })
    
    # Sort by F1 score
    df = df.sort_values('f1_score', ascending=False).reset_index(drop=True)
    
    return df


def print_metrics_summary(metrics: Dict, class_names: List[str] = None):
    """
    Print formatted metrics summary
    
    Args:
        metrics: Metrics dict from evaluate_model
        class_names: List of class names
    """
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)")
    print(f"  Top-3 Accuracy:    {metrics['top3_accuracy']:.4f} ({metrics['top3_accuracy']*100:.2f}%)")
    print(f"  Top-5 Accuracy:    {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
    
    print(f"\nMacro-averaged Metrics:")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  F1-Score:  {metrics['f1_macro']:.4f}")
    
    print(f"\nWeighted-averaged Metrics:")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
    
    # Per-class summary
    if class_names:
        print(f"\nTop 5 Best Performing Classes (by F1-score):")
        df = get_per_class_metrics_df(metrics, class_names)
        print(df.head(5).to_string(index=False))
        
        print(f"\nBottom 5 Worst Performing Classes (by F1-score):")
        print(df.tail(5).to_string(index=False))
    
    print("\n" + "=" * 60)


def compute_confidence_metrics(probabilities: np.ndarray, predictions: np.ndarray) -> Dict:
    """
    Compute confidence-related metrics
    
    Args:
        probabilities: Predicted probabilities (N, num_classes)
        predictions: Predicted class indices (N,)
        
    Returns:
        dict with confidence metrics
    """
    # Get confidence for predicted class
    predicted_probs = probabilities[np.arange(len(predictions)), predictions]
    
    return {
        'mean_confidence': predicted_probs.mean(),
        'std_confidence': predicted_probs.std(),
        'min_confidence': predicted_probs.min(),
        'max_confidence': predicted_probs.max(),
        'median_confidence': np.median(predicted_probs)
    }


def get_misclassified_samples(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str],
    top_k: int = 20
) -> pd.DataFrame:
    """
    Get most confident misclassifications
    
    Args:
        predictions: Predicted labels
        labels: True labels
        probabilities: Prediction probabilities
        class_names: List of class names
        top_k: Number of top misclassifications to return
        
    Returns:
        DataFrame with misclassified samples
    """
    # Find misclassified samples
    misclassified_mask = predictions != labels
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) == 0:
        return pd.DataFrame()
    
    # Get confidence for predicted class
    predicted_probs = probabilities[misclassified_indices, predictions[misclassified_indices]]
    
    # Sort by confidence (descending)
    sorted_indices = np.argsort(predicted_probs)[::-1][:top_k]
    
    # Create DataFrame
    results = []
    for idx in sorted_indices:
        sample_idx = misclassified_indices[idx]
        results.append({
            'sample_index': sample_idx,
            'true_class': class_names[labels[sample_idx]],
            'predicted_class': class_names[predictions[sample_idx]],
            'confidence': predicted_probs[idx],
            'true_prob': probabilities[sample_idx, labels[sample_idx]]
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    """Test metrics computation"""
    
    # Create dummy predictions
    num_samples = 1000
    num_classes = 51
    
    np.random.seed(42)
    labels = np.random.randint(0, num_classes, num_samples)
    predictions = labels.copy()
    
    # Add some errors
    error_indices = np.random.choice(num_samples, size=100, replace=False)
    predictions[error_indices] = np.random.randint(0, num_classes, 100)
    
    # Create dummy probabilities
    probabilities = np.random.rand(num_samples, num_classes)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # Create dummy class names
    class_names = [f"class_{i}" for i in range(num_classes)]
    
    # Compute metrics (simulated)
    print("Testing metrics computation...")
    print("=" * 60)
    
    accuracy = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    top3_acc = top_k_accuracy_score(labels, probabilities, k=3)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    
    # Confidence metrics
    conf_metrics = compute_confidence_metrics(probabilities, predictions)
    print(f"\nConfidence Metrics:")
    for key, value in conf_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Misclassified samples
    print(f"\nTop 5 Misclassified Samples:")
    misclassified = get_misclassified_samples(
        predictions, labels, probabilities, class_names, top_k=5
    )
    if len(misclassified) > 0:
        print(misclassified.to_string(index=False))
