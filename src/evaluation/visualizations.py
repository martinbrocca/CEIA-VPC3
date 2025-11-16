"""
Visualization utilities
Plotting functions for training results and evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
):
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
    normalize: bool = False
):
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        normalize: Whether to normalize by true labels
    """
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm = confusion_matrix
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=False,  # Don't annotate for large matrices
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_per_class_metrics(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
    top_k: int = 20
):
    """
    Plot per-class metrics (precision, recall, f1)
    
    Args:
        metrics_df: DataFrame with class metrics
        save_path: Path to save figure
        figsize: Figure size
        top_k: Show top K classes
    """
    # Select top K classes by F1 score
    df_plot = metrics_df.head(top_k).copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(df_plot))
    width = 0.25
    
    ax.bar(x - width, df_plot['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, df_plot['recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, df_plot['f1_score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Top {top_k} Classes - Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['class'], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class metrics to {save_path}")
    
    plt.show()


def plot_class_distribution(
    class_counts: Dict[str, int],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6)
):
    """
    Plot class distribution
    
    Args:
        class_counts: Dict mapping class names to counts
        save_path: Path to save figure
        figsize: Figure size
    """
    # Sort by count
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_items)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    ax.bar(range(len(classes)), counts, color=colors, alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean line
    mean_count = np.mean(counts)
    ax.axhline(y=mean_count, color='red', linestyle='--', 
               label=f'Mean: {mean_count:.0f}', linewidth=2)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved class distribution to {save_path}")
    
    plt.show()


def plot_learning_rate_schedule(
    history: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5)
):
    """
    Plot learning rate schedule over epochs
    
    Args:
        history: Dict with 'lr' key
        save_path: Path to save figure
        figsize: Figure size
    """
    if 'lr' not in history:
        print("No learning rate history found")
        return
    
    epochs = range(1, len(history['lr']) + 1)
    
    plt.figure(figsize=figsize)
    plt.plot(epochs, history['lr'], 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved LR schedule to {save_path}")
    
    plt.show()


def plot_top_errors(
    misclassified_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
    top_k: int = 15
):
    """
    Plot top misclassified samples by confidence
    
    Args:
        misclassified_df: DataFrame from get_misclassified_samples
        save_path: Path to save figure
        figsize: Figure size
        top_k: Number of top errors to show
    """
    if len(misclassified_df) == 0:
        print("No misclassifications found")
        return
    
    df_plot = misclassified_df.head(top_k)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(df_plot))
    
    ax.bar(x, df_plot['confidence'], alpha=0.7, color='red', label='Predicted Confidence')
    ax.bar(x, df_plot['true_prob'], alpha=0.7, color='green', label='True Class Prob')
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'Top {top_k} Most Confident Misclassifications', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['true_class'][:8]}\n->\n{row['predicted_class'][:8]}" 
                        for _, row in df_plot.iterrows()], 
                       rotation=0, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved top errors to {save_path}")
    
    plt.show()


def create_evaluation_report(
    metrics: Dict,
    class_names: List[str],
    history: Dict,
    output_dir: str = 'outputs/figures'
):
    """
    Create complete evaluation report with all visualizations
    
    Args:
        metrics: Metrics dict from evaluate_model
        class_names: List of class names
        history: Training history dict
        output_dir: Directory to save figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating evaluation report...")
    print("=" * 60)
    
    # Training history
    if history:
        plot_training_history(
            history,
            save_path=output_path / 'training_history.png'
        )
    
    # Confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path=output_path / 'confusion_matrix.png',
        normalize=False
    )
    
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path=output_path / 'confusion_matrix_normalized.png',
        normalize=True
    )
    
    # Per-class metrics
    from .metrics import get_per_class_metrics_df
    metrics_df = get_per_class_metrics_df(metrics, class_names)
    
    plot_per_class_metrics(
        metrics_df,
        save_path=output_path / 'per_class_metrics.png'
    )
    
    # Learning rate schedule
    if history and 'lr' in history:
        plot_learning_rate_schedule(
            history,
            save_path=output_path / 'lr_schedule.png'
        )
    
    # Top errors
    from .metrics import get_misclassified_samples
    misclassified = get_misclassified_samples(
        metrics['predictions'],
        metrics['labels'],
        metrics['probabilities'],
        class_names,
        top_k=15
    )
    
    if len(misclassified) > 0:
        plot_top_errors(
            misclassified,
            save_path=output_path / 'top_errors.png'
        )
    
    print(f"\nAll figures saved to {output_path}/")
    print("=" * 60)


if __name__ == "__main__":
    """Test visualizations"""
    
    # Create dummy data
    num_epochs = 50
    num_classes = 51
    
    # Dummy history
    history = {
        'train_loss': np.linspace(2.0, 0.5, num_epochs) + np.random.randn(num_epochs) * 0.1,
        'val_loss': np.linspace(2.2, 0.7, num_epochs) + np.random.randn(num_epochs) * 0.15,
        'train_acc': np.linspace(30, 85, num_epochs) + np.random.randn(num_epochs) * 2,
        'val_acc': np.linspace(28, 80, num_epochs) + np.random.randn(num_epochs) * 3,
        'lr': np.logspace(-5, -6, num_epochs)
    }
    
    print("Testing visualizations...")
    plot_training_history(history)
    plot_learning_rate_schedule(history)
