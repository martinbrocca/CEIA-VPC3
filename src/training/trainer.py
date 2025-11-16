"""
Trainer class for training Vision Transformers
Includes MLflow logging for Databricks
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, Optional, Callable
import mlflow


class Trainer:
    """
    Trainer class for Vision Transformers with MLflow logging
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        output_dir: Directory to save checkpoints
        use_mlflow: Whether to use MLflow logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        output_dir: str = 'outputs/models',
        use_mlflow: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return {'train_loss': epoch_loss, 'train_acc': epoch_acc}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return {'val_loss': epoch_loss, 'val_acc': epoch_acc}
    
    def fit(self, epochs: int, log_interval: int = 1):
        """
        Train the model for specified epochs
        Note: MLflow run should already be active when calling this method
        
        Args:
            epochs: Number of epochs to train
            log_interval: Log metrics every N epochs
        """
        
        print(f"\nTraining for {epochs} epochs...")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['train_acc'].append(train_metrics['train_acc'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_acc'].append(val_metrics['val_acc'])
            self.history['lr'].append(current_lr)
            
            # Log to MLflow
            if self.use_mlflow and epoch % log_interval == 0:
                mlflow.log_metrics({
                    'train_loss': train_metrics['train_loss'],
                    'train_acc': train_metrics['train_acc'],
                    'val_loss': val_metrics['val_loss'],
                    'val_acc': val_metrics['val_acc'],
                    'learning_rate': current_lr
                }, step=epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_acc']:.2f}%")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_metrics['val_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['val_acc']
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                print(f"  Saved best model (Val Acc: {self.best_val_acc:.2f}%)")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best Val Acc: {self.best_val_acc:.2f}%")
        
        # Log final artifacts if using MLflow
        if self.use_mlflow:
            self._log_artifacts()
    
    def _log_artifacts(self):
        """Log artifacts to MLflow"""
        # Log best model
        best_model_path = self.output_dir / 'best_model.pth'
        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path))
        
        # Log final metrics
        mlflow.log_metrics({
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.history['train_loss'][-1],
            'final_train_acc': self.history['train_acc'][-1]
        })
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Metrics: {checkpoint['metrics']}")