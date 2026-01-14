"""
Training utilities for MNIST classifier.
"""
import os
from typing import Dict, Optional
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from mnist_classifier.visualization.gradient_visualizer import GradientVisualizer


class Trainer:
    """
    Trainer class for training and evaluating the MNIST classifier.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_dir: str = './checkpoints',
        save_best_only: bool = True,
        visualize_gradients: bool = False,
        gradient_vis_dir: str = './gradient_frames',
        gradient_vis_frequency: int = 100
    ):
        """
        Initialize the Trainer.

        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on (CPU or CUDA)
            scheduler: Learning rate scheduler (optional)
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Whether to save only the best model
            visualize_gradients: Whether to save gradient visualizations
            gradient_vis_dir: Directory to save gradient frames
            gradient_vis_frequency: Save gradient visualization every N batches
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only

        # Gradient visualization
        self.visualize_gradients = visualize_gradients
        self.gradient_vis_frequency = gradient_vis_frequency
        self.gradient_visualizer = None
        if visualize_gradients:
            self.gradient_visualizer = GradientVisualizer(model, gradient_vis_dir)
            print(f'Gradient visualization enabled: saving every {gradient_vis_frequency} batches to {gradient_vis_dir}/')

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        # Best metrics for checkpointing
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()

            # Visualize gradients before optimizer step
            if self.visualize_gradients and (batch_idx % self.gradient_vis_frequency == 0):
                current_lr = self.optimizer.param_groups[0]['lr']
                self.gradient_visualizer.visualize_gradients(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    loss=loss.item(),
                    learning_rate=current_lr,
                    save=True
                )

            self.optimizer.step()

            # Calculate metrics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx + 1}/{len(self.train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}%')

        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        print(f'Train Epoch {epoch} completed in {epoch_time:.2f}s | '
              f'Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')

        return {'loss': avg_loss, 'accuracy': accuracy}

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)

                # Calculate metrics
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        print(f'Validation Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')

        return {'loss': avg_loss, 'accuracy': accuracy}

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save checkpoint
        if not self.save_best_only or is_best:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_epoch_{epoch}.pth' if not is_best else 'best_model.pth'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        print(f'Checkpoint loaded from: {checkpoint_path}')
        return checkpoint.get('epoch', 0)

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, list]:
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait for improvement before stopping

        Returns:
            Training history dictionary
        """
        print(f'Starting training for {num_epochs} epochs on {self.device}')
        print('=' * 80)

        epochs_without_improvement = 0

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['lr'].append(current_lr)

            # Check if this is the best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                epochs_without_improvement = 0
                print(f'New best model! Validation accuracy: {self.best_val_acc:.2f}%')
            else:
                epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch} epochs')
                print(f'Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}')
                break

            print('-' * 80)

        print('=' * 80)
        print(f'Training completed!')
        print(f'Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}')

        # Save gradient visualization summary
        if self.visualize_gradients and self.gradient_visualizer:
            self.gradient_visualizer.save_summary()

        return self.history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the model on test data.

    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Dictionary containing test metrics
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Calculate metrics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total

    print(f'Test Results | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')

    return {'loss': avg_loss, 'accuracy': accuracy}
