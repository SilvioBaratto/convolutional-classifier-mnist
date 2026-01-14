"""
Configuration settings for MNIST classifier training.
"""
import torch
from dataclasses import dataclass, field
from typing import Optional


def get_best_device() -> str:
    """
    Detect and return the best available device.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_device_info(device: str) -> dict:
    """
    Get information about the specified device.

    Args:
        device: Device string ('cuda', 'mps', or 'cpu')

    Returns:
        Dictionary with device information
    """
    info = {'device': device, 'name': device.upper()}

    if device == 'cuda':
        info['name'] = torch.cuda.get_device_name(0)
        info['memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    elif device == 'mps':
        info['name'] = 'Apple Silicon GPU (MPS)'
        info['memory'] = 'Unified Memory'
    else:
        info['name'] = 'CPU'
        info['memory'] = 'System RAM'

    return info


@dataclass
class Config:
    """
    Configuration class for training hyperparameters and settings.
    """

    # Model parameters
    num_classes: int = 10
    dropout_rate: float = 0.5

    # Data parameters
    data_dir: str = './data'
    batch_size: int = 64  # Reduced for CPU efficiency
    val_split: float = 0.1
    num_workers: int = 2  # Reduced for CPU to avoid overhead

    # Training parameters
    num_epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD optimizer

    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_step_size: int = 7
    scheduler_gamma: float = 0.1

    # Optimizer choice: 'adam' or 'sgd'
    optimizer_type: str = 'adam'

    # Early stopping
    early_stopping_patience: Optional[int] = 5

    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_best_only: bool = True

    # Device - Auto-detect best available (CUDA > MPS > CPU)
    device: str = field(default_factory=get_best_device)

    # Random seed for reproducibility
    random_seed: int = 42

    # Logging
    log_interval: int = 50  # Print training stats every N batches

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.num_classes > 0, "num_classes must be positive"
        assert 0 < self.dropout_rate < 1, "dropout_rate must be between 0 and 1"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0 < self.val_split < 1, "val_split must be between 0 and 1"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.optimizer_type in ['adam', 'sgd'], "optimizer_type must be 'adam' or 'sgd'"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'data_dir': self.data_dir,
            'batch_size': self.batch_size,
            'val_split': self.val_split,
            'num_workers': self.num_workers,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'use_scheduler': self.use_scheduler,
            'scheduler_step_size': self.scheduler_step_size,
            'scheduler_gamma': self.scheduler_gamma,
            'optimizer_type': self.optimizer_type,
            'early_stopping_patience': self.early_stopping_patience,
            'checkpoint_dir': self.checkpoint_dir,
            'save_best_only': self.save_best_only,
            'device': self.device,
            'random_seed': self.random_seed,
            'log_interval': self.log_interval
        }

    def __repr__(self) -> str:
        """String representation of config."""
        lines = ["Configuration:"]
        lines.append("=" * 60)
        for key, value in self.to_dict().items():
            lines.append(f"  {key:30s}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)


def get_default_config() -> Config:
    """
    Get default configuration.

    Returns:
        Config object with default settings
    """
    return Config()


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA-specific seeding
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # MPS-specific seeding (Apple Silicon)
    if torch.backends.mps.is_available():
        # MPS uses the same manual_seed as CPU
        # No additional configuration needed for MPS determinism
        pass
