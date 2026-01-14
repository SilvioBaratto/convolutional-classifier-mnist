"""
Data loading utilities for MNIST dataset with augmentation.
"""
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_transforms(train: bool = True) -> transforms.Compose:
    """
    Get data transforms for MNIST dataset.

    Args:
        train: Whether to apply training augmentations

    Returns:
        Composed transforms
    """
    if train:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.RandomRotation(10),  # Random rotation up to 10 degrees
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Random translation
                scale=(0.9, 1.1)  # Random scaling
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    return transform


def get_data_loaders(
    data_dir: str = './data',
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 2,
    pin_memory: bool = False,  # Disabled for CPU
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get MNIST data loaders for training, validation, and testing.

    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform = get_mnist_transforms(train=True)
    test_transform = get_mnist_transforms(train=False)

    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    # Download and load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # Split training data into train and validation sets
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # Use generator for reproducibility
    generator = torch.Generator().manual_seed(random_seed)
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader, test_loader


def get_dataset_info(train_loader: DataLoader) -> dict:
    """
    Get information about the dataset.

    Args:
        train_loader: Training data loader

    Returns:
        Dictionary containing dataset information
    """
    dataset = train_loader.dataset
    if hasattr(dataset, 'dataset'):
        # Handle Subset case
        dataset = dataset.dataset

    info = {
        'num_samples': len(train_loader.dataset),
        'num_batches': len(train_loader),
        'batch_size': train_loader.batch_size,
        'num_classes': len(dataset.classes) if hasattr(dataset, 'classes') else 10,
        'input_shape': tuple(dataset[0][0].shape)
    }

    return info
