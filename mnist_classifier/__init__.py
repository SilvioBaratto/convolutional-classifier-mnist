"""
MNIST Classifier - A PyTorch CNN for digit classification with visualization tools.
"""

from mnist_classifier.core.model import MNISTConvNet, count_parameters
from mnist_classifier.core.config import Config, get_default_config, set_random_seed
from mnist_classifier.core.data_loader import get_data_loaders, get_mnist_transforms, get_dataset_info
from mnist_classifier.core.trainer import Trainer, evaluate_model

__version__ = '1.0.0'

__all__ = [
    'MNISTConvNet',
    'count_parameters',
    'Config',
    'get_default_config',
    'set_random_seed',
    'get_data_loaders',
    'get_mnist_transforms',
    'get_dataset_info',
    'Trainer',
    'evaluate_model',
]
