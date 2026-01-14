"""
Convolutional Neural Network model for MNIST classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTConvNet(nn.Module):
    """
    Modern CNN architecture for MNIST digit classification.

    Architecture:
        - 3 Convolutional blocks with BatchNorm and MaxPooling
        - Dropout for regularization
        - 2 Fully connected layers
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        """
        Initialize the CNN model.

        Args:
            num_classes: Number of output classes (10 for MNIST)
            dropout_rate: Dropout probability for regularization
        """
        super(MNISTConvNet, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        # After 3 pooling layers: 28x28 -> 14x14 -> 7x7 -> 3x3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First conv block: 28x28x1 -> 28x28x32 -> 14x14x32
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Second conv block: 14x14x32 -> 14x14x64 -> 7x7x64
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Third conv block: 7x7x64 -> 7x7x128 -> 3x3x128
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten: 3x3x128 -> 1152
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions from input.

        Args:
            x: Input tensor

        Returns:
            Predicted class indices
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
