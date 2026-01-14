"""
Utility functions for MNIST classifier.
"""
import os
import json
import torch
import matplotlib.pyplot as plt
from typing import Dict, List


def plot_training_history(history: Dict[str, List], save_path: str = None):
    """
    Plot training history (loss and accuracy curves).

    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Training history plot saved to: {save_path}')
    else:
        plt.show()

    plt.close()


def load_checkpoint_history(checkpoint_path: str) -> Dict[str, List]:
    """
    Load training history from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Training history dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint.get('history', {})


def save_history_to_json(history: Dict[str, List], save_path: str):
    """
    Save training history to JSON file.

    Args:
        history: Training history dictionary
        save_path: Path to save JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training history saved to: {save_path}')


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 1, 28, 28)):
    """
    Print model summary.

    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    from torchinfo import summary
    summary(model, input_size=input_size, col_names=['input_size', 'output_size', 'num_params'])


def get_misclassified_samples(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 10
) -> List[Dict]:
    """
    Get misclassified samples from the dataset.

    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to run inference on
        num_samples: Number of misclassified samples to return

    Returns:
        List of dictionaries containing misclassified samples
    """
    model.eval()
    misclassified = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)

            # Find misclassified samples
            incorrect = predicted.ne(target)
            incorrect_indices = incorrect.nonzero(as_tuple=True)[0]

            for idx in incorrect_indices:
                if len(misclassified) >= num_samples:
                    return misclassified

                misclassified.append({
                    'image': data[idx].cpu(),
                    'true_label': target[idx].item(),
                    'predicted_label': predicted[idx].item(),
                    'probabilities': torch.softmax(output[idx], dim=0).cpu().numpy()
                })

    return misclassified


def visualize_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 16,
    save_path: str = None
):
    """
    Visualize model predictions on random samples.

    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Path to save the plot (optional)
    """
    model.eval()

    # Get a batch of data
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)

    # Make predictions
    with torch.no_grad():
        output = model(data)
        _, predicted = output.max(1)

    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()

    for i in range(min(num_samples, len(data))):
        img = data[i].cpu().squeeze()
        true_label = target[i].item()
        pred_label = predicted[i].item()

        axes[i].imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}', color=color, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Predictions plot saved to: {save_path}')
    else:
        plt.show()

    plt.close()


def analyze_class_performance(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10
) -> Dict[int, Dict[str, float]]:
    """
    Analyze per-class performance metrics.

    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to run inference on
        num_classes: Number of classes

    Returns:
        Dictionary containing per-class metrics
    """
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)

            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    # Calculate metrics
    results = {}
    for i in range(num_classes):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        results[i] = {
            'accuracy': accuracy,
            'correct': class_correct[i],
            'total': class_total[i]
        }

    return results


def print_class_performance(class_metrics: Dict[int, Dict[str, float]]):
    """
    Print per-class performance metrics.

    Args:
        class_metrics: Dictionary containing per-class metrics
    """
    print('\n' + '=' * 70)
    print('Per-Class Performance Analysis')
    print('=' * 70)
    print(f'{"Class":>8} {"Correct":>10} {"Total":>10} {"Accuracy":>12}')
    print('-' * 70)

    for class_id, metrics in sorted(class_metrics.items()):
        print(f'{class_id:>8} {metrics["correct"]:>10} {metrics["total"]:>10} '
              f'{metrics["accuracy"]:>11.2f}%')

    print('=' * 70)
