"""
Visualization script for MNIST ConvNet architecture.
Shows the complete data flow through the network layers.
"""
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

from mnist_classifier.core.model import MNISTConvNet


class ModelVisualizer:
    """
    Visualizes the CNN model's internal representations and data flow.
    """

    def __init__(self, model, device):
        """
        Initialize the visualizer.

        Args:
            model: MNISTConvNet model
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.activations = {}

    def get_activations(self, x):
        """
        Extract activations from all layers during forward pass.

        Args:
            x: Input tensor (1, 1, 28, 28)

        Returns:
            Dictionary containing all intermediate activations
        """
        activations = {}

        # Input
        activations['input'] = x.clone()

        # First conv block: 28x28x1 -> 28x28x32 -> 14x14x32
        conv1_out = self.model.conv1(x)
        activations['conv1'] = conv1_out.clone()

        bn1_out = self.model.bn1(conv1_out)
        activations['bn1'] = bn1_out.clone()

        relu1_out = F.relu(bn1_out)
        activations['relu1'] = relu1_out.clone()

        pool1_out = self.model.pool(relu1_out)
        activations['pool1'] = pool1_out.clone()

        # Second conv block: 14x14x32 -> 14x14x64 -> 7x7x64
        conv2_out = self.model.conv2(pool1_out)
        activations['conv2'] = conv2_out.clone()

        bn2_out = self.model.bn2(conv2_out)
        activations['bn2'] = bn2_out.clone()

        relu2_out = F.relu(bn2_out)
        activations['relu2'] = relu2_out.clone()

        pool2_out = self.model.pool(relu2_out)
        activations['pool2'] = pool2_out.clone()

        # Third conv block: 7x7x64 -> 7x7x128 -> 3x3x128
        conv3_out = self.model.conv3(pool2_out)
        activations['conv3'] = conv3_out.clone()

        bn3_out = self.model.bn3(conv3_out)
        activations['bn3'] = bn3_out.clone()

        relu3_out = F.relu(bn3_out)
        activations['relu3'] = relu3_out.clone()

        pool3_out = self.model.pool(relu3_out)
        activations['pool3'] = pool3_out.clone()

        # Flatten
        flattened = pool3_out.view(pool3_out.size(0), -1)
        activations['flatten'] = flattened.clone()

        # FC1
        fc1_out = self.model.fc1(flattened)
        activations['fc1'] = fc1_out.clone()

        relu_fc1 = F.relu(fc1_out)
        activations['relu_fc1'] = relu_fc1.clone()

        dropout_out = self.model.dropout(relu_fc1)
        activations['dropout'] = dropout_out.clone()

        # FC2 (logits)
        logits = self.model.fc2(dropout_out)
        activations['logits'] = logits.clone()

        # Probabilities
        probabilities = torch.softmax(logits, dim=1)
        activations['probabilities'] = probabilities.clone()

        # Prediction
        prediction = torch.argmax(probabilities, dim=1)
        activations['prediction'] = prediction.clone()

        return activations

    def visualize_feature_maps(self, feature_maps, title, num_maps=16, save_path=None):
        """
        Visualize feature maps from a convolutional layer.

        Args:
            feature_maps: Tensor of shape (1, C, H, W)
            title: Title for the plot
            num_maps: Number of feature maps to display
            save_path: Path to save the plot (optional)
        """
        feature_maps = feature_maps.squeeze(0).cpu().detach().numpy()
        num_channels = feature_maps.shape[0]
        num_maps = min(num_maps, num_channels)

        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(num_maps)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle(f'{title}\nShape: {feature_maps.shape}', fontsize=16, fontweight='bold')

        for idx in range(grid_size * grid_size):
            row = idx // grid_size
            col = idx % grid_size
            ax = axes[row, col] if grid_size > 1 else axes

            if idx < num_maps:
                feature_map = feature_maps[idx]
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'Filter {idx}', fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Feature maps saved to: {save_path}')
        else:
            plt.show()

        plt.close()

    def visualize_complete_flow(self, image, label, save_path=None):
        """
        Visualize the complete data flow through the network.
        Creates separate images for each processing step.

        Args:
            image: Input image tensor (1, 28, 28)
            label: Ground truth label
            save_path: Directory path to save the plots (if None, displays instead)
        """
        import os

        # Prepare input
        image_input = image.unsqueeze(0).to(self.device)  # (1, 1, 28, 28)

        # Get all activations
        activations = self.get_activations(image_input)

        # Create output directory if save_path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            print(f'\nSaving visualizations to: {save_path}/')

        # Step 0: Show training dataset structure (before processing individual image)
        self._plot_training_dataset_structure(save_path, '00a_training_dataset.png')

        # Step 1: Input Image with pixel coordinates
        fig, ax = plt.subplots(1, 1, figsize=(14, 14))
        img = image.squeeze().cpu().numpy()

        # Display image
        im = ax.imshow(img, cmap='gray', interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Pixel Value', rotation=270, labelpad=20, fontsize=12)

        # Set up the grid
        ax.set_xticks(np.arange(28))
        ax.set_yticks(np.arange(28))
        ax.set_xticklabels(np.arange(28), fontsize=7)
        ax.set_yticklabels(np.arange(28), fontsize=7)

        # Add grid lines
        ax.set_xticks(np.arange(28) - 0.5, minor=True)
        ax.set_yticks(np.arange(28) - 0.5, minor=True)
        ax.grid(which='minor', color='red', linestyle='-', linewidth=0.5, alpha=0.3)

        # Add major grid every 4 pixels for easier reading
        ax.set_xticks(np.arange(0, 28, 4), minor=False)
        ax.set_yticks(np.arange(0, 28, 4), minor=False)
        ax.grid(which='major', color='blue', linestyle='-', linewidth=1.5, alpha=0.5)

        # Labels
        ax.set_xlabel('X Coordinate (Width)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Coordinate (Height)', fontsize=12, fontweight='bold')
        ax.set_title(f'Step 0b: Input Image with Pixel Coordinates\nTrue Label: {label} | Shape: (1, 1, 28, 28)',
                     fontsize=14, fontweight='bold', pad=20)

        # Add pixel value annotations for a subset (every 4th pixel to avoid clutter)
        for i in range(0, 28, 4):
            for j in range(0, 28, 4):
                text_color = 'white' if img[i, j] < 0.5 else 'black'
                ax.text(j, i, f'{img[i, j]:.2f}',
                       ha='center', va='center',
                       color=text_color, fontsize=6, fontweight='bold')

        self._save_or_show(fig, save_path, '00b_input.png')

        # Step 2: Conv1 Output
        self._plot_feature_map_grid(activations['conv1'],
                                    'Step 1: Conv1 Output\n32 filters, 28x28, after 3x3 convolution',
                                    save_path, '01_conv1.png')

        # Step 3: After BatchNorm1
        self._plot_feature_map_grid(activations['bn1'],
                                    'Step 2: After BatchNorm1\n32 filters, 28x28, normalized',
                                    save_path, '02_batchnorm1.png')

        # Step 4: After ReLU1
        self._plot_feature_map_grid(activations['relu1'],
                                    'Step 3: After ReLU1\n32 filters, 28x28, ReLU activation',
                                    save_path, '03_relu1.png')

        # Step 5: After MaxPool1
        self._plot_feature_map_grid(activations['pool1'],
                                    'Step 4: After MaxPool1\n32 filters, 14x14, 2x2 max pooling',
                                    save_path, '04_pool1.png')

        # Step 6: Conv2 Output
        self._plot_feature_map_grid(activations['conv2'],
                                    'Step 5: Conv2 Output\n64 filters, 14x14, after 3x3 convolution',
                                    save_path, '05_conv2.png')

        # Step 7: After BatchNorm2
        self._plot_feature_map_grid(activations['bn2'],
                                    'Step 6: After BatchNorm2\n64 filters, 14x14, normalized',
                                    save_path, '06_batchnorm2.png')

        # Step 8: After ReLU2
        self._plot_feature_map_grid(activations['relu2'],
                                    'Step 7: After ReLU2\n64 filters, 14x14, ReLU activation',
                                    save_path, '07_relu2.png')

        # Step 9: After MaxPool2
        self._plot_feature_map_grid(activations['pool2'],
                                    'Step 8: After MaxPool2\n64 filters, 7x7, 2x2 max pooling',
                                    save_path, '08_pool2.png')

        # Step 10: Conv3 Output
        self._plot_feature_map_grid(activations['conv3'],
                                    'Step 9: Conv3 Output\n128 filters, 7x7, after 3x3 convolution',
                                    save_path, '09_conv3.png')

        # Step 11: After BatchNorm3
        self._plot_feature_map_grid(activations['bn3'],
                                    'Step 10: After BatchNorm3\n128 filters, 7x7, normalized',
                                    save_path, '10_batchnorm3.png')

        # Step 12: After ReLU3
        self._plot_feature_map_grid(activations['relu3'],
                                    'Step 11: After ReLU3\n128 filters, 7x7, ReLU activation',
                                    save_path, '11_relu3.png')

        # Step 13: After MaxPool3
        self._plot_feature_map_grid(activations['pool3'],
                                    'Step 12: After MaxPool3\n128 filters, 3x3, 2x2 max pooling',
                                    save_path, '12_pool3.png')

        # Step 14: Flattened Vector
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        flatten_data = activations['flatten'].squeeze().cpu().detach().numpy()
        ax.bar(range(len(flatten_data)), flatten_data, width=1.0)
        ax.set_title(f'Step 13: Flattened Vector\n{len(flatten_data)} values (128 × 3 × 3)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        self._save_or_show(fig, save_path, '13_flatten.png')

        # Step 15: FC1 Output
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        fc1_data = activations['fc1'].squeeze().cpu().detach().numpy()
        ax.bar(range(len(fc1_data)), fc1_data)
        ax.set_title(f'Step 14: FC1 Output\n256 neurons, fully connected layer',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Neuron Index', fontsize=12)
        ax.set_ylabel('Activation Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        self._save_or_show(fig, save_path, '14_fc1.png')

        # Step 16: After ReLU (FC1)
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        relu_fc1_data = activations['relu_fc1'].squeeze().cpu().detach().numpy()
        ax.bar(range(len(relu_fc1_data)), relu_fc1_data)
        ax.set_title(f'Step 15: After ReLU\n256 neurons, non-negative activations',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Neuron Index', fontsize=12)
        ax.set_ylabel('Activation Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        self._save_or_show(fig, save_path, '15_relu_fc1.png')

        # Step 17: After Dropout
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        dropout_data = activations['dropout'].squeeze().cpu().detach().numpy()
        ax.bar(range(len(dropout_data)), dropout_data)
        ax.set_title(f'Step 16: After Dropout\n256 neurons, dropout regularization (rate=0.5)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Neuron Index', fontsize=12)
        ax.set_ylabel('Activation Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        self._save_or_show(fig, save_path, '16_dropout.png')

        # Step 18: Logits (FC2 Output)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        logits = activations['logits'].squeeze().cpu().detach().numpy()
        colors = ['green' if i == label else 'blue' for i in range(10)]
        bars = ax.bar(range(10), logits, color=colors)
        ax.set_title('Step 17: Logits (Raw Output from FC2)\n10 output neurons, one per class',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Class (Digit)', fontsize=12)
        ax.set_ylabel('Logit Value', fontsize=12)
        ax.set_xticks(range(10))
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
        ax.legend([bars[label]], ['True Label'], loc='upper right')
        self._save_or_show(fig, save_path, '17_logits.png')

        # Step 19: Probabilities (After Softmax)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        probs = activations['probabilities'].squeeze().cpu().detach().numpy()
        colors = ['green' if i == label else 'lightblue' for i in range(10)]
        bars = ax.bar(range(10), probs * 100, color=colors)
        ax.set_title('Step 18: Probabilities (After Softmax)\nConverted logits to probabilities',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Class (Digit)', fontsize=12)
        ax.set_ylabel('Probability (%)', fontsize=12)
        ax.set_xticks(range(10))
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend([bars[label]], ['True Label'], loc='upper right')
        # Add probability values on top of bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            if height > 1:  # Only show if > 1%
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob*100:.1f}%',
                       ha='center', va='bottom', fontsize=9)
        self._save_or_show(fig, save_path, '18_probabilities.png')

        # Step 19: Training - How Labels Drive Learning
        self._plot_training_explanation(probs, activations['logits'].squeeze().cpu().detach().numpy(),
                                        label, save_path, '19_learning_from_labels.png')

        # Step 21: Final Prediction Summary
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        prediction = activations['prediction'].item()
        confidence = probs[prediction] * 100

        # Create prediction summary text
        summary_text = f"""FINAL PREDICTION
{'=' * 70}

Ground Truth Label:  {label}
Predicted Label:     {prediction}
Confidence:          {confidence:.2f}%

Result:  {'✓ CORRECT PREDICTION' if prediction == label else '✗ INCORRECT PREDICTION'}

{'=' * 70}

Top 5 Predictions:
"""

        top5_indices = np.argsort(probs)[::-1][:5]
        for i, idx in enumerate(top5_indices, 1):
            marker = '→ ' if idx == prediction else '  '
            check = '✓' if idx == label else ' '
            summary_text += f"\n{marker}#{i}  Digit {idx}  |  {probs[idx]*100:6.2f}%  {check}"

        summary_text += f"\n\n{'=' * 70}\n"
        summary_text += f"\nModel Pipeline Summary:\n"
        summary_text += f"  Input: 28×28 grayscale image\n"
        summary_text += f"  → Conv Block 1: 32 filters → 14×14\n"
        summary_text += f"  → Conv Block 2: 64 filters → 7×7\n"
        summary_text += f"  → Conv Block 3: 128 filters → 3×3\n"
        summary_text += f"  → Flatten: 1152 values\n"
        summary_text += f"  → FC1: 256 neurons\n"
        summary_text += f"  → FC2: 10 output classes\n"
        summary_text += f"  → Softmax: Probabilities\n"
        summary_text += f"  → Prediction: argmax\n"

        ax.text(0.5, 0.5, summary_text,
                fontsize=11,
                family='monospace',
                ha='center',
                va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        ax.set_title('Step 20: Final Prediction Summary', fontsize=16, fontweight='bold', pad=20)
        self._save_or_show(fig, save_path, '20_prediction_summary.png')

        if save_path:
            print(f'\n✓ All visualization steps saved to: {save_path}/')
            print(f'  - Training dataset structure')
            print(f'  - Forward pass activations (00-18)')
            print(f'  - Training process explanation (19)')
            print(f'  - Final prediction summary (20)')

    def _plot_feature_map_grid(self, feature_maps, title, save_path, filename):
        """
        Plot feature maps in a grid and save/show.

        Args:
            feature_maps: Tensor of shape (1, C, H, W)
            title: Title for the plot
            save_path: Directory to save (None to show)
            filename: Filename for saving
        """
        feature_maps = feature_maps.squeeze(0).cpu().detach().numpy()
        num_channels = feature_maps.shape[0]

        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(num_channels)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
        fig.suptitle(f'{title}\nTotal Filters: {num_channels}', fontsize=14, fontweight='bold')

        if grid_size == 1:
            axes = np.array([[axes]])
        elif len(axes.shape) == 1:
            axes = axes.reshape(-1, 1)

        for idx in range(grid_size * grid_size):
            row = idx // grid_size
            col = idx % grid_size
            ax = axes[row, col]

            if idx < num_channels:
                feature_map = feature_maps[idx]
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'Filter {idx}', fontsize=8)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis('off')

        plt.tight_layout()
        self._save_or_show(fig, save_path, filename)

    def _save_or_show(self, fig, save_path, filename):
        """Save figure to file or show it."""
        if save_path:
            import os
            full_path = os.path.join(save_path, filename)
            fig.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f'  ✓ {filename}')
        else:
            plt.show()
        plt.close(fig)

    def _plot_training_dataset_structure(self, save_path, filename):
        """
        Visualize how a training dataset is structured with images and labels.
        Shows that images go through the network, then labels are used to improve it.

        Args:
            save_path: Directory to save (None to show)
            filename: Filename for saving
        """
        # Load 3 sample images from the dataset for demonstration
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)

        # Get 3 different examples with different labels
        sample_indices = [0, 12, 25]  # These typically give different digits
        samples = [dataset[idx] for idx in sample_indices]

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 1, hspace=0.4, height_ratios=[1.5, 1.5])

        # Section 1: Training Dataset Structure
        ax1 = fig.add_subplot(gs[0])
        ax1.axis('off')

        # Display 3 sample images with their labels
        inner_gs1 = gs[0].subgridspec(1, 3, wspace=0.3)

        for idx, (image, label) in enumerate(samples):
            ax_img = fig.add_subplot(inner_gs1[idx])
            img = image.squeeze().cpu().numpy()
            ax_img.imshow(img, cmap='gray')
            ax_img.set_title(f'Training Example {idx+1}\nLabel: {label}',
                           fontsize=14, fontweight='bold', pad=10)
            ax_img.axis('off')

            # Add label box below
            ax_img.text(0.5, -0.15, f'Label = {label}',
                       transform=ax_img.transAxes,
                       ha='center', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#2ecc71',
                                edgecolor='black', linewidth=2))

        # Add title for this section
        fig.text(0.5, 0.95, 'TRAINING DATASET STRUCTURE',
                ha='center', fontsize=18, fontweight='bold')

        # Section 2: Visual Flow Diagram
        ax3 = fig.add_subplot(gs[1])
        ax3.axis('off')
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 4)

        # Draw the flow
        # Image box
        rect1 = plt.Rectangle((0.5, 2.5), 1.5, 1, facecolor='#3498db', edgecolor='black', linewidth=2)
        ax3.add_patch(rect1)
        ax3.text(1.25, 3, 'IMAGE', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

        # Arrow to network
        ax3.annotate('', xy=(2.5, 3), xytext=(2, 3),
                    arrowprops=dict(arrowstyle='->', lw=3, color='black'))

        # Network box
        rect2 = plt.Rectangle((2.5, 2), 2, 2, facecolor='#9b59b6', edgecolor='black', linewidth=2)
        ax3.add_patch(rect2)
        ax3.text(3.5, 3, 'NEURAL\nNETWORK\n(Forward Pass)', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # Arrow to prediction
        ax3.annotate('', xy=(5, 3), xytext=(4.5, 3),
                    arrowprops=dict(arrowstyle='->', lw=3, color='black'))

        # Prediction box
        rect3 = plt.Rectangle((5, 2.5), 1.5, 1, facecolor='#e67e22', edgecolor='black', linewidth=2)
        ax3.add_patch(rect3)
        ax3.text(5.75, 3, 'PREDICTION', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Label box (separate, below)
        rect4 = plt.Rectangle((0.5, 0.5), 1.5, 1, facecolor='#2ecc71', edgecolor='black', linewidth=2)
        ax3.add_patch(rect4)
        ax3.text(1.25, 1, 'LABEL\n(Target)', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        # Arrow from label to loss
        ax3.annotate('', xy=(5.75, 2.2), xytext=(1.25, 1.5),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#2ecc71', linestyle='dashed'))

        # Loss comparison box
        rect5 = plt.Rectangle((5, 1.2), 1.5, 0.8, facecolor='#e74c3c', edgecolor='black', linewidth=2)
        ax3.add_patch(rect5)
        ax3.text(5.75, 1.6, 'COMPUTE\nLOSS', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Arrow back to network (backpropagation)
        ax3.annotate('', xy=(3.5, 1.9), xytext=(5.5, 1.2),
                    arrowprops=dict(arrowstyle='->', lw=3, color='#e74c3c'))
        ax3.text(4.5, 1.2, 'Backprop\n(Update Weights)', ha='center', va='top',
                fontsize=10, fontweight='bold', color='#e74c3c')

        # Add annotations
        ax3.text(1.25, 2.3, 'Only image\ngoes through', ha='center', va='top',
                fontsize=9, style='italic', color='#3498db')
        ax3.text(1.25, 0.2, 'Label waits here\nuntil needed', ha='center', va='top',
                fontsize=9, style='italic', color='#2ecc71')
        ax3.text(5.75, 0.4, 'Label is used HERE\nto compute error', ha='center', va='top',
                fontsize=9, style='italic', color='#e74c3c')

        plt.tight_layout()
        self._save_or_show(fig, save_path, filename)

    def _plot_training_explanation(self, probs, logits, true_label, save_path, filename):
        """
        Visualize label vs prediction - visual only version.

        Args:
            probs: Predicted probabilities (10,)
            logits: Raw logits (10,)
            true_label: Ground truth label (int)
            save_path: Directory to save (None to show)
            filename: Filename for saving
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        predicted_class = np.argmax(probs)

        # 1. Label (Target)
        target = np.zeros(10)
        target[true_label] = 1.0
        colors_target = ['#2ecc71' if i == true_label else '#ecf0f1' for i in range(10)]
        bars1 = ax1.bar(range(10), target, color=colors_target, edgecolor='black', linewidth=2)
        ax1.set_xlabel('Digit Class', fontsize=11)
        ax1.set_ylabel('Target', fontsize=11)
        ax1.set_xticks(range(10))
        ax1.set_ylim([0, 1.3])
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Model Prediction
        colors_pred = ['#2ecc71' if i == true_label else '#e74c3c' if i == predicted_class else '#3498db'
                      for i in range(10)]
        bars2 = ax2.bar(range(10), probs, color=colors_pred, edgecolor='black', linewidth=1)
        ax2.set_xlabel('Digit Class', fontsize=11)
        ax2.set_ylabel('Probability', fontsize=11)
        ax2.set_xticks(range(10))
        ax2.set_ylim([0, 1.3])
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save_or_show(fig, save_path, filename)

    def _plot_feature_map_summary(self, feature_maps, ax, title):
        """
        Plot a summary of feature maps (first 8 channels).

        Args:
            feature_maps: Tensor of shape (1, C, H, W)
            ax: Matplotlib axis
            title: Title for the plot
        """
        feature_maps = feature_maps.squeeze(0).cpu().detach().numpy()
        num_channels = min(8, feature_maps.shape[0])

        # Create a grid showing first 8 feature maps
        grid_h = 2
        grid_w = 4

        combined = np.zeros((feature_maps.shape[1] * grid_h, feature_maps.shape[2] * grid_w))

        for idx in range(num_channels):
            row = idx // grid_w
            col = idx % grid_w
            h, w = feature_maps.shape[1], feature_maps.shape[2]
            combined[row*h:(row+1)*h, col*w:(col+1)*w] = feature_maps[idx]

        im = ax.imshow(combined, cmap='viridis')
        ax.set_title(f'{title}\nShape: {feature_maps.shape}\n(Showing first 8 filters)',
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def visualize_conv_layers_detailed(self, image, label, save_dir='./visualizations'):
        """
        Create detailed visualizations for each convolutional layer.

        Args:
            image: Input image tensor (1, 28, 28)
            label: Ground truth label
            save_dir: Directory to save visualizations
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Prepare input
        image_input = image.unsqueeze(0).to(self.device)

        # Get activations
        activations = self.get_activations(image_input)

        # Visualize input
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(image.squeeze().cpu().numpy(), cmap='gray')
        ax.set_title(f'Input Image (Label: {label})', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.savefig(f'{save_dir}/00_input.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Visualize each conv layer
        self.visualize_feature_maps(
            activations['conv1'],
            'Conv Layer 1 (32 filters, 28x28)',
            num_maps=32,
            save_path=f'{save_dir}/01_conv1.png'
        )

        self.visualize_feature_maps(
            activations['pool1'],
            'After MaxPool 1 (32 filters, 14x14)',
            num_maps=32,
            save_path=f'{save_dir}/02_pool1.png'
        )

        self.visualize_feature_maps(
            activations['conv2'],
            'Conv Layer 2 (64 filters, 14x14)',
            num_maps=64,
            save_path=f'{save_dir}/03_conv2.png'
        )

        self.visualize_feature_maps(
            activations['pool2'],
            'After MaxPool 2 (64 filters, 7x7)',
            num_maps=64,
            save_path=f'{save_dir}/04_pool2.png'
        )

        self.visualize_feature_maps(
            activations['conv3'],
            'Conv Layer 3 (128 filters, 7x7)',
            num_maps=128,
            save_path=f'{save_dir}/05_conv3.png'
        )

        self.visualize_feature_maps(
            activations['pool3'],
            'After MaxPool 3 (128 filters, 3x3)',
            num_maps=128,
            save_path=f'{save_dir}/06_pool3.png'
        )

        print(f'\nDetailed visualizations saved to: {save_dir}/')

    def compare_forward_vs_predict(self, image, label):
        """
        Compare forward() vs predict() methods.

        Args:
            image: Input image tensor (1, 28, 28)
            label: Ground truth label
        """
        image_input = image.unsqueeze(0).to(self.device)

        print("\n" + "=" * 80)
        print("COMPARING forward() vs predict() METHODS")
        print("=" * 80)

        with torch.no_grad():
            # Using forward()
            logits = self.model.forward(image_input)
            probabilities = torch.softmax(logits, dim=1)
            pred_from_forward = torch.argmax(probabilities, dim=1)

            # Using predict()
            pred_from_predict = self.model.predict(image_input)

            print(f"\nGround Truth: {label}")
            print(f"\nforward() method:")
            print(f"  - Returns: Logits (raw scores)")
            print(f"  - Logits shape: {logits.shape}")
            print(f"  - Logits values: {logits.squeeze().cpu().numpy()}")
            print(f"  - After softmax: {probabilities.squeeze().cpu().numpy()}")
            print(f"  - Prediction (argmax): {pred_from_forward.item()}")

            print(f"\npredict() method:")
            print(f"  - Returns: Class prediction directly")
            print(f"  - Prediction shape: {pred_from_predict.shape}")
            print(f"  - Prediction value: {pred_from_predict.item()}")

            print(f"\nAre predictions equal? {pred_from_forward.item() == pred_from_predict.item()}")
            print(f"Match ground truth? {pred_from_predict.item() == label}")

        print("=" * 80)


def load_sample_image(data_dir='./data', sample_idx=0):
    """
    Load a sample image from MNIST dataset.

    Args:
        data_dir: Directory containing MNIST data
        sample_idx: Index of sample to load

    Returns:
        Tuple of (image, label)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    image, label = dataset[sample_idx]

    return image, label


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize MNIST ConvNet Architecture')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional, will use random weights if not provided)')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for MNIST data')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Index of sample to visualize (default: 0)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default='./visualizations_output',
                       help='Directory to save step-by-step visualizations (activations + weights)')
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed layer-by-layer visualizations')
    parser.add_argument('--detailed-dir', type=str, default='./visualizations',
                       help='Directory for detailed visualizations')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    print(f'Using device: {device}')

    # Load model
    model = MNISTConvNet()

    if args.checkpoint:
        print(f'\nLoading model from checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Model loaded successfully')
    else:
        print('\nNo checkpoint provided - using randomly initialized weights')
        print('Note: Predictions will be random. Use --checkpoint for trained model.')

    # Load sample image
    print(f'\nLoading sample image (index: {args.sample_idx})...')
    image, label = load_sample_image(args.data_dir, args.sample_idx)
    print(f'Image shape: {image.shape}')
    print(f'Label: {label}')

    # Create visualizer
    visualizer = ModelVisualizer(model, device)

    # Compare forward vs predict
    visualizer.compare_forward_vs_predict(image, label)

    # Visualize complete flow (step-by-step)
    print(f'\nGenerating step-by-step visualizations (activations + weights)...')
    visualizer.visualize_complete_flow(image, label, save_path=args.output)

    # Detailed visualizations if requested
    if args.detailed:
        print(f'\nGenerating detailed layer-by-layer visualizations (showing all feature maps)...')
        visualizer.visualize_conv_layers_detailed(image, label, save_dir=args.detailed_dir)

    print('\n' + '=' * 80)
    print('Visualization completed successfully!')
    if args.output:
        print(f'Step-by-step visualizations saved to: {args.output}/')
    if args.detailed:
        print(f'Detailed layer visualizations saved to: {args.detailed_dir}/')
    print('=' * 80)


if __name__ == '__main__':
    main()
