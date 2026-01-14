"""
Gradient visualization utilities for tracking how gradients change during training.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Dict, List, Optional


class GradientVisualizer:
    """
    Visualizer for tracking and saving gradient information during training.
    Designed to create frames for GIF creation.
    """

    def __init__(self, model: nn.Module, save_dir: str = './gradient_frames'):
        """
        Initialize the gradient visualizer.

        Args:
            model: The neural network model
            save_dir: Directory to save gradient visualization frames
        """
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Track gradient statistics over time
        self.frame_count = 0

        # Layer names to track
        self.layer_names = []
        self._collect_layer_names()

        # History tracking for time-series visualization
        self.gradient_history = {name: [] for name in self.layer_names}
        self.loss_history = []
        self.batch_indices = []

    def _collect_layer_names(self):
        """Collect names of layers with gradients to track."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'weight' in name:  # Only track weights, not biases
                # Simplify names for better readability
                simplified_name = name.replace('model.', '').replace('.weight', '')
                self.layer_names.append(simplified_name)

    def collect_gradients(self) -> Dict[str, float]:
        """
        Collect current gradient statistics from the model.

        Returns:
            Dictionary mapping layer names to mean gradient magnitude
        """
        gradients = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None and 'weight' in name:
                grad = param.grad.detach().cpu().numpy()
                simplified_name = name.replace('model.', '').replace('.weight', '')

                # Calculate mean absolute gradient (most important metric)
                gradients[simplified_name] = np.mean(np.abs(grad))

        return gradients

    def visualize_gradients(
        self,
        epoch: int,
        batch_idx: int,
        loss: float,
        learning_rate: float,
        save: bool = True
    ):
        """
        Create simple 2D line plot showing gradient evolution over time.

        Args:
            epoch: Current epoch number
            batch_idx: Current batch index
            loss: Current loss value
            learning_rate: Current learning rate
            save: Whether to save the visualization
        """
        # Collect current gradients
        gradients = self.collect_gradients()

        if not gradients:
            return

        # Store in history
        for name in self.layer_names:
            if name in gradients:
                self.gradient_history[name].append(gradients[name])
            else:
                self.gradient_history[name].append(0.0)

        self.loss_history.append(loss)
        self.batch_indices.append(self.frame_count)

        # Create simple, clean visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Define colors for each layer
        colors = {
            'conv1.0': '#3498db',      # Blue
            'conv2.0': '#2980b9',      # Darker blue
            'conv3.0': '#1f618d',      # Even darker blue
            'fc1': '#e74c3c',          # Red
            'fc2': '#c0392b'           # Darker red
        }

        # Plot 1: Gradient magnitude over time (LINE PLOT)
        ax1.set_title(f'Gradient Evolution During Training (Epoch {epoch}, Batch {batch_idx})',
                     fontsize=14, fontweight='bold', pad=15)

        for name in self.layer_names:
            color = colors.get(name, '#95a5a6')
            ax1.plot(self.batch_indices, self.gradient_history[name],
                    label=name, color=color, linewidth=2.5, marker='o', markersize=4)

        ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Gradient Magnitude (Mean Absolute)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(bottom=0)

        # Plot 2: Loss over time
        ax2.plot(self.batch_indices, self.loss_history,
                color='#e67e22', linewidth=2.5, marker='o', markersize=4, label='Training Loss')
        ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax2.set_title('Loss Curve', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(bottom=0)

        # Add info text
        info = f'Learning Rate: {learning_rate:.6f} | Current Loss: {loss:.6f}'
        fig.text(0.5, 0.96, info, ha='center', fontsize=11, style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save:
            filename = f'gradient_frame_{self.frame_count:05d}.png'
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            if self.frame_count % 10 == 0:
                print(f'Gradient frame saved: {filename}')
            self.frame_count += 1
        else:
            plt.show()

        plt.close()

    def save_summary(self):
        """Save a summary of gradient visualization settings."""
        summary_path = os.path.join(self.save_dir, 'README.txt')
        with open(summary_path, 'w') as f:
            f.write('GRADIENT VISUALIZATION FRAMES\n')
            f.write('=' * 50 + '\n\n')
            f.write(f'Total frames saved: {self.frame_count}\n')
            f.write(f'Location: {self.save_dir}\n\n')
            f.write('TO CREATE A GIF:\n')
            f.write('-' * 50 + '\n')
            f.write('Using ImageMagick:\n')
            f.write(f'  convert -delay 20 -loop 0 {self.save_dir}/gradient_frame_*.png gradients.gif\n\n')
            f.write('Using Python (imageio):\n')
            f.write('  import imageio\n')
            f.write('  import glob\n')
            f.write(f'  images = [imageio.imread(f) for f in sorted(glob.glob("{self.save_dir}/gradient_frame_*.png"))]\n')
            f.write('  imageio.mimsave("gradients.gif", images, duration=0.2)\n\n')
            f.write('Using ffmpeg:\n')
            f.write(f'  ffmpeg -framerate 5 -pattern_type glob -i "{self.save_dir}/gradient_frame_*.png" -c:v libx264 gradients.mp4\n')

        print(f'\nGradient visualization summary saved to: {summary_path}')
