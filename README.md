# MNIST Digit Classifier

A PyTorch CNN for MNIST digit classification with visualization tools to understand how neural networks learn.

## Features

- **High Accuracy**: >99% on MNIST test set
- **Beautiful CLI**: Rich console interface with progress bars and tables
- **Educational Visualizations**: Step-by-step network data flow
- **MPS Support**: GPU acceleration on Apple Silicon

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Train the model (auto-detects GPU)
python cli.py train --epochs 10

# Predict from image
python cli.py predict --image ./test/foto.jpeg

# Visualize how the network processes an image
python cli.py visualize --checkpoint ./checkpoints/best_model.pth
```

---

## How Convolutional Neural Networks Work

This project generates visualizations that show exactly what happens inside a CNN. Run:

```bash
python cli.py visualize --checkpoint ./checkpoints/best_model.pth --output ./visualizations
```

### Step 0: Training Data Structure

![Training Dataset](docs/images/00a_training_dataset.png)

The network learns from labeled examples. Each training sample has:
- **Image**: 28×28 grayscale pixels (input)
- **Label**: The correct digit 0-9 (target)

The image flows through the network, and the label is used *only at the end* to compute the error and update weights.

### Step 1: Input Image

![Input](docs/images/00b_input.png)

A 28×28 grayscale image where each pixel is a value between 0 (black) and 1 (white). This is what the network "sees".

---

### Convolutional Block 1: Feature Detection

#### Convolution (32 filters, 3×3)

![Conv1](docs/images/01_conv1.png)

**What is convolution?** A small 3×3 filter slides across the image, computing dot products at each position. Each filter learns to detect a specific pattern (edges, curves, etc.).

- Input: (1, 28, 28) - single channel image
- Output: (32, 28, 28) - 32 different feature maps

#### Batch Normalization

![BatchNorm1](docs/images/02_batchnorm1.png)

Normalizes the activations to have zero mean and unit variance. This stabilizes training and allows higher learning rates.

#### ReLU Activation

![ReLU1](docs/images/03_relu1.png)

`ReLU(x) = max(0, x)` - Sets all negative values to zero. This introduces non-linearity, allowing the network to learn complex patterns.

#### Max Pooling (2×2)

![Pool1](docs/images/04_pool1.png)

Reduces spatial dimensions by taking the maximum value in each 2×2 region:
- Input: (32, 28, 28)
- Output: (32, 14, 14)

This provides translation invariance and reduces computation.

---

### Convolutional Block 2: Higher-Level Features

![Conv2](docs/images/05_conv2.png)
![Pool2](docs/images/08_pool2.png)

The second block detects combinations of the basic features from Block 1:
- Input: (32, 14, 14)
- After Conv2: (64, 14, 14) - 64 filters detecting more complex patterns
- After Pool2: (64, 7, 7)

---

### Convolutional Block 3: Abstract Features

![Conv3](docs/images/09_conv3.png)
![Pool3](docs/images/12_pool3.png)

The third block learns the most abstract features:
- Input: (64, 7, 7)
- After Conv3: (128, 7, 7)
- After Pool3: (128, 3, 3)

At this stage, the network has learned digit-specific patterns.

---

### Fully Connected Layers: Classification

#### Flatten

![Flatten](docs/images/13_flatten.png)

Converts the 3D feature maps (128, 3, 3) into a 1D vector of 1,152 values.

#### FC1: Feature Combination

![FC1](docs/images/14_fc1.png)

A fully connected layer with 256 neurons. Each neuron sees all 1,152 input features and learns to combine them.

#### Dropout (0.5)

![Dropout](docs/images/16_dropout.png)

During training, randomly sets 50% of neurons to zero. This prevents overfitting by forcing the network to learn redundant representations.

---

### Output: Making a Prediction

#### Logits (Raw Scores)

![Logits](docs/images/17_logits.png)

The final layer outputs 10 values (one per digit). These are raw scores called **logits** - higher values indicate more confidence.

#### Softmax → Probabilities

![Probabilities](docs/images/18_probabilities.png)

Softmax converts logits to probabilities that sum to 1:

```
P(class_i) = exp(logit_i) / Σ exp(logit_j)
```

The predicted digit is the one with highest probability.

---

### Training: How the Network Learns

![Learning](docs/images/19_learning_from_labels.png)

1. **Forward Pass**: Image flows through network → prediction
2. **Loss Calculation**: Compare prediction with true label using Cross-Entropy Loss
3. **Backward Pass**: Compute gradients (how much each weight contributed to error)
4. **Update Weights**: Adjust weights to reduce error (gradient descent)

This cycle repeats for thousands of images until the network learns.

#### Final Prediction

![Prediction](docs/images/20_prediction_summary.png)

---

## Model Architecture

```
Input (1, 28, 28)
    │
    ▼
Conv1 (32 filters) → BatchNorm → ReLU → MaxPool → (32, 14, 14)
    │
    ▼
Conv2 (64 filters) → BatchNorm → ReLU → MaxPool → (64, 7, 7)
    │
    ▼
Conv3 (128 filters) → BatchNorm → ReLU → MaxPool → (128, 3, 3)
    │
    ▼
Flatten (1152) → FC1 (256) → ReLU → Dropout → FC2 (10) → Softmax
```

**Total Parameters**: ~390,000

---

## CLI Commands

### Train

```bash
python cli.py train                          # Basic training (20 epochs)
python cli.py train --epochs 10 --device mps # Use Apple Silicon GPU
python cli.py train --visualize-gradients    # Save gradient evolution frames
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 20 | Training epochs |
| `--batch-size` | 128 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--device` | auto | Device (cuda/mps/cpu) |
| `--visualize-gradients` | false | Save gradient frames |

### Predict

```bash
python cli.py predict --image ./test/foto.jpeg
python cli.py predict --image ./test/foto.jpeg --output result.png
```

### Visualize

```bash
python cli.py visualize --checkpoint ./checkpoints/best_model.pth
```

### Info

```bash
python cli.py info  # Show model architecture
```

---

## Project Structure

```
├── cli.py                      # Main CLI entry point
├── mnist_classifier/
│   ├── core/
│   │   ├── model.py           # CNN architecture
│   │   ├── config.py          # Configuration & device detection
│   │   ├── data_loader.py     # MNIST loading with augmentation
│   │   └── trainer.py         # Training loop
│   ├── visualization/
│   │   ├── visualize.py       # Network visualization (21 images)
│   │   └── gradient_visualizer.py
│   └── utils/
├── docs/images/                # Documentation images (CNN explanation)
├── checkpoints/                # Saved model weights
└── test/                       # Sample test images
```

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | >99% |
| Training Time | ~2 min (MPS) / ~10 min (CPU) |
| Model Size | ~1.5 MB |

---

## License

MIT License - see [LICENSE](LICENSE) for details.
