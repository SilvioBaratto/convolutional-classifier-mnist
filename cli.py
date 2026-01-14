#!/usr/bin/env python3
"""
MNIST Classifier CLI - A beautiful command-line interface for training and inference.
"""

import argparse
import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

console = Console()


def print_banner():
    """Print the application banner."""
    banner = """
    ███╗   ███╗███╗   ██╗██╗███████╗████████╗
    ████╗ ████║████╗  ██║██║██╔════╝╚══██╔══╝
    ██╔████╔██║██╔██╗ ██║██║███████╗   ██║
    ██║╚██╔╝██║██║╚██╗██║██║╚════██║   ██║
    ██║ ╚═╝ ██║██║ ╚████║██║███████║   ██║
    ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝╚══════╝   ╚═╝

    Convolutional Neural Network Classifier
    """
    console.print(Panel(banner, style="bold blue", box=box.DOUBLE))


def print_config_table(config):
    """Print configuration as a rich table."""
    table = Table(title="Training Configuration", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Epochs", str(config.num_epochs))
    table.add_row("Batch Size", str(config.batch_size))
    table.add_row("Learning Rate", str(config.learning_rate))
    table.add_row("Optimizer", config.optimizer_type.upper())
    table.add_row("Device", config.device)
    table.add_row("Early Stopping", str(config.early_stopping_patience) if config.early_stopping_patience else "Disabled")
    table.add_row("Dropout", str(config.dropout_rate))

    console.print(table)


def cmd_train(args):
    """Train the MNIST classifier."""
    from mnist_classifier.core.model import MNISTConvNet, count_parameters
    from mnist_classifier.core.data_loader import get_data_loaders, get_dataset_info
    from mnist_classifier.core.trainer import Trainer, evaluate_model
    from mnist_classifier.core.config import Config, set_random_seed, get_device_info

    print_banner()

    # Build config
    config = Config()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.optimizer_type = args.optimizer
    config.early_stopping_patience = args.early_stopping if args.early_stopping > 0 else None
    config.checkpoint_dir = args.checkpoint_dir

    # Override device if specified
    if args.device:
        config.device = args.device

    print_config_table(config)

    # Set random seed
    set_random_seed(config.random_seed)
    device = torch.device(config.device)

    # Show device info
    device_info = get_device_info(config.device)
    console.print(f"\n[bold]Device:[/bold] {device_info['name']}", style="yellow")
    console.print(f"[bold]Memory:[/bold] {device_info['memory']}", style="yellow")

    # Load data
    with console.status("[bold green]Loading MNIST dataset..."):
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            val_split=config.val_split,
            num_workers=config.num_workers,
            random_seed=config.random_seed
        )

    dataset_info = get_dataset_info(train_loader)

    # Dataset info table
    table = Table(title="Dataset Information", box=box.ROUNDED)
    table.add_column("Split", style="cyan")
    table.add_column("Samples", style="green")
    table.add_row("Training", str(dataset_info["num_samples"]))
    table.add_row("Validation", str(len(val_loader.dataset)))
    table.add_row("Test", str(len(test_loader.dataset)))
    console.print(table)

    # Create model
    console.print("\n[bold green]Creating model...[/bold green]")
    model = MNISTConvNet(
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate
    )

    num_params = count_parameters(model)
    console.print(f"[bold]Parameters:[/bold] {num_params:,}", style="cyan")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if config.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

    scheduler = None
    if config.use_scheduler:
        scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=config.checkpoint_dir,
        save_best_only=config.save_best_only,
        visualize_gradients=args.visualize_gradients,
        gradient_vis_dir=args.gradient_dir,
        gradient_vis_frequency=args.gradient_frequency
    )

    # Resume if specified
    if args.resume:
        console.print(f"\n[yellow]Resuming from checkpoint: {args.resume}[/yellow]")
        trainer.load_checkpoint(args.resume)

    # Train
    console.print("\n[bold green]Starting training...[/bold green]\n")
    history = trainer.train(
        num_epochs=config.num_epochs,
        early_stopping_patience=config.early_stopping_patience
    )

    # Evaluate on test set
    console.print("\n[bold green]Evaluating on test set...[/bold green]")
    best_model_path = f'{config.checkpoint_dir}/best_model.pth'
    trainer.load_checkpoint(best_model_path)
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    # Final results
    console.print(Panel(
        f"[bold green]Training Complete![/bold green]\n\n"
        f"Best Model: {best_model_path}\n"
        f"Test Accuracy: [bold]{test_metrics['accuracy']:.2f}%[/bold]",
        title="Results",
        box=box.DOUBLE
    ))


def cmd_predict(args):
    """Predict digit from a custom image."""
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    from mnist_classifier.core.model import MNISTConvNet
    from mnist_classifier.core.config import get_best_device, get_device_info

    print_banner()

    # Auto-detect device if not specified
    device_str = args.device if args.device else get_best_device()
    device = torch.device(device_str)
    device_info = get_device_info(device_str)
    console.print(f"[bold]Device:[/bold] {device_info['name']}", style="yellow")

    # Load model
    with console.status("[bold green]Loading model..."):
        model = MNISTConvNet(num_classes=10, dropout_rate=0.5)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

    console.print(f"[green]Model loaded from:[/green] {args.checkpoint}")

    # Load and preprocess image
    with console.status("[bold green]Processing image..."):
        img = Image.open(args.image).convert('L')
        img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)

        # Invert if needed (MNIST has white on black)
        if img_array.mean() > 127:
            img_array = 255 - img_array
            console.print("[yellow]Image inverted (detected black digit on white background)[/yellow]")

        img_pil = Image.fromarray(img_array)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])

    # Results table
    table = Table(title="Prediction Results", box=box.DOUBLE)
    table.add_column("Digit", style="cyan", justify="center")
    table.add_column("Probability", style="green", justify="right")
    table.add_column("Bar", style="blue")

    for i in range(10):
        prob = probs[i]
        bar = "█" * int(prob * 30)
        marker = " ◄" if i == predicted else ""
        if i == predicted:
            table.add_row(
                f"[bold green]{i}[/bold green]",
                f"[bold green]{prob*100:5.2f}%[/bold green]",
                f"[bold green]{bar}{marker}[/bold green]"
            )
        else:
            table.add_row(str(i), f"{prob*100:5.2f}%", bar)

    console.print(table)

    console.print(Panel(
        f"[bold]Predicted Digit:[/bold] [bold green]{predicted}[/bold green]\n"
        f"[bold]Confidence:[/bold] [cyan]{confidence*100:.2f}%[/cyan]",
        title="Result",
        box=box.DOUBLE
    ))

    # Save visualization if requested
    if args.output:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.imshow(img_array, cmap='gray')
        ax1.set_title(f'Input Image (28x28)', fontsize=14, fontweight='bold')
        ax1.axis('off')

        colors = ['green' if i == predicted else 'lightblue' for i in range(10)]
        ax2.bar(range(10), probs, color=colors, edgecolor='black')
        ax2.set_xlabel('Digit', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title(f'Prediction: {predicted} ({confidence*100:.1f}%)', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(10))
        ax2.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        plt.close()

        console.print(f"\n[green]Visualization saved to:[/green] {args.output}")


def cmd_visualize(args):
    """Visualize the neural network data flow."""
    from mnist_classifier.core.model import MNISTConvNet
    from mnist_classifier.visualization.visualize import ModelVisualizer, load_sample_image
    from mnist_classifier.core.config import get_best_device, get_device_info

    print_banner()

    # Auto-detect device if not specified
    device_str = args.device if args.device else get_best_device()
    device = torch.device(device_str)
    device_info = get_device_info(device_str)
    console.print(f"[bold]Device:[/bold] {device_info['name']}", style="yellow")

    # Load model
    model = MNISTConvNet()

    if args.checkpoint:
        with console.status("[bold green]Loading model..."):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        console.print(f"[green]Model loaded from:[/green] {args.checkpoint}")
    else:
        console.print("[yellow]No checkpoint provided - using random weights[/yellow]")

    # Load sample image
    with console.status("[bold green]Loading sample image..."):
        image, label = load_sample_image(args.data_dir, args.sample_idx)

    console.print(f"[bold]Sample:[/bold] Index {args.sample_idx}, Label {label}")

    # Create visualizer
    visualizer = ModelVisualizer(model, device)

    # Generate visualizations
    console.print(f"\n[bold green]Generating visualizations...[/bold green]")
    visualizer.visualize_complete_flow(image, label, save_path=args.output)

    console.print(Panel(
        f"[bold green]Visualizations Complete![/bold green]\n\n"
        f"Output Directory: {args.output}\n"
        f"Generated 21 step-by-step images showing the complete forward pass.",
        title="Done",
        box=box.DOUBLE
    ))


def cmd_gif(args):
    """Create a GIF from gradient visualization frames."""
    import glob

    print_banner()

    # Find frames
    frame_pattern = os.path.join(args.input_dir, "gradient_*.png")
    frames = sorted(glob.glob(frame_pattern))

    if not frames:
        console.print(f"[red]No gradient frames found in {args.input_dir}[/red]")
        console.print("[yellow]Run training with --visualize-gradients first[/yellow]")
        return

    console.print(f"[bold]Found:[/bold] {len(frames)} frames")

    try:
        import imageio

        with console.status("[bold green]Creating GIF..."):
            images = [imageio.imread(f) for f in frames]
            imageio.mimsave(args.output, images, duration=args.duration)

        console.print(Panel(
            f"[bold green]GIF Created![/bold green]\n\n"
            f"Output: {args.output}\n"
            f"Frames: {len(frames)}\n"
            f"Duration: {args.duration}s per frame",
            title="Done",
            box=box.DOUBLE
        ))

    except ImportError:
        console.print("[red]imageio not installed. Install with: pip install imageio[/red]")


def cmd_info(args):
    """Show model architecture information."""
    from mnist_classifier.core.model import MNISTConvNet, count_parameters

    print_banner()

    model = MNISTConvNet()
    num_params = count_parameters(model)

    # Architecture table
    table = Table(title="Model Architecture", box=box.DOUBLE)
    table.add_column("Layer", style="cyan")
    table.add_column("Output Shape", style="green")
    table.add_column("Parameters", style="yellow", justify="right")

    table.add_row("Input", "(1, 28, 28)", "-")
    table.add_row("Conv1 + BN + ReLU + Pool", "(32, 14, 14)", "352")
    table.add_row("Conv2 + BN + ReLU + Pool", "(64, 7, 7)", "18,560")
    table.add_row("Conv3 + BN + ReLU + Pool", "(128, 3, 3)", "73,984")
    table.add_row("Flatten", "(1152,)", "-")
    table.add_row("FC1 + ReLU + Dropout", "(256,)", "295,168")
    table.add_row("FC2 (Output)", "(10,)", "2,570")
    table.add_section()
    table.add_row("[bold]Total[/bold]", "", f"[bold]{num_params:,}[/bold]")

    console.print(table)

    # Features
    console.print("\n[bold]Features:[/bold]")
    console.print("  • 3 Convolutional blocks with BatchNorm")
    console.print("  • MaxPooling after each conv block")
    console.print("  • Dropout (0.5) for regularization")
    console.print("  • He weight initialization")
    console.print("  • Cross-entropy loss with Adam/SGD optimizer")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MNIST Classifier - Train, predict, and visualize CNN for digit classification",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the MNIST classifier")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs (default: 20)")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    train_parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optimizer (default: adam)")
    train_parser.add_argument("--early-stopping", type=int, default=5, help="Early stopping patience (default: 5, 0 to disable)")
    train_parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
    train_parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    train_parser.add_argument("--device", type=str, default=None, help="Device: cuda, mps, cpu (default: auto-detect)")
    train_parser.add_argument("--visualize-gradients", action="store_true", help="Save gradient visualizations")
    train_parser.add_argument("--gradient-dir", default="./gradient_frames", help="Gradient frames directory")
    train_parser.add_argument("--gradient-frequency", type=int, default=100, help="Save gradients every N batches")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict digit from image")
    predict_parser.add_argument("--image", required=True, help="Path to input image")
    predict_parser.add_argument("--checkpoint", default="./checkpoints/best_model.pth", help="Model checkpoint")
    predict_parser.add_argument("--output", help="Save visualization to file")
    predict_parser.add_argument("--device", default=None, help="Device: cuda, mps, cpu (default: auto-detect)")

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize network data flow")
    viz_parser.add_argument("--checkpoint", help="Model checkpoint (optional)")
    viz_parser.add_argument("--output", default="./visualizations", help="Output directory")
    viz_parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to visualize")
    viz_parser.add_argument("--data-dir", default="./data", help="MNIST data directory")
    viz_parser.add_argument("--device", default=None, help="Device: cuda, mps, cpu (default: auto-detect)")

    # GIF command
    gif_parser = subparsers.add_parser("gif", help="Create gradient evolution GIF")
    gif_parser.add_argument("--input-dir", default="./gradient_frames", help="Input frames directory")
    gif_parser.add_argument("--output", default="./gradient_evolution.gif", help="Output GIF path")
    gif_parser.add_argument("--duration", type=float, default=0.3, help="Duration per frame in seconds")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model architecture info")

    args = parser.parse_args()

    if args.command is None:
        print_banner()
        parser.print_help()
        console.print("\n[bold]Examples:[/bold]")
        console.print("  python cli.py train --epochs 10")
        console.print("  python cli.py predict --image ./test/foto.jpeg")
        console.print("  python cli.py visualize --checkpoint ./checkpoints/best_model.pth")
        console.print("  python cli.py info")
        return

    commands = {
        "train": cmd_train,
        "predict": cmd_predict,
        "visualize": cmd_visualize,
        "gif": cmd_gif,
        "info": cmd_info,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
