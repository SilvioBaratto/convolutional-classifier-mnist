"""
Script to create a GIF from gradient visualization frames.
"""
import argparse
import glob
import os


def create_gif_imageio(input_dir, output_file, duration=0.2):
    """
    Create GIF using imageio library.

    Args:
        input_dir: Directory containing gradient frames
        output_file: Output GIF filename
        duration: Duration per frame in seconds
    """
    try:
        import imageio
    except ImportError:
        print("Error: imageio not installed. Install with: pip install imageio")
        return False

    # Find all gradient frames
    pattern = os.path.join(input_dir, "gradient_frame_*.png")
    image_files = sorted(glob.glob(pattern))

    if not image_files:
        print(f"No gradient frames found in {input_dir}")
        return False

    print(f"Found {len(image_files)} frames")
    print(f"Creating GIF: {output_file}")

    # Load images and create GIF
    images = []
    for i, filepath in enumerate(image_files):
        if (i + 1) % 10 == 0:
            print(f"Loading frame {i + 1}/{len(image_files)}")
        images.append(imageio.imread(filepath))

    imageio.mimsave(output_file, images, duration=duration)
    print(f"GIF created successfully: {output_file}")
    return True


def create_gif_pillow(input_dir, output_file, duration=200):
    """
    Create GIF using Pillow library.

    Args:
        input_dir: Directory containing gradient frames
        output_file: Output GIF filename
        duration: Duration per frame in milliseconds
    """
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow not installed. Install with: pip install Pillow")
        return False

    # Find all gradient frames
    pattern = os.path.join(input_dir, "gradient_frame_*.png")
    image_files = sorted(glob.glob(pattern))

    if not image_files:
        print(f"No gradient frames found in {input_dir}")
        return False

    print(f"Found {len(image_files)} frames")
    print(f"Creating GIF: {output_file}")

    # Load images
    images = []
    for i, filepath in enumerate(image_files):
        if (i + 1) % 10 == 0:
            print(f"Loading frame {i + 1}/{len(image_files)}")
        images.append(Image.open(filepath))

    # Save as GIF
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF created successfully: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create GIF from gradient visualization frames")
    parser.add_argument("--input-dir", type=str, default="./gradient_frames",
                        help="Directory containing gradient frames (default: ./gradient_frames)")
    parser.add_argument("--output", type=str, default="./gradient_evolution.gif",
                        help="Output GIF filename (default: ./gradient_evolution.gif)")
    parser.add_argument("--duration", type=float, default=0.2,
                        help="Duration per frame in seconds (default: 0.2)")
    parser.add_argument("--library", type=str, choices=["imageio", "pillow"], default="pillow",
                        help="Library to use for GIF creation (default: pillow)")

    args = parser.parse_args()

    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return

    # Create GIF using selected library
    if args.library == "imageio":
        success = create_gif_imageio(args.input_dir, args.output, args.duration)
    else:
        duration_ms = int(args.duration * 1000)
        success = create_gif_pillow(args.input_dir, args.output, duration_ms)

    if success:
        # Get file size
        file_size = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\nGIF Details:")
        print(f"  File: {args.output}")
        print(f"  Size: {file_size:.2f} MB")
        print(f"  Frame duration: {args.duration}s")
        print(f"\nYou can now use this GIF to visualize how gradients evolved during training!")


if __name__ == "__main__":
    main()
