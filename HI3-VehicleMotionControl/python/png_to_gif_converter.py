import os
import glob
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Method 1: Using PIL (Pillow) - Simple and effective
def png_to_gif_pil(input_folder, output_gif, duration=100):
    """
    Convert PNG images to GIF using PIL
    
    Args:
        input_folder: Path to folder containing PNG files
        output_gif: Output GIF filename
        duration: Duration between frames in milliseconds
    """
    # Get all PNG files and sort them
    png_files = sorted(glob.glob(os.path.join(input_folder, "*.png")))
    
    if not png_files:
        print("No PNG files found!")
        return
    
    # Load images
    images = []
    for png_file in png_files:
        img = Image.open(png_file)
        images.append(img)
    
    # Save as GIF
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0  # 0 means infinite loop
    )
    print(f"GIF saved as {output_gif}")

# Example usage functions
def example_usage():
    """Examples of how to use the functions"""
    
    # Example 1: Convert pursuit plot PNGs to GIF
    print("Converting pursuit plot images...")
    png_to_gif_pil("./sf/", "SF.gif", duration=50)
    

if __name__ == "__main__":
    # Run examples
    example_usage()