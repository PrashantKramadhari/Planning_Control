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

# Method 2: Using imageio - Great for scientific/technical images
def png_to_gif_imageio(input_folder, output_gif, duration=0.1):
    """
    Convert PNG images to GIF using imageio
    
    Args:
        input_folder: Path to folder containing PNG files
        output_gif: Output GIF filename
        duration: Duration between frames in seconds
    """
    png_files = sorted(glob.glob(os.path.join(input_folder, "*.png")))
    
    if not png_files:
        print("No PNG files found!")
        return
    
    with imageio.get_writer(output_gif, mode='I', duration=duration, loop=0) as writer:
        for png_file in png_files:
            image = imageio.imread(png_file)
            writer.append_data(image)
    
    print(f"GIF saved as {output_gif}")

# Method 3: Using matplotlib animation - For plots specifically
def create_gif_from_function(data, output_gif, interval=100):
    """
    Create GIF animation directly from data using matplotlib
    
    Args:
        data: List of data arrays or function to generate frames
        output_gif: Output GIF filename
        interval: Interval between frames in milliseconds
    """
    fig, ax = plt.subplots()
    
    def animate(frame):
        ax.clear()
        # Your plotting code here - example with random data
        ax.plot(data[frame])
        ax.set_title(f'Frame {frame}')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=interval)
    anim.save(output_gif, writer='pillow')
    plt.close()
    print(f"GIF saved as {output_gif}")

# Method 4: Advanced function with customization options
def png_to_gif_advanced(input_pattern, output_gif, duration=100, 
                       resize=None, optimize=True, quality=95):
    """
    Advanced PNG to GIF converter with optimization options
    
    Args:
        input_pattern: Glob pattern for input files (e.g., "./pp/pp*.png")
        output_gif: Output GIF filename
        duration: Duration between frames in milliseconds
        resize: Tuple (width, height) to resize images, None to keep original
        optimize: Whether to optimize the GIF
        quality: Quality setting (0-100)
    """
    png_files = sorted(glob.glob(input_pattern))
    
    if not png_files:
        print(f"No files found matching pattern: {input_pattern}")
        return
    
    images = []
    for png_file in png_files:
        img = Image.open(png_file)
        
        # Resize if specified
        if resize:
            img = img.resize(resize, Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary (GIFs don't support RGBA well)
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            else:
                background.paste(img)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        images.append(img)
    
    # Save optimized GIF
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=optimize,
        quality=quality
    )
    print(f"Optimized GIF saved as {output_gif} ({len(images)} frames)")

# Example usage functions
def example_usage():
    """Examples of how to use the functions"""
    
    # Example 1: Convert pursuit plot PNGs to GIF
    print("Converting pursuit plot images...")
    png_to_gif_pil("./sf/", "SF.gif", duration=50)
    
    # Example 2: Using imageio with custom duration
    #print("Creating GIF with imageio...")
    #png_to_gif_imageio("./pp/", "pursuit_animation_imageio.gif", duration=0.05)
    
    # Example 3: Advanced conversion with resizing and optimization
    #print("Creating optimized GIF...")
    '''
    png_to_gif_advanced(
        "./pp/pp*.png", 
        "pursuit_optimized.gif", 
        duration=80,
        resize=(800, 600),  # Resize to 800x600
        optimize=True,
        quality=90
    )'''

if __name__ == "__main__":
    # Run examples
    example_usage()
    
    # Uncomment to clean up PNG files after creating GIF
    # cleanup_pngs("./pp/", keep_every_nth=5)  # Keep every 5th frame