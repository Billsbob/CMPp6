import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

def create_histograms(measurements, mask_name, output_dir):
    """
    Create individual histograms for each image's measurements.
    
    Args:
        measurements (dict): Image name to ROI intensity values.
        mask_name (str): Name of the mask used.
        output_dir (str): Directory to save the histogram images.
        
    Returns:
        list of str: List of filenames of the generated histograms.
    """
    if not measurements:
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    for image_name, values in measurements.items():
        if len(values) == 0:
            continue
            
        plt.figure(figsize=(10, 6))
        sns.histplot(values, kde=True)
        plt.title(f"Histogram of {image_name} under {mask_name}")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        
        # Extract probe from image name: <Sample>_<Slide ##>_<Owner Initials>_<ObjectiveMag>_<Well Position>_<Probe>
        parts = image_name.split('_')
        probe = parts[5].split('.')[0] if len(parts) >= 6 else "Unknown"
        
        # Clean up mask_name for filename
        safe_mask_name = "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in mask_name])
        
        # New naming convention: <Mask Name>_<Probe>
        hist_filename = f"{safe_mask_name}_{probe}.png"
        hist_path = os.path.join(output_dir, hist_filename)
        plt.savefig(hist_path)
        plt.close()
        generated_files.append(hist_filename)
        
    return generated_files

def create_overlaid_histogram(measurements, mask_name, output_dir):
    """
    Create one histogram with all measurements overlaid as different series.
    Adjust Y-axis to the maximum value of all histograms.
    
    Args:
        measurements (dict): Image name to ROI intensity values.
        mask_name (str): Name of the mask used.
        output_dir (str): Directory to save the histogram image.
        
    Returns:
        str: Filename of the generated overlaid histogram.
    """
    if not measurements:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Use a color palette for multiple images
    palette = sns.color_palette("husl", len(measurements))
    
    max_freq = 0
    
    for i, (image_name, values) in enumerate(measurements.items()):
        if len(values) == 0:
            continue
        
        # We need to find the max frequency to adjust Y-axis. 
        # sns.histplot returns an Axes object, we can inspect it after each call or just let it auto-scale.
        # But the requirement says "Adjust Y-axis to histogram with the highest value." 
        # Seaborn/Matplotlib usually does this automatically if multiple series are plotted on the same axes.
        
        ax = sns.histplot(values, kde=True, label=image_name, color=palette[i], element="step")
        
        # Calculate max bin height manually if we want to be explicit, 
        # but histplot on the same Axes will grow to accommodate the largest.
        # To be sure we satisfy "highest value" requirement across all:
        counts, bins = np.histogram(values, bins='auto')
        max_freq = max(max_freq, counts.max())

    plt.title(f"Overlaid Histograms under {mask_name}")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.legend(title="Images")
    
    # Seaborn's histplot auto-scales Y, but let's ensure it's at least max_freq
    plt.ylim(0, max_freq * 1.1) # Add some margin

    safe_mask_name = "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in mask_name])
    hist_filename = f"Hist_Overlay_{safe_mask_name}.png"
    hist_path = os.path.join(output_dir, hist_filename)
    plt.savefig(hist_path)
    plt.close()
    
    return hist_filename

def create_dynamic_overlaid_histogram(items_measurements, title="Combined Histograms", output_path=None):
    """
    Create a histogram overlay from a list of (label, values) tuples.
    
    Args:
        items_measurements (list of tuples): List of (label, values) to plot.
        title (str): Plot title.
        output_path (str, optional): If provided, save the plot to this path.
        
    Returns:
        np.ndarray: The plot as an RGB image array.
    """
    if not items_measurements:
        return None
        
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("husl", len(items_measurements))
    
    max_freq = 0
    for i, (label, values) in enumerate(items_measurements):
        if len(values) == 0:
            continue
            
        sns.histplot(values, kde=True, label=label, color=palette[i], element="step")
        
        counts, _ = np.histogram(values, bins='auto')
        if len(counts) > 0:
            max_freq = max(max_freq, counts.max())
            
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.legend(title="Items")
    
    if max_freq > 0:
        plt.ylim(0, max_freq * 1.1)
        
    if output_path:
        plt.savefig(output_path)
        
    # Convert plot to image array
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    # In newer Matplotlib versions, use buffer_rgba() or similar
    try:
        rgba = np.array(canvas.buffer_rgba())
    except AttributeError:
        # Fallback for older versions if needed
        rgba = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        width, height = canvas.get_width_height()
        rgba = rgba.reshape((height, width, 3))
        # Add alpha channel or just use as is
        plt.close()
        return rgba
    
    rgb = rgba[:, :, :3]
    
    plt.close()

    return rgb
