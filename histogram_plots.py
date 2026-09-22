import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from naming_utils import build_histogram_identity, sanitize_name, strip_extension

def create_histograms(measurements, mask_name, output_dir, source_masks=None, show_kde=True, normalization=None, color_palette=None):
    """
    Create individual histograms for each image's measurements.
    
    Args:
        measurements (dict): Image name to ROI intensity values.
        mask_name (str): Name of the mask used.
        output_dir (str): Directory to save the histogram images.
        source_masks (list, optional): List of source masks if the mask is a merged one.
        show_kde (bool): Whether to show KDE in the PNG.
        normalization (str, optional): Type of normalization applied.
        color_palette (dict, optional): Maps '<well_position>_<probe>' to a color string.
            Example: {'3_DAPI': '#377eb8', '7_GFP': '#4daf4a'}
        
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
            
        identity = build_histogram_identity(image_name, mask_name, source_masks)
        hist_color = None
        if color_palette:
            hist_color = color_palette.get(identity.color_key)

        plt.figure(figsize=(10, 6))
        if hist_color:
            ax = sns.histplot(values, kde=show_kde, stat="density", color=hist_color)
        else:
            ax = sns.histplot(values, kde=show_kde, stat="density")
        
        # Add mean and median lines
        mean_val = np.mean(values)
        median_val = np.median(values)
        ax.axvline(mean_val, color='r', linestyle='--', alpha=0.3, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='g', linestyle='-', alpha=0.3, label=f'Median: {median_val:.2f}')
        ax.legend()
        
        title = identity.display_name
        if identity.color_key:
            title += f"\n(Color key: {identity.color_key})"
        if identity.source_masks:
            title += f"\n(Sources: {', '.join(identity.source_masks)})"
        if normalization and normalization != "None":
            title += f"\n(Normalization: {normalization})"
        plt.title(title)
        plt.xlabel("Normalized Intensity" if normalization and normalization != "None" else "Intensity")
        plt.ylabel("Frequency")
        
        hist_filename = f"{identity.safe_filename_base}.png"
        
        hist_path = os.path.join(output_dir, hist_filename)
        plt.savefig(hist_path)
        plt.close()
        
        metadata_path = os.path.splitext(hist_path)[0] + ".json"
        with open(metadata_path, "w") as f:
            import json
            json.dump(identity.to_metadata(color=hist_color), f, indent=2)
            
        generated_files.append(hist_filename)
        
    return generated_files

def create_overlaid_histogram(measurements, mask_name, output_dir, source_masks=None):
    """
    Create one histogram with all measurements overlaid as different series.
    Adjust Y-axis to the maximum value of all histograms.
    
    Args:
        measurements (dict): Image name to ROI intensity values.
        mask_name (str): Name of the mask used.
        output_dir (str): Directory to save the histogram image.
        source_masks (list, optional): List of source masks if the mask is a merged one.
        
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
        
        identity = build_histogram_identity(image_name, mask_name)
        ax = sns.histplot(values, kde=True, label=identity.image.base_name, color=palette[i], element="step", stat="density")
        
        # Add mean and median lines
        mean_val = np.mean(values)
        median_val = np.median(values)
        ax.axvline(mean_val, color=palette[i], linestyle='--', alpha=0.3)
        ax.axvline(median_val, color=palette[i], linestyle='-', alpha=0.3)

    # Strip extension from mask name for title
    mask_display_name = strip_extension(mask_name)
    title = f"Overlaid Histograms under {mask_display_name}"
    if source_masks:
        title += f"\n(Sources: {', '.join(source_masks)})"
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.legend(title="Images")
    
    # Seaborn's histplot auto-scales Y, but let's ensure it's at least max_freq
    plt.ylim(0, max_freq * 1.1) # Add some margin

    # Strip extension from mask name for filename
    safe_mask_name = sanitize_name(strip_extension(mask_name))
    hist_filename = f"{safe_mask_name}_Overlay.png"
    hist_path = os.path.join(output_dir, hist_filename)
    plt.savefig(hist_path)
    plt.close()
    
    return hist_filename

def create_dynamic_overlaid_histogram(items_measurements, title="Combined Histograms", output_path=None, source_masks=None, show_kde=True):
    """
    Create a high-quality histogram overlay from a list of (label, values) tuples.
    
    Args:
        items_measurements (list of tuples): List of (label, values) to plot.
        title (str): Plot title.
        output_path (str, optional): If provided, save the plot to this path.
        source_masks (list, optional): List of source masks if applicable.
        show_kde (bool): Whether to show KDE.
        
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
            
        ax = sns.histplot(values, kde=show_kde, label=label, color=palette[i], element="step", stat="density")
        
        # Add mean and median lines
        mean_val = np.mean(values)
        median_val = np.median(values)
        ax.axvline(mean_val, color=palette[i], linestyle='--', alpha=0.3)
        ax.axvline(median_val, color=palette[i], linestyle='-', alpha=0.3)
            
    if source_masks:
        title += f"\n(Sources: {', '.join(source_masks)})"
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

def render_fast_overlay(items_counts, title="Combined Histograms (Fast Preview)", source_masks=None):
    """
    Render a fast overlay using pre-calculated binned data (counts).
    
    Args:
        items_counts (list of tuples): List of (label, counts) to plot. 
                                      Counts should be an array of size 256.
        title (str): Plot title.
        source_masks (list, optional): List of source masks if applicable.
        
    Returns:
        np.ndarray: The plot as an RGB image array.
    """
    if not items_counts:
        return None
        
    # Use standard Matplotlib (no Seaborn) for speed
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(256)
    
    max_val = 0
    for label, counts in items_counts:
        if len(counts) == 0:
            continue
        
        # Normalize counts to density
        bin_width = 1.0 # Assuming 0-255 range with 256 bins
        total = np.sum(counts)
        density = counts / (total * bin_width) if total > 0 else counts
        
        ax.step(x, density, label=label, where='mid', alpha=0.7)
        max_val = max(max_val, np.max(density))
        
        # Estimate mean/median from binned data
        mean_est = np.sum(x * counts) / total if total > 0 else 0
        cumulative = np.cumsum(counts)
        median_est = np.searchsorted(cumulative, total / 2.0)
        
        line = ax.step(x, density, where='mid', alpha=0.0)[0] # dummy to get color if needed, but ax.step above already did
        color = ax.get_lines()[-1].get_color()
        ax.axvline(mean_est, color=color, linestyle='--', alpha=0.3)
        ax.axvline(median_est, color=color, linestyle='-', alpha=0.3)
        
    if source_masks:
        title += f"\n(Sources: {', '.join(source_masks)})"
    ax.set_title(title)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    ax.legend(title="Items")
    
    if max_val > 0:
        ax.set_ylim(0, max_val * 1.1)
        
    # Convert plot to image array
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    try:
        rgba = np.array(canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
    except AttributeError:
        # Fallback
        rgb = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        w, h = canvas.get_width_height()
        rgb = rgb.reshape((h, w, 3))
    
    plt.close(fig)
    return rgb

def render_mask_separated_fast_overlay(items_counts, title="Combined Histograms (Fast Preview)", probe_colors=None):
    """
    Render a fast overlay using pre-calculated binned data (counts),
    separated into rows by mask.
    
    Args:
        items_counts (list): List of (label, counts, mask_name, original_image_name)
        title (str): Plot title.
        probe_colors (dict, optional): Mapping of probe name to color.
        
    Returns:
        np.ndarray: The plot as an RGB image array.
    """
    if not items_counts:
        return None
        
    # Group by mask
    masks = {}
    for label, counts, mask_name, img_name in items_counts:
        if mask_name not in masks:
            masks[mask_name] = []
        masks[mask_name].append((label, counts, img_name))
        
    sorted_mask_names = sorted(masks.keys())
    num_masks = len(sorted_mask_names)
    fig, axes = plt.subplots(num_masks, 1, figsize=(10, num_masks), squeeze=False)
    
    x = np.arange(256)
    for idx, mask_name in enumerate(sorted_mask_names):
        ax = axes[idx, 0]
        mask_items = masks[mask_name]
        
        # Use a color palette for multiple images within the same mask
        palette = sns.color_palette("husl", len(mask_items))
        
        for i, (label, counts, img_name) in enumerate(mask_items):
            # Format legend label: <well_position>_<Probe>
            identity = build_histogram_identity(img_name, mask_name)
            legend_label = label
            probe = identity.image.probe
            if identity.image.well_position and identity.image.probe:
                legend_label = identity.image.well_probe_key
            
            # Determine color
            color = palette[i]
            if probe_colors and probe in probe_colors:
                color = probe_colors[probe]

            # Normalize counts to density
            total = np.sum(counts)
            density = counts / total if total > 0 else counts
            
            ax.step(x, density, label=legend_label, where='mid', alpha=0.7, color=color)
            
            # Add mean and median lines (estimated from binned data)
            mean_est = np.sum(x * counts) / total if total > 0 else 0
            cumulative = np.cumsum(counts)
            median_est = np.searchsorted(cumulative, total / 2.0)
            
            ax.axvline(mean_est, color=color, linestyle='--', alpha=0.3)
            ax.axvline(median_est, color=color, linestyle='-', alpha=0.3)
            
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Remove top and right spines
        sns.despine(ax=ax, left=True)
        
        # Set title to include mask name (strip extension)
        mask_display_name = strip_extension(mask_name)
        ax.set_title(f"Mask: {mask_display_name}", loc='left', fontsize=9)
            
    plt.tight_layout()
    
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    try:
        rgba = np.array(canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
    except AttributeError:
        # Fallback
        rgb = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        w, h = canvas.get_width_height()
        rgb = rgb.reshape((h, w, 3))
    
    plt.close(fig)
    return rgb

def create_mask_separated_histograms(items_to_render, title="Mask Separated Histograms", output_path=None, show_kde=True, probe_colors=None):
    """
    Create histograms displayed in separated rows organized by mask.
    Graph height is approx 1 inch per row.
    Legend is shown outside the graph area with <well_position>_<Probe>.
    
    Args:
        items_to_render (list): List of (label, values, mask_name, original_image_name)
        title (str): Plot title.
        output_path (str, optional): Path to save the plot.
        show_kde (bool): Whether to show KDE.
        probe_colors (dict, optional): Mapping of probe name to color.
        
    Returns:
        np.ndarray: The plot as an RGB image array.
    """
    if not items_to_render:
        return None
        
    # Group by mask
    masks = {}
    for label, values, mask_name, img_name in items_to_render:
        if mask_name not in masks:
            masks[mask_name] = []
        masks[mask_name].append((label, values, img_name))
    
    sorted_mask_names = sorted(masks.keys())
    num_masks = len(sorted_mask_names)
    
    # 1 inch height per mask row, width 10 inches
    fig, axes = plt.subplots(num_masks, 1, figsize=(10, num_masks), squeeze=False)
    
    for idx, mask_name in enumerate(sorted_mask_names):
        ax = axes[idx, 0]
        mask_items = masks[mask_name]
        
        # Use a color palette for multiple images within the same mask
        palette = sns.color_palette("husl", len(mask_items))
        
        for i, (label, values, img_name) in enumerate(mask_items):
            # Format legend label: <well_position>_<Probe>
            identity = build_histogram_identity(img_name, mask_name)
            legend_label = label
            probe = identity.image.probe
            if identity.image.well_position and identity.image.probe:
                legend_label = identity.image.well_probe_key
            
            # Determine color
            color = palette[i]
            if probe_colors and probe in probe_colors:
                color = probe_colors[probe]

            sns.histplot(values, kde=show_kde, ax=ax, label=legend_label, element="step", color=color, stat="density")
            
            # Add mean and median lines
            mean_val = np.mean(values)
            median_val = np.median(values)
            ax.axvline(mean_val, color=color, linestyle='--', alpha=0.3)
            ax.axvline(median_val, color=color, linestyle='-', alpha=0.3)
            
        ax.set_ylabel("") # No y-axis label
        ax.set_yticks([]) # No y-axis ticks
        
        # Show legend outside the graph area
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Remove top and right spines
        sns.despine(ax=ax, left=True)
        
        # Set title to include mask name (strip extension)
        mask_display_name = strip_extension(mask_name)
        ax.set_title(f"Mask: {mask_display_name}", loc='left', fontsize=9)
            
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        
    # Convert plot to image array
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    try:
        rgba = np.array(canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
    except AttributeError:
        # Fallback
        rgb = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        w, h = canvas.get_width_height()
        rgb = rgb.reshape((h, w, 3))
    
    plt.close(fig)
    return rgb
