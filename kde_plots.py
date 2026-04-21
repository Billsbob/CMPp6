import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

def truncate_image_name(name):
    """
    Truncate the image name to the last two signifiers (e.g., ##_AA).
    If the name has less than two signifiers separated by underscores,
    it returns the entire name without extension.
    """
    # Remove extension if present
    base_name = os.path.splitext(name)[0]
    parts = base_name.split('_')
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return base_name

def create_joint_kde_plot(all_measurements, output_dir, user_filename=None):
    """
    Create a Joint KDE plot (Jointplot) for up to 3 sets of images under masks.
    
    Args:
        all_measurements (list of dict): List of measurement dicts. Each dict has:
            - 'image1_name': str
            - 'image2_name': str
            - 'mask_name': str
            - 'x_values': list
            - 'y_values': list
        output_dir (str): Directory to save the plot image.
        user_filename (str, optional): Custom filename for the plot.
        
    Returns:
        str: Filename of the generated jointplot.
    """
    if not all_measurements:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the global limits for X and Y axes to keep them consistent
    all_x = []
    all_y = []
    for m in all_measurements:
        all_x.extend(m['x_values'])
        all_y.extend(m['y_values'])
    
    if not all_x or not all_y:
        return None

    # Create a list of descriptions for each set and truncate names
    descriptions = []
    for i, m in enumerate(all_measurements):
        t_img1 = truncate_image_name(m['image1_name'])
        t_img2 = truncate_image_name(m['image2_name'])
        mask_name = m['mask_name']
        # The user mentioned "image1 vs image2 under mask ##.npy, image3 vs image4 under mask ##.py"
        # We will follow this format for the description.
        descriptions.append(f"Set {i+1}: {t_img1} vs {t_img2} under {mask_name}")

    plt.figure(figsize=(12, 12))
    # We use a color palette for multiple sets
    palette = sns.color_palette("husl", len(all_measurements))
    
    # Create the JointGrid
    g = sns.JointGrid()
    
    for i, m in enumerate(all_measurements):
        x = m['x_values']
        y = m['y_values']
        label = descriptions[i]
        
        # Plot KDE in the joint area
        sns.kdeplot(x=x, y=y, ax=g.ax_joint, fill=True, color=palette[i], label=label, alpha=0.5)
        # Plot KDE in the marginal areas
        sns.kdeplot(x=x, ax=g.ax_marg_x, color=palette[i], fill=True)
        sns.kdeplot(y=y, ax=g.ax_marg_y, color=palette[i], fill=True)

    # Use the JointGrid's figure to add the legend below
    handles, labels = g.ax_joint.get_legend_handles_labels()
    if handles:
        g.figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize='small')
    
    # Adjust the plot to make room for the legend at the bottom
    g.figure.subplots_adjust(top=0.9, bottom=0.2)
    
    # Set axis labels to image names
    x_label = "Intensity 1"
    y_label = "Intensity 2"
    if len(all_measurements) == 1:
        x_label = truncate_image_name(all_measurements[0]['image1_name'])
        y_label = truncate_image_name(all_measurements[0]['image2_name'])
    else:
        # Check if all sets have the same image1 and image2
        first_x = all_measurements[0]['image1_name']
        first_y = all_measurements[0]['image2_name']
        all_same_x = all(m['image1_name'] == first_x for m in all_measurements)
        all_same_y = all(m['image2_name'] == first_y for m in all_measurements)
        if all_same_x:
            x_label = truncate_image_name(first_x)
        if all_same_y:
            y_label = truncate_image_name(first_y)

    g.set_axis_labels(x_label, y_label)
    
    # Construct the title using the descriptions
    # We will manually add text for each description line with its corresponding color
    num_sets = len(all_measurements)
    # Each line takes about 0.03 coordinate units
    # Start y at 0.96 (since suptitle is removed)
    start_y = 0.96
    for i, desc in enumerate(descriptions):
        line_y = start_y - (i * 0.03)
        g.figure.text(0.5, line_y, desc, color=palette[i], 
                      ha='center', va='center', fontsize='medium')
    
    # Adjust the top margin to accommodate the title and lines
    # suptitle at 0.96, then lines at 0.93, 0.90, 0.87
    # So top margin should be around start_y - (num_sets * 0.03) - a bit extra
    top_margin = start_y - (num_sets * 0.03) - 0.04
    g.figure.subplots_adjust(top=top_margin, bottom=0.2)
    
    if user_filename:
        filename = f"JointPlot_{user_filename}.png"
    elif len(all_measurements) == 1:
        m = all_measurements[0]
        safe_image1 = "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in truncate_image_name(m['image1_name'])])
        safe_image2 = "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in truncate_image_name(m['image2_name'])])
        safe_mask_name = "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in m['mask_name']])
        filename = f"JointPlot_{safe_image1}_vs_{safe_image2}_{safe_mask_name}.png"
    else:
        import time
        timestamp = int(time.time())
        filename = f"JointPlot_{timestamp}.png"

    path = os.path.join(output_dir, filename)
    g.savefig(path)
    
    plt.close()
    
    return filename
