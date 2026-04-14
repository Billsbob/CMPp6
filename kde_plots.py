import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

def create_joint_kde_plot(all_measurements, output_dir):
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

    plt.figure(figsize=(10, 10))
    # We use a color palette for multiple sets
    palette = sns.color_palette("husl", len(all_measurements))
    
    # Create the JointGrid
    g = sns.JointGrid()
    
    for i, m in enumerate(all_measurements):
        x = m['x_values']
        y = m['y_values']
        label = f"Set {i+1}: {m['image1_name']} vs {m['image2_name']} ({m['mask_name']})"
        
        # Plot KDE in the joint area
        sns.kdeplot(x=x, y=y, ax=g.ax_joint, fill=True, color=palette[i], label=label, alpha=0.5)
        # Plot KDE in the marginal areas
        sns.kdeplot(x=x, ax=g.ax_marg_x, color=palette[i], fill=True)
        sns.kdeplot(y=y, ax=g.ax_marg_y, color=palette[i], fill=True)

    g.ax_joint.legend(loc='upper right', fontsize='small')
    
    # Set axis labels to image names
    x_label = "Intensity 1"
    y_label = "Intensity 2"
    if len(all_measurements) == 1:
        x_label = all_measurements[0]['image1_name']
        y_label = all_measurements[0]['image2_name']
    else:
        # Check if all sets have the same image1 and image2
        first_x = all_measurements[0]['image1_name']
        first_y = all_measurements[0]['image2_name']
        all_same_x = all(m['image1_name'] == first_x for m in all_measurements)
        all_same_y = all(m['image2_name'] == first_y for m in all_measurements)
        if all_same_x:
            x_label = first_x
        if all_same_y:
            y_label = first_y

    g.set_axis_labels(x_label, y_label)
    
    title = "Comparison of Joint KDE Plots"
    if len(all_measurements) == 1:
        m = all_measurements[0]
        title = f"Joint KDE Plot: {m['image1_name']} vs {m['image2_name']} under {m['mask_name']}"
    
    g.fig.suptitle(title)
    g.fig.subplots_adjust(top=0.9)
    
    # Generate a unique filename
    import time
    timestamp = int(time.time())
    filename = f"JointPlot_Comparison_{timestamp}.png"
    if len(all_measurements) == 1:
        m = all_measurements[0]
        safe_image1 = "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in m['image1_name']])
        safe_image2 = "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in m['image2_name']])
        safe_mask_name = "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in m['mask_name']])
        filename = f"JointPlot_{safe_image1}_vs_{safe_image2}_{safe_mask_name}.png"

    path = os.path.join(output_dir, filename)
    g.savefig(path)
    
    plt.close()
    
    return filename
