from __future__ import annotations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import Qt
import io
import scipy.stats as stats

def create_kdeplot_pixmap(
    image_data: np.ndarray, 
    color: QColor, 
    width: int, 
    height: int,
    y_limit: float | None = None
) -> QPixmap:
    """
    Generate a univariate Seaborn KDE plot for the given image data.
    
    Args:
        image_data: Numpy array of the image (grayscale)
        color: QColor to use for the plot line
        width: Desired width of the output pixmap
        height: Desired height of the output pixmap
        y_limit: Optional fixed maximum for the Y-axis (Density)
        
    Returns:
        QPixmap containing the KDE plot
    """
    # Flatten the image and exclude intensity 0 if desired (consistent with graphs)
    data = image_data.flatten()
    data = data[data > 0]
    
    if len(data) == 0:
        # Return empty transparent pixmap if no data
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        return pixmap

    # Create figure
    # We use a small figure size and then scale it to the requested pixmap size
    # to maintain performance and look consistent.
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # Convert QColor to hex or RGB for Seaborn
    hex_color = color.name()
    
    # Generate KDE plot
    sns.kdeplot(data, ax=ax, color=hex_color, fill=True, alpha=0.4, linewidth=2)
    
    # Set static intensity axis 0-255
    ax.set_xlim(0, 255)
    
    # Apply Y-limit for normalization across multiple plots
    if y_limit is not None:
        ax.set_ylim(0, y_limit)
    
    # Clean up the plot to look like an overlay
    text_bbox = dict(facecolor='lightgray', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
    ax.set_xlabel('Intensity', color='black', bbox=text_bbox)
    ax.set_ylabel('Density', color='black', bbox=text_bbox)
    
    # Tick labels background
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_bbox(text_bbox)

    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
        
    # Remove background
    fig.tight_layout()
    
    # Convert plot to QPixmap
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    width_canvas, height_canvas = canvas.get_width_height()
    buffer = canvas.buffer_rgba()
    image = QImage(buffer, width_canvas, height_canvas, width_canvas * 4, QImage.Format_RGBA8888)
    image.buffer = buffer # Keep reference
    pixmap = QPixmap.fromImage(image.copy())
    
    plt.close(fig)
    return pixmap

def create_joint_kdeplot_pixmap(
    images: list[np.ndarray] | dict, 
    colors: list[QColor], 
    width: int, 
    height: int,
    names: list[str] | None = None,
    max_points: int = 100_000,
    gridsize: int = 50,
    fill: bool = True
) -> QPixmap:
    """
    Generate a Seaborn joint KDE plot for the given list of images or pre-calculated values.
    """
    if isinstance(images, dict):
        x = np.array(images.get('x', []))
        y = np.array(images.get('y', []))
    elif len(images) >= 2:
        # Flatten the images for pixel-to-pixel comparison
        img1 = images[0].flatten()
        img2 = images[1].flatten()
        
        # Create common mask: keep only overlapping signal pixels (> 0 in both)
        common_mask = (img1 > 0) & (img2 > 0)
        x = img1[common_mask]
        y = img2[common_mask]
    else:
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        return pixmap

    n = len(x)
    if n == 0:
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        return pixmap

    # Randomly sample if the number of points is too large
    if n > max_points:
        rng = np.random.default_rng()
        idx = rng.choice(n, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
    
    # Convert to float for KDE, keeping values in 0-255 range
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    import pandas as pd
    label1 = names[0] if names and len(names) > 0 else "Image 1"
    label2 = names[1] if names and len(names) > 1 else "Image 2"
    df = pd.DataFrame({label1: x, label2: y})

    # Use first color for the bivariate plot
    color = colors[0].name()
    
    # Generate Joint KDE plot
    g = sns.jointplot(
        data=df, x=label1, y=label2,
        kind="kde",
        color=color,
        fill=fill,
        thresh=0.05,
        gridsize=gridsize,
        marginal_kws=dict(fill=True)
    )
    
    fig = g.fig
    fig.patch.set_facecolor('none')
    for ax in fig.axes:
        ax.set_facecolor('none')

    # Set axes limits
    g.ax_joint.set_xlim(0, 255)
    g.ax_joint.set_ylim(0, 255)
    
    # Clean up the plot
    text_bbox = dict(facecolor='lightgray', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
    g.ax_joint.set_xlabel(f"{label1} Intensity", color='black', bbox=text_bbox)
    g.ax_joint.set_ylabel(f"{label2} Intensity", color='black', bbox=text_bbox)
    
    # Tick labels background for all axes
    for ax in fig.axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_bbox(text_bbox)
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2)
        
    fig.set_size_inches(width/100, height/100)
    fig.tight_layout()
    
    # Convert plot to QPixmap
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    width_canvas, height_canvas = canvas.get_width_height()
    buffer = canvas.buffer_rgba()
    image = QImage(buffer, width_canvas, height_canvas, width_canvas * 4, QImage.Format_RGBA8888)
    image.buffer = buffer # Keep reference
    pixmap = QPixmap.fromImage(image.copy())
    
    plt.close(fig)
    return pixmap

def get_kde_max_density(image_data: np.ndarray) -> float:
    """
    Calculate the maximum density value of a KDE plot for the given image data,
    using seaborn/matplotlib without requiring SciPy.
    
    Args:
        image_data: Numpy array of the image (grayscale)
        
    Returns:
        The maximum density value found in the KDE.
    """
    data = image_data.flatten()
    data = data[data > 0]
    if len(data) == 0:
        return 0.0

    # Create a tiny offscreen figure to compute KDE and read back the curve
    try:
        fig, ax = plt.subplots(figsize=(2, 1), dpi=100)
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        sns.kdeplot(data, ax=ax, color='black', fill=False)
        ax.set_xlim(0, 255)
        ymax = 0.0
        # Extract max from plotted line(s)
        for line in ax.lines:
            xdata, ydata = line.get_data()
            if ydata is not None and len(ydata) > 0:
                m = float(np.nanmax(ydata))
                if m > ymax:
                    ymax = m
        plt.close(fig)
        return ymax if ymax > 0 else 0.0
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return 0.0

def get_bivariate_kde_data(
    images: list[np.ndarray], 
    max_points: int = 200_000
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract the sampled data points used for a bivariate KDE plot.
    """
    if len(images) < 2:
        return None

    # Flatten the images for pixel-to-pixel comparison
    img1 = images[0].flatten()
    img2 = images[1].flatten()
    
    # Create common mask: keep only overlapping signal pixels (> 0 in both)
    common_mask = (img1 > 0) & (img2 > 0)
    x = img1[common_mask]
    y = img2[common_mask]

    n = len(x)
    if n == 0:
        return None

    # Randomly sample if the number of points is too large
    if n > max_points:
        rng = np.random.default_rng()
        idx = rng.choice(n, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
    
    # Convert to float for KDE, keeping values in 0-255 range
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    
    return x, y

def create_bivariate_kdeplot_pixmap(
    images: list[np.ndarray] | dict, 
    colors: list[QColor], 
    width: int, 
    height: int,
    names: list[str] | None = None,
    max_points: int = 200_000,
    gridsize: int = 50,
    fill: bool = True
) -> QPixmap:
    """
    Generate a bivariate Seaborn KDE plot for the given list of images or pre-calculated values.
    """
    if isinstance(images, dict):
        x = np.array(images.get('x', []))
        y = np.array(images.get('y', []))
    elif len(images) >= 2:
        # Flatten the images for pixel-to-pixel comparison
        img1 = images[0].flatten()
        img2 = images[1].flatten()
        
        # Create common mask: keep only overlapping signal pixels (> 0 in both)
        common_mask = (img1 > 0) & (img2 > 0)
        x = img1[common_mask]
        y = img2[common_mask]
    else:
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        return pixmap

    n = len(x)
    if n == 0:
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        return pixmap

    # Randomly sample if the number of points is too large
    if n > max_points:
        rng = np.random.default_rng()
        idx = rng.choice(n, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
    
    # Convert to float for KDE, keeping values in 0-255 range
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # Use first color for the bivariate plot
    color = colors[0].name()
    
    # Generate Bivariate KDE plot with optimized parameters
    # gridsize reduces evaluation resolution for speed
    sns.kdeplot(
        x=x, y=y, 
        ax=ax, 
        color=color, 
        fill=fill, 
        alpha=0.6, 
        thresh=0.05,
        gridsize=gridsize
    )
    
    # Clean up the plot
    text_bbox = dict(facecolor='lightgray', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
    label1 = f"{names[0]} Intensity" if names and len(names) > 0 else "Image 1 Intensity"
    label2 = f"{names[1]} Intensity" if names and len(names) > 1 else "Image 2 Intensity"
    ax.set_xlabel(label1, color='black', bbox=text_bbox)
    ax.set_ylabel(label2, color='black', bbox=text_bbox)
    
    # Tick labels background
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_bbox(text_bbox)

    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
        
    fig.tight_layout()
    
    # Convert plot to QPixmap
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    width_canvas, height_canvas = canvas.get_width_height()
    buffer = canvas.buffer_rgba()
    image = QImage(buffer, width_canvas, height_canvas, width_canvas * 4, QImage.Format_RGBA8888)
    image.buffer = buffer # Keep reference
    pixmap = QPixmap.fromImage(image.copy())
    
    plt.close(fig)
    return pixmap

def create_ridgeplot_pixmap(
    images: list[np.ndarray | list], 
    colors: list[QColor], 
    width: int, 
    height: int,
    names: list[str] | None = None,
    max_points: int = 100_000
) -> QPixmap:
    """
    Generate a Seaborn ridge plot for the given list of images or pre-sampled values.
    """
    if not images:
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        return pixmap

    import pandas as pd
    
    # Prepare data for Seaborn: a long-form DataFrame
    all_data = []
    for i, img in enumerate(images):
        if isinstance(img, np.ndarray):
            data = img.flatten()
            data = data[data > 0] # Keep only signal
        else:
            # Assume it's a list of pre-sampled values
            data = np.array(img)
        
        if len(data) > max_points:
            rng = np.random.default_rng()
            data = rng.choice(data, size=max_points, replace=False)
        
        name = names[i] if names and i < len(names) else f"Image {i+1}"
        df = pd.DataFrame({
            'Intensity': data,
            'Image': name
        })
        all_data.append(df)
        
    if not all_data:
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        return pixmap
        
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Create the ridge plot using FacetGrid
    # For a monochromatic look, we use a single color for all rows instead of a palette.
    # We'll use the first color in the list (if any) or default to black.
    main_color = colors[0].name() if colors else "black"
    
    sns.set_theme(style="white", rc={"axes.facecolor": "none"})
    g = sns.FacetGrid(df_combined, row="Image", hue="Image", aspect=15, height=.5)
    g.figure.patch.set_facecolor('none')

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "Intensity",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5, color=main_color)
    g.map(sns.kdeplot, "Intensity", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping (which we've overridden with main_color in kdeplot)
    # but to be safe, we explicitly set color to main_color here too.
    g.refline(y=0, linewidth=2, linestyle="-", color=main_color, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label, **kwargs):
        ax = plt.gca()
        # Use a slightly larger negative offset and ha="right" to place to the side
        # Use main_color for labels to keep it monochromatic
        text_bbox = dict(facecolor='lightgray', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
        ax.text(-0.02, .2, label, fontweight="bold", color=main_color,
                ha="right", va="center", transform=ax.transAxes,
                bbox=text_bbox)


    g.map(label, "Intensity")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25, left=0.15)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    
    # Set X axis range
    text_bbox = dict(facecolor='lightgray', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
    for ax in g.axes.flat:
        ax.set_xlim(0, 255)
        # Tick labels background
        for tick_label in ax.get_xticklabels():
            tick_label.set_bbox(text_bbox)
        ax.tick_params(axis='x', colors='black')
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2)
    
    g.figure.set_size_inches(width/100, height/100)
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied")
        g.figure.tight_layout()

    # Convert plot to QPixmap
    canvas = FigureCanvas(g.figure)
    canvas.draw()
    
    width_canvas, height_canvas = canvas.get_width_height()
    image_data = canvas.buffer_rgba()
    image = QImage(image_data, width_canvas, height_canvas, width_canvas * 4, QImage.Format_RGBA8888)
    image.ndarray = image_data # Keep reference
    pixmap = QPixmap.fromImage(image.copy())
    
    plt.close(g.figure)
    sns.set_theme() # Reset to default
    
    return pixmap

def create_graph_plot_pixmap(
    image_data: np.ndarray,
    mask_data: np.ndarray,
    color: QColor,
    width: int,
    height: int,
    bins: int = 256
) -> QPixmap:
    """
    Generate a Seaborn graph plot for the given image data, restricted by mask.
    """
    # Apply mask and flatten
    data = image_data[mask_data > 0].flatten()
    
    if len(data) == 0:
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        return pixmap

    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    hex_color = color.name()
    
    sns.histplot(data, bins=bins, binrange=(0, 255), ax=ax, color=hex_color, element="step", alpha=0.4)
    
    ax.set_xlim(0, 255)
    
    # Clean up the plot
    text_bbox = dict(facecolor='lightgray', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
    ax.set_xlabel('Intensity', color='black', bbox=text_bbox)
    ax.set_ylabel('Count', color='black', bbox=text_bbox)
    
    # Tick labels background
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_bbox(text_bbox)

    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
        
    fig.tight_layout()
    
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    width_canvas, height_canvas = canvas.get_width_height()
    buffer = canvas.buffer_rgba()
    image = QImage(buffer, width_canvas, height_canvas, width_canvas * 4, QImage.Format_RGBA8888)
    image.buffer = buffer # Keep reference
    pixmap = QPixmap.fromImage(image.copy())
    
    plt.close(fig)
    return pixmap

def create_multi_graph_plot_pixmap(
    datasets: list[np.ndarray | dict],
    colors: list[QColor],
    labels: list[str],
    width: int,
    height: int,
    bins: int = 256
) -> QPixmap:
    """
    Generate a combined Seaborn graph plot for multiple datasets.
    The y-axis is shared across all plots based on the largest y-axis value.
    datasets can be list of raw data or list of dicts with 'counts' and 'bins'.
    """
    if not datasets:
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        return pixmap

    import pandas as pd
    
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    any_plotted = False
    for data, label, color in zip(datasets, labels, colors):
        if isinstance(data, dict) and 'counts' in data and 'bins' in data:
            # Pre-calculated histogram
            counts = np.array(data['counts'])
            bin_edges = np.array(data['bins'])
            # Center of bins
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.step(bin_centers, counts, where='mid', color=color.name(), label=label, alpha=0.7)
            ax.fill_between(bin_centers, counts, step='mid', alpha=0.3, color=color.name())
            any_plotted = True
        elif isinstance(data, np.ndarray) and len(data) > 0:
            # Raw data
            sns.histplot(x=data, bins=bins, binrange=(0, 255), 
                         ax=ax, color=color.name(), element="step", alpha=0.3, label=label)
            any_plotted = True
        
    if not any_plotted:
        plt.close(fig)
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        return pixmap
    
    ax.set_xlim(0, 255)
    
    # Clean up the plot
    text_bbox = dict(facecolor='lightgray', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
    ax.set_xlabel('Intensity', color='black', bbox=text_bbox)
    ax.set_ylabel('Count', color='black', bbox=text_bbox)
    
    # Tick labels background
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_bbox(text_bbox)

    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
        
    # Counting how many datasets were actually added to labels for legend
    num_plotted = 0
    for data in datasets:
        if isinstance(data, dict) and 'counts' in data and 'bins' in data:
            num_plotted += 1
        elif isinstance(data, np.ndarray) and len(data) > 0:
            num_plotted += 1
            
    if num_plotted > 1:
        legend = ax.legend()
        legend.get_frame().set_facecolor('lightgray')
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_edgecolor('none')
        
    fig.tight_layout()
    
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    width_canvas, height_canvas = canvas.get_width_height()
    buffer = canvas.buffer_rgba()
    image = QImage(buffer, width_canvas, height_canvas, width_canvas * 4, QImage.Format_RGBA8888)
    image.buffer = buffer # Keep reference
    pixmap = QPixmap.fromImage(image.copy())
    
    plt.close(fig)
    return pixmap
