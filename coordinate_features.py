import numpy as np

def get_scaled_coordinates(shape, x_weight=1.0, y_weight=1.0):
    """
    Generate scaled X and Y coordinate grids for a given image shape.
    
    Args:
        shape (tuple): Shape of the image (height, width).
        x_weight (float): Weight factor to scale the X coordinates.
        y_weight (float): Weight factor to scale the Y coordinates.
        
    Returns:
        tuple: (X, Y) coordinate grids of shape (height, width), 
               scaled to [0, x_weight] and [0, y_weight] respectively.
    """
    height, width = shape
    
    # Generate linear grids
    y_coords = np.linspace(0, y_weight, height, dtype=np.float32)
    x_coords = np.linspace(0, x_weight, width, dtype=np.float32)
    
    # Create 2D grids
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    return xx, yy

def add_coordinate_features(data, height, width, x_weight=1.0, y_weight=1.0):
    """
    Append scaled X and Y coordinates as features to the pixel data.
    
    Args:
        data (np.ndarray): Pixel data of shape (H*W, N).
        height (int): Height of the image.
        width (int): Width of the image.
        x_weight (float): Weight factor for the X coordinates.
        y_weight (float): Weight factor for the Y coordinates.
        
    Returns:
        np.ndarray: Augmented data of shape (H*W, N+2).
    """
    xx, yy = get_scaled_coordinates((height, width), x_weight=x_weight, y_weight=y_weight)
    
    # Flatten and reshape for concatenation
    xx_flat = xx.reshape(-1, 1)
    yy_flat = yy.reshape(-1, 1)
    
    # Append to data
    augmented_data = np.hstack([data, xx_flat, yy_flat])
    
    return augmented_data
