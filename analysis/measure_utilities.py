import os
import numpy as np

def calculate_mask_measurements(asset_manager, image_names, mask_path):
    """
    Calculate measurements (intensity values) under the cluster mask ROI for a list of images.
    
    Args:
        asset_manager (AssetManager): Manager to get image data.
        image_names (list of str): Names of images to measure.
        mask_path (str): Path to the .npy mask file.
        
    Returns:
        dict: Dictionary mapping image name to the list of pixel intensities under the mask ROI.
    """
    if not os.path.exists(mask_path):
        return None
    
    try:
        mask = np.load(mask_path)
    except Exception:
        return None
    
    # Ensure mask is binary (0 or 1)
    mask = (mask > 0).astype(np.uint8)
    
    measurements = {}
    for name in image_names:
        asset = asset_manager.get_image_by_name(name)
        if asset:
            data = asset.get_rendered_data(data_only=True)
            if data is not None:
                # Get pixel values where mask is 1
                roi_values = data[mask == 1].flatten()
                measurements[name] = roi_values.tolist()
                
    return measurements
