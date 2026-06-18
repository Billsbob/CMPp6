import os
import numpy as np

def calculate_mask_measurements(asset_manager, image_names, mask_name):
    """
    Calculate measurements (intensity values) under the cluster mask ROI for a list of images.
    
    Args:
        asset_manager (AssetManager): Manager to get image data and mask data.
        image_names (list of str): Names of images to measure.
        mask_name (str): Name of the mask asset.
        
    Returns:
        dict: Dictionary mapping image name to the list of pixel intensities under the mask ROI.
    """
    mask_asset = asset_manager.get_mask_by_name(mask_name)
    if not mask_asset:
        return None
    
    mask = mask_asset.get_rendered_data(data_only=True)
    if mask is None:
        return None
    
    # Ensure mask is binary (0 or 1)
    mask = (mask > 0).astype(np.uint8)
    
    measurements = {}
    for name in image_names:
        asset = asset_manager.get_image_by_name(name)
        if asset:
            data = asset.get_rendered_data(data_only=True)
            if data is not None:
                # Handle potential size mismatch if mask and image have different shapes
                if data.shape[:2] != mask.shape[:2]:
                    import cv2
                    resized_mask = cv2.resize(mask, (data.shape[1], data.shape[0]), interpolation=cv2.INTER_NEAREST)
                    roi_values = data[resized_mask == 1].flatten()
                else:
                    # Get pixel values where mask is 1
                    roi_values = data[mask == 1].flatten()
                measurements[name] = roi_values.tolist()
                
    return measurements
