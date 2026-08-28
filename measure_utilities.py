import os
import numpy as np

def calculate_mask_measurements(asset_manager, image_names, mask_name, normalize=False, normalize_stack=False):
    """
    Calculate measurements (intensity values) under the cluster mask ROI for a list of images.
    
    Args:
        asset_manager (AssetManager): Manager to get image data and mask data.
        image_names (list of str): Names of images to measure.
        mask_name (str): Name of the mask asset.
        normalize (bool): Whether to normalize each image individually (local normalization).
        normalize_stack (bool): Whether to normalize the entire stack (global normalization).
        
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
    
    # Collect all image data first if normalization is required
    image_data_list = []
    valid_image_names = []
    
    for name in image_names:
        asset = asset_manager.get_image_by_name(name)
        if asset:
            data = asset.get_rendered_data(data_only=True)
            if data is not None:
                image_data_list.append(data)
                valid_image_names.append(name)
    
    if not image_data_list:
        return {}

    # Apply normalization if requested
    if normalize or normalize_stack:
        import clustering
        try:
            # Try to stack images to (N, H, W)
            stack = np.stack(image_data_list, axis=0)
            normalized_stack = clustering._apply_normalization(stack, normalize_stack=normalize_stack, normalize=normalize)
            # Convert back to list of arrays
            image_data_list = [normalized_stack[i] for i in range(normalized_stack.shape[0])]
        except ValueError:
            # Fallback if images have different shapes
            if normalize_stack:
                all_min = min(np.min(img) for img in image_data_list)
                all_max = max(np.max(img) for img in image_data_list)
                if all_max > all_min:
                    image_data_list = [(img - all_min) / (all_max - all_min) for img in image_data_list]
                else:
                    image_data_list = [np.zeros_like(img) for img in image_data_list]
            
            if normalize:
                new_data_list = []
                for img in image_data_list:
                    d_min, d_max = np.min(img), np.max(img)
                    if d_max > d_min:
                        new_data_list.append((img - d_min) / (d_max - d_min))
                    else:
                        new_data_list.append(np.zeros_like(img))
                image_data_list = new_data_list

    measurements = {}
    for name, data in zip(valid_image_names, image_data_list):
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
