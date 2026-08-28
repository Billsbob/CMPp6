import os
import numpy as np
from scipy.signal import find_peaks
from scipy import stats

def calculate_histogram_features(values, bins=256):
    """
    Calculate histogram-based features: mode, peak intensity, peak prominence,
    number of peaks, and secondary peak location.
    
    Args:
        values (list or np.ndarray): Pixel intensity values.
        bins (int): Number of bins for histogram calculation.
        
    Returns:
        dict: Dictionary containing the calculated features.
    """
    if not values or len(values) == 0:
        return {
            'mode': 0,
            'peak intensity': 0,
            'peak prominence': 0,
            'number of peaks': 0,
            'secondary peak location': 0
        }
    
    v = np.array(values)
    
    # Exact mode for discrete values, but bin-based mode is safer for normalized/float data
    if np.issubdtype(v.dtype, np.integer):
        m = stats.mode(v, keepdims=True)
        mode_val = float(m.mode[0])
    else:
        # For float data, use the peak of the histogram as mode
        counts, bin_edges = np.histogram(v, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mode_val = float(bin_centers[np.argmax(counts)])
        
    # Peak detection using histogram
    counts, bin_edges = np.histogram(v, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize counts for prominence thresholding
    max_count = np.max(counts)
    if max_count == 0:
        return {
            'mode': mode_val,
            'peak intensity': 0,
            'peak prominence': 0,
            'number of peaks': 0,
            'secondary peak location': 0
        }
        
    peaks, properties = find_peaks(counts, prominence=max_count * 0.05)
    
    num_peaks = len(peaks)
    peak_intensity = 0.0
    peak_prominence = 0.0
    secondary_peak_location = 0.0
    
    if num_peaks > 0:
        # Find indices of peaks sorted by their height (counts)
        sorted_peak_indices = peaks[np.argsort(counts[peaks])[::-1]]
        
        main_peak_idx = sorted_peak_indices[0]
        peak_intensity = float(bin_centers[main_peak_idx])
        
        # find index in 'peaks' array to get prominence from properties
        main_peak_in_peaks_idx = np.where(peaks == main_peak_idx)[0][0]
        peak_prominence = float(properties['prominences'][main_peak_in_peaks_idx])
        
        if num_peaks > 1:
            secondary_peak_idx = sorted_peak_indices[1]
            secondary_peak_location = float(bin_centers[secondary_peak_idx])
            
    return {
        'mode': mode_val,
        'peak intensity': peak_intensity,
        'peak prominence': peak_prominence,
        'number of peaks': num_peaks,
        'secondary peak location': secondary_peak_location
    }

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
