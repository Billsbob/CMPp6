import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

def refine_mask(mask, min_area=0, min_circularity=0.0, max_eccentricity=1.0, 
                min_solidity=0.0, min_extent=0.0, euler_number=None):
    """
    Refines a binary mask based on area, circularity, eccentricity, solidity, extent, and euler number.
    
    Args:
        mask (np.ndarray): Binary mask to refine.
        min_area (int): Minimum area (in pixels) for an object to keep.
        min_circularity (float): Minimum circularity (4*pi*area/perimeter^2) to keep.
        max_eccentricity (float): Maximum eccentricity to keep.
        min_solidity (float): Minimum solidity (ratio of pixels in the region to pixels of the convex hull image) to keep.
        min_extent (float): Minimum extent (ratio of pixels in the region to pixels in the total bounding box) to keep.
        euler_number (int, optional): Euler number of the region to keep. If None, it's ignored.
        
    Returns:
        np.ndarray: Refined binary mask.
    """
    if mask is None:
        return None
        
    # Ensure binary mask
    binary_mask = (mask > 0).astype(int)
    
    # Label connected components
    label_image = label(binary_mask)
    refined_mask = np.zeros_like(binary_mask)
    
    # Iterate through each region
    for region in regionprops(label_image):
        # Calculate circularity
        # Avoid division by zero if perimeter is 0
        if region.perimeter == 0:
            circularity = 0.0
        else:
            circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
        
        # Check criteria
        keep = (region.area >= min_area and 
                circularity >= min_circularity and 
                region.eccentricity <= max_eccentricity and
                region.solidity >= min_solidity and
                region.extent >= min_extent)
        
        if euler_number is not None:
            keep = keep and (region.euler_number == euler_number)

        if keep:
            # Keep this region
            refined_mask[label_image == region.label] = 1
            
    return refined_mask.astype(np.uint8)

def get_mask_properties(mask):
    """
    Extracts properties for each component in the mask.
    
    Args:
        mask (np.ndarray): Binary mask.
        
    Returns:
        list of dict: List containing properties for each labeled region.
    """
    if mask is None:
        return []
        
    binary_mask = (mask > 0).astype(int)
    label_image = label(binary_mask)
    
    properties = []
    for region in regionprops(label_image):
        if region.perimeter == 0:
            circularity = 0.0
        else:
            circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
            
        properties.append({
            "Label": region.label,
            "Area": region.area,
            "Eccentricity": round(region.eccentricity, 4),
            "Circularity": round(circularity, 4),
            "Solidity": round(region.solidity, 4),
            "Extent": round(region.extent, 4),
            "Euler Number": region.euler_number
        })
        
    return properties

def merge_masks(masks):
    """
    Merges multiple binary masks into one using logical OR.
    Resizes masks to match the first mask's shape if necessary.
    
    Args:
        masks (list of np.ndarray): List of binary masks.
        
    Returns:
        np.ndarray: Merged binary mask (uint8).
    """
    if not masks:
        return None
        
    import cv2
    merged_mask = None
    
    for mask in masks:
        if mask is None:
            continue
            
        if merged_mask is None:
            merged_mask = (mask > 0)
        else:
            if mask.shape[:2] != merged_mask.shape[:2]:
                mask_resized = cv2.resize(mask.astype(np.uint8), 
                                         (merged_mask.shape[1], merged_mask.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
                mask_bool = (mask_resized > 0)
            else:
                mask_bool = (mask > 0)
            merged_mask = np.logical_or(merged_mask, mask_bool)
            
    if merged_mask is None:
        return None
        
    return merged_mask.astype(np.uint8)

def create_threshold_mask(stack, threshold, normalize=False):
    """
    Creates a threshold mask from an image stack.
    
    Args:
        stack (np.ndarray): Image stack (N, H, W).
        threshold (float): Threshold value.
        normalize (bool): Whether to normalize the stack before thresholding.
        
    Returns:
        np.ndarray: Binary mask (uint8).
    """
    if stack is None or len(stack) == 0:
        return None
        
    processed_stack = stack.astype(np.float32)
    
    if normalize:
        s_min, s_max = processed_stack.min(), processed_stack.max()
        if s_max > s_min:
            processed_stack = (processed_stack - s_min) / (s_max - s_min)
        else:
            processed_stack = np.zeros_like(processed_stack)
            
    max_projection = np.max(processed_stack, axis=0)
    binary_mask = (max_projection > threshold).astype(np.uint8)
    
    return binary_mask
