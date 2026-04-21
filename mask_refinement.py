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
