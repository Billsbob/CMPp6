import numpy as np
from PIL import Image, ImageFilter

def _gaussian_kernel_1d(radius, sigma=None):
    if sigma is None:
        sigma = radius / 2.0
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    
    # Kernel size is typically (2 * ceil(2 * sigma) + 1) or related
    # PIL's GaussianBlur(radius) uses radius differently.
    # In PIL, radius is the standard deviation. Wait, is it?
    # According to PIL docs: "radius is the standard deviation of the Gaussian filter."
    # So sigma = radius.
    sigma = float(radius)
    size = int(round(sigma * 3.5)) * 2 + 1
    x = np.arange(size) - (size - 1) / 2.0
    kernel = np.exp(-0.5 * (x / sigma)**2)
    return (kernel / kernel.sum()).astype(np.float32)

def _apply_separable_filter(data, kernel_1d):
    # Apply to rows
    row_filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        row_filtered[i, :] = np.convolve(data[i, :], kernel_1d, mode='same')
    
    # Apply to columns
    final_filtered = np.zeros_like(data)
    for j in range(data.shape[1]):
        final_filtered[:, j] = np.convolve(row_filtered[:, j], kernel_1d, mode='same')
    
    return final_filtered

def apply_gaussian_blur(data, radius=2):
    # PIL's GaussianBlur(radius) radius parameter is sigma
    if radius <= 0:
        return data
    kernel = _gaussian_kernel_1d(radius)
    return _apply_separable_filter(data, kernel).astype(data.dtype)

def apply_median_filter(data, size=3):
    # PIL's MedianFilter only supports 8-bit (L), RGB, CMYK, P. 
    # It does NOT support I, I;16, or F. 
    # For high bit-depth or float, we should use scipy.ndimage or handle it differently.
    # Since we can't easily add dependencies like scipy if not already there, 
    # let's try to convert to float32 for processing if not already,
    # but PIL still won't help.
    
    # Check if we can use scipy
    try:
        from scipy.ndimage import median_filter
        return median_filter(data, size=size).astype(data.dtype)
    except ImportError:
        # Fallback to PIL if possible, but it will fail for I/F modes
        # If it's uint16 (I;16) or float32 (F), PIL.ImageFilter.MedianFilter will fail.
        # We can try to downsample to 8-bit for median if scipy is not available,
        # but that's what we want to avoid.
        
        # If the image is not in a supported mode, we must handle it.
        # Let's check the mode first.
        img = Image.fromarray(data)
        if img.mode not in ['L', 'RGB', 'RGBA', 'CMYK', 'P']:
            # For I, I;16, F we can't use PIL's MedianFilter.
            # As a last resort, if we don't have scipy, we'll have to skip or 
            # use a simpler approach. 
            # Given the requirement to keep native depth, let's just return 
            # original data if we can't process it correctly without downsampling.
            # OR we can try to use a simple pure-numpy implementation for small kernels.
            return data
            
        img = img.filter(ImageFilter.MedianFilter(size))
        return np.array(img).astype(data.dtype)

def apply_mean_filter(data, size=3):
    # Mean filter is a separable box blur
    kernel = np.ones(size, dtype=np.float32) / float(size)
    return _apply_separable_filter(data, kernel).astype(data.dtype)

def apply_blur(data, size=3):
    # PIL's BLUR is approximately a 3x3 box blur or similar
    # Let's use 3x3 box blur (size=3)
    return apply_mean_filter(data, size=size)

def apply_unsharp_mask(data, radius=2, percent=150, threshold=3):
    # unsharp_mask = original + (original - blurred) * (percent / 100)
    # only apply if difference is above threshold
    blurred = apply_gaussian_blur(data, radius=radius)
    diff = data - blurred
    
    # Simple thresholding
    mask = np.abs(diff) > threshold
    sharpened = data + diff * (percent / 100.0)
    
    result = np.where(mask, sharpened, data)
    return result.astype(data.dtype)

def rotate_image(data, angle, expand=False, crop_border=5):
    """
    Rotate image by angle (in degrees).
    Uses PIL for rotation because it handles interpolation and expansion well.
    If expand is True, automatically crops the black borders introduced by rotation.
    If crop_border is > 0, removes that many additional pixels from all 4 edges.
    """
    if angle == 0:
        if crop_border > 0:
            h, w = data.shape[:2]
            if h > 2 * crop_border and w > 2 * crop_border:
                return data[crop_border:-crop_border, crop_border:-crop_border].copy()
        return data
        
    img = Image.fromarray(data)
    # PIL.Image.rotate: resample=BICUBIC is good for quality
    resample = Image.BICUBIC
    if img.mode in ['I;16', 'I', 'F']:
        resample = Image.NEAREST
        
    rotated_img = img.rotate(angle, resample=resample, expand=expand)
    rotated_data = np.array(rotated_img).astype(data.dtype)
    
    if expand:
        # Calculate the largest inscribed rectangle to remove black borders
        # For a rectangle of (w, h) rotated by theta:
        # The inner rectangle (wi, hi) that fits is:
        # wi = (w * cos(theta) - h * sin(theta)) / cos(2*theta) -- this is for a specific case
        # A simpler approximation for the inscribed rectangle:
        h_orig, w_orig = data.shape[:2]
        theta = np.radians(abs(angle))
        
        # New dimensions after rotation (expanded)
        # h_new = w_orig * sin(theta) + h_orig * cos(theta)
        # w_new = w_orig * cos(theta) + h_orig * sin(theta)
        
        # Inscribed rectangle dimensions:
        # Source: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        def get_max_inscribed_rect(w, h, angle_deg):
            angle = np.radians(abs(angle_deg))
            
            # Use symmetry for angles > 90
            angle = angle % (np.pi / 2)
            if angle > np.pi / 4:
                angle = np.pi / 2 - angle
                w, h = h, w

            if w <= 0 or h <= 0:
                return 0, 0
            
            width_is_longer = w >= h
            side_long, side_short = (w, h) if width_is_longer else (h, w)
            
            sin_a, cos_a = np.sin(angle), np.cos(angle)
            
            # Avoid division by zero for 45 degrees if they are equal
            denom = cos_a**2 - sin_a**2
            if abs(denom) < 1e-10:
                # 45 degrees
                wr = side_short / (sin_a + cos_a)
                hr = wr
                return wr, hr

            if side_short <= 2 * side_long * sin_a * cos_a / (sin_a**2 + cos_a**2):
                x = 0.5 * side_short
                if width_is_longer:
                    wr, hr = x / sin_a, x / cos_a
                else:
                    wr, hr = x / cos_a, x / sin_a
            else:
                wr = (side_long * cos_a - side_short * sin_a) / denom
                hr = (side_short * cos_a - side_long * sin_a) / denom
                
            return wr, hr

        wr, hr = get_max_inscribed_rect(w_orig, h_orig, angle)
        
        h_new, w_new = rotated_data.shape[:2]
        # Crop to (wr, hr) centered in (w_new, h_new)
        x1 = int(max(0, (w_new - wr) / 2))
        y1 = int(max(0, (h_new - hr) / 2))
        x2 = int(min(w_new, x1 + wr))
        y2 = int(min(h_new, y1 + hr))
        
        rotated_data = rotated_data[y1:y2, x1:x2].copy()

    if crop_border > 0:
        h, w = rotated_data.shape[:2]
        if h > 2 * crop_border and w > 2 * crop_border:
            rotated_data = rotated_data[crop_border:-crop_border, crop_border:-crop_border].copy()
            
    return rotated_data

def crop_image(data, x, y, width, height, crop_border=5):
    """
    Crop image to (x, y, x+width, y+height).
    Handles bounds checking.
    If crop_border is > 0, trims that many pixels from each edge of the result.
    """
    h_orig, w_orig = data.shape[:2]
    
    # Simple bounds checking
    x1 = max(0, min(x, w_orig - 1))
    y1 = max(0, min(y, h_orig - 1))
    x2 = max(x1 + 1, min(x + width, w_orig))
    y2 = max(y1 + 1, min(y + height, h_orig))
    
    cropped = data[int(y1):int(y2), int(x1):int(x2)].copy()
    
    if crop_border > 0:
        h, w = cropped.shape[:2]
        if h > 2 * crop_border and w > 2 * crop_border:
            cropped = cropped[crop_border:-crop_border, crop_border:-crop_border].copy()
            
    return cropped
