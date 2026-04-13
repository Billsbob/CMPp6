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
    # PIL's MedianFilter supports mode 'F' (float32)
    img = Image.fromarray(data)
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

def rotate_image(data, angle, expand=False):
    """
    Rotate image by angle (in degrees).
    Uses PIL for rotation because it handles interpolation and expansion well.
    """
    if angle == 0:
        return data
        
    img = Image.fromarray(data)
    # PIL.Image.rotate: resample=BICUBIC is good for quality
    rotated_img = img.rotate(angle, resample=Image.BICUBIC, expand=expand)
    return np.array(rotated_img).astype(data.dtype)

def crop_image(data, x, y, width, height):
    """
    Crop image to (x, y, x+width, y+height).
    Handles bounds checking.
    """
    h_orig, w_orig = data.shape[:2]
    
    # Simple bounds checking
    x1 = max(0, min(x, w_orig - 1))
    y1 = max(0, min(y, h_orig - 1))
    x2 = max(x1 + 1, min(x + width, w_orig))
    y2 = max(y1 + 1, min(y + height, h_orig))
    
    return data[int(y1):int(y2), int(x1):int(x2)].copy()
