import numpy as np
import cv2

def _apply_separable_filter(data, kernel_1d):
    # Use OpenCV for fast separable filter
    return cv2.sepFilter2D(data, -1, kernel_1d, kernel_1d)

def apply_gaussian_blur(data, radius=2):
    if radius <= 0:
        return data
    # In OpenCV, ksize should be odd. 
    # radius in PIL was sigma.
    sigma = float(radius)
    ksize = int(round(sigma * 3.5)) * 2 + 1
    if ksize % 2 == 0: ksize += 1
    return cv2.GaussianBlur(data, (ksize, ksize), sigma)

def apply_median_filter(data, size=3):
    if size % 2 == 0: size += 1
    # OpenCV's medianBlur supports uint8, uint16, float32
    return cv2.medianBlur(data, size)

def apply_mean_filter(data, size=3):
    if size <= 0: return data
    return cv2.blur(data, (size, size))

def apply_blur(data, size=3):
    return apply_mean_filter(data, size=size)

def apply_unsharp_mask(data, radius=2, percent=150, threshold=3):
    blurred = apply_gaussian_blur(data, radius=radius)
    diff = data - blurred
    mask = np.abs(diff) > threshold
    sharpened = data + diff * (percent / 100.0)
    return np.where(mask, sharpened, data).astype(data.dtype)

def rotate_image(data, angle, expand=False, crop_border=5):
    if angle == 0:
        if crop_border > 0:
            h, w = data.shape[:2]
            if h > 2 * crop_border and w > 2 * crop_border:
                return data[crop_border:-crop_border, crop_border:-crop_border].copy()
        return data

    h, w = data.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    if expand:
        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]
        rotated = cv2.warpAffine(data, M, (nW, nH))
    else:
        rotated = cv2.warpAffine(data, M, (w, h))

    if crop_border > 0:
        rh, rw = rotated.shape[:2]
        if rh > 2 * crop_border and rw > 2 * crop_border:
            rotated = rotated[crop_border:-crop_border, crop_border:-crop_border].copy()
            
    return rotated

def crop_image(data, x, y, width, height, crop_border=5):
    h_orig, w_orig = data.shape[:2]
    x1 = max(0, min(int(x), w_orig - 1))
    y1 = max(0, min(int(y), h_orig - 1))
    x2 = max(x1 + 1, min(int(x + width), w_orig))
    y2 = max(y1 + 1, min(int(y + height), h_orig))
    cropped = data[y1:y2, x1:x2].copy()
    if crop_border > 0:
        ch, cw = cropped.shape[:2]
        if ch > 2 * crop_border and cw > 2 * crop_border:
            cropped = cropped[crop_border:-crop_border, crop_border:-crop_border].copy()
    return cropped
