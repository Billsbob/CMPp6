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
        
        # Crop to max inscribed rectangle to match PIL behavior if requested
        # Actually, let's just use the rotated image if they want expand.
        # But wait, the original code had a get_max_inscribed_rect.
        # Let's keep that logic but use OpenCV for the transformation.
        
        def get_max_inscribed_rect(w, h, angle_deg):
            angle = np.radians(abs(angle_deg))
            angle = angle % (np.pi / 2)
            if angle > np.pi / 4:
                angle = np.pi / 2 - angle
                w, h = h, w
            if w <= 0 or h <= 0: return 0, 0
            width_is_longer = w >= h
            side_long, side_short = (w, h) if width_is_longer else (h, w)
            sin_a, cos_a = np.sin(angle), np.cos(angle)
            denom = cos_a**2 - sin_a**2
            if abs(denom) < 1e-10:
                wr = side_short / (sin_a + cos_a)
                return wr, wr
            if side_short <= 2 * side_long * sin_a * cos_a / (sin_a**2 + cos_a**2):
                x = 0.5 * side_short
                if width_is_longer: wr, hr = x / sin_a, x / cos_a
                else: wr, hr = x / cos_a, x / sin_a
            else:
                wr = (side_long * cos_a - side_short * sin_a) / denom
                hr = (side_short * cos_a - side_long * sin_a) / denom
            return wr, hr

        wr, hr = get_max_inscribed_rect(w, h, angle)
        h_new, w_new = rotated.shape[:2]
        x1 = int(max(0, (w_new - wr) / 2))
        y1 = int(max(0, (h_new - hr) / 2))
        x2 = int(min(w_new, x1 + wr))
        y2 = int(min(h_new, y1 + hr))
        rotated = rotated[y1:y2, x1:x2].copy()
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
