import numpy as np
import qimage2ndarray
from PySide6.QtGui import QImage
import os
import cv2
import json

class ImageDisplayHandler:
    COLORS = {
        "grayscale": (1, 1, 1),
        "red": (1, 0, 0),
        "green": (0, 1, 0),
        "blue": (0, 0, 1),
        "cyan": (0, 1, 1),
        "magenta": (1, 0, 1),
        "yellow": (1, 1, 0),
        "white": (1, 1, 1),
    }

    def __init__(self):
        self.visible_assets = set()
        self.asset_colors = {}
        self._cached_image_composite = None
        self._cached_mask_overlay = None
        self._visible_masks_cache_key = None
        self._mask_opacity_cache = 1.0

    def get_default_color(self, name):
        # Deterministically pick a color for a name if not already set
        available = list(self.COLORS.keys())
        if "grayscale" in available:
            available.remove("grayscale")
        
        # Use simple hash-based index
        idx = hash(name) % len(available)
        return available[idx]

    def set_asset_color(self, name, color_name):
        self.asset_colors[name] = color_name
        self._cached_image_composite = None
        self._cached_mask_overlay = None

    def get_asset_color(self, name):
        return self.asset_colors.get(name, "grayscale")

    def toggle_visibility(self, name):
        if name in self.visible_assets:
            self.visible_assets.remove(name)
        else:
            self.visible_assets.add(name)
        self._cached_image_composite = None
        self._cached_mask_overlay = None

    def is_visible(self, name):
        return name in self.visible_assets

    def rename_asset(self, old_name, new_name):
        if old_name in self.visible_assets:
            self.visible_assets.remove(old_name)
            self.visible_assets.add(new_name)
        
        if old_name in self.asset_colors:
            self.asset_colors[new_name] = self.asset_colors.pop(old_name)

    def remove_asset(self, name):
        if name in self.visible_assets:
            self.visible_assets.remove(name)
        
        if name in self.asset_colors:
            del self.asset_colors[name]

    def clear(self):
        self.visible_assets.clear()
        self.asset_colors.clear()

    @staticmethod
    def array_to_qimage(data):
        """
        Converts a NumPy array to a QImage.
        Expects data in float 0-1 range or uint8 0-255.
        """
        if data is None:
            return None

        # Normalize data to 0-255 for display if needed
        if data.max() <= 1.01 and data.min() >= -0.01:
            display_data = (data * 255).astype(np.uint8)
        else:
            d_min, d_max = data.min(), data.max()
            if d_max > d_min:
                display_data = ((data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                display_data = np.zeros_like(data, dtype=np.uint8)

        display_data = np.ascontiguousarray(display_data)
        qimg = qimage2ndarray.array2qimage(display_data)
        if qimg.isNull():
            # Fallback for failed array conversion
            qimg = QImage(display_data.shape[1], display_data.shape[0], QImage.Format_Grayscale8)
            qimg.fill(0)
        return qimg.copy()

    @staticmethod
    def apply_color_to_qimage(qimg, color_rgb):
        """Applies a color tint to a grayscale QImage."""
        if qimg.isNull():
            return qimg

        if qimg.format() != QImage.Format_RGB32 and qimg.format() != QImage.Format_ARGB32:
            qimg = qimg.convertToFormat(QImage.Format_ARGB32)

        # Convert to ndarray for faster processing
        try:
            arr = qimage2ndarray.rgb_view(qimg).astype(np.float32)
        except (ValueError, TypeError):
            # Fallback if qimage2ndarray fails
            return qimg

        # Multiply by color_rgb
        # color_rgb is (R, G, B) in 0-1 range
        for i in range(3):
            arr[:, :, i] *= color_rgb[i]

        # Clip and convert back to uint8
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return qimage2ndarray.array2qimage(arr).copy()

    def render_composite(self, asset_manager):
        if self._cached_image_composite is not None:
            return self._cached_image_composite

        num_images = len(self.visible_assets)
        if num_images == 0:
            return None

        # Determine scaling factor to prevent washout when many image layers are visible
        scaling_factor = 1.0
        if num_images >= 4:
            scaling_factor = 4.0 / num_images

        composite_rgb = None
        target_shape = None

        for name in sorted(self.visible_assets):
            image_asset = asset_manager.get_image_by_name(name)
            if not image_asset:
                continue
            
            data = image_asset.get_rendered_data(for_display=True)
            if data is None:
                continue

            norm_data = data 
            
            # Since for_display=True already normalizes if contrast_stretch is not on, 
            # and we need values in 0-1 range for composite building.
            if norm_data.max() > 1.01 or norm_data.min() < -0.01:
                d_min, d_max = norm_data.min(), norm_data.max()
                if d_max > d_min:
                    norm_data = (norm_data - d_min) / (d_max - d_min)
                else:
                    norm_data = np.zeros_like(norm_data)

            norm_data *= scaling_factor

            if target_shape is None:
                target_shape = data.shape[:2]
                composite_rgb = np.zeros((*target_shape, 3), dtype=np.float32)

            if data.shape[:2] != target_shape:
                # Resize image to match current composite
                norm_data = cv2.resize(norm_data, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

            color_name = image_asset.pipeline.config.get("color")
            if not color_name or color_name == "grayscale":
                color_name = self.get_asset_color(name)
            color_rgb = self.COLORS.get(color_name, (1, 1, 1))

            for i in range(3):
                channel_img = norm_data * color_rgb[i]
                composite_rgb[:, :, i] = np.maximum(composite_rgb[:, :, i], channel_img)

        if composite_rgb is None:
            return None

        composite_rgb = np.clip(composite_rgb, 0, 1)

        display_img = np.ascontiguousarray((composite_rgb * 255).astype(np.uint8))
        qimg = qimage2ndarray.array2qimage(display_img)
        if qimg.isNull():
            # Create a small black image as fallback
            qimg = QImage(100, 100, QImage.Format_RGB32)
            qimg.fill(0)
        
        self._cached_image_composite = qimg.copy()
        return self._cached_image_composite

    def render_mask_overlay(self, image_composite, asset_manager, visible_masks, mask_opacity, get_mask_color_func):
        """
        Renders an overlay of masks onto an existing image composite.
        image_composite: QImage
        visible_masks: set of mask names
        mask_opacity: float 0-1
        get_mask_color_func: function that returns QColor for a mask name
        """
        if not visible_masks:
            return image_composite

        # Create a cache key for masks
        cache_key = (tuple(sorted(visible_masks)), mask_opacity)
        if self._cached_mask_overlay is not None and self._visible_masks_cache_key == cache_key and self._cached_image_composite == image_composite:
            return self._cached_mask_overlay

        if image_composite is None or image_composite.isNull():
            # Determine size from first mask
            mask_asset = asset_manager.get_mask_by_name(list(visible_masks)[0])
            if mask_asset:
                m = mask_asset.get_rendered_data(for_display=True)
                if m is not None:
                    h, w = m.shape[:2]
                else:
                    h, w = 1000, 1000
            else:
                h, w = 1000, 1000
            composite_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            composite_rgb = qimage2ndarray.rgb_view(image_composite).copy()

        for mask_name in sorted(visible_masks):
            mask_asset = asset_manager.get_mask_by_name(mask_name)
            if not mask_asset:
                continue
            
            mask_data = mask_asset.get_rendered_data(for_display=True)
            if mask_data is None:
                continue
            
            if mask_data.shape[:2] != composite_rgb.shape[:2]:
                mask_data = cv2.resize(mask_data, (composite_rgb.shape[1], composite_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            color = get_mask_color_func(mask_name)
            r, g, b = color.red(), color.green(), color.blue()
            
            mask_bool = mask_data.astype(bool)
            overlay = composite_rgb.copy()
            overlay[mask_bool] = [r, g, b]
            cv2.addWeighted(overlay, mask_opacity, composite_rgb, 1 - mask_opacity, 0, composite_rgb)

        self._cached_mask_overlay = qimage2ndarray.array2qimage(composite_rgb).copy()
        self._visible_masks_cache_key = cache_key
        return self._cached_mask_overlay

    def save_visible(self, asset_manager, output_dir, filename, image_format):
        composite_qimg = self.render_composite(asset_manager)
        if not composite_qimg:
            return False
            
        save_path = os.path.join(output_dir, f"{filename}.{image_format}")
        
        # Use OpenCV to save for consistency
        if hasattr(composite_qimg, 'ndarray'):
            data = composite_qimg.ndarray
            # data is RGB, OpenCV wants BGR
            if len(data.shape) == 3 and data.shape[2] == 3:
                data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            elif len(data.shape) == 3 and data.shape[2] == 4:
                data = cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA)
            return cv2.imwrite(save_path, data)
        
        return composite_qimg.save(save_path)
