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

    def get_asset_color(self, name):
        return self.asset_colors.get(name, "grayscale")

    def toggle_visibility(self, name):
        if name in self.visible_assets:
            self.visible_assets.remove(name)
        else:
            self.visible_assets.add(name)

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

    def render_composite(self, asset_manager):
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
            
            data = image_asset.get_rendered_data()
            if data is None:
                continue

            norm_data = data 
            
            if norm_data.max() > 1.0 or norm_data.min() < 0.0:
                d_min, d_max = norm_data.min(), norm_data.max()
                if d_max > d_min:
                    norm_data = (norm_data - d_min) / (d_max - d_min)
                else:
                    norm_data = np.zeros_like(norm_data)

            norm_data *= scaling_factor

            if target_shape is None:
                target_shape = data.shape
                composite_rgb = np.zeros((*target_shape, 3), dtype=np.float32)

            color_name = image_asset.pipeline.config.get("color") or self.get_asset_color(name)
            color_rgb = self.COLORS.get(color_name, (1, 1, 1))

            for i in range(3):
                channel_img = norm_data * color_rgb[i]
                composite_rgb[:, :, i] = np.maximum(composite_rgb[:, :, i], channel_img)

        if composite_rgb is None:
            return None

        composite_rgb = np.clip(composite_rgb, 0, 1)

        display_img = np.ascontiguousarray((composite_rgb * 255).astype(np.uint8))
        return qimage2ndarray.array2qimage(display_img).copy()

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
