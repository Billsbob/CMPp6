import numpy as np
from PySide6.QtGui import QImage
import os
from PIL import Image
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
        height, width, _ = display_img.shape
        qimg = QImage(display_img.data, width, height, display_img.strides[0], QImage.Format_RGB888)
        qimg.ndarray = display_img
        return qimg.copy()

    def save_visible(self, asset_manager, output_dir, filename, image_format):
        composite_qimg = self.render_composite(asset_manager)
        if not composite_qimg:
            return False
            
        # Convert QImage to PIL and save
        width = composite_qimg.width()
        height = composite_qimg.height()
        
        ptr = composite_qimg.constBits()
        # For RGB888, bytes_per_pixel is 3
        # PIL.Image.frombuffer('RGB', (width, height), data, 'raw', 'RGB', 0, 1)
        # But wait, we have display_img already in render_composite? 
        # Actually it's simpler if we just use the QImage's save method.
        # But let's use PIL to have more control over metadata or other formats.
        
        # It's better to just call qimg.save() if we want simple formats.
        # For more complex stuff, we'd need to convert.
        
        # Let's use QImage.save for simplicity
        save_path = os.path.join(output_dir, f"{filename}.{image_format}")
        return composite_qimg.save(save_path)
