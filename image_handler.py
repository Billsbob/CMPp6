import numpy as np
from PySide6.QtGui import QImage
import os

class ImageDisplayHandler:
    COLORS = {
        "red": (1, 0, 0),
        "green": (0, 1, 0),
        "blue": (0, 0, 1),
        "magenta": (1, 0, 1),
        "cyan": (0, 1, 1),
        "yellow": (1, 1, 0),
        "orange": (1, 0.5, 0),
        "violet": (0.5, 0, 1),
        "teal": (0, 0.5, 0.5),
        "lime": (0.5, 1, 0),
        "rose": (1, 0, 0.5),
        "azure": (0, 0.5, 1)
    }

    def __init__(self):
        self.asset_colors = {}  # asset_name -> color_name
        self.visible_assets = set() # set of asset names

    def get_default_color(self, name):
        if "_YY" in name:
            return "red"
        elif "_TT" in name:
            return "green"
        elif "_E" in name:
            return "blue"
        return "red" # default fallback

    def set_asset_color(self, name, color_name):
        if color_name in self.COLORS:
            self.asset_colors[name] = color_name

    def get_asset_color(self, name):
        if name not in self.asset_colors:
            self.asset_colors[name] = self.get_default_color(name)
        return self.asset_colors[name]

    def toggle_visibility(self, name):
        if name in self.visible_assets:
            self.visible_assets.remove(name)
        else:
            self.visible_assets.add(name)

    def is_visible(self, name):
        return name in self.visible_assets

    def render_composite(self, asset_manager):
        num_visible = len(self.visible_assets)
        if num_visible == 0:
            return None

        # Determine scaling factor to prevent washout when many layers are visible
        # If 4 or more images are selected, scale down each layer
        scaling_factor = 1.0
        if num_visible >= 4:
            scaling_factor = 4.0 / num_visible

        composite_rgb = None
        target_shape = None

        for name in sorted(self.visible_assets):
            image_asset, _ = asset_manager.get_asset_pair(name)
            if not image_asset:
                continue
            
            data = image_asset.data
            if data is None:
                continue

            # Normalize data to [0, 1] for display version
            d_min, d_max = data.min(), data.max()
            if d_max > d_min:
                norm_data = (data - d_min) / (d_max - d_min)
            else:
                norm_data = np.zeros_like(data, dtype=np.float32)

            # Apply brightness limit
            norm_data *= scaling_factor

            if target_shape is None:
                target_shape = data.shape
                composite_rgb = np.zeros((*target_shape, 3), dtype=np.float32)

            color_name = self.get_asset_color(name)
            color_rgb = self.COLORS[color_name]

            # Apply color and composite (per-channel max)
            for i in range(3):
                channel_img = norm_data * color_rgb[i]
                composite_rgb[:, :, i] = np.maximum(composite_rgb[:, :, i], channel_img)

        if composite_rgb is None:
            return None

        # Convert to 8-bit QImage
        display_img = (composite_rgb * 255).astype(np.uint8)
        height, width, _ = display_img.shape
        return QImage(display_img.data, width, height, width * 3, QImage.Format_RGB888)
