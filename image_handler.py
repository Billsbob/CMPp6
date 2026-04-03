import numpy as np
from PySide6.QtGui import QImage
import os
import tifffile
from PIL import Image
import json

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
        "azure": (0, 0.5, 1),
        "grayscale": (1, 1, 1)
    }

    def __init__(self):
        self.asset_colors = {}  # asset_name -> color_name
        self.visible_assets = set() # set of image asset names
        self.visible_masks = set() # set of mask asset names
        self.visible_graphs = set() # set of graph names
        self.mask_opacity = 0.5 # default opacity for masks

    def get_default_color(self, name):
        if "_YY." in name:
            return "red"
        elif "_TT." in name:
            return "green"
        elif "_E." in name:
            return "blue"
        elif ".csv" in name.lower():
            return "magenta"
        elif ".bmp" in name.lower() or ".png" in name.lower():
            return "yellow" # masks are often yellow by default
        return "grayscale" # default fallback

    def set_asset_color(self, name, color_name):
        if color_name in self.COLORS:
            self.asset_colors[name] = color_name

    def get_asset_color(self, name):
        if name not in self.asset_colors:
            self.asset_colors[name] = self.get_default_color(name)
        return self.asset_colors[name]

    def toggle_visibility(self, name, is_mask=False, is_graph=False):
        if is_graph:
            target_set = self.visible_graphs
        else:
            target_set = self.visible_masks if is_mask else self.visible_assets
            
        if name in target_set:
            target_set.remove(name)
        else:
            target_set.add(name)

    def is_visible(self, name, is_mask=False, is_graph=False):
        if is_graph:
            target_set = self.visible_graphs
        else:
            target_set = self.visible_masks if is_mask else self.visible_assets
        return name in target_set

    def rename_asset(self, old_name, new_name, is_mask=False, is_graph=False):
        if is_graph:
            target_set = self.visible_graphs
        else:
            target_set = self.visible_masks if is_mask else self.visible_assets
            
        if old_name in target_set:
            target_set.remove(old_name)
            target_set.add(new_name)
        
        if old_name in self.asset_colors:
            self.asset_colors[new_name] = self.asset_colors.pop(old_name)

    def remove_asset(self, name, is_mask=False, is_graph=False):
        if is_graph:
            target_set = self.visible_graphs
        else:
            target_set = self.visible_masks if is_mask else self.visible_assets
            
        if name in target_set:
            target_set.remove(name)
        
        if name in self.asset_colors:
            del self.asset_colors[name]

    def render_composite(self, asset_manager, graphs=None):
        num_images = len(self.visible_assets)
        num_masks = len(self.visible_masks)
        
        if num_images == 0 and num_masks == 0:
            return None

        # Determine scaling factor to prevent washout when many image layers are visible
        scaling_factor = 1.0
        if num_images >= 4:
            scaling_factor = 4.0 / num_images

        composite_rgb = None
        target_shape = None

        # Render images
        for name in sorted(self.visible_assets):
            image_asset, _ = asset_manager.get_asset_pair(name)
            if not image_asset:
                continue
            
            # Get rendered data from asset (which applies the pipeline)
            data = image_asset.get_rendered_data(for_clustering=False)
            if data is None:
                continue

            # data is already float32 from TransformPipeline
            norm_data = data 
            
            # If pipeline didn't normalize, we might still want to ensure [0, 1]
            if norm_data.max() > 1.0 or norm_data.min() < 0.0:
                d_min, d_max = norm_data.min(), norm_data.max()
                if d_max > d_min:
                    norm_data = (norm_data - d_min) / (d_max - d_min)
                else:
                    norm_data = np.zeros_like(norm_data)

            # Apply brightness limit
            norm_data *= scaling_factor

            if target_shape is None:
                target_shape = data.shape
                composite_rgb = np.zeros((*target_shape, 3), dtype=np.float32)

            color_name = image_asset.pipeline.config.get("color") or self.get_asset_color(name)
            color_rgb = self.COLORS.get(color_name, (1, 1, 1))

            # Apply color and composite (per-channel max)
            for i in range(3):
                channel_img = norm_data * color_rgb[i]
                composite_rgb[:, :, i] = np.maximum(composite_rgb[:, :, i], channel_img)

        # If no images, but masks, initialize composite_rgb
        if composite_rgb is None and num_masks > 0:
            # We need the shape of the masks
            for name in self.visible_masks:
                mask_asset = asset_manager.get_mask_by_name(name)
                if mask_asset:
                    data = mask_asset.get_rendered_data(for_clustering=False)
                    if data is not None:
                        target_shape = data.shape
                        composite_rgb = np.zeros((*target_shape, 3), dtype=np.float32)
                        break

        # Render masks
        if composite_rgb is not None:
            for name in sorted(self.visible_masks):
                mask_asset = asset_manager.get_mask_by_name(name)
                if not mask_asset:
                    continue
                
                # Get rendered data from mask (which applies the pipeline transformations)
                mask_data_raw = mask_asset.get_rendered_data(for_clustering=False)
                if mask_data_raw is None:
                    continue

                # Masks are often binary or label images. 
                # Let's assume non-zero values are part of the mask
                mask_data = (mask_data_raw > 0).astype(np.float32)
                
                color_name = self.get_asset_color(name)
                color_rgb = self.COLORS[color_name]
                
                # Overlay mask with opacity
                # result = (1 - alpha) * background + alpha * mask_color
                alpha = mask_asset.pipeline.config.get("opacity", self.mask_opacity)
                
                for i in range(3):
                    mask_channel = mask_data * color_rgb[i]
                    # We only apply the overlay where the mask is active
                    overlay = (1 - alpha) * composite_rgb[:, :, i] + alpha * mask_channel
                    composite_rgb[:, :, i] = np.where(mask_data > 0, overlay, composite_rgb[:, :, i])

        if composite_rgb is None:
            return None

        # Clip values to [0, 1] before conversion to 8-bit
        composite_rgb = np.clip(composite_rgb, 0, 1)

        display_img = np.ascontiguousarray((composite_rgb * 255).astype(np.uint8))
        height, width, _ = display_img.shape
        qimg = QImage(display_img.data, width, height, display_img.strides[0], QImage.Format_RGB888)
        qimg.ndarray = display_img
        return qimg.copy()

    def save_visible(self, asset_manager, output_dir, filename, image_format):
        """
        Saves a blended representation of what is currently displayed (screenshot style).
        :param asset_manager: The asset manager containing the data.
        :param output_dir: The directory to save the file.
        :param filename: The name for the saved file.
        :param image_format: 'tif', 'png', 'jpg', or 'bmp'.
        :return: List containing the saved file path.
        """
        composite_qimage = self.render_composite(asset_manager)
        if composite_qimage is None:
            return []

        # Convert QImage to numpy array for saving with metadata
        if composite_qimage.format() != QImage.Format_RGB888:
            composite_qimage = composite_qimage.convertToFormat(QImage.Format_RGB888)

        h, w = composite_qimage.height(), composite_qimage.width()
        ptr = composite_qimage.constBits()
        stride = composite_qimage.bytesPerLine()

        # np.frombuffer on constBits() returns a 1D array of bytes
        arr_1d = np.frombuffer(ptr, np.uint8)

        # Correctly handle stride (padding at the end of each row)
        # Each row takes 'stride' bytes, but we only care about the first 'w * 3' bytes
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            row_start = y * stride
            row_end = row_start + (w * 3)
            arr[y, :, :] = arr_1d[row_start:row_end].reshape((w, 3))

        # Prepare metadata about all visible layers
        metadata = {
            "type": "composite_view",
            "visible_images": [],
            "visible_masks": []
        }
        for name in sorted(self.visible_assets):
            image_asset, _ = asset_manager.get_asset_pair(name)
            if image_asset:
                metadata["visible_images"].append(self._get_asset_metadata(image_asset))
        for name in sorted(self.visible_masks):
            mask_asset = asset_manager.get_mask_by_name(name)
            if mask_asset:
                metadata["visible_masks"].append(self._get_asset_metadata(mask_asset))

        metadata_str = json.dumps(metadata)
        ext = image_format.lower()
        
        # Avoid double extensions if filename already has it
        final_filename = filename
        if final_filename.lower().endswith(f".{ext}"):
            final_filename = final_filename[:-len(ext)-1]
            
        out_path = os.path.join(output_dir, f"{final_filename}.{ext}")

        if ext == 'tif':
            tifffile.imwrite(out_path, arr, description=metadata_str)
        elif ext == 'png':
            img = Image.fromarray(arr)
            from PIL import PngImagePlugin
            meta = PngImagePlugin.PngInfo()
            meta.add_text("Description", metadata_str)
            img.save(out_path, pnginfo=meta)
        else: # jpg or bmp
            img = Image.fromarray(arr)
            if ext == 'jpg':
                img.save(out_path, quality=95)
            else: # bmp
                img.save(out_path)

        return [out_path]

    def export_images(self, asset_manager, output_dir, filename, image_format):
        """
        Saves individual modified versions of visible images, preserving data accuracy.
        Applies invert and filters, but NOT color changes or normalization.
        :param asset_manager: The asset manager containing the data.
        :param output_dir: The directory to save the files.
        :param filename: The base name for the saved file(s).
        :param image_format: 'tif', 'png', 'jpg', or 'bmp'.
        :return: List of saved file paths.
        """
        saved_files = []

        for name in sorted(self.visible_assets):
            image_asset, _ = asset_manager.get_asset_pair(name)
            if not image_asset:
                continue

            # Use for_clustering=True to apply filters and invert but skip normalization/stretch
            data = image_asset.get_rendered_data(for_clustering=True)
            if data is None:
                continue

            metadata = self._get_asset_metadata(image_asset)
            # Remove normalization/stretch from metadata as they were not applied
            metadata["modifications"] = [m for m in metadata["modifications"] 
                                        if m not in ["normalized", "contrast_stretched"]]
            
            # Additional metadata for exported images as requested
            metadata["export_modedisplaydata"] = "uint16" 
            
            metadata_str = json.dumps(metadata)

            # Determine extension and path
            ext = image_format.lower()
            base_asset_name = image_asset.name
            
            # Strip any existing image extensions to avoid double extensions (e.g., .tif.tif or .tif.png)
            while True:
                stripped = False
                for e in ['.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg']:
                    if base_asset_name.lower().endswith(e):
                        base_asset_name = base_asset_name[:-len(e)]
                        stripped = True
                        break
                if not stripped:
                    break
            
            out_name = f"{filename}_{base_asset_name}.{ext}"
            out_path = os.path.join(output_dir, out_name)

            if ext == 'tif':
                # Convert to uint16 as requested. 
                # Preserving "physical values" but in uint16.
                # If values were float, we scale to uint16 range to avoid white appearance
                save_data = np.nan_to_num(data)
                d_min, d_max = save_data.min(), save_data.max()
                
                if d_max > d_min:
                    save_data = ((save_data - d_min) / (d_max - d_min) * 65535).astype(np.uint16)
                else:
                    save_data = save_data.astype(np.uint16)
                    
                tifffile.imwrite(out_path, save_data, description=metadata_str)
            else:
                # For non-TIF formats, we must scale to 8-bit.
                # Since we want to preserve data accuracy, we use the full range of the processed data.
                d_min, d_max = data.min(), data.max()
                if d_max > d_min:
                    save_data = ((data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                else:
                    save_data = np.zeros_like(data, dtype=np.uint8)
                
                img = Image.fromarray(save_data)
                if ext == 'png':
                    from PIL import PngImagePlugin
                    meta = PngImagePlugin.PngInfo()
                    meta.add_text("Description", metadata_str)
                    img.save(out_path, pnginfo=meta)
                else: # jpg or bmp
                    if ext == 'jpg':
                        img.save(out_path, quality=95)
                    else: # bmp
                        img.save(out_path)

            saved_files.append(out_path)
        return saved_files

    def export_masks(self, asset_manager, output_dir, filename, image_format):
        """
        Saves individual modified versions of visible masks.
        :param asset_manager: The asset manager containing the data.
        :param output_dir: The directory to save the files.
        :param filename: The base name for the saved file(s).
        :param image_format: 'tif', 'png', 'jpg', or 'bmp'.
        :return: List of saved file paths.
        """
        saved_files = []

        for name in sorted(self.visible_masks):
            mask_asset = asset_manager.get_mask_by_name(name)
            if not mask_asset:
                continue

            data = mask_asset.data
            if data is None:
                continue

            metadata = self._get_asset_metadata(mask_asset)
            metadata_str = json.dumps(metadata)

            ext = image_format.lower()
            base_mask_name = mask_asset.name
            
            # Strip any existing image extensions to avoid double extensions
            while True:
                stripped = False
                for e in ['.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg']:
                    if base_mask_name.lower().endswith(e):
                        base_mask_name = base_mask_name[:-len(e)]
                        stripped = True
                        break
                if not stripped:
                    break
            
            out_name = f"{filename}_{base_mask_name}.{ext}"
            out_path = os.path.join(output_dir, out_name)

            color_name = mask_asset.pipeline.config.get("color") or self.get_asset_color(name)
            color_rgb = self.COLORS.get(color_name, (1, 1, 1))
            opacity = mask_asset.pipeline.config.get("opacity", self.mask_opacity)
            
            h, w = data.shape
            
            if ext == 'tif':
                # For TIF masks, we can save them as colored or just bitmask.
                # Usually masks are exported as images that look like the mask.
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                mask_indices = data > 0
                rgba[mask_indices, 0] = int(color_rgb[0] * 255)
                rgba[mask_indices, 1] = int(color_rgb[1] * 255)
                rgba[mask_indices, 2] = int(color_rgb[2] * 255)
                rgba[mask_indices, 3] = int(opacity * 255)
                tifffile.imwrite(out_path, rgba, description=metadata_str)
            else:
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                mask_indices = data > 0
                rgba[mask_indices, 0] = int(color_rgb[0] * 255)
                rgba[mask_indices, 1] = int(color_rgb[1] * 255)
                rgba[mask_indices, 2] = int(color_rgb[2] * 255)
                rgba[mask_indices, 3] = int(opacity * 255)
                
                img = Image.fromarray(rgba, 'RGBA')
                if ext == 'png':
                    from PIL import PngImagePlugin
                    meta = PngImagePlugin.PngInfo()
                    meta.add_text("Description", metadata_str)
                    img.save(out_path, pnginfo=meta)
                else: # jpg or bmp - Note: JPG doesn't support alpha, so it will be black background
                    if ext == 'jpg':
                        img = img.convert("RGB")
                    img.save(out_path)
            
            saved_files.append(out_path)

        return saved_files

    def _get_asset_metadata(self, asset):
        """Extracts metadata from an asset's pipeline."""
        metadata = {
            "name": asset.name,
            "original_file": asset.base_name,
            "source_dtype": str(asset.data.dtype),
            "source_min": float(asset.data.min()),
            "source_max": float(asset.data.max()),
            "modifications": [],
            "processing_flags": []
        }
        
        config = asset.pipeline.config
        filters = config.get("filters", [])
        params = config.get("filter_params", {})
        
        for f in filters:
            p = params.get(f, {})
            p_str = ", ".join([f"{k}: {v}" for k, v in p.items()])
            metadata["modifications"].append(f"{f} ({p_str})" if p_str else f)
            metadata["processing_flags"].append(f)
            
        if config.get("invert"):
            metadata["modifications"].append("inverted")
            metadata["processing_flags"].append("inverted")
        if config.get("contrast_stretch"):
            metadata["modifications"].append("contrast_stretched")
        if config.get("normalize"):
            metadata["modifications"].append("normalized")
            
        return metadata
