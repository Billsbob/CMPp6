import os
import tifffile
import numpy as np
import json
import image_manipulation
from PIL import Image, ImageFilter
from PySide6.QtGui import QImage, QPixmap

class TransformPipeline:
    def __init__(self, config=None):
        self.config = config or {
            "display_name": None,
            "filters": [], # list of filter names, e.g., ["blur", "sharpen"]
            "filter_params": {}, # e.g., {"gaussian": {"radius": 2}}
            "invert": False,
            "color": "grayscale",
            "contrast_stretch": False,
            "opacity": 1.0,
            "normalize": False,
            "transforms": [] # list of {"type": "crop", "params": [x, y, w, h]} or {"type": "rotate", "angle": angle}
        }

    def apply(self, data, for_clustering=False):
        if data is None:
            return None
            
        processed = data.copy().astype(np.float32)
        
        # Apply iterative transformations
        transforms = self.config.get("transforms", [])
        
        # Backward compatibility for old single crop/rotate
        if not transforms:
            crop = self.config.get("crop")
            if crop:
                transforms.append({"type": "crop", "params": crop})
            rotate = self.config.get("rotate", 0.0)
            if rotate != 0.0:
                transforms.append({"type": "rotate", "angle": rotate})

        for t in transforms:
            if t["type"] == "crop":
                processed = image_manipulation.crop_image(processed, *t["params"])
            elif t["type"] == "rotate":
                processed = image_manipulation.rotate_image(processed, t["angle"], expand=True)

        # Apply clustering-level transforms (also applies to display)
        # Filters
        filters = self.config.get("filters", [])
        params = self.config.get("filter_params", {})
        
        if "gaussian" in filters:
            p = params.get("gaussian", {"radius": 2})
            processed = image_manipulation.apply_gaussian_blur(processed, radius=p.get("radius", 2))
        if "median" in filters:
            p = params.get("median", {"size": 3})
            processed = image_manipulation.apply_median_filter(processed, size=p.get("size", 3))
        if "mean" in filters:
            p = params.get("mean", {"size": 3})
            processed = image_manipulation.apply_mean_filter(processed, size=p.get("size", 3))
        if "blur" in filters:
            p = params.get("blur", {"size": 3})
            processed = image_manipulation.apply_blur(processed, size=p.get("size", 3))
        if "unsharp" in filters:
            p = params.get("unsharp", {"radius": 2, "percent": 150, "threshold": 3})
            processed = image_manipulation.apply_unsharp_mask(
                processed, 
                radius=p.get("radius", 2), 
                percent=p.get("percent", 150), 
                threshold=p.get("threshold", 3)
            )
        
        if "blur" in self.config.get("filters", []): # Backward compatibility if needed, but we replaced it above
            pass # Already handled
        if "sharpen" in self.config.get("filters", []):
            img = Image.fromarray(processed)
            img = img.filter(ImageFilter.SHARPEN)
            processed = np.array(img).astype(np.float32)

        # Invert
        if self.config.get("invert", False):
            d_min, d_max = processed.min(), processed.max()
            processed = d_max - (processed - d_min)

        if for_clustering:
            return processed

        # Apply display-only transforms
        # Contrast stretch
        if self.config.get("contrast_stretch", False):
            p2, p98 = np.percentile(processed, (2, 98))
            if p98 > p2:
                processed = np.clip((processed - p2) / (p98 - p2), 0, 1)
                # After clip it is [0, 1], we might want to scale back if needed, 
                # but usually display wants [0, 1] or [0, 255]
            else:
                processed = np.zeros_like(processed)

        # Normalization
        if self.config.get("normalize", False):
            d_min, d_max = processed.min(), processed.max()
            if d_max > d_min:
                processed = (processed - d_min) / (d_max - d_min)
            else:
                processed = np.zeros_like(processed)

        return processed

    def to_json(self):
        return json.dumps(self.config, indent=4)

    @classmethod
    def from_json(cls, json_str):
        return cls(json.loads(json_str))

class Asset:
    def __init__(self, path, is_mask=False):
        self.path = path
        self.is_mask = is_mask
        self.base_name = os.path.basename(path)
        self._data = None
        self.project_path = self.path + ".json"
        self.pipeline = self.load_project()
        # Use base name without extension by default to avoid double extensions on export
        self.name = self.pipeline.config.get("display_name") or os.path.splitext(self.base_name)[0]
        # Clean display name if it somehow got double extensions or should be without extension
        while True:
            stripped = False
            for e in ['.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg']:
                if self.name.lower().endswith(e):
                    self.name = self.name[:-len(e)]
                    stripped = True
                    break
            if not stripped:
                break
        self._cache = {} # key -> rendered_data

    def load_project(self):
        if os.path.exists(self.project_path):
            try:
                with open(self.project_path, 'r') as f:
                    return TransformPipeline.from_json(f.read())
            except:
                pass
        return TransformPipeline()

    def save_project(self):
        self.pipeline.config["display_name"] = self.name
        with open(self.project_path, 'w') as f:
            f.write(self.pipeline.to_json())

    @property
    def data(self):
        if self._data is None:
            self.load()
        return self._data

    def load(self):
        if self.is_mask:
            # Masks are bitmaps with color maps
            img = Image.open(self.path)
            self._data = np.array(img)
        else:
            # Images are grayscale 8 or 16 bit uncompressed tifs
            self._data = tifffile.imread(self.path)
        return self._data

    def get_rendered_data(self, for_clustering=False):
        cache_key = "clustering" if for_clustering else "display"
        # In a real app we might want to invalidate cache if pipeline changes
        # For simplicity, let's just re-render if it's not cached or if we want it fresh
        # Actually, let's not cache indefinitely to avoid memory issues if pipeline changes
        return self.pipeline.apply(self.data, for_clustering=for_clustering)

    def to_qimage(self, for_display=True):
        if for_display:
            data = self.get_rendered_data(for_clustering=False)
        else:
            data = self.data

        if self.is_mask:
            img = Image.open(self.path)
            if img.mode == 'P':
                img = img.convert('RGBA')
            elif img.mode == '1':
                img = img.convert('L')
            
            data = np.ascontiguousarray(np.array(img))
            format = QImage.Format_RGBA8888 if img.mode == 'RGBA' else QImage.Format_Grayscale8
            qimg = QImage(data.data, img.size[0], img.size[1], data.strides[0], format)
            qimg.ndarray = data
            return qimg.copy()
        else:
            # Rendered data is float32 usually [0, 1] if normalized or stretched
            # Let's ensure it's in a good range for display
            if data.max() <= 1.01:
                display_data = (data * 255).astype(np.uint8)
            else:
                d_min, d_max = data.min(), data.max()
                if d_max > d_min:
                    display_data = ((data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                else:
                    display_data = np.zeros_like(data, dtype=np.uint8)
            
            display_data = np.ascontiguousarray(display_data)
            height, width = display_data.shape
            qimg = QImage(display_data.data, width, height, display_data.strides[0], QImage.Format_Grayscale8)
            qimg.ndarray = display_data
            return qimg.copy()

class AssetManager:
    def __init__(self):
        self.images = {} # name -> Asset
        self.masks = {}  # name -> Asset
        self.working_dir = None

    def set_working_dir(self, path):
        self.working_dir = path
        self.scan_assets()

    def scan_assets(self):
        if not self.working_dir:
            return

        images_dir = os.path.join(self.working_dir, "Images")
        clusters_dir = os.path.join(self.working_dir, "Clusters")

        self.images = {}
        if os.path.exists(images_dir):
            for f in os.listdir(images_dir):
                if f.lower().endswith(('.tif', '.tiff')):
                    self.images[f] = Asset(os.path.join(images_dir, f), is_mask=False)

        self.masks = {}
        if os.path.exists(clusters_dir):
            for f in os.listdir(clusters_dir):
                if f.lower().endswith(('.bmp', '.png', '.tif', '.tiff')):
                    self.masks[f] = Asset(os.path.join(clusters_dir, f), is_mask=True)

    def get_image_list(self):
        return sorted([img.name for img in self.images.values()])

    def get_mask_list(self):
        return sorted([mask.name for mask in self.masks.values()])

    def get_image_by_name(self, name):
        if name in self.images:
            return self.images[name]
        for img in self.images.values():
            if img.name == name:
                return img
        return None

    def get_mask_by_name(self, name):
        if name in self.masks:
            return self.masks[name]
        for mask in self.masks.values():
            if mask.name == name:
                return mask
        return None

    def get_asset_pair(self, name):
        # name might be display name
        image = self.get_image_by_name(name)
        
        if not image:
            return None, None

        # Often masks might have similar names but different extensions
        image_base = os.path.splitext(image.base_name)[0]
        
        # Try to find a matching mask
        mask = None
        for m_name in self.masks:
            m_asset = self.masks[m_name]
            mask_base = os.path.splitext(m_asset.base_name)[0]
            if mask_base == image_base or mask_base.startswith(image_base + "_"):
                mask = m_asset
                break
        
        return image, mask

    def rename_mask(self, old_name, new_name):
        # Find the actual key (filename) for the given display name
        old_file_name = None
        if old_name in self.masks:
            old_file_name = old_name
        else:
            for key, asset in self.masks.items():
                if asset.name == old_name:
                    old_file_name = key
                    break
        
        if not old_file_name:
            return False, f"Mask '{old_name}' not found."
        
        # Ensure new name has correct extension if user didn't provide it
        asset = self.masks[old_file_name]
        old_path = asset.path
        dir_name = os.path.dirname(old_path)
        old_ext = os.path.splitext(old_path)[1]
        
        if not new_name.lower().endswith(old_ext.lower()):
            new_file_name = new_name + old_ext
        else:
            new_file_name = new_name
            
        # Strip all image extensions from the display name
        while True:
            stripped = False
            for e in ['.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg']:
                if new_name.lower().endswith(e):
                    new_name = new_name[:-len(e)]
                    stripped = True
                    break
            if not stripped:
                break
            
        if new_file_name in self.masks and new_file_name != old_file_name:
            return False, f"A mask with name '{new_file_name}' already exists."
        
        new_path = os.path.join(dir_name, new_file_name)
        
        # Check if new file already exists on disk
        if os.path.exists(new_path) and new_path.lower() != old_path.lower():
            return False, f"File '{new_file_name}' already exists on disk."
            
        try:
            if old_path.lower() != new_path.lower():
                os.rename(old_path, new_path)
                
                # Also rename project file if it exists
                old_project = old_path + ".json"
                new_project = new_path + ".json"
                if os.path.exists(old_project):
                    if os.path.exists(new_project):
                        os.remove(new_project)
                    os.rename(old_project, new_project)
            
            asset.path = new_path
            asset.project_path = new_path + ".json"
            asset.base_name = new_file_name
            asset.name = new_name
            if old_file_name != new_file_name:
                self.masks[new_file_name] = self.masks.pop(old_file_name)
            asset.save_project()
            return True, (new_file_name, new_name)
        except Exception as e:
            return False, str(e)

    def delete_mask(self, name):
        """
        Delete a mask from the system.
        :param name: display name or filename of the mask
        :return: (success, message)
        """
        file_name = None
        if name in self.masks:
            file_name = name
        else:
            for key, asset in self.masks.items():
                if asset.name == name:
                    file_name = key
                    break
        
        if not file_name:
            return False, f"Mask '{name}' not found."

        asset = self.masks[file_name]
        try:
            # Delete mask file
            if os.path.exists(asset.path):
                os.remove(asset.path)
            
            # Delete project file
            if os.path.exists(asset.project_path):
                os.remove(asset.project_path)
            
            # Remove from internal dictionary
            del self.masks[file_name]
            return True, "Mask deleted successfully."
        except Exception as e:
            return False, str(e)

    def apply_global_crop(self, x, y, width, height):
        crop_transform = {"type": "crop", "params": [x, y, width, height]}
        for asset in self.images.values():
            if "transforms" not in asset.pipeline.config:
                asset.pipeline.config["transforms"] = []
            asset.pipeline.config["transforms"].append(crop_transform)
            # Remove old legacy params to avoid confusion, though apply() handles them
            asset.pipeline.config.pop("crop", None) 
            asset.save_project()
        for asset in self.masks.values():
            if "transforms" not in asset.pipeline.config:
                asset.pipeline.config["transforms"] = []
            asset.pipeline.config["transforms"].append(crop_transform)
            asset.pipeline.config.pop("crop", None)
            asset.save_project()

    def apply_global_rotate(self, angle):
        rotate_transform = {"type": "rotate", "angle": angle}
        for asset in self.images.values():
            if "transforms" not in asset.pipeline.config:
                asset.pipeline.config["transforms"] = []
            asset.pipeline.config["transforms"].append(rotate_transform)
            asset.pipeline.config.pop("rotate", None)
            asset.save_project()
        for asset in self.masks.values():
            if "transforms" not in asset.pipeline.config:
                asset.pipeline.config["transforms"] = []
            asset.pipeline.config["transforms"].append(rotate_transform)
            asset.pipeline.config.pop("rotate", None)
            asset.save_project()

    def add_new_mask(self, name, data):
        """
        Save a numpy array as a PNG mask and register it as an asset.
        :param name: display name for the mask
        :param data: 2D numpy array (uint8, 0 or 255)
        :return: the new Asset
        """
        if not self.working_dir:
            return None

        clusters_dir = os.path.join(self.working_dir, "Clusters")
        if not os.path.exists(clusters_dir):
            os.makedirs(clusters_dir)

        # Ensure filename is safe
        safe_filename = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        mask_path = os.path.normpath(os.path.join(clusters_dir, f"{safe_filename}.png"))
        
        # Save as PNG
        img = Image.fromarray(data.astype(np.uint8), mode='L')
        img.save(mask_path)

        # Create Asset and save its project file
        new_asset = Asset(mask_path, is_mask=True)
        new_asset.name = name
        new_asset.save_project()
        
        # Add to local tracking
        self.masks[name] = new_asset
        return new_asset
