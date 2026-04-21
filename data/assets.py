import os
import numpy as np
import json
from . import image_manipulation
import cv2
import qimage2ndarray
from PySide6.QtGui import QImage, QPixmap

class TransformPipeline:
    def __init__(self, config=None):
        self.config = config or {
            "filters": [],
            "filter_params": {},
            "invert": False,
            "color": "grayscale",
            "contrast_stretch": False,
            "opacity": 1.0,
            "normalize": False,
            "transforms": []
        }

    def apply(self, data, data_only=False):
        if data is None:
            return None
            
        processed = data.copy().astype(np.float32)
        
        transforms = self.config.get("transforms", [])
        
        if not transforms:
            crop = self.config.get("crop")
            if crop:
                transforms.append({"type": "crop", "params": crop})
            rotate = self.config.get("rotate", 0.0)
            if rotate != 0.0:
                transforms.append({"type": "rotate", "angle": rotate})

        for t in transforms:
            if t["type"] == "crop":
                processed = image_manipulation.crop_image(processed, *t["params"], crop_border=5)
            elif t["type"] == "rotate":
                processed = image_manipulation.rotate_image(processed, t["angle"], expand=True, crop_border=5)

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
        
        if "sharpen" in self.config.get("filters", []):
            # Use OpenCV for sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)

        if self.config.get("invert", False):
            d_min, d_max = processed.min(), processed.max()
            processed = d_max - (processed - d_min)

        if data_only:
            return processed

        if self.config.get("contrast_stretch", False):
            p2, p98 = np.percentile(processed, (2, 98))
            if p98 > p2:
                processed = np.clip((processed - p2) / (p98 - p2), 0, 1)
            else:
                processed = np.zeros_like(processed)

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
    def __init__(self, path, working_dir=None):
        self.path = path
        self.working_dir = working_dir
        self.base_name = os.path.basename(path)
        self.name = self.base_name
        self._data = None
        self.pipeline = TransformPipeline()

    def get_json_path(self):
        if not self.working_dir:
            return self.path + ".json"
        
        json_dir = os.path.join(self.working_dir, "JSON", "Image JSONs")
        return os.path.join(json_dir, self.base_name + ".json")

    def load_project(self):
        project_file = self.get_json_path()
        if os.path.exists(project_file):
            with open(project_file, 'r') as f:
                self.pipeline = TransformPipeline.from_json(f.read())

    def save_project(self):
        project_file = self.get_json_path()
        with open(project_file, 'w') as f:
            f.write(self.pipeline.to_json())

    @property
    def data(self):
        if self._data is None:
            self.load()
        return self._data

    def load(self):
        if not os.path.exists(self.path):
            return None
        
        # Use OpenCV to load images, including high bit depth
        # IMREAD_UNCHANGED keeps bit depth and channels
        data = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if data is None:
            return None

        # Convert to float32 for processing
        if data.dtype == np.uint8:
            self._data = data.astype(np.float32)
        elif data.dtype == np.uint16:
            self._data = data.astype(np.float32)
        else:
            self._data = data.astype(np.float32)
            
        return self._data

    def get_rendered_data(self, data_only=False):
        return self.pipeline.apply(self.data, data_only=data_only)

    def to_qimage(self, for_display=True):
        if for_display:
            data = self.get_rendered_data()
        else:
            data = self.data

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
        
        # Use qimage2ndarray to handle the conversion safely
        return qimage2ndarray.array2qimage(display_data).copy()

class AssetManager:
    def __init__(self):
        self.images = {}
        self.working_dir = None

    def set_working_dir(self, path):
        self.working_dir = path
        
        # Create folder structure
        os.makedirs(os.path.join(path, "Cluster Masks"), exist_ok=True)
        os.makedirs(os.path.join(path, "Graphs"), exist_ok=True)
        json_dir = os.path.join(path, "JSON")
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(os.path.join(json_dir, "Image JSONs"), exist_ok=True)
        
        self.scan_assets()

    def scan_assets(self):
        if not self.working_dir:
            return

        self.images = {}
        for f in os.listdir(self.working_dir):
            if f.lower().endswith(('.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg')):
                path = os.path.join(self.working_dir, f)
                self.images[f] = Asset(path, working_dir=self.working_dir)
                self.images[f].load_project()

    def get_image_list(self):
        return sorted([img.name for img in self.images.values()])

    def get_image_by_name(self, name):
        if name in self.images:
            return self.images[name]
        for img in self.images.values():
            if img.name == name:
                return img
        return None

    def delete_image(self, name):
        asset = self.get_image_by_name(name)
        if asset:
            if name in self.images:
                del self.images[name]
            else:
                for k, v in list(self.images.items()):
                    if v.name == name:
                        del self.images[k]
                        break
