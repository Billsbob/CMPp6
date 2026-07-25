import os
import tempfile
import shutil
import numpy as np
import json
import image_manipulation
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

    def apply(self, data, data_only=False, for_display=False):
        if data is None:
            return None
            
        processed = data.copy().astype(np.float32)
        
        transforms = self.config.get("transforms", [])
        
        # Backward compatibility for old config format
        if not transforms:
            crop = self.config.get("crop")
            if crop:
                transforms.append({"type": "crop", "params": crop})
            rotate = self.config.get("rotate", 0.0)
            if rotate != 0.0:
                # Use default black for old rotate format
                transforms.append({"type": "rotate", "angle": rotate, "fill_color": "black"})

        for t in transforms:
            if t["type"] == "crop":
                processed = image_manipulation.crop_image(processed, *t["params"], crop_border=5)
            elif t["type"] == "rotate":
                angle = t.get("angle", 0)
                fill_color_name = t.get("fill_color", "black")
                fill_value = 0
                if fill_color_name.lower() == "white":
                    if data.dtype == np.uint8:
                        fill_value = 255
                    elif data.dtype == np.uint16:
                        fill_value = 65535
                    else:
                        # Fallback for float or other types, assuming 0..1 if max <= 1 else 255
                        fill_value = 1.0 if data.max() <= 1.01 else 255
                
                processed = image_manipulation.rotate_image(processed, angle, expand=True, crop_border=5, fill_color=fill_value)

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

        # Display-only transforms
        if self.config.get("contrast_stretch", False):
            p2, p98 = np.percentile(processed, (2, 98))
            if p98 > p2:
                processed = np.clip((processed - p2) / (p98 - p2), 0, 1)
            else:
                processed = np.zeros_like(processed)

        if self.config.get("normalize", False) or (for_display and not self.config.get("contrast_stretch", False)):
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

    def get_rendered_data(self, data_only=False, for_display=False):
        return self.pipeline.apply(self.data, data_only=data_only, for_display=for_display)

    def to_qimage(self, for_display=True):
        from image_handler import ImageDisplayHandler
        if for_display:
            data = self.get_rendered_data(for_display=True)
        else:
            data = self.data

        img = ImageDisplayHandler.array_to_qimage(data)

        if img is None or img.isNull():
            # Create a small empty image as ultimate fallback
            img = QImage(100, 100, QImage.Format_ARGB32)
            img.fill(0)

        # Apply color if specified in pipeline config
        color_name = self.pipeline.config.get("color", "grayscale")
        if color_name != "grayscale":
            color_rgb = ImageDisplayHandler.COLORS.get(color_name)
            if color_rgb:
                return ImageDisplayHandler.apply_color_to_qimage(img, color_rgb)
        
        return img

class MaskAsset(Asset):
    def load(self):
        if not os.path.exists(self.path):
            return None
        
        # Load as image
        data = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if data is None:
            # Fallback for .npy if it still exists
            if self.path.lower().endswith('.npy'):
                try:
                    data = np.load(self.path)
                except Exception:
                    return None
            else:
                return None
        
        # If color, convert to grayscale
        if len(data.shape) == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        
        # Ensure it's treated as float32 for processing
        self._data = data.astype(np.float32)
            
        return self._data

    def get_json_path(self):
        if not self.working_dir:
            return self.path + ".json"
        
        json_dir = os.path.join(self.working_dir, "JSON", "Mask JSONs")
        os.makedirs(json_dir, exist_ok=True)
        return os.path.join(json_dir, self.base_name + ".json")

class AssetManager:
    def __init__(self):
        self.images = {}
        self.masks = {}
        self.working_dir = None

    def set_working_dir(self, path):
        self.working_dir = path
        
        # Create folder structure
        os.makedirs(os.path.join(path, "Cluster Masks"), exist_ok=True)
        os.makedirs(os.path.join(path, "Graphs"), exist_ok=True)
        json_dir = os.path.join(path, "JSON")
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(os.path.join(json_dir, "Image JSONs"), exist_ok=True)
        os.makedirs(os.path.join(json_dir, "Mask JSONs"), exist_ok=True)
        
        self.scan_assets()
        self.update_project_json()

    def get_project_json_path(self):
        if not self.working_dir:
            return None

        image_list = self.get_image_list()
        if not image_list:
            # Try to find any image in the directory even if not scanned yet
            files = [f for f in os.listdir(self.working_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg'))]
            if not files:
                return None
            image_list = files

        # Use the first image to determine the project name
        first_image = image_list[0]
        parts = first_image.split('_')
        if len(parts) >= 2:
            project_name = f"{parts[0]}_{parts[1]}_"
        else:
            project_name = os.path.basename(self.working_dir) + "_"

        json_dir = os.path.join(self.working_dir, "JSON")
        return os.path.join(json_dir, f"{project_name}.json")

    def move_to_deleted_assets(self, asset_name, asset_type, asset_data=None):
        """
        Moves an asset reference to the 'Deleted Assets' section in project JSON.
        asset_type: 'Image', 'Mask', or 'Histogram'
        """
        project_json_path = self.get_project_json_path()
        if not project_json_path or not os.path.exists(project_json_path):
            return

        try:
            with open(project_json_path, 'r') as f:
                project_data = json.load(f)
        except:
            return

        if "Deleted Assets" not in project_data:
            project_data["Deleted Assets"] = []

        entry = {
            "name": asset_name,
            "type": asset_type,
            "deletion_date": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # If it's an image, we might want to move its paths from main sections
        if asset_type == "Image":
            if "Image Paths" in project_data and asset_name in project_data["Image Paths"]:
                entry["path"] = project_data["Image Paths"].pop(asset_name)
            if "Image JSON Paths" in project_data and asset_name in project_data["Image JSON Paths"]:
                entry["json_path"] = project_data["Image JSON Paths"].pop(asset_name)
            if "Image IDs" in project_data and asset_name in project_data["Image IDs"]:
                project_data["Image IDs"].remove(asset_name)

        elif asset_type == "Mask":
            if "Masks" in project_data and asset_name in project_data["Masks"]:
                entry["data"] = project_data["Masks"].pop(asset_name)
            elif asset_data:
                entry["data"] = asset_data

        elif asset_type == "Histogram":
            if "Histograms" in project_data and asset_name in project_data["Histograms"]:
                entry["data"] = project_data["Histograms"].pop(asset_name)
            elif asset_data:
                entry["data"] = asset_data

        project_data["Deleted Assets"].append(entry)

        with open(project_json_path, 'w') as f:
            json.dump(project_data, f, indent=4)

    def update_project_json(self):
        if not self.working_dir:
            return

        image_list = self.get_image_list()
        if not image_list:
            return

        project_json_path = self.get_project_json_path()
        if not project_json_path:
            return

        project_data = {}
        if os.path.exists(project_json_path):
            try:
                with open(project_json_path, 'r') as f:
                    project_data = json.load(f)
            except:
                project_data = {}

        # Preserve Deleted Assets
        deleted_assets = project_data.get("Deleted Assets", [])

        # Update Image IDs and Paths
        project_data["Image IDs"] = image_list
        if "Image Paths" not in project_data:
            project_data["Image Paths"] = {}
        if "Image JSON Paths" not in project_data:
            project_data["Image JSON Paths"] = {}
        
        for img_name in image_list:
            asset = self.get_image_by_name(img_name)
            if asset:
                project_data["Image Paths"][img_name] = os.path.abspath(asset.path)
                project_data["Image JSON Paths"][img_name] = os.path.abspath(asset.get_json_path())
        
        if "Masks" not in project_data:
            project_data["Masks"] = {}

        project_data["Deleted Assets"] = deleted_assets

        with open(project_json_path, 'w') as f:
            json.dump(project_data, f, indent=4)

    def validate_filenames(self):
        """
        Validates filenames in the working directory against the convention:
        <Sample>_<Slide ##>_<Owner Initials>_<ObjectiveMag>_<Well Position>_<Probe>
        
        Rules:
        - <Sample>: Numbers
        - <Slide ##>: Numbers
        - <Owner Initials>: Letters
        - <ObjectiveMag>: Number followed by 'x' or 'X'
        - <Well Position>: Number between 1 and 12
        - Probe: Letters, no numbers
        """
        invalid_files = []
        if not self.working_dir:
            return invalid_files

        import re
        # Pattern components:
        # ^(\d+)                  : <Sample> (numbers)
        # _(\d+)                  : <Slide ##> (numbers)
        # _([a-zA-Z]+)            : <Owner Initials> (letters)
        # _(\d+[xX])              : <ObjectiveMag> (number followed by x or X)
        # _(\d+)                  : <Well Position> (we'll check 1-12 range manually or with regex)
        # _([a-zA-Z]+)            : <Probe> (letters, no numbers)
        # \.[^.]+$                : file extension
        
        pattern = re.compile(r'^(\d+)_(\d+)_([a-zA-Z]+)_(\d+[xX])_(\d+)_([a-zA-Z]+)\.[^.]+$')

        for f in os.listdir(self.working_dir):
            if f.lower().endswith(('.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg')):
                match = pattern.match(f)
                if not match:
                    invalid_files.append(f)
                    continue
                
                # Check Well Position range (1-12)
                try:
                    well_pos = int(match.group(5))
                    if not (1 <= well_pos <= 12):
                        invalid_files.append(f)
                except ValueError:
                    invalid_files.append(f)

        return invalid_files

    def scan_assets(self):
        if not self.working_dir:
            return

        self.images = {}
        for f in os.listdir(self.working_dir):
            if f.lower().endswith(('.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg')):
                path = os.path.join(self.working_dir, f)
                self.images[f] = Asset(path, working_dir=self.working_dir)
                self.images[f].load_project()

        self.masks = {}
        mask_dir = os.path.join(self.working_dir, "Cluster Masks")
        if os.path.exists(mask_dir):
            for f in os.listdir(mask_dir):
                if f.lower().endswith(('.png', '.npy', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    path = os.path.join(mask_dir, f)
                    self.masks[f] = MaskAsset(path, working_dir=self.working_dir)
                    self.masks[f].load_project()

    def get_image_list(self):
        return sorted([img.name for img in self.images.values()])

    def get_mask_list(self):
        return sorted([m.name for m in self.masks.values()])

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
        for m in self.masks.values():
            if m.name == name:
                return m
        return None

    def delete_image(self, name):
        asset = self.get_image_by_name(name)
        if asset:
            # Move to deleted assets in JSON
            self.move_to_deleted_assets(name, "Image")

            if name in self.images:
                del self.images[name]
            else:
                for k, v in list(self.images.items()):
                    if v.name == name:
                        del self.images[k]
                        break

    def delete_mask(self, name):
        asset = self.get_mask_by_name(name)
        if asset:
            # Move to deleted assets in JSON
            self.move_to_deleted_assets(name, "Mask")

            if name in self.masks:
                del self.masks[name]
            else:
                for k, v in list(self.masks.items()):
                    if v.name == name:
                        del self.masks[k]
                        break
