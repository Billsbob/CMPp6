import os
import tifffile
import numpy as np
from PIL import Image
from PySide6.QtGui import QImage, QPixmap

class Asset:
    def __init__(self, path, is_mask=False):
        self.path = path
        self.is_mask = is_mask
        self.name = os.path.basename(path)
        self._data = None

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

    def to_qimage(self):
        data = self.data
        if self.is_mask:
            # For masks, we should ideally use the palette/colormap
            img = Image.open(self.path)
            if img.mode == 'P':
                img = img.convert('RGBA')
            elif img.mode == '1':
                img = img.convert('L')
            
            qimg = QImage(img.tobytes(), img.size[0], img.size[1], QImage.Format_RGBA8888 if img.mode == 'RGBA' else QImage.Format_Grayscale8)
            return qimg
        else:
            # Handle grayscale 8 or 16 bit
            if data.dtype == np.uint16:
                # Normalize to 8-bit for display if needed, or use QImage.Format_Grayscale16 if supported
                # PySide6 supports Grayscale16
                height, width = data.shape
                return QImage(data.data, width, height, width * 2, QImage.Format_Grayscale16)
            elif data.dtype == np.uint8:
                height, width = data.shape
                return QImage(data.data, width, height, width, QImage.Format_Grayscale8)
            else:
                # Fallback normalization
                norm_data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
                height, width = norm_data.shape
                return QImage(norm_data.data, width, height, width, QImage.Format_Grayscale8)

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
        return sorted(self.images.keys())

    def get_asset_pair(self, name):
        # Often masks might have similar names but different extensions
        base_name = os.path.splitext(name)[0]
        image = self.images.get(name)
        
        # Try to find a matching mask
        mask = None
        for m_name in self.masks:
            if os.path.splitext(m_name)[0] == base_name:
                mask = self.masks[m_name]
                break
        
        return image, mask
