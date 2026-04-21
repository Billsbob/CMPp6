import os
os.environ['QT_API'] = 'pyside6'
import numpy as np
import cv2
import os
from .assets import Asset, TransformPipeline
import clustering

def create_test_images():
    # 1. 16-bit TIFF image (grayscale)
    data_16 = np.linspace(0, 65535, 100*100, dtype=np.uint16).reshape((100, 100))
    cv2.imwrite("test_16bit.tif", data_16)
    
    # 2. 32-bit float TIFF image (grayscale)
    data_32f = np.linspace(0.0, 1.0, 100*100, dtype=np.float32).reshape((100, 100))
    cv2.imwrite("test_32f.tif", data_32f)
    
    # 3. 8-bit RGB image (should be converted to grayscale, but what about others?)
    data_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    data_rgb[:, :, 2] = 100 # Red channel in BGR is 2? Wait, OpenCV is BGR. 
    # Let's just set one channel.
    cv2.imwrite("test_rgb.png", data_rgb)

def test_bit_depth_preservation():
    create_test_images()
    
    # Test 16-bit
    print("\n--- Testing 16-bit TIFF ---")
    asset_16 = Asset("test_16bit.tif")
    data_16 = asset_16.data
    print(f"Loaded dtype = {data_16.dtype}")
    rendered_16 = asset_16.get_rendered_data(data_only=True)
    print(f"Rendered (clustering) dtype = {rendered_16.dtype}")
    
    # Check K-means on 16-bit
    labels_kmeans_16 = clustering.kmeans_clustering(rendered_16[np.newaxis, ...], n_clusters=3, normalize=False)
    print(f"K-means labels shape: {labels_kmeans_16.shape}")
    
    # Check ISODATA on 16-bit
    labels_iso_16 = clustering.isodata_clustering(rendered_16[np.newaxis, ...], n_clusters=3, normalize=False)
    print(f"ISODATA labels shape: {labels_iso_16.shape}")

    # Test 32-bit float
    print("\n--- Testing 32-bit float TIFF ---")
    asset_32f = Asset("test_32f.tif")
    data_32f = asset_32f.data
    print(f"Loaded dtype = {data_32f.dtype}")
    rendered_32f = asset_32f.get_rendered_data(data_only=True)
    print(f"Rendered (clustering) dtype = {rendered_32f.dtype}")
    
    # Check K-means on 32-bit float
    labels_kmeans_32f = clustering.kmeans_clustering(rendered_32f[np.newaxis, ...], n_clusters=3, normalize=True)
    print(f"K-means (normalized) labels shape: {labels_kmeans_32f.shape}")
    
    # Clean up
    for f in ["test_16bit.tif", "test_32f.tif", "test_rgb.png"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    test_bit_depth_preservation()
