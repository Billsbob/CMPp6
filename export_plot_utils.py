import os
import json
import numpy as np
import pandas as pd
import shutil
from naming_utils import build_histogram_identity, sanitize_name, strip_extension

def save_measurements_json(measurements, mask_name, output_dir):
    """
    Save the measurements dictionary as a JSON file.
    
    Args:
        measurements (dict): Image name to ROI intensity values.
        mask_name (str): Name of the mask used.
        output_dir (str): Directory to save the JSON file.
        
    Returns:
        str: Path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    safe_mask_name = sanitize_name(strip_extension(mask_name))
    json_filename = f"Histograms_{safe_mask_name}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(measurements, f)
        
    return json_path

def get_safe_histogram_name(image_name, mask_name):
    """
    Generate a name following the convention <Mask Name>_<Well Position>_<Probe>
    based on the image name and mask name.
    """
    identity = build_histogram_identity(image_name, mask_name)
    return identity.safe_filename_base

def save_group_csv(measurements, mask_name, output_dir):
    """
    Save measurements as a CSV file where rows are intensity values (0-255)
    and columns are the frequency of pixels at each intensity for each image.
    
    Args:
        measurements (dict): Image name to ROI intensity values.
        mask_name (str): Name of the mask used.
        output_dir (str): Directory to save the CSV file.
        
    Returns:
        str: Path to the saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    safe_mask_name = sanitize_name(strip_extension(mask_name))
    csv_filename = f"Histograms_{safe_mask_name}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Create a dictionary for frequencies
    # Initialize with all intensities 0-255
    hist_data = {"Intensity": list(range(256))}
    
    for img_name, values in measurements.items():
        column_header = get_safe_histogram_name(img_name, mask_name)
        if len(values) > 0:
            # Calculate frequency for each intensity (0-255)
            # We assume values are in range 0-255. 
            # If values are normalized (0-1), we scale them to 0-255 for the fixed-bin CSV export.
            v = np.array(values)
            if v.max() <= 1.0001 and v.min() >= -0.0001 and len(v) > 0 and v.max() > v.min():
                counts, _ = np.histogram(v * 255.0, bins=range(257))
            else:
                counts, _ = np.histogram(v, bins=range(257))
            hist_data[column_header] = counts
        else:
            hist_data[column_header] = [0] * 256
            
    df = pd.DataFrame(hist_data)
    df.to_csv(csv_path, index=False)
    
    return csv_path

def export_png_files(source_files, output_dir):
    """
    Copy selected png files to the output directory.
    
    Args:
        source_files (list of str): Full paths to the source png files.
        output_dir (str): Directory to save the png files.
        
    Returns:
        list of str: Paths to the copied files.
    """
    os.makedirs(output_dir, exist_ok=True)
    copied_files = []
    for src in source_files:
        if os.path.exists(src):
            filename = os.path.basename(src)
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            copied_files.append(dst)
    return copied_files
