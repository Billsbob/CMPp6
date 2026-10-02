import os
import json
import numpy as np
import pandas as pd
import shutil
from naming_utils import build_histogram_identity, sanitize_name, strip_extension

def save_measurements_json(measurements, mask_name, output_dir):
    """
    Save the measurements dictionary as a JSON file, merging with existing data if it exists.
    
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
    
    # Load existing data if file exists
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
            # Update existing data with new measurements
            existing_data.update(measurements)
            measurements = existing_data
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not read existing JSON at {json_path}: {e}")
    
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
    Merges with existing CSV if it exists.
    
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
    
    # Load existing data if file exists
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            # Ensure "Intensity" is the first column and index
            if "Intensity" in df_existing.columns:
                df_existing.set_index("Intensity", inplace=True)
        except Exception as e:
            print(f"Warning: Could not read existing CSV at {csv_path}: {e}")
            df_existing = pd.DataFrame(index=range(256))
    else:
        df_existing = pd.DataFrame(index=range(256))
    
    # Create a dictionary for new frequencies
    new_hist_data = {}
    
    for img_name, values in measurements.items():
        column_header = get_safe_histogram_name(img_name, mask_name)
        if len(values) > 0:
            v = np.array(values)
            if v.max() <= 1.0001 and v.min() >= -0.0001 and len(v) > 0 and v.max() > v.min():
                counts, _ = np.histogram(v * 255.0, bins=range(257))
            else:
                counts, _ = np.histogram(v, bins=range(257))
            new_hist_data[column_header] = counts
        else:
            new_hist_data[column_header] = [0] * 256
            
    df_new = pd.DataFrame(new_hist_data, index=range(256))
    
    # Merge: update existing or add new columns
    for col in df_new.columns:
        df_existing[col] = df_new[col]
    
    # Reset index to make Intensity a column again
    df_final = df_existing.reset_index().rename(columns={"index": "Intensity"})
    
    # Ensure Intensity is the first column if it was somehow moved
    cols = df_final.columns.tolist()
    if "Intensity" in cols:
        cols.insert(0, cols.pop(cols.index("Intensity")))
        df_final = df_final[cols]
        
    df_final.to_csv(csv_path, index=False)
    
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
