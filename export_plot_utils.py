import os
import json
import pandas as pd
import shutil

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
    # Strip .npy from mask name if present
    if mask_name.lower().endswith(".npy"):
        mask_name = mask_name[:-4]
    
    safe_mask_name = "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in mask_name])
    json_filename = f"Measurements_{safe_mask_name}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(measurements, f)
        
    return json_path

def save_group_csv(measurements, mask_name, output_dir):
    """
    Save measurements as a CSV file where each column is an image.
    
    Args:
        measurements (dict): Image name to ROI intensity values.
        mask_name (str): Name of the mask used.
        output_dir (str): Directory to save the CSV file.
        
    Returns:
        str: Path to the saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Strip .npy from mask name if present
    if mask_name.lower().endswith(".npy"):
        mask_name = mask_name[:-4]
    
    safe_mask_name = "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in mask_name])
    csv_filename = f"Measurements_{safe_mask_name}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Create a DataFrame from measurements
    # Note: different images might have different number of pixels if masks were different,
    # but here they use the same mask, so lengths should be equal.
    # If not, it will pad with NaN.
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in measurements.items() ]))
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
