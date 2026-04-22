import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json

def calculate_image_stats(asset_manager, image_names):
    """
    Calculate statistics for a list of images (per image).
    
    Args:
        asset_manager (AssetManager): Manager to get image data.
        image_names (list of str): Names of images to measure.
        
    Returns:
        pd.DataFrame: DataFrame containing the statistics.
    """
    stats_list = []
    
    for name in image_names:
        asset = asset_manager.get_image_by_name(name)
        if asset:
            data = asset.get_rendered_data(data_only=True)
            if data is not None:
                data_flat = data.flatten()
                
                img_min = np.min(data_flat)
                img_max = np.max(data_flat)
                img_mean = np.mean(data_flat)
                img_std = np.std(data_flat)
                
                # Fraction of zero pixels
                fraction_zero = np.sum(data_flat == 0) / data_flat.size
                
                # Fraction of pixels near the max (within 5% of max value range)
                # If max is 0, then range is 0.
                if img_max > 0:
                    threshold = img_max * 0.95
                    fraction_near_max = np.sum(data_flat >= threshold) / data_flat.size
                else:
                    fraction_near_max = 0.0
                
                stats_list.append({
                    'Image': name,
                    'Min': img_min,
                    'Max': img_max,
                    'Mean': img_mean,
                    'Std': img_std,
                    'Fraction of Zero Pixels': fraction_zero,
                    'Fraction Near Max': fraction_near_max
                })
                
    df = pd.DataFrame(stats_list)
    if not df.empty:
        df = df.sort_values(by='Std', ascending=True)
    return df

def calculate_stack_stats(asset_manager, image_names):
    """
    Calculate statistics for the totality of the image stack (aggregated).
    
    Args:
        asset_manager (AssetManager): Manager to get image data.
        image_names (list of str): Names of images to measure.
        
    Returns:
        pd.DataFrame: DataFrame containing the statistics (single row).
    """
    all_data = []
    
    for name in image_names:
        asset = asset_manager.get_image_by_name(name)
        if asset:
            data = asset.get_rendered_data(data_only=True)
            if data is not None:
                all_data.append(data.flatten())
                
    if not all_data:
        return pd.DataFrame()
        
    stack_data = np.concatenate(all_data)
    
    stack_min = np.min(stack_data)
    stack_max = np.max(stack_data)
    stack_mean = np.mean(stack_data)
    stack_std = np.std(stack_data)
    
    # Fraction of zero pixels
    fraction_zero = np.sum(stack_data == 0) / stack_data.size
    
    # Fraction of pixels near the max
    if stack_max > 0:
        threshold = stack_max * 0.95
        fraction_near_max = np.sum(stack_data >= threshold) / stack_data.size
    else:
        fraction_near_max = 0.0
        
    df = pd.DataFrame([{
        'Image': 'Stack (All Selected)',
        'Min': stack_min,
        'Max': stack_max,
        'Mean': stack_mean,
        'Std': stack_std,
        'Fraction of Zero Pixels': fraction_zero,
        'Fraction Near Max': fraction_near_max
    }])
    
    return df

def save_stats_and_graphs(df, output_dir, base_name):
    """
    Save statistics to CSV and generate graphs.
    
    Args:
        df (pd.DataFrame): Statistics DataFrame.
        output_dir (str): Directory to save results.
        base_name (str): Base name for files ("All Img Stats" or "Img Stack").
    """
    if df.empty:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{base_name}.json")
    df.to_json(json_path, orient='records', indent=4)
    
    # Generate Graphs
    # We'll create a bar chart for each statistic except 'Image'
    metrics = ['Min', 'Max', 'Mean', 'Std', 'Fraction of Zero Pixels', 'Fraction Near Max']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        # Use bar chart only if multiple items, otherwise use a point or just label
        if len(df) > 1:
            plt.bar(df['Image'], df[metric])
            plt.xticks(rotation=45, ha='right')
        else:
            plt.bar([df['Image'].iloc[0]], [df[metric].iloc[0]])
            
        plt.title(f"{metric} ({base_name})")
        plt.xlabel("Image/Stack")
        plt.ylabel(metric)
        plt.tight_layout()
        
        graph_path = os.path.join(output_dir, f"{base_name}_{metric.replace(' ', '_')}.png")
        plt.savefig(graph_path)
        plt.close()

def diagnose_pairwise_similarity(stack):
    """
    Print average correlation of each image to the rest.
    Very high correlation across all others may indicate redundancy.
    """
    if stack is None or stack.ndim != 3:
        raise ValueError("Expected stack with shape (N, H, W).")

    n_images = stack.shape[0]
    flat = stack.reshape(n_images, -1).astype(np.float32)

    # Normalize each image
    flat -= flat.mean(axis=1, keepdims=True)
    denom = flat.std(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    flat /= denom

    corr = np.corrcoef(flat)

    print("\n--- Pairwise similarity diagnostic ---")
    for i in range(n_images):
        others = np.delete(corr[i], i)
        print(f"Image {i}: mean corr to others = {others.mean():.4f}, max corr = {others.max():.4f}")
    print("--------------------------------------\n")
