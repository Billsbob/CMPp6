import numpy as np
import pandas as pd
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def calculate_cluster_statistics(stack, cluster_mask, mask_root_name, image_names, output_dir, mask=None):
    """
    Calculate cluster statistics and evaluation metrics.
    
    Args:
        stack (np.ndarray): Image stack (N, H, W)
        cluster_mask (np.ndarray): Cluster labels (H, W) where -1 is ignored.
        mask_root_name (str): Base name for the output CSV.
        image_names (list of str): Names of the images in the stack.
        output_dir (str): Directory to save the CSV.
        mask (np.ndarray, optional): Original mask used to restrict clustering.
        
    Returns:
        str: Path to the saved CSV file.
    """
    n_images, height, width = stack.shape
    
    # Flatten stack and mask for easier processing
    # stack: (N, H, W) -> data: (H*W, N)
    data = stack.transpose(1, 2, 0).reshape(-1, n_images)
    labels = cluster_mask.flatten()
    
    # Identify valid indices (where labels are not -1)
    valid_indices = np.where(labels != -1)[0]
    if valid_indices.size == 0:
        return None
    
    valid_data = data[valid_indices]
    valid_labels = labels[valid_indices]
    unique_labels = np.unique(valid_labels)
    
    total_area_pixels = valid_indices.size
    
    stats_rows = []
    
    # Calculate per-cluster statistics
    for label in unique_labels:
        cluster_indices = np.where(valid_labels == label)[0]
        cluster_data = valid_data[cluster_indices] # (pixels_in_cluster, N)
        
        cluster_size_pixels = cluster_indices.size
        cluster_percentage = (cluster_size_pixels / total_area_pixels) * 100
        
        # Base row data
        row_base = {
            'Cluster': f"Cluster {label + 1}",
            'Size (pixels)': cluster_size_pixels,
            'Size (%)': cluster_percentage
        }
        
        # Per probe (image) statistics
        for i, img_name in enumerate(image_names):
            probe_data = cluster_data[:, i]
            row_base[f'{img_name}_Mean'] = np.mean(probe_data)
            row_base[f'{img_name}_Min'] = np.min(probe_data)
            row_base[f'{img_name}_Max'] = np.max(probe_data)
            row_base[f'{img_name}_Std'] = np.std(probe_data)
            
        # Within cluster variance (mean variance across all probes)
        # Or variance of the cluster in the N-dimensional space
        cluster_variance = np.var(cluster_data, axis=0).mean()
        row_base['Within Cluster Variance'] = cluster_variance
        
        # Within cluster std mean (mean of standard deviations across all probes)
        cluster_std_mean = np.std(cluster_data, axis=0).mean()
        row_base['Within Cluster Std Mean'] = cluster_std_mean
        
        stats_rows.append(row_base)
        
    # Create DataFrame
    df = pd.DataFrame(stats_rows)
    
    # Overall evaluation metrics
    eval_metrics = {}
    
    # These metrics can be slow for large datasets, consider sampling if needed
    # But for now, we'll try to calculate them on the full valid data
    # silhouette_score is particularly slow O(n^2)
    sample_size = min(10000, valid_data.shape[0])
    if sample_size < valid_data.shape[0]:
        # Use a random sample for silhouette score to avoid memory/time issues
        idx = np.random.choice(valid_data.shape[0], sample_size, replace=False)
        sample_data = valid_data[idx]
        sample_labels = valid_labels[idx]
        if len(np.unique(sample_labels)) > 1:
            eval_metrics['Silhouette Score'] = silhouette_score(sample_data, sample_labels)
        else:
            eval_metrics['Silhouette Score'] = np.nan
    else:
        if len(np.unique(valid_labels)) > 1:
            eval_metrics['Silhouette Score'] = silhouette_score(valid_data, valid_labels)
        else:
            eval_metrics['Silhouette Score'] = np.nan

    if len(np.unique(valid_labels)) > 1:
        eval_metrics['Davies-Bouldin Index'] = davies_bouldin_score(valid_data, valid_labels)
        eval_metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(valid_data, valid_labels)
    else:
        eval_metrics['Davies-Bouldin Index'] = np.nan
        eval_metrics['Calinski-Harabasz Index'] = np.nan
        
    # Add evaluation metrics to the dataframe or as a separate section?
    # Usually better to have them as a separate block or in every row if they are global
    for metric, value in eval_metrics.items():
        df[metric] = value
        
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{mask_root_name}_statistics.csv")
    df.to_csv(csv_path, index=False)
    
    return csv_path
