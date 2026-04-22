import json
import os
import numpy as np

def package_data_to_json(output_path, images=None, masks=None, histograms=None, project_metadata=None):
    """
    Package selected images, masks, and histograms into a JSON file.
    
    Args:
        output_path (str): Path to save the JSON file.
        images (list of dict): List of {'name': str, 'path': str}
        masks (list of dict): List of {'name': str, 'path': str}
        histograms (list of dict): List of {'name': str, 'data': np.ndarray, 'source_image': str, 'source_mask': str}
        project_metadata (dict): Additional project metadata to include.
    """
    data_to_save = {
        "project_metadata": project_metadata if project_metadata else {},
        "images": images if images else [],
        "masks": masks if masks else [],
        "histograms": []
    }
    
    if histograms:
        for hist in histograms:
            values = hist['data']
            # Try to get extra info from project_metadata if available
            h_name = os.path.splitext(hist['name'])[0]
            extra_info = {}
            if project_metadata and "histograms" in project_metadata:
                extra_info = project_metadata["histograms"].get(h_name, {})

            hist_entry = {
                "name": hist['name'],
                "source_image": hist.get('source_image'),
                "source_mask": hist.get('source_mask'),
                "mask_used": extra_info.get("mask_used"),
                "probe": extra_info.get("probe"),
                "cluster_method": extra_info.get("cluster_method"),
                "cluster_parameters": extra_info.get("cluster_parameters"),
                "statistics": {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "std": float(np.std(values))
                }
            }
            data_to_save["histograms"].append(hist_entry)

    # Also enrich masks if project_metadata has info
    if project_metadata and "masks" in project_metadata:
        for mask_entry in data_to_save["masks"]:
            m_name = os.path.splitext(mask_entry['name'])[0]
            if m_name in project_metadata["masks"]:
                mask_entry.update(project_metadata["masks"][m_name])
            
    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    
    return output_path
