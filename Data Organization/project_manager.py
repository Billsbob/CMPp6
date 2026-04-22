import os
import json
import numpy as np

def create_project_json(working_dir, images):
    """
    Create or update the main project JSON file that tracks image IDs.
    """
    if not images:
        return None

    # Collect IDs (filenames)
    image_ids = [os.path.splitext(f)[0] for f in images]
    
    # Determine project name from first image: <Sample>_<Slide ##>_...
    first_image = images[0]
    parts = first_image.split('_')
    if len(parts) >= 2:
        project_name = f"{parts[0]}_{parts[1]}"
    else:
        project_name = os.path.splitext(first_image)[0]

    project_json_path = os.path.join(working_dir, "JSON", f"{project_name}.json")
    
    project_data = {
        "project_name": project_name,
        "image_ids": image_ids,
        "masks": {},
        "histograms": {}
    }
    
    # Load existing if any to preserve other info
    if os.path.exists(project_json_path):
        try:
            with open(project_json_path, 'r') as f:
                existing_data = json.load(f)
                # Update image_ids but keep other fields
                existing_data["image_ids"] = image_ids
                # Ensure other keys exist
                for key in ["masks", "histograms"]:
                    if key not in existing_data:
                        existing_data[key] = project_data[key]
                project_data = existing_data
        except Exception as e:
            print(f"Error loading existing project JSON: {e}")

    with open(project_json_path, 'w') as f:
        json.dump(project_data, f, indent=4)
    
    return project_json_path

def update_mask_metadata(working_dir, image_names, mask_name, algorithm, params):
    """
    Update project JSON with mask metadata.
    """
    if not image_names:
        return

    # Extract probes
    probes = []
    for name in image_names:
        parts = name.split('_')
        if len(parts) >= 6:
            probes.append(parts[5].split('.')[0])
        else:
            probes.append("Unknown")
    probes = sorted(list(set(probes)))

    # Determine project JSON path
    parts = image_names[0].split('_')
    if len(parts) >= 2:
        project_name = f"{parts[0]}_{parts[1]}"
        project_json_path = os.path.join(working_dir, "JSON", f"{project_name}.json")
        
        if os.path.exists(project_json_path):
            try:
                with open(project_json_path, 'r') as f:
                    project_data = json.load(f)
                
                project_data.setdefault("masks", {})[mask_name] = {
                    "probes": probes,
                    "cluster_method": algorithm,
                    "cluster_parameters": params
                }
                
                with open(project_json_path, 'w') as f:
                    json.dump(project_data, f, indent=4)
            except Exception as e:
                print(f"Error updating project JSON with mask info: {e}")

def update_histogram_metadata(working_dir, image_names, mask_name, hist_files):
    """
    Update project JSON with histogram metadata.
    """
    if not image_names:
        return

    parts = image_names[0].split('_')
    if len(parts) >= 2:
        project_name = f"{parts[0]}_{parts[1]}"
        project_json_path = os.path.join(working_dir, "JSON", f"{project_name}.json")
        
        if os.path.exists(project_json_path):
            try:
                with open(project_json_path, 'r') as f:
                    project_data = json.load(f)
                
                base_mask_name = os.path.splitext(mask_name)[0]
                mask_info = project_data.get("masks", {}).get(base_mask_name, {})
                
                for h_file in hist_files:
                    h_base = os.path.splitext(h_file)[0]
                    h_parts = h_base.split('_')
                    probe = h_parts[-1] if len(h_parts) >= 2 else "Unknown"
                    
                    project_data.setdefault("histograms", {})[h_base] = {
                        "mask_used": base_mask_name,
                        "probe": probe,
                        "cluster_method": mask_info.get("cluster_method"),
                        "cluster_parameters": mask_info.get("cluster_parameters")
                    }
                
                with open(project_json_path, 'w') as f:
                    json.dump(project_data, f, indent=4)
            except Exception as e:
                print(f"Error updating project JSON for histograms: {e}")

def get_project_metadata(working_dir, image_names):
    """
    Load project metadata for packaging.
    """
    if not image_names:
        return {}
        
    parts = image_names[0].split('_')
    if len(parts) >= 2:
        project_name = f"{parts[0]}_{parts[1]}"
        project_json_path = os.path.join(working_dir, "JSON", f"{project_name}.json")
        if os.path.exists(project_json_path):
            try:
                with open(project_json_path, 'r') as f:
                    return json.load(f)
            except:
                pass
    return {}
