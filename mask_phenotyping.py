import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import cv2
import measure_utilities
import histogram_plots
import export_plot_utils
import json
import scipy.stats

def phenotype_masks(asset_manager, mask_names, output_dir, tissue_mask_name, tissue_type, image_names=None, normalization_type="None", project_name=None):
    """
    Perform phenotyping on selected masks and output a condensed feature vector for each mask.
    
    Args:
        asset_manager (AssetManager): Manager to get mask data.
        mask_names (list of str): Names of the masks to phenotype.
        output_dir (str): Directory to save the output file.
        tissue_mask_name (str): Name of the mask representing the tissue area.
        tissue_type (str): "Retina" or "Organoid".
        image_names (list of str, optional): Names of images to calculate histogram features.
        normalization_type (str, optional): Normalization type for histograms.
        project_name (str, optional): Name of the project to use for the CSV filename.
        
    Returns:
        list of str: List containing the path to the generated CSV if successful, None otherwise.
    """
    all_mask_vectors = []
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get tissue mask area for denominator
    tissue_mask_asset = asset_manager.get_mask_by_name(tissue_mask_name)
    if not tissue_mask_asset:
        return None
    
    tissue_mask_data = tissue_mask_asset.get_rendered_data(data_only=True)
    if tissue_mask_data is None:
        return None
    
    tissue_area = np.sum(tissue_mask_data > 0)
    if tissue_area == 0:
        tissue_area = 1 # Avoid division by zero

    # Pre-calculate tissue metrics for spatial localization
    tissue_coords = np.argwhere(tissue_mask_data > 0)
    if tissue_coords.size > 0:
        t_min_y, t_min_x = tissue_coords.min(axis=0)
        t_max_y, t_max_x = tissue_coords.max(axis=0)
        tissue_width = t_max_x - t_min_x + 1
        tissue_height = t_max_y - t_min_y + 1
        tissue_centroid_y, tissue_centroid_x = tissue_coords.mean(axis=0)
    else:
        tissue_width = tissue_height = 1
        t_min_y = t_min_x = 0
        tissue_centroid_y = tissue_centroid_x = 0

    R_lookup = None
    if tissue_type == "Organoid" and tissue_coords.size > 0:
        # Precompute R(theta) for Organoid
        contours, _ = cv2.findContours((tissue_mask_data > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            all_boundary_points = np.vstack(contours).reshape(-1, 2)
            b_dy = all_boundary_points[:, 1] - tissue_centroid_y
            b_dx = all_boundary_points[:, 0] - tissue_centroid_x
            b_r = np.sqrt(b_dx**2 + b_dy**2)
            b_theta = np.arctan2(b_dy, b_dx)
            
            num_bins = 360
            theta_bins = np.linspace(-np.pi, np.pi, num_bins + 1)
            R_lookup = np.zeros(num_bins)
            
            bin_indices = np.digitize(b_theta, theta_bins) - 1
            for i in range(num_bins):
                mask = (bin_indices == i)
                if np.any(mask):
                    R_lookup[i] = np.max(b_r[mask])
            
            # Fill holes in R_lookup
            if np.any(R_lookup == 0):
                nz = np.where(R_lookup > 0)[0]
                if nz.size > 0:
                    indices = np.arange(num_bins)
                    ext_nz = np.concatenate([nz - num_bins, nz, nz + num_bins])
                    ext_vals = np.concatenate([R_lookup[nz], R_lookup[nz], R_lookup[nz]])
                    R_lookup = np.interp(indices, ext_nz, ext_vals)
                else:
                    R_lookup[:] = 1

    for mask_name in mask_names:
        mask_asset = asset_manager.get_mask_by_name(mask_name)
        if not mask_asset:
            continue
        
        mask_data = mask_asset.get_rendered_data(data_only=True)
        if mask_data is None:
            continue
        
        # Ensure binary (0 or 1)
        binary_mask = (mask_data > 0).astype(np.uint8)
            
        # Label the mask to get connected components
        labeled_mask, num_labels = label(binary_mask, return_num=True)
        props = regionprops(labeled_mask)
        
        if not props:
            # Add a row with zeros for empty masks to satisfy "one row per mask"
            base_vector = {
                'mask_name': mask_name,
                'relative_area_total': 0,
                'mask_area': 0,
                'component_count': 0,
                'component_density': 0,
                'largest_component_area': 0,
                'largest_component_fraction': 0,
                'small_component_fraction': 0,
                'solidity_weighted_mean': 0,
                'extent_weighted_mean': 0,
                'eccentricity_weighted_mean': 0,
                'circularity_weighted_mean': 0,
                'major_axis_weighted_mean': 0,
                'minor_axis_weighted_mean': 0,
                'feret_diameter_weighted_mean': 0,
                'area_median': 0,
                'area_p90': 0,
                'area_std': 0,
                'perimeter_p90': 0,
                'circularity_min': 0,
                'solidity_min': 0,
                'skeleton_length_total': 0,
                'skeleton_length_per_area': 0,
                'branch_count_total': 0,
                'branch_count_per_area': 0,
                'branch_density': 0
            }
            if tissue_type == "Retina":
                base_vector.update({'normalized_x': 0, 'normalized_y': 0})
            elif tissue_type == "Organoid":
                base_vector.update({
                    'normalized_radius_mean': 0, 'normalized_radius_std': 0,
                    'radial_p10': 0, 'radial_p25': 0, 'radial_p50': 0, 'radial_p75': 0, 'radial_p90': 0,
                    'angular_coverage': 0, 'angular_entropy': 0, 'radial_entropy': 0,
                    'circumferentiality': 0, 'radiality': 0, 'boundary_proximity': 0
                })
                for i in range(1, 11): base_vector[f'radial_occupancy_bin_{i}'] = 0
                for i in range(1, 13): base_vector[f'angular_occupancy_bin_{i}'] = 0
            total_mask_area = 0
        else:
            comp_areas = []
            comp_perimeters = []
            comp_solidities = []
            comp_extents = []
            comp_eccentricities = []
            comp_circularities = []
            comp_major_axes = []
            comp_minor_axes = []
            comp_ferets = []
            comp_skeletons = []
            comp_branches = []
            
            total_mask_area = 0
            
            for prop in props:
                area = prop.area
                total_mask_area += area
                comp_areas.append(area)
                comp_perimeters.append(prop.perimeter)
                comp_solidities.append(prop.solidity)
                comp_extents.append(prop.extent)
                comp_eccentricities.append(prop.eccentricity)
                comp_major_axes.append(prop.axis_major_length)
                comp_minor_axes.append(prop.axis_minor_length)
                
                # Feret diameter
                try:
                    feret = prop.feret_diameter_max
                except AttributeError:
                    feret = 0
                comp_ferets.append(feret)
                
                # Circularity: 4 * PI * area / (perimeter^2)
                if prop.perimeter > 0:
                    circularity = (4 * np.pi * area) / (prop.perimeter ** 2)
                else:
                    circularity = 0
                comp_circularities.append(circularity)
                
                # Skeleton length and branch count
                component_image = prop.image
                skeleton = skeletonize(component_image)
                skel_len = np.sum(skeleton)
                comp_skeletons.append(skel_len)
                
                # Branch count (junction points)
                neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, 
                                              np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), 
                                              borderType=cv2.BORDER_CONSTANT)
                branch_points = (skeleton > 0) & (neighbor_count > 2)
                comp_branches.append(np.sum(branch_points))

            # Convert to numpy for calculations
            areas = np.array(comp_areas)
            
            def weighted_mean(values):
                if total_mask_area == 0:
                    return 0
                return np.sum(np.array(values) * areas) / total_mask_area

            largest_area = np.max(areas)
            # Small components defined as area < 50 pixels
            small_comp_total_area = np.sum(areas[areas < 50])
            
            base_vector = {
                'mask_name': mask_name,
                'relative_area_total': total_mask_area / tissue_area,
                'mask_area': total_mask_area,
                'component_count': len(props),
                'component_density': len(props) / tissue_area,
                'largest_component_area': largest_area,
                'largest_component_fraction': largest_area / total_mask_area if total_mask_area > 0 else 0,
                'small_component_fraction': small_comp_total_area / total_mask_area if total_mask_area > 0 else 0,
                'solidity_weighted_mean': weighted_mean(comp_solidities),
                'extent_weighted_mean': weighted_mean(comp_extents),
                'eccentricity_weighted_mean': weighted_mean(comp_eccentricities),
                'circularity_weighted_mean': weighted_mean(comp_circularities),
                'major_axis_weighted_mean': weighted_mean(comp_major_axes),
                'minor_axis_weighted_mean': weighted_mean(comp_minor_axes),
                'feret_diameter_weighted_mean': weighted_mean(comp_ferets),
                'area_median': np.median(areas),
                'area_p90': np.percentile(areas, 90),
                'area_std': np.std(areas),
                'perimeter_p90': np.percentile(comp_perimeters, 90),
                'circularity_min': np.min(comp_circularities),
                'solidity_min': np.min(comp_solidities),
                'skeleton_length_total': np.sum(comp_skeletons),
                'skeleton_length_per_area': np.sum(comp_skeletons) / total_mask_area if total_mask_area > 0 else 0,
                'branch_count_total': np.sum(comp_branches),
                'branch_count_per_area': np.sum(comp_branches) / total_mask_area if total_mask_area > 0 else 0,
                'branch_density': np.sum(comp_branches) / tissue_area
            }

            # Add Spatial Localization Data
            if tissue_type == "Retina":
                target_coords = np.argwhere(binary_mask > 0)
                if target_coords.size > 0:
                    cY, cX = target_coords.mean(axis=0)
                    base_vector['normalized_x'] = (cX - t_min_x) / tissue_width
                    base_vector['normalized_y'] = (cY - t_min_y) / tissue_height
                else:
                    base_vector['normalized_x'] = 0
                    base_vector['normalized_y'] = 0
            elif tissue_type == "Organoid" and R_lookup is not None:
                target_coords = np.argwhere(binary_mask > 0)
                if target_coords.size > 0:
                    dy = target_coords[:, 0] - tissue_centroid_y
                    dx = target_coords[:, 1] - tissue_centroid_x
                    r = np.sqrt(dx**2 + dy**2)
                    theta = np.arctan2(dy, dx)
                    
                    theta_bins = np.linspace(-np.pi, np.pi, len(R_lookup) + 1)
                    target_bin_indices = np.digitize(theta, theta_bins) - 1
                    target_bin_indices = np.clip(target_bin_indices, 0, len(R_lookup) - 1)
                    R_target = R_lookup[target_bin_indices]
                    
                    r_normalized = r / np.where(R_target > 0, R_target, 1)
                    r_normalized = np.clip(r_normalized, 0, 1)
                    
                    base_vector['normalized_radius_mean'] = np.mean(r_normalized)
                    base_vector['normalized_radius_std'] = np.std(r_normalized)
                    for p in [10, 25, 50, 75, 90]:
                        base_vector[f'radial_p{p}'] = np.percentile(r_normalized, p)
                    
                    base_vector['angular_coverage'] = np.unique(target_bin_indices).size / len(R_lookup)
                    
                    def calculate_entropy(values, bins):
                        hist, _ = np.histogram(values, bins=bins)
                        p = hist / (np.sum(hist) + 1e-10)
                        p = p[p > 0]
                        return -np.sum(p * np.log2(p))
                    
                    base_vector['angular_entropy'] = calculate_entropy(theta, theta_bins)
                    base_vector['radial_entropy'] = calculate_entropy(r_normalized, np.linspace(0, 1, 11))
                    
                    # Circumferentiality and Radiality
                    rad_vals = []
                    circ_vals = []
                    for prop in props:
                        phi = prop.orientation
                        pY, pX = prop.centroid
                        p_theta = np.arctan2(pY - tissue_centroid_y, pX - tissue_centroid_x)
                        rad_vals.append(np.abs(np.sin(phi + p_theta)))
                        circ_vals.append(np.abs(np.cos(phi + p_theta)))
                    
                    base_vector['radiality'] = np.sum(np.array(rad_vals) * areas) / total_mask_area if total_mask_area > 0 else 0
                    base_vector['circumferentiality'] = np.sum(np.array(circ_vals) * areas) / total_mask_area if total_mask_area > 0 else 0
                    
                    # Occupancy profiles
                    rad_hist, _ = np.histogram(r_normalized, bins=np.linspace(0, 1, 11))
                    for i, val in enumerate(rad_hist):
                        base_vector[f'radial_occupancy_bin_{i+1}'] = val / (np.sum(rad_hist) + 1e-10)
                    
                    ang_hist, _ = np.histogram(theta, bins=np.linspace(-np.pi, np.pi, 13))
                    for i, val in enumerate(ang_hist):
                        base_vector[f'angular_occupancy_bin_{i+1}'] = val / (np.sum(ang_hist) + 1e-10)
                    
                    base_vector['boundary_proximity'] = np.mean(1 - r_normalized)

        if image_names:
            normalize = (normalization_type == "Local (per image)")
            normalize_stack = (normalization_type == "Global (entire stack)")
            
            measurements = measure_utilities.calculate_mask_measurements(
                asset_manager, image_names, mask_name,
                normalize=normalize, normalize_stack=normalize_stack
            )
            
            if measurements:
                graph_dir = os.path.join(asset_manager.working_dir, "Graphs")
                os.makedirs(graph_dir, exist_ok=True)
                
                project_json_path = asset_manager.get_project_json_path()
                project_data = {}
                if project_json_path and os.path.exists(project_json_path):
                    try:
                        with open(project_json_path, 'r') as f:
                            project_data = json.load(f)
                    except: pass
                
                mask_metadata = project_data.get("Masks", {}).get(mask_name, {})
                source_masks = mask_metadata.get("source_masks")
                
                # Create plots and saved data
                hist_files = histogram_plots.create_histograms(
                    measurements, mask_name, graph_dir,
                    source_masks=source_masks, show_kde=True,
                    normalization=normalization_type
                )
                export_plot_utils.save_measurements_json(measurements, mask_name, graph_dir)
                export_plot_utils.save_group_csv(measurements, mask_name, graph_dir)
                
                # Update project JSON (matching UI behavior)
                if "Histograms" not in project_data:
                    project_data["Histograms"] = {}
                
                cluster_method = mask_metadata.get("cluster_method", "Unknown")
                for hist_file in hist_files:
                    hist_path = os.path.join(graph_dir, hist_file)
                    # Find corresponding image
                    for img_name, values in measurements.items():
                        from export_plot_utils import get_safe_histogram_name
                        expected_name = f"{get_safe_histogram_name(img_name, mask_name)}.png"
                        if hist_file == expected_name:
                            # Calculate stats for project JSON
                            v = np.array(values)
                            stats_dict = {"mean": 0, "median": 0, "std": 0, "skewness": 0, "kurtosis": 0,
                                     "q05": 0, "q25": 0, "q75": 0, "q95": 0, "entropy": 0}
                            if v.size > 0:
                                stats_dict = {
                                    "mean": float(np.mean(v)),
                                    "median": float(np.median(v)),
                                    "std": float(np.std(v)),
                                    "skewness": float(scipy.stats.skew(v)) if v.size > 1 else 0.0,
                                    "kurtosis": float(scipy.stats.kurtosis(v)) if v.size > 1 else 0.0,
                                    "q05": float(np.percentile(v, 5)),
                                    "q25": float(np.percentile(v, 25)),
                                    "q75": float(np.percentile(v, 75)),
                                    "q95": float(np.percentile(v, 95)),
                                    "entropy": float(scipy.stats.differential_entropy(v)) if v.size > 1 else 0.0
                                }
                            
                            sample_name, slide_number, well_position, probe_name = "Unknown", "Unknown", "Unknown", "Unknown"
                            parts = img_name.split('_')
                            if len(parts) >= 6:
                                sample_name, slide_number, well_position = parts[0], parts[1], parts[4]
                                probe_name = os.path.splitext(parts[5])[0]
                            
                            project_data["Histograms"][hist_file] = {
                                "name": hist_file,
                                "path": os.path.abspath(hist_path),
                                "sample": sample_name,
                                "slide": slide_number,
                                "well": well_position,
                                "probe": probe_name,
                                "linked_mask": mask_name,
                                "cluster_method": cluster_method,
                                "normalization": normalization_type,
                                "histograms_json": os.path.abspath(os.path.join(graph_dir, f"Histograms_{mask_name}.json")),
                                "histograms_csv": os.path.abspath(os.path.join(graph_dir, f"Histograms_{mask_name}.csv")),
                                **stats_dict
                            }
                            break
                
                if project_json_path:
                    with open(project_json_path, 'w') as f:
                        json.dump(project_data, f, indent=4)
                
                # Add rows to all_mask_vectors
                for image_name in image_names:
                    row = base_vector.copy()
                    row['image_name'] = image_name
                    if image_name in measurements:
                        hist_features = measure_utilities.calculate_histogram_features(measurements[image_name])
                        row.update(hist_features)
                    else:
                        row.update({'mode': 0, 'peak intensity': 0, 'peak prominence': 0, 'number of peaks': 0, 'secondary peak location': 0})
                    all_mask_vectors.append(row)
            else:
                # No measurements found
                for image_name in image_names:
                    row = base_vector.copy()
                    row['image_name'] = image_name
                    row.update({'mode': 0, 'peak intensity': 0, 'peak prominence': 0, 'number of peaks': 0, 'secondary peak location': 0})
                    all_mask_vectors.append(row)
        else:
            all_mask_vectors.append(base_vector)

    if all_mask_vectors:
        df = pd.DataFrame(all_mask_vectors)
        if project_name:
            csv_filename = f"{project_name}_feature_vectors.csv"
        else:
            csv_filename = "mask_feature_vectors.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        return [csv_path]
            
    return None
