import numpy as np
import os
import json
import datetime
from PySide6.QtCore import QObject, Signal
import clustering

class ClusteringWorker(QObject):
    finished = Signal(object, str, int, object, str, dict)  # result_mask, mask_root_name, n_clusters, stats_info, method, params

    def __init__(self, algorithm, stack, mask, params, mask_root_name, image_names=None, output_dir=None):
        super().__init__()
        self.algorithm = algorithm
        self.stack = stack.astype(np.float32) if stack is not None else None
        self.mask = mask
        # Store a copy of params before they are popped
        self.params = params.copy()
        self.mask_root_name = mask_root_name
        self.image_names = image_names
        self.output_dir = output_dir

    def run(self):
        try:
            # Save debug log before clustering
            self._save_debug_log()

            # Create working copy of params for clustering functions
            run_params = self.params.copy()
            include_coords = run_params.pop("include_coords", False)
            x_weight = run_params.pop("x_weight", 1.0)
            y_weight = run_params.pop("y_weight", 1.0)

            if self.algorithm == "kmeans":
                result_mask = clustering.kmeans_clustering(
                    self.stack, mask=self.mask, 
                    include_coords=include_coords, 
                    x_weight=x_weight, y_weight=y_weight,
                    **run_params
                )
                n_clusters = run_params["n_clusters"]
            elif self.algorithm == "gmm":
                result_mask = clustering.gaussian_mixture_clustering(
                    self.stack, mask=self.mask, 
                    include_coords=include_coords, 
                    x_weight=x_weight, y_weight=y_weight,
                    **run_params
                )
                n_clusters = run_params["n_components"]
            elif self.algorithm == "isodata":
                result_mask = clustering.isodata_clustering(
                    self.stack, mask=self.mask, 
                    include_coords=include_coords, 
                    x_weight=x_weight, y_weight=y_weight,
                    **run_params
                )
                # For ISODATA, the number of clusters can change. 
                # We need to find the unique labels in result_mask (excluding -1)
                unique_labels = np.unique(result_mask)
                n_clusters = len(unique_labels[unique_labels != -1])
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
            # Generate statistics
            stats_csv_path = None
            if self.stack is not None and self.image_names is not None and self.output_dir is not None:
                import cluster_statistics
                stats_csv_path = cluster_statistics.calculate_cluster_statistics(
                    self.stack, result_mask, self.mask_root_name, 
                    self.image_names, self.output_dir, mask=self.mask
                )
            
            self.finished.emit(result_mask, self.mask_root_name, n_clusters, stats_csv_path, self.algorithm, self.params)
        except Exception as e:
            self.error.emit(str(e))

    def _save_debug_log(self):
        if self.stack is None:
            return

        try:
            # Working directory is the parent of the output_dir (Graphs)
            if self.output_dir:
                working_dir = os.path.dirname(self.output_dir)
                json_dir = os.path.join(working_dir, "JSON")
            else:
                # Fallback if output_dir is not set
                json_dir = "JSON"

            os.makedirs(json_dir, exist_ok=True)

            # Determine number of pixels after mask
            if self.mask is not None:
                # If mask is RGB, convert to grayscale
                mask_data = self.mask
                if len(mask_data.shape) == 3:
                    mask_data = np.max(mask_data, axis=2)
                num_pixels_after_mask = int(np.sum(mask_data > 0))
            else:
                num_pixels_after_mask = int(np.prod(self.stack.shape[1:]))

            # Unique values (this can be slow for large stacks, but it's a debug log)
            # Use a subset of pixels if the stack is too large to speed up np.unique?
            # For now, just do it as requested.
            # Convert to float64 for unique if needed, but float32 should be fine.
            num_unique_values = int(len(np.unique(self.stack)))

            # Number of components/clusters
            # For ISODATA, it's n_clusters, for GMM it's n_components, for KMeans it's n_clusters
            num_components = self.params.get("n_clusters") or self.params.get("n_components") or "N/A"
            
            # Covariance type
            covariance_type = self.params.get("covariance_type", "N/A")

            stats = {
                "stack.shape": list(self.stack.shape),
                "stack.dtype": str(self.stack.dtype),
                "stack.min": float(np.min(self.stack)),
                "stack.max": float(np.max(self.stack)),
                "has_nan": bool(np.isnan(self.stack).any()),
                "has_inf": bool(np.isinf(self.stack).any()),
                "num_pixels_after_mask": num_pixels_after_mask,
                "num_unique_values": num_unique_values,
                "covariance_type": covariance_type,
                "num_components": num_components,
                "algorithm": self.algorithm,
                "timestamp": datetime.datetime.now().isoformat()
            }

            filename = f"debug_log_{self.algorithm}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path = os.path.join(json_dir, filename)

            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=4)
        except Exception as e:
            print(f"Error saving debug log: {e}")
