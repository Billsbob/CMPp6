import numpy as np
from PySide6.QtCore import QObject, Signal
import clustering
from clustering import cluster_statistics

class ClusteringWorker(QObject):
    finished = Signal(object, str, int, object)  # result_mask, mask_root_name, n_clusters, stats_info
    error = Signal(str)

    def __init__(self, algorithm, stack, mask, params, mask_root_name, image_names=None, output_dir=None):
        super().__init__()
        self.algorithm = algorithm
        self.stack = stack.astype(np.float32) if stack is not None else None
        self.mask = mask
        self.params = params
        self.mask_root_name = mask_root_name
        self.image_names = image_names
        self.output_dir = output_dir

    def run(self):
        try:
            # Extract coordinate parameters
            include_coords = self.params.pop("include_coords", False)
            x_weight = self.params.pop("x_weight", 1.0)
            y_weight = self.params.pop("y_weight", 1.0)

            if self.algorithm == "kmeans":
                result_mask = clustering.kmeans_clustering(
                    self.stack, mask=self.mask, 
                    include_coords=include_coords, 
                    x_weight=x_weight, y_weight=y_weight,
                    **self.params
                )
                n_clusters = self.params["n_clusters"]
            elif self.algorithm == "gmm":
                result_mask = clustering.gaussian_mixture_clustering(
                    self.stack, mask=self.mask, 
                    include_coords=include_coords, 
                    x_weight=x_weight, y_weight=y_weight,
                    **self.params
                )
                n_clusters = self.params["n_components"]
            elif self.algorithm == "isodata":
                result_mask = clustering.isodata_clustering(
                    self.stack, mask=self.mask, 
                    include_coords=include_coords, 
                    x_weight=x_weight, y_weight=y_weight,
                    **self.params
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
                stats_csv_path = cluster_statistics.calculate_cluster_statistics(
                    self.stack, result_mask, self.mask_root_name, 
                    self.image_names, self.output_dir, mask=self.mask
                )
            
            self.finished.emit(result_mask, self.mask_root_name, n_clusters, stats_csv_path)
        except Exception as e:
            self.error.emit(str(e))
