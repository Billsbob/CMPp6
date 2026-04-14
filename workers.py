import numpy as np
from PySide6.QtCore import QObject, Signal
import clustering

class ClusteringWorker(QObject):
    finished = Signal(object, str, int)  # result_mask, mask_root_name, n_clusters
    error = Signal(str)

    def __init__(self, algorithm, stack, mask, params, mask_root_name):
        super().__init__()
        self.algorithm = algorithm
        self.stack = stack.astype(np.float32) if stack is not None else None
        self.mask = mask
        self.params = params
        self.mask_root_name = mask_root_name

    def run(self):
        try:
            if self.algorithm == "kmeans":
                result_mask = clustering.kmeans_clustering(self.stack, mask=self.mask, **self.params)
                n_clusters = self.params["n_clusters"]
            elif self.algorithm == "gmm":
                result_mask = clustering.gaussian_mixture_clustering(self.stack, mask=self.mask, **self.params)
                n_clusters = self.params["n_components"]
            elif self.algorithm == "isodata":
                result_mask = clustering.isodata_clustering(self.stack, mask=self.mask, **self.params)
                # For ISODATA, the number of clusters can change. 
                # We need to find the unique labels in result_mask (excluding -1)
                unique_labels = np.unique(result_mask)
                n_clusters = len(unique_labels[unique_labels != -1])
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
            self.finished.emit(result_mask, self.mask_root_name, n_clusters)
        except Exception as e:
            self.error.emit(str(e))
