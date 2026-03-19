import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    import cupyx.scipy.sparse as sparse
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components as cpu_connected_components
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    logger.warning("cupy or scipy not found. CUDA support for DBSCAN will be unavailable.")

def is_cuda_available():
    if not CUDA_AVAILABLE:
        return False
    try:
        # Check if a CUDA-capable device is present and accessible
        cp.cuda.Device(0).use()
        return True
    except Exception as e:
        logger.warning(f"CUDA device not accessible: {e}")
        return False

def apply_dbscan_cuda(data, eps=0.5, min_samples=5, metric='euclidean', batch_size=2000):
    """
    Apply DBSCAN clustering to the image data using CUDA (cupy).
    :param data: numpy array of shape (H, W) or (H, W, C)
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :param metric: The metric to use when calculating distance. Currently only 'euclidean' is optimized.
    :param batch_size: Batch size for tiled computation to save memory.
    :return: cluster labels as a numpy array of shape (H, W)
    """
    if not is_cuda_available():
        raise RuntimeError("CUDA is not available or no CUDA device found.")

    h, w = data.shape[:2]
    if data.dtype != np.float32 and data.dtype != np.float64:
        data_float = data.astype(np.float32)
    else:
        data_float = data

    if data_float.ndim == 2:
        features = data_float.reshape(-1, 1)
    else:
        features = data_float.reshape(-1, data_float.shape[2])

    X = cp.asarray(features)
    N = X.shape[0]
    
    # 1. Identify core points
    # We compute neighbor counts in batches to stay within GPU memory
    neighbor_counts = cp.zeros(N, dtype=cp.int32)
    eps_sq = eps ** 2
    
    X_sq = cp.sum(X**2, axis=1)

    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        X_batch = X[i:end_i]
        X_batch_sq = X_sq[i:end_i, cp.newaxis]
        
        for j in range(0, N, batch_size):
            end_j = min(j + batch_size, N)
            X_all_batch = X[j:end_j]
            X_all_batch_sq = X_sq[cp.newaxis, j:end_j]
            
            # Compute squared Euclidean distance: x^2 + y^2 - 2xy
            dists_sq = X_batch_sq + X_all_batch_sq - 2 * cp.dot(X_batch, X_all_batch.T)
            
            neighbor_counts[i:end_i] += cp.sum(dists_sq <= eps_sq, axis=1)
            
    is_core = neighbor_counts >= min_samples
    core_indices = cp.where(is_core)[0]
    num_core = len(core_indices)
    
    if num_core == 0:
        # All points are noise (-1)
        return np.full((h, w), -1, dtype=np.int32)

    # 2. Build adjacency for core points and find connected components
    # We use a mapping to work with a smaller adjacency matrix of size num_core x num_core
    row_indices = []
    col_indices = []
    
    for i in range(0, num_core, batch_size):
        end_i = min(i + batch_size, num_core)
        X_core_batch = X[core_indices[i:end_i]]
        X_core_batch_sq = X_sq[core_indices[i:end_i], cp.newaxis]
        
        for j in range(i, num_core, batch_size):
            end_j = min(j + batch_size, num_core)
            X_core_all_batch = X[core_indices[j:end_j]]
            X_core_all_batch_sq = X_sq[core_indices[j:end_j], cp.newaxis].T
            
            dists_sq = X_core_batch_sq + X_core_all_batch_sq - 2 * cp.dot(X_core_batch, X_core_all_batch.T)
            mask = dists_sq <= eps_sq
            
            # Extract local indices
            local_row, local_col = cp.where(mask)
            
            # Global core indices
            row_indices.append(local_row + i)
            col_indices.append(local_col + j)
            
    if not row_indices:
        # This shouldn't happen as each point is its own neighbor
        return np.full((h, w), -1, dtype=np.int32)
        
    all_rows = cp.concatenate(row_indices)
    all_cols = cp.concatenate(col_indices)
    
    # Create sparse symmetric adjacency matrix
    # Move indices to CPU for connected_components (since cupy's version requires pylibcugraph)
    all_rows_cpu = all_rows.get()
    all_cols_cpu = all_cols.get()
    
    adj_cpu = sp.csr_matrix((np.ones(len(all_rows_cpu), dtype=bool), (all_rows_cpu, all_cols_cpu)), shape=(num_core, num_core))
    n_components, core_labels_cpu = cpu_connected_components(adj_cpu, directed=False)
    core_labels = cp.asarray(core_labels_cpu)
    
    # 3. Assign labels to original points
    # Initialize with -1 (noise)
    labels = cp.full(N, -1, dtype=cp.int32)
    labels[core_indices] = core_labels
    
    # 4. Assign non-core points to nearby core clusters (border points)
    non_core_indices = cp.where(~is_core)[0]
    if len(non_core_indices) > 0:
        # We process non-core points in batches
        for i in range(0, len(non_core_indices), batch_size):
            end_i = min(i + batch_size, len(non_core_indices))
            idx_batch = non_core_indices[i:end_i]
            X_nc_batch = X[idx_batch]
            X_nc_batch_sq = X_sq[idx_batch, cp.newaxis]
            
            # For each non-core batch, find any core point within eps
            for j in range(0, num_core, batch_size):
                end_j = min(j + batch_size, num_core)
                X_core_all_batch = X[core_indices[j:end_j]]
                X_core_all_batch_sq = X_sq[core_indices[j:end_j], cp.newaxis].T
                
                dists_sq = X_nc_batch_sq + X_core_all_batch_sq - 2 * cp.dot(X_nc_batch, X_core_all_batch.T)
                mask = dists_sq <= eps_sq
                
                # Check which non-core points in batch have a neighbor in this core batch
                has_neighbor = cp.any(mask, axis=1)
                if cp.any(has_neighbor):
                    # For points that found a neighbor, get the index of the first neighbor
                    neighbor_local_idx = cp.argmax(mask, axis=1)
                    # Get the label of that core point
                    found_labels = core_labels[neighbor_local_idx + j]
                    
                    # Update labels for points that were previously unassigned
                    # or follow a deterministic rule (e.g. first core cluster found)
                    current_batch_labels = labels[idx_batch]
                    update_mask = has_neighbor & (current_batch_labels == -1)
                    if cp.any(update_mask):
                        current_batch_labels[update_mask] = found_labels[update_mask]
                        labels[idx_batch] = current_batch_labels

    return labels.reshape(h, w).get()
