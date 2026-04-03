import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    logger.warning("cupy not found. CUDA support for ISODATA will be unavailable.")

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

def apply_isodata_cuda(data, initial_clusters=3, max_iter=100, min_samples=20, max_stddev=10, min_dist=20, max_merge_pairs=2, random_state=None, include_coords=False, coord_weight=1.0):
    """
    Apply ISODATA clustering to the image data using CUDA (cupy).
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

    if include_coords:
        y, x = cp.mgrid[0:h, 0:w]
        # Normalize coordinates to [0, 1] and apply weight
        x = (x.astype(cp.float32) / max(1, w - 1)) * coord_weight
        y = (y.astype(cp.float32) / max(1, h - 1)) * coord_weight
        coords = cp.stack([x.ravel(), y.ravel()], axis=-1)
        # Move features to GPU first
        features_gpu = cp.asarray(features)
        features_gpu = cp.hstack([features_gpu, coords])
    else:
        # Move to GPU
        features_gpu = cp.asarray(features)
    
    rng = cp.random.RandomState(random_state)

    # 1. Initialize means
    n_samples, n_features = features_gpu.shape
    if n_samples < initial_clusters:
        initial_clusters = n_samples
    
    indices = rng.choice(n_samples, initial_clusters, replace=False)
    means_gpu = features_gpu[indices]

    for iteration in range(max_iter):
        # 2. Assign samples to the nearest mean
        # Compute distances using norm for better precision matching CPU version
        # (features_gpu[:, np.newaxis] - means_gpu) has shape (n_samples, n_clusters, n_features)
        diff = features_gpu[:, cp.newaxis, :] - means_gpu[cp.newaxis, :, :]
        dist_sq = cp.sum(diff**2, axis=2)
        labels_gpu = cp.argmin(dist_sq, axis=1)

        # 3. Discard clusters with fewer than min_samples samples
        unique_labels, counts = cp.unique(labels_gpu, return_counts=True)
        # Handle case where some clusters might have 0 samples
        all_counts = cp.zeros(len(means_gpu), dtype=cp.int32)
        all_counts[unique_labels] = counts
        valid_mask = all_counts >= min_samples
        
        if not cp.all(valid_mask):
            valid_indices = cp.where(valid_mask)[0]
            if len(valid_indices) == 0:
                break
            means_gpu = means_gpu[valid_indices]
            # Re-assign
            diff = features_gpu[:, cp.newaxis, :] - means_gpu[cp.newaxis, :, :]
            dist_sq = cp.sum(diff**2, axis=2)
            labels_gpu = cp.argmin(dist_sq, axis=1)
            unique_labels, counts = cp.unique(labels_gpu, return_counts=True)
            # Update counts for the new set of means
            all_counts = cp.zeros(len(means_gpu), dtype=cp.int32)
            all_counts[unique_labels] = counts

        # 4. Update cluster means
        num_clusters = len(means_gpu)
        new_means_gpu = cp.zeros((num_clusters, n_features), dtype=features_gpu.dtype)
        for i in range(num_clusters):
            mask = (labels_gpu == i)
            if cp.any(mask):
                new_means_gpu[i] = cp.mean(features_gpu[mask], axis=0)
            else:
                new_means_gpu[i] = means_gpu[i]
        means_gpu = new_means_gpu

        num_clusters = len(means_gpu)
        if num_clusters == 0:
            break
            
        do_split = False
        do_merge = False
        
        if num_clusters <= initial_clusters // 2:
            do_split = True
        elif iteration % 2 == 0 or num_clusters >= initial_clusters * 2:
            do_merge = True
        else:
            do_split = True

        if do_split:
            # 5. Splitting
            new_means_gpu_list = []
            for i in range(num_clusters):
                mask = (labels_gpu == i)
                cluster_features = features_gpu[mask]
                if len(cluster_features) > 2 * (min_samples + 1):
                    std_devs = cp.std(cluster_features, axis=0)
                    max_std_idx = cp.argmax(std_devs)
                    max_std = std_devs[max_std_idx]
                    
                    if max_std > max_stddev:
                        # Split cluster
                        mean = means_gpu[i]
                        v = 0.5 * max_std
                        m1 = mean.copy()
                        m2 = mean.copy()
                        m1[max_std_idx] += v
                        m2[max_std_idx] -= v
                        new_means_gpu_list.append(m1)
                        new_means_gpu_list.append(m2)
                    else:
                        new_means_gpu_list.append(means_gpu[i])
                else:
                    new_means_gpu_list.append(means_gpu[i])
            means_gpu = cp.stack(new_means_gpu_list)
        
        elif do_merge and num_clusters > 1:
            # 6. Merging
            means_gpu_sq = cp.sum(means_gpu**2, axis=1, keepdims=True)
            pair_dist_sq = means_gpu_sq + means_gpu_sq.T - 2 * cp.dot(means_gpu, means_gpu.T)
            pair_dist_sq = cp.maximum(pair_dist_sq, 0)
            pair_distances_gpu = cp.sqrt(pair_dist_sq)
            
            pair_distances = cp.asnumpy(pair_distances_gpu)
            
            pairs = []
            for i in range(num_clusters):
                for j in range(i + 1, num_clusters):
                    dist = pair_distances[i, j]
                    if dist < min_dist:
                        pairs.append((i, j, dist))
            
            if pairs:
                pairs.sort(key=lambda x: x[2])
                merged_indices = set()
                new_means_gpu_list = []
                num_merges = 0
                
                for i, j, dist in pairs:
                    if num_merges >= max_merge_pairs:
                        break
                    if i not in merged_indices and j not in merged_indices:
                        merged_indices.add(i)
                        merged_indices.add(j)
                        m1 = means_gpu[i]
                        m2 = means_gpu[j]
                        n1 = all_counts[i]
                        n2 = all_counts[j]
                        m = (n1 * m1 + n2 * m2) / (n1 + n2)
                        new_means_gpu_list.append(m)
                        num_merges += 1
                
                for i in range(num_clusters):
                    if i not in merged_indices:
                        new_means_gpu_list.append(means_gpu[i])
                
                means_gpu = cp.stack(new_means_gpu_list)

    # Final labels
    features_gpu_sq = cp.sum(features_gpu**2, axis=1, keepdims=True)
    means_gpu_sq = cp.sum(means_gpu**2, axis=1, keepdims=True).T
    dist_sq = features_gpu_sq + means_gpu_sq - 2 * cp.dot(features_gpu, means_gpu.T)
    dist_sq = cp.maximum(dist_sq, 0)
    labels_gpu = cp.argmin(dist_sq, axis=1)
    
    labels = cp.asnumpy(labels_gpu)
    return labels.reshape(h, w)
