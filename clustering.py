import numpy as np
from sklearn.cluster import KMeans
def is_cuda_available():
    return False

def prepare_features(data, include_coords=False, coord_weight=1.0, mask=None):
    """
    Reshape image data into a feature vector for each pixel.
    :param data: numpy array of shape (H, W) or (H, W, C)
    :param include_coords: Whether to include (x, y) coordinates as features.
    :param coord_weight: Scaling factor for coordinates.
    :param mask: Optional mask of shape (H, W) or (H, W, 1). If provided, only pixels where mask > 0 are included.
    :return: (features, h, w, mask_indices)
    """
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
        y, x = np.mgrid[0:h, 0:w]
        # Normalize coordinates to [0, 1] and apply weight
        x = (x.astype(np.float32) / max(1, w - 1)) * coord_weight
        y = (y.astype(np.float32) / max(1, h - 1)) * coord_weight
        coords = np.stack([x.ravel(), y.ravel()], axis=-1)
        features = np.hstack([features, coords])

    mask_indices = None
    if mask is not None:
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_flat = mask.ravel()
        mask_indices = np.where(mask_flat > 0)[0]
        features = features[mask_indices]

    return features, h, w, mask_indices

def apply_kmeans(data, n_clusters=3, max_iter=300, tol=1e-4, init='k-means++', random_state=None, include_coords=False, coord_weight=1.0, mask=None):
    """
    Apply K-Means clustering to the image data.
    """
    features, h, w, mask_indices = prepare_features(data, include_coords, coord_weight, mask)
    
    if len(features) == 0:
        return np.full((h, w), -9999, dtype=np.int32)

    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol, init=init, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(features)
    
    if mask_indices is not None:
        full_labels = np.full(h * w, -9999, dtype=np.int32)
        full_labels[mask_indices] = labels
        return full_labels.reshape(h, w)
    
    return labels.reshape(h, w)

def apply_isodata(data, initial_clusters=3, max_iter=100, min_samples=20, max_stddev=10, min_dist=20, max_merge_pairs=2, random_state=None, include_coords=False, coord_weight=1.0, mask=None):
    """
    Apply ISODATA clustering to the image data.
    """
    features, h, w, mask_indices = prepare_features(data, include_coords, coord_weight, mask)

    if len(features) == 0:
        return np.full((h, w), -9999, dtype=np.int32)

    rng = np.random.RandomState(random_state)

    # 1. Initialize means
    n_samples, n_features = features.shape
    indices = rng.choice(n_samples, initial_clusters, replace=False)
    means = features[indices]

    for iteration in range(max_iter):
        # 2. Assign samples to the nearest mean
        distances = np.linalg.norm(features[:, np.newaxis] - means, axis=2)
        labels = np.argmin(distances, axis=1)

        # 3. Discard clusters with fewer than min_samples samples
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_mask = counts >= min_samples
        if not np.all(valid_mask):
            valid_labels = unique_labels[valid_mask]
            if len(valid_labels) == 0:
                break
            means = means[valid_labels]
            # Re-assign
            distances = np.linalg.norm(features[:, np.newaxis] - means, axis=2)
            labels = np.argmin(distances, axis=1)
            unique_labels, counts = np.unique(labels, return_counts=True)

        # 4. Update cluster means
        new_means = np.zeros((len(unique_labels), n_features))
        for i, label in enumerate(unique_labels):
            new_means[i] = np.mean(features[labels == label], axis=0)
        means = new_means

        num_clusters = len(means)
        
        # Determine if we should split or merge
        # Standard ISODATA rules:
        # If num_clusters <= initial_clusters / 2, split
        # If num_clusters >= initial_clusters * 2, merge
        # Otherwise, alternate split/merge on odd/even iterations
        
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
            new_means = []
            for i in range(num_clusters):
                cluster_features = features[labels == i]
                if len(cluster_features) > 2 * (min_samples + 1):
                    std_devs = np.std(cluster_features, axis=0)
                    max_std_idx = np.argmax(std_devs)
                    max_std = std_devs[max_std_idx]
                    
                    if max_std > max_stddev:
                        # Split cluster
                        mean = means[i]
                        v = 0.5 * max_std # Splitting factor
                        m1 = mean.copy()
                        m2 = mean.copy()
                        m1[max_std_idx] += v
                        m2[max_std_idx] -= v
                        new_means.append(m1)
                        new_means.append(m2)
                    else:
                        new_means.append(means[i])
                else:
                    new_means.append(means[i])
            means = np.array(new_means)
        
        elif do_merge and num_clusters > 1:
            # 6. Merging
            # Calculate distances between all pairs of cluster centers
            pair_distances = []
            for i in range(num_clusters):
                for j in range(i + 1, num_clusters):
                    dist = np.linalg.norm(means[i] - means[j])
                    if dist < min_dist:
                        pair_distances.append((i, j, dist))
            
            if pair_distances:
                pair_distances.sort(key=lambda x: x[2])
                merged_indices = set()
                new_means = []
                num_merges = 0
                
                # Merge up to max_merge_pairs
                for i, j, dist in pair_distances:
                    if num_merges >= max_merge_pairs:
                        break
                    if i not in merged_indices and j not in merged_indices:
                        # Merge clusters i and j
                        n_i = counts[i]
                        n_j = counts[j]
                        new_mean = (n_i * means[i] + n_j * means[j]) / (n_i + n_j)
                        new_means.append(new_mean)
                        merged_indices.add(i)
                        merged_indices.add(j)
                        num_merges += 1
                
                # Add remaining clusters
                for i in range(num_clusters):
                    if i not in merged_indices:
                        new_means.append(means[i])
                means = np.array(new_means)

    # Final assignment
    distances = np.linalg.norm(features[:, np.newaxis] - means, axis=2)
    labels = np.argmin(distances, axis=1)

    if mask_indices is not None:
        full_labels = np.full(h * w, -9999, dtype=np.int32)
        full_labels[mask_indices] = labels
        return full_labels.reshape(h, w)

    return labels.reshape(h, w)



