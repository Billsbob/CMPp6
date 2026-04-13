import numpy as np
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, OPTICS
try:
    from isodata_cuda import apply_isodata_cuda, is_cuda_available
    from dbscan_cuda import apply_dbscan_cuda
except ImportError:
    apply_isodata_cuda = None
    apply_dbscan_cuda = None
    def is_cuda_available():
        return False

def prepare_features(data, include_coords=False, coord_weight=1.0):
    """
    Reshape image data into a feature vector for each pixel.
    :param data: numpy array of shape (H, W) or (H, W, C)
    :param include_coords: Whether to include (x, y) coordinates as features.
    :param coord_weight: Scaling factor for coordinates.
    :return: (features, h, w)
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

    return features, h, w

def apply_kmeans(data, n_clusters=3, max_iter=300, tol=1e-4, init='k-means++', random_state=None, include_coords=False, coord_weight=1.0):
    """
    Apply K-Means clustering to the image data.
    :param data: numpy array of shape (H, W) or (H, W, C)
    :param n_clusters: number of clusters
    :param max_iter: maximum number of iterations
    :param tol: tolerance
    :param init: initialization method ('k-means++' or 'random')
    :param random_state: random seed
    :param include_coords: Whether to include (x, y) coordinates as features.
    :param coord_weight: Scaling factor for coordinates.
    :return: cluster labels as a numpy array of shape (H, W)
    """
    features, h, w = prepare_features(data, include_coords, coord_weight)
    
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol, init=init, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(features)
    
    return labels.reshape(h, w)

def apply_isodata(data, initial_clusters=3, max_iter=100, min_samples=20, max_stddev=10, min_dist=20, max_merge_pairs=2, random_state=None, include_coords=False, coord_weight=1.0):
    """
    Apply ISODATA clustering to the image data.
    """
    features, h, w = prepare_features(data, include_coords, coord_weight)

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
    return labels.reshape(h, w)

def apply_dbscan(data, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', p=None, include_coords=False, coord_weight=1.0):
    """
    Apply DBSCAN clustering to the image data.
    :param data: numpy array of shape (H, W) or (H, W, C)
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    :param metric: The metric to use when calculating distance between instances in a feature array.
    :param algorithm: The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors.
    :param p: The power of the Minkowski metric to be used for the distance computation.
    :param include_coords: Whether to include (x, y) coordinates as features.
    :param coord_weight: Scaling factor for coordinates.
    :return: cluster labels as a numpy array of shape (H, W)
    """
    features, h, w = prepare_features(data, include_coords, coord_weight)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, p=p)
    labels = dbscan.fit_predict(features)

    return labels.reshape(h, w)

def apply_hdbscan(data, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, max_cluster_size=None, metric='euclidean', metric_params=None, alpha=1.0, algorithm='auto', leaf_size=40, n_jobs=None, cluster_selection_method='eom', allow_single_cluster=False, store_centers=None, copy=False, include_coords=False, coord_weight=1.0):
    """
    Apply HDBSCAN clustering to the image data.
    """
    features, h, w = prepare_features(data, include_coords, coord_weight)

    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size,
        metric=metric,
        metric_params=metric_params,
        alpha=alpha,
        algorithm=algorithm,
        leaf_size=leaf_size,
        n_jobs=n_jobs,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        store_centers=store_centers,
        copy=copy
    )
    labels = hdbscan.fit_predict(features)

    return labels.reshape(h, w)

def apply_optics(data, min_samples=5, max_eps=np.inf, metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, memory=None, n_jobs=None, include_coords=False, coord_weight=1.0):
    """
    Apply OPTICS clustering to the image data.
    """
    features, h, w = prepare_features(data, include_coords, coord_weight)

    optics = OPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        metric=metric,
        p=p,
        metric_params=metric_params,
        cluster_method=cluster_method,
        eps=eps,
        xi=xi,
        predecessor_correction=predecessor_correction,
        min_cluster_size=min_cluster_size,
        algorithm=algorithm,
        leaf_size=leaf_size,
        memory=memory,
        n_jobs=n_jobs
    )
    labels = optics.fit_predict(features)

    return labels.reshape(h, w)
