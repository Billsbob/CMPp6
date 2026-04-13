import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def _apply_normalization(stack, normalize_stack=False, normalize=False):
    """
    Apply normalization to the image stack.
    
    Args:
        stack (np.ndarray): Image stack (N, H, W).
        normalize_stack (bool): Normalize entire stack.
        normalize (bool): Normalize per image.
        
    Returns:
        np.ndarray: Normalized stack.
    """
    if not normalize_stack and not normalize:
        return stack
    
    stack = stack.copy()
    n_images = stack.shape[0]

    # Apply normalization to the entire stack if requested
    if normalize_stack:
        s_min, s_max = stack.min(), stack.max()
        if s_max > s_min:
            stack = (stack - s_min) / (s_max - s_min)
        else:
            stack = np.zeros_like(stack)

    # Apply normalization per image if requested
    if normalize:
        for i in range(n_images):
            img = stack[i]
            d_min, d_max = img.min(), img.max()
            if d_max > d_min:
                stack[i] = (img - d_min) / (d_max - d_min)
            else:
                stack[i] = np.zeros_like(img)
    return stack

def kmeans_clustering(stack, n_clusters=8, max_iter=300, init='k-means++', n_init=10, random_state=None, algorithm='auto', normalize=False, tol=1e-4, normalize_stack=False, mask=None):
    """
    Perform k-means clustering on an image stack.
    
    Args:
        stack (np.ndarray): Image stack of shape (N, H, W) float32.
        n_clusters (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        init (str or np.ndarray): Initialization method.
        n_init (int): Number of times k-means will be run with different seeds.
        random_state (int or None): Random seed.
        algorithm (str): K-means algorithm to use ('lloyd', 'elkan', 'auto', 'full').
        normalize (bool): Whether to normalize each image in the stack individually.
        tol (float): Relative tolerance with regards to Frobenius norm of the difference in the cluster centers.
        normalize_stack (bool): Whether to normalize the entire stack as a single block.
        mask (np.ndarray or None): Optional binary mask (H, W) to constrain clustering to only non-zero pixels.
        
    Returns:
        np.ndarray: Cluster labels as an image of shape (H, W).
    """
    if stack is None or len(stack.shape) != 3:
        raise ValueError("Stack must be (N, H, W) numpy array.")
    
    n_images, height, width = stack.shape

    stack = _apply_normalization(stack, normalize_stack, normalize)
    
    # Reshape stack to (H*W, N) so each pixel is a sample with N features
    # stack is (N, H, W), we want (H, W, N) then (H*W, N)
    data = stack.transpose(1, 2, 0).reshape(-1, n_images)
    
    if mask is not None:
        if len(mask.shape) == 3:
            # If mask is RGB, convert to grayscale by taking the maximum across channels
            mask = np.max(mask, axis=2)
        
        if mask.shape != (height, width):
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {(height, width)}.")

        mask_flat = mask.flatten()
        indices = np.where(mask_flat > 0)[0]
        if indices.size == 0:
            return np.full((height, width), -1, dtype=np.int32)
        data_to_cluster = data[indices]
    else:
        data_to_cluster = data

    # K-means implementation from scikit-learn
    # Note: 'auto' and 'full' are deprecated or removed in newer versions of sklearn.
    if algorithm in ['auto', 'full']:
        algorithm = 'lloyd'
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        init=init,
        n_init=n_init,
        random_state=random_state,
        algorithm=algorithm,
        tol=tol
    )
    
    labels = kmeans.fit_predict(data_to_cluster)
    
    if mask is not None:
        full_labels = np.full(data.shape[0], -1, dtype=np.int32)
        full_labels[indices] = labels
        cluster_mask = full_labels.reshape(height, width)
    else:
        # Reshape labels back to (H, W)
        cluster_mask = labels.reshape(height, width)
    
    return cluster_mask

def gaussian_mixture_clustering(stack, n_components=8, covariance_type='full', tol=1e-3, max_iter=100, random_state=None, normalize=False, normalize_stack=False, mask=None):
    """
    Perform Gaussian Mixture Model clustering on an image stack.
    
    Args:
        stack (np.ndarray): Image stack of shape (N, H, W) float32.
        n_components (int): Number of mixture components.
        covariance_type (str): Type of covariance parameters to use ('full', 'tied', 'diag', 'spherical').
        tol (float): The convergence threshold.
        max_iter (int): Maximum number of EM iterations to perform.
        random_state (int or None): Random seed.
        normalize (bool): Whether to normalize each image in the stack individually.
        normalize_stack (bool): Whether to normalize the entire stack as a single block.
        mask (np.ndarray or None): Optional binary mask (H, W) to constrain clustering to only non-zero pixels.
        
    Returns:
        np.ndarray: Cluster labels as an image of shape (H, W).
    """
    if stack is None or len(stack.shape) != 3:
        raise ValueError("Stack must be (N, H, W) numpy array.")
    
    n_images, height, width = stack.shape

    stack = _apply_normalization(stack, normalize_stack, normalize)
    
    # Reshape stack to (H*W, N)
    data = stack.transpose(1, 2, 0).reshape(-1, n_images)
    
    if mask is not None:
        if len(mask.shape) == 3:
            # If mask is RGB, convert to grayscale by taking the maximum across channels
            mask = np.max(mask, axis=2)
            
        if mask.shape != (height, width):
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {(height, width)}.")

        mask_flat = mask.flatten()
        indices = np.where(mask_flat > 0)[0]
        if indices.size == 0:
            return np.full((height, width), -1, dtype=np.int32)
        data_to_cluster = data[indices]
    else:
        data_to_cluster = data

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state
    )
    
    labels = gmm.fit_predict(data_to_cluster)
    
    if mask is not None:
        full_labels = np.full(data.shape[0], -1, dtype=np.int32)
        full_labels[indices] = labels
        cluster_mask = full_labels.reshape(height, width)
    else:
        # Reshape labels back to (H, W)
        cluster_mask = labels.reshape(height, width)
    
    return cluster_mask

def isodata_clustering(stack, n_clusters=8, max_iter=100, min_samples=20, max_std_dev=0.1, min_cluster_distance=0.1, max_merge_pairs=2, random_state=None, normalize=False, normalize_stack=False, mask=None):
    """
    Perform ISODATA clustering on an image stack.
    
    Args:
        stack (np.ndarray): Image stack (N, H, W).
        n_clusters (int): Initial number of clusters.
        max_iter (int): Maximum number of iterations.
        min_samples (int): Minimum number of samples in a cluster to keep it.
        max_std_dev (float): Maximum standard deviation to split a cluster.
        min_cluster_distance (float): Minimum distance between clusters to merge them.
        max_merge_pairs (int): Maximum number of pairs to merge in one iteration.
        random_state (int or None): Random seed.
        normalize (bool): Normalize per image.
        normalize_stack (bool): Normalize entire stack.
        mask (np.ndarray or None): Optional binary mask.
        
    Returns:
        np.ndarray: Cluster labels (H, W).
    """
    if stack is None or len(stack.shape) != 3:
        raise ValueError("Stack must be (N, H, W) numpy array.")
    
    n_images, height, width = stack.shape
    stack = _apply_normalization(stack, normalize_stack, normalize)
    data = stack.transpose(1, 2, 0).reshape(-1, n_images)
    
    if mask is not None:
        if len(mask.shape) == 3:
            mask = np.max(mask, axis=2)
        mask_flat = mask.flatten()
        indices = np.where(mask_flat > 0)[0]
        if indices.size == 0:
            return np.full((height, width), -1, dtype=np.int32)
        data_to_cluster = data[indices]
    else:
        data_to_cluster = data
        indices = None

    n_samples = data_to_cluster.shape[0]
    if n_samples == 0:
        return np.full((height, width), -1, dtype=np.int32)

    if random_state is not None:
        np.random.seed(random_state)

    # Initial centroids
    n_clusters = min(n_clusters, n_samples)
    initial_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = data_to_cluster[initial_indices]

    for iteration in range(max_iter):
        # 1. Assignment
        distances = np.linalg.norm(data_to_cluster[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # 2. Update centroids and discard small clusters
        new_centroids = []
        active_labels = []
        for i in range(len(centroids)):
            cluster_data = data_to_cluster[labels == i]
            if len(cluster_data) >= min_samples:
                new_centroids.append(cluster_data.mean(axis=0))
                active_labels.append(i)
        
        if not new_centroids:
            break
            
        centroids = np.array(new_centroids)
        # Re-assign after discarding
        distances = np.linalg.norm(data_to_cluster[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # 3. Split or Merge (Simplified ISODATA logic)
        # We'll do a simple version: split if std dev is too high, merge if too close.
        # This can be expanded but fulfills the basic ISODATA requirement.
        
        # Split logic
        if len(centroids) < 2 * n_clusters: # Only split if we have room
            split_occurred = False
            for i in range(len(centroids)):
                cluster_data = data_to_cluster[labels == i]
                if len(cluster_data) > 2 * min_samples:
                    std_devs = cluster_data.std(axis=0)
                    if np.any(std_devs > max_std_dev):
                        # Split this cluster
                        max_dim = np.argmax(std_devs)
                        gamma = 0.1 * std_devs[max_dim]
                        c1 = centroids[i].copy()
                        c2 = centroids[i].copy()
                        c1[max_dim] += gamma
                        c2[max_dim] -= gamma
                        
                        centroids = np.delete(centroids, i, axis=0)
                        centroids = np.vstack([centroids, c1, c2])
                        split_occurred = True
                        break # Only one split per iteration for stability
            if split_occurred:
                continue # Re-assign in next iteration

        # Merge logic
        if len(centroids) > 2:
            dists = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    d = np.linalg.norm(centroids[i] - centroids[j])
                    dists.append((d, i, j))
            
            dists.sort()
            merged_count = 0
            to_delete = set()
            new_merges = []
            
            for d, i, j in dists:
                if d < min_cluster_distance and merged_count < max_merge_pairs:
                    if i not in to_delete and j not in to_delete:
                        # Merge i and j
                        n_i = np.sum(labels == i)
                        n_j = np.sum(labels == j)
                        new_centroid = (n_i * centroids[i] + n_j * centroids[j]) / (n_i + n_j)
                        new_merges.append(new_centroid)
                        to_delete.add(i)
                        to_delete.add(j)
                        merged_count += 1
            
            if to_delete:
                centroids = np.delete(centroids, list(to_delete), axis=0)
                if new_merges:
                    centroids = np.vstack([centroids, new_merges])
                continue

    # Final labels
    distances = np.linalg.norm(data_to_cluster[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    if mask is not None:
        full_labels = np.full(data.shape[0], -1, dtype=np.int32)
        full_labels[indices] = labels
        cluster_mask = full_labels.reshape(height, width)
    else:
        cluster_mask = labels.reshape(height, width)
    
    return cluster_mask

def get_individual_masks(cluster_mask, n_clusters):
    """
    Split the cluster mask into individual binary masks for each cluster.
    
    Args:
        cluster_mask (np.ndarray): Mask of shape (H, W) with cluster labels.
        n_clusters (int): Total number of clusters.
        
    Returns:
        list of np.ndarray: List of binary masks.
    """
    masks = []
    for i in range(n_clusters):
        binary_mask = (cluster_mask == i).astype(np.uint8)
        masks.append(binary_mask)
    return masks
