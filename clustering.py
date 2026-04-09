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

def kmeans_clustering(stack, n_clusters=8, max_iter=300, init='k-means++', n_init=10, random_state=None, algorithm='auto', normalize=False, tol=1e-4, normalize_stack=False):
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
    
    labels = kmeans.fit_predict(data)
    
    # Reshape labels back to (H, W)
    cluster_mask = labels.reshape(height, width)
    
    return cluster_mask

def gaussian_mixture_clustering(stack, n_components=8, covariance_type='full', tol=1e-3, max_iter=100, random_state=None, normalize=False, normalize_stack=False):
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
        
    Returns:
        np.ndarray: Cluster labels as an image of shape (H, W).
    """
    if stack is None or len(stack.shape) != 3:
        raise ValueError("Stack must be (N, H, W) numpy array.")
    
    n_images, height, width = stack.shape

    stack = _apply_normalization(stack, normalize_stack, normalize)
    
    # Reshape stack to (H*W, N)
    data = stack.transpose(1, 2, 0).reshape(-1, n_images)
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state
    )
    
    labels = gmm.fit_predict(data)
    
    # Reshape labels back to (H, W)
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
