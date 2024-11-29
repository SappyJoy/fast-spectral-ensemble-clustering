import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.neighbors import NearestNeighbors

def recursive_bkmeans(data, num_anchors, current_depth=0, max_depth=None, use_mini_batch=False):
    if max_depth is None:
        max_depth = int(np.ceil(np.log2(num_anchors)))
    
    # Base cases
    if num_anchors == 1 or len(data) <= 1 or current_depth >= max_depth:
        if len(data) == 0:
            return []
        else:
            return [np.mean(data, axis=0)]
    
    # Check if data has at least 2 samples
    if len(data) < 2:
        return [np.mean(data, axis=0)]
    
    # Choose clustering method
    if use_mini_batch:
        kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=1000)
    else:
        kmeans = KMeans(n_clusters=2, random_state=42)
    
    labels = kmeans.fit_predict(data)
    left = data[labels == 0]
    right = data[labels == 1]
    
    # Allocate anchors to each split
    num_left = num_anchors // 2
    num_right = num_anchors - num_left
    
    # Handle edge cases where splits may have insufficient data
    anchors_left = recursive_bkmeans(left, num_left, current_depth + 1, max_depth, use_mini_batch)
    anchors_right = recursive_bkmeans(right, num_right, current_depth + 1, max_depth, use_mini_batch)
    
    return anchors_left + anchors_right

def BKHK(data, num_anchors, use_mini_batch=False):
    """
    Balanced K-means-Based Hierarchical K-means (BKHK) implementation.

    Parameters:
    - data: numpy array of shape (n_samples, n_features)
    - num_anchors: desired number of anchors
    - use_mini_batch: whether to use MiniBatchKMeans for large datasets

    Returns:
    - anchors: array of anchor points of shape (num_anchors, n_features)
    - anchor_assignments: array of anchor indices for each sample
    """
    anchors = recursive_bkmeans(data, num_anchors, use_mini_batch=use_mini_batch)
    anchors = np.array(anchors[:num_anchors])
    
    # Assign each sample to the nearest anchor
    anchor_assignments = pairwise_distances_argmin(data, anchors)
    
    return anchors, anchor_assignments

def compute_anchor_neighbors(anchors, K_prime):
    """
    Computes the K'-nearest neighbors for each anchor, excluding itself.

    Parameters:
    - anchors: numpy array of shape (p, d), where p is the number of anchors.
    - K_prime: Number of nearest neighbors to find for each anchor.

    Returns:
    - anchor_neighbors: numpy array of shape (p, K_prime) containing indices of K'-nearest neighbors for each anchor.
    """
    p = anchors.shape[0]
    K_prime = min(K_prime, p - 1)  # Ensure K_prime does not exceed p-1

    # Choose the appropriate algorithm based on data characteristics
    nbrs = NearestNeighbors(n_neighbors=K_prime + 1, algorithm='ball_tree').fit(anchors)
    distances, indices = nbrs.kneighbors(anchors)
    anchor_neighbors = indices[:, 1:]  # Exclude self
    return anchor_neighbors
