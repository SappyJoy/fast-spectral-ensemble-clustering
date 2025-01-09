import dask.array as da
import numpy as np
from dask_ml.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin
from sklearn.neighbors import NearestNeighbors


def recursive_bkmeans_dask(data, num_anchors, current_depth=0, max_depth=None):
    if max_depth is None:
        max_depth = int(np.ceil(np.log2(num_anchors)))

    if num_anchors == 1 or data.shape[0] <= 1 or current_depth >= max_depth:
        if data.shape[0] == 0:
            return []
        return [data.mean(axis=0)]

    clusterer = KMeans(n_clusters=2, random_state=42)
    clusterer.fit(data)  # OK as data has known chunk sizes
    labels = clusterer.predict(data)

    # Must compute labels to know how big left/right will be
    labels_np = labels.compute()
    data_np = data.compute()

    # Split in memory
    left_np = data_np[labels_np == 0]
    right_np = data_np[labels_np == 1]

    # Convert back to dask arrays if you want
    left_da = da.from_array(left_np, chunks=(left_np.shape[0], data_np.shape[1]))
    right_da = da.from_array(right_np, chunks=(right_np.shape[0], data_np.shape[1]))

    num_left = num_anchors // 2
    num_right = num_anchors - num_left

    anchors_left = recursive_bkmeans_dask(left_da, num_left, current_depth + 1, max_depth)
    anchors_right = recursive_bkmeans_dask(right_da, num_right, current_depth + 1, max_depth)

    return anchors_left + anchors_right

def BKHK_dask(data, num_anchors):
    anchors_list = recursive_bkmeans_dask(data, num_anchors)
    # anchors_list is a Python list of Dask arrays
    anchors_np = [a.compute() for a in anchors_list]
    anchors_np = anchors_np[:num_anchors]
    anchors = np.stack(anchors_np, axis=0)

    # Final assignment
    data_np = data.compute()
    assignments = pairwise_distances_argmin(data_np, anchors)
    return anchors, assignments

def DBSCAN_anchor_selection(data, num_anchors, eps=0.5, min_samples=5):
    """
    Select anchors using DBSCAN to capture dense data regions.

    Parameters:
    - data: numpy array of shape (n_samples, n_features)
    - eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    - num_anchors: Desired number of anchors. If None, all core samples are used.

    Returns:
    - anchors: numpy array of anchor points.
    - anchor_assignments: array of anchor indices for each sample.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data)
    
    # Identify core samples
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    
    # Extract core samples
    core_samples = data[core_samples_mask]
    
    if num_anchors is not None and len(core_samples) > num_anchors:
        # If more core samples than desired anchors, perform KMeans on core samples
        kmeans = KMeans(n_clusters=num_anchors, random_state=42)
        kmeans.fit(core_samples)
        anchors = kmeans.cluster_centers_
    else:
        # Use all core samples as anchors
        anchors = core_samples
    
    # If no core samples found, fallback to BKHK
    if len(anchors) == 0:
        print("No core samples found by DBSCAN. Falling back to BKHK for anchor selection.")
        anchors, anchor_assignments = BKHK(data, num_anchors)
    else:
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
