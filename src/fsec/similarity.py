from scipy.sparse import lil_matrix
import numpy as np

def compute_sample_anchor_similarities(data, anchors, anchor_assignments, anchor_neighbors, K):
    """
    Computes the similarity matrix W based on Equation (3) with improved stability.

    Parameters:
    - data: numpy array of shape (n_samples, n_features)
    - anchors: numpy array of shape (p, n_features)
    - anchor_assignments: array of shape (n_samples,), indicating assigned anchor for each sample
    - anchor_neighbors: numpy array of shape (p, K_prime), indices of K'-nearest neighbors for each anchor
    - K: Number of nearest anchors to retain for each sample

    Returns:
    - W: scipy.sparse.csr_matrix of shape (n_samples, p) with similarity scores
    """
    n_samples = data.shape[0]
    n_anchors = anchors.shape[0]
    W = lil_matrix((n_samples, n_anchors))
    
    epsilon = 1e-8  # Small value to prevent division by zero
    
    for i in range(n_samples):
        # Find the anchor assignment for this sample
        anchor_idx = anchor_assignments[i]
        
        # Candidate anchors are the K' nearest anchors to this anchor
        candidate_anchor_indices = anchor_neighbors[anchor_idx]
        
        # Include the assigned anchor itself
        candidate_anchor_indices = np.concatenate(([anchor_idx], candidate_anchor_indices))
        
        # Compute distances to candidate anchors
        candidate_anchors = anchors[candidate_anchor_indices]
        distances = np.linalg.norm(data[i] - candidate_anchors, axis=1)
        
        # If more than K + 1 candidates, select the top K
        if len(distances) > K + 1:
            sorted_indices = np.argsort(distances)
            distances = distances[sorted_indices][:K + 1]
            candidate_anchor_indices = candidate_anchor_indices[sorted_indices][:K + 1]
        
        # Sort distances and select top K
        sorted_indices = np.argsort(distances)
        K_nearest_indices = sorted_indices[:K]
        K_anchor_indices = candidate_anchor_indices[K_nearest_indices]
        K_distances = distances[K_nearest_indices]
        
        # Compute d(i, K+1)
        if len(distances) > K:
            d_i_K_plus_1 = distances[sorted_indices[K]]
        else:
            d_i_K_plus_1 = distances[sorted_indices[-1]] + epsilon  # Avoid zero denominator
        
        # Compute denominator
        sum_d_i_l = np.sum(K_distances)
        denominator = K * d_i_K_plus_1 - sum_d_i_l
        denominator = denominator if denominator > epsilon else epsilon  # Prevent division by zero
        
        # Compute similarities using Equation (3)
        similarities = (d_i_K_plus_1 - K_distances) / denominator
        similarities = np.maximum(similarities, 0)  # Ensure non-negative similarities
        similarities /= (np.sum(similarities) + epsilon)  # Normalize to sum to 1
        
        # Assign to W
        W[i, K_anchor_indices] = similarities
    
    return W, distances

