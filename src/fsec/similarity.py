import dask.array as da
import numpy as np
from dask import compute, delayed
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, vstack


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
    
    return W



def _compute_similarities_chunk(
    data_chunk, 
    start_idx, 
    anchor_assignments_chunk, 
    anchors, 
    anchor_neighbors,
    K
):
    """
    Process a single chunk of the data to compute partial W and store full distances.

    Parameters
    ----------
    ...  # (same as before)

    Returns
    -------
    rows, cols, vals, distances : np.ndarray, np.ndarray, np.ndarray, np.ndarray
        Triplets for sparse matrix and full distances for each sample.
    """
    chunk_size = data_chunk.shape[0]
    epsilon = 1e-8

    row_coords = []
    col_coords = []
    values = []
    distances_list = []

    for i in range(chunk_size):
        global_i = start_idx + i

        # Instead of using neighbor-based selection, use all anchors
        candidate_anchor_indices = np.arange(anchors.shape[0])
        candidate_anchors = anchors[candidate_anchor_indices]
        
        # Compute distances to all anchors for the full distance vector
        distances = np.linalg.norm(data_chunk[i] - candidate_anchors, axis=1)
        full_distances = distances.copy()  # Save the full distances before any truncation

        # Truncate distances for similarity computations if necessary
        if len(distances) > K + 1:
            sort_idx = np.argsort(distances)
            distances = distances[sort_idx][:K + 1]
            candidate_anchor_indices = candidate_anchor_indices[sort_idx][:K + 1]

        sorted_idx = np.argsort(distances)
        K_nearest_idx = sorted_idx[:K]
        K_anchors = candidate_anchor_indices[K_nearest_idx]
        K_dists = distances[K_nearest_idx]

        if len(distances) > K:
            d_i_K_plus_1 = distances[sorted_idx[K]]
        else:
            d_i_K_plus_1 = distances[sorted_idx[-1]] + epsilon

        sum_d = np.sum(K_dists)
        denom = K * d_i_K_plus_1 - sum_d
        if denom <= epsilon:
            denom = epsilon

        similarities = (d_i_K_plus_1 - K_dists) / denom
        similarities = np.maximum(similarities, 0)
        similarities /= (similarities.sum() + epsilon)

        for j, sim in zip(K_anchors, similarities):
            row_coords.append(global_i)
            col_coords.append(j)
            values.append(sim)
        
        # Append the full (untruncated) distances for this sample
        distances_list.append(full_distances)

    # Convert collected distances to a 2D array: shape (chunk_size, num_anchors)
    distances_array = np.array(distances_list)

    return (np.array(row_coords), np.array(col_coords), np.array(values), distances_array)


def compute_sample_anchor_similarities_dask(data, anchors, anchor_assignments, anchor_neighbors, K):
    # Ensure data and assignments are dask arrays with row-wise chunks
    if not isinstance(data, da.Array):
        data = da.from_array(data, chunks=(1000, data.shape[1]))
    if not isinstance(anchor_assignments, da.Array):
        anchor_assignments = da.from_array(anchor_assignments, chunks=(1000,))
        
    # Convert data and assignments to lists of delayed chunks
    data_delayed_chunks = data.to_delayed().ravel()  
    assignments_delayed_chunks = anchor_assignments.to_delayed().ravel()
    
    # Compute chunk sizes and starting indices
    chunk_sizes = data.chunks[0]
    start_indices = np.cumsum((0,) + chunk_sizes[:-1])
    
    tasks = []
    for i, (d_chunk_delayed, a_chunk_delayed) in enumerate(zip(data_delayed_chunks, assignments_delayed_chunks)):
        start_idx = int(start_indices[i])
        # Create a delayed task for each chunk
        task = delayed(_compute_similarities_chunk)(
            d_chunk_delayed,
            start_idx,
            a_chunk_delayed,
            anchors,
            anchor_neighbors,
            K
        )
        tasks.append(task)

    # Execute all tasks in parallel and gather results
    results = compute(*tasks)
    
    # Initialize lists to collect triplets from each chunk
    row_coords_list = []
    col_coords_list = []
    vals_list = []
    distances_list = []

    for rows_arr, cols_arr, vals_arr, distances_arr in results:
        row_coords_list.append(rows_arr)
        col_coords_list.append(cols_arr)
        vals_list.append(vals_arr)
        distances_list.append(distances_arr)

    # Concatenate results from all chunks
    row_coords = np.concatenate(row_coords_list) if row_coords_list else np.array([], dtype=int)
    col_coords = np.concatenate(col_coords_list) if col_coords_list else np.array([], dtype=int)
    vals = np.concatenate(vals_list) if vals_list else np.array([], dtype=float)
    distances = np.concatenate(distances_list) if distances_list else np.array([], dtype=float)

    # Construct the final sparse matrix
    n_samples = data.shape[0]
    p = anchors.shape[0]
    W_coo = coo_matrix((vals, (row_coords, col_coords)), shape=(n_samples, p))

    return W_coo.tocsr(), distances
