import dask.array as da
import numpy as np
from dask import compute, delayed
from joblib import Parallel, delayed
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from fsec.qr_evd_mr import compute_evd_map_reduce, qr_eigenvalues_map_reduce

# def generate_base_clusterings(U, num_clusters_list, n_init=10, n_jobs=-1):
#     """
#     Generates multiple base clusterings by running k-means with different numbers of clusters.
#
#     Parameters:
#     - U: numpy array of shape (p, k), spectral embedding matrix.
#     - num_clusters_list: list of integers, specifying the number of clusters for each base clustering.
#     - n_init: Number of initializations for k-means to ensure robustness.
#     - n_jobs: Number of parallel jobs to run. -1 means using all processors.
#
#     Returns:
#     - base_clusterings: list of numpy arrays, each containing cluster labels for a base clustering.
#     """
#     if not num_clusters_list:
#         raise ValueError("num_clusters_list must contain at least one cluster number.")
#     
#     # Define a function to run k-means for a given number of clusters
#     def run_kmeans(k):
#         kmeans = KMeans(n_clusters=k, init='k-means++', n_init=n_init, random_state=None)
#         labels = kmeans.fit_predict(U)
#         return labels
#     
#     # Run k-means in parallel
#     base_clusterings = Parallel(n_jobs=n_jobs)(
#         delayed(run_kmeans)(k) for k in num_clusters_list
#     )
#     
#     return base_clusterings


import dask
from dask import delayed, compute
from sklearn.cluster import KMeans
import numpy as np

def generate_base_clusterings_dask(U, num_clusters_list, n_init=10):
    """
    Generates multiple base clusterings by running k-means with different numbers of clusters, 
    distributed using Dask.

    Parameters:
    - U: numpy array of shape (p, k), spectral embedding matrix.
    - num_clusters_list: list of integers, specifying the number of clusters for each base clustering.
    - n_init: Number of initializations for k-means to ensure robustness.

    Returns:
    - base_clusterings: list of numpy arrays, each containing cluster labels for a base clustering.
    """
    if not num_clusters_list:
        raise ValueError("num_clusters_list must contain at least one cluster number.")

    @delayed
    def run_kmeans(k):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=n_init, random_state=None)
        return kmeans.fit_predict(U)

    # Create a list of delayed tasks, one for each k in num_clusters_list
    tasks = [run_kmeans(k) for k in num_clusters_list]

    # Compute all tasks in parallel and gather results
    # The '*' unpacks the list of delayed tasks as separate arguments to compute
    base_clusterings = compute(*tasks)

    # compute returns a tuple of results corresponding to each delayed task
    return list(base_clusterings)


def build_bipartite_graph(base_clusterings):
    """
    Constructs a bipartite graph between samples and clusters based on multiple base clusterings.

    Parameters:
    - base_clusterings: List of NumPy arrays, each containing cluster labels for all samples.

    Returns:
    - H: A CSR sparse matrix representing the bipartite graph.
    """
    n_samples = base_clusterings[0].shape[0]
    row_indices = []
    col_indices = []
    data = []
    current_cluster_id = 0

    for labels in base_clusterings:
        unique_labels, inverse = np.unique(labels, return_inverse=True)
        n_clusters = unique_labels.size

        # Assign unique cluster indices across all base clusterings
        mapped_labels = inverse + current_cluster_id

        # Append data for the bipartite graph
        row_indices.extend(np.arange(n_samples))
        col_indices.extend(mapped_labels)
        data.extend([1] * n_samples)

        current_cluster_id += n_clusters

    total_clusters = current_cluster_id
    H = coo_matrix((data, (row_indices, col_indices)), shape=(n_samples, total_clusters))
    return H.tocsr()

def consensus_clustering(H, n_clusters):
    # Compute the degree matrix for clusters
    D_c = np.array(H.sum(axis=0)).flatten()
    D_c_inv = diags(1.0 / D_c)
    
    # Compute the Laplacian for the simplified graph
    D_r_inv = diags(1.0 / H.sum(axis=1).A.flatten())
    L_tilde = H.T.dot(D_r_inv).dot(H)
    
    n_components = n_clusters
    L_tilde = L_tilde.toarray()
    vecs = compute_evd_map_reduce(L_tilde, n_components)
    
    # Map back to the original samples
    f = H.dot(D_c_inv).dot(vecs)
    
    # Use the eigenvectors for clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    final_labels = kmeans.fit_predict(f)
    return final_labels
