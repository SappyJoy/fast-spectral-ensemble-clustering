import numpy as np

# =====================
# Helper Functions
# =====================

def map_compute_projections(qj_part, ai_part):
    """
    Compute partial projection of vector ai onto vector qj.
    :param qj_part: Partition of orthogonalized vector qj (shape: m_part).
    :param ai_part: Partition of current column vector ai (shape: m_part).
    :return: Partial scalar projection (shape: scalar).
    """
    proj = np.dot(qj_part, ai_part)
    if np.isnan(proj):
        raise ValueError("NaN encountered in projection computation.")
    return proj

def reduce_aggregate_projections(partial_projections):
    """
    Aggregate partial projections into a single scalar value.
    :param partial_projections: List of partial projections (shape: scalars).
    :return: Full scalar projection.
    """
    return np.sum(partial_projections)

def map_compute_norm_part(qi_part):
    """
    Compute partial squared norm for a partition of qi.
    :param qi_part: Partition of vector qi (shape: m_part).
    :return: Partial squared norm (shape: scalar).
    """
    return np.sum(qi_part**2)

def reduce_norm(partial_norms):
    """
    Aggregate partial squared norms and compute the full norm.
    :param partial_norms: List of partial squared norms (shape: scalars).
    :return: Full norm (shape: scalar).
    """
    total_norm = np.sqrt(np.sum(partial_norms))
    if total_norm == 0:
        raise ValueError("Norm is zero; cannot normalize a zero vector.")
    return total_norm

def map_matrix_multiply(R_part, Q):
    """
    Compute partial matrix multiplication for distributed computation.
    :param R_part: Partition of matrix R (shape: n_part x n).
    :param Q: Full matrix Q (shape: n x n).
    :return: Partial product (shape: n_part x n).
    """
    if np.isnan(R_part).any() or np.isnan(Q).any():
        raise ValueError("NaN encountered in matrix multiplication inputs.")
    return R_part @ Q

def reduce_matrix_multiply(partial_results):
    """
    Aggregate partial results of matrix multiplication row-wise.
    :param partial_results: List of partial products (shape: n_part x n).
    :return: Full product (shape: n x n).
    """
    return np.vstack(partial_results)

def map_compute_ZTZ_part(Z_part):
    """
    Compute partial product Z^T Z for distributed computation.
    :param Z_part: Partition of matrix Z (shape: n_part x p).
    :return: Partial product (shape: p x p).
    """
    return Z_part.T @ Z_part

def reduce_sum_ZTZ(partial_ZTZ):
    """
    Aggregate partial Z^T Z results into a single matrix.
    :param partial_ZTZ: List of partial products (shape: p x p).
    :return: Full Z^T Z matrix (shape: p x p).
    """
    return np.sum(partial_ZTZ, axis=0)

def map_compute_Uk_part(Z_part, V_k, Lambda_inv_sqrt):
    """
    Compute partial U_k matrix for distributed computation.
    :param Z_part: Partition of matrix Z (shape: n_part x p).
    :param V_k: Top eigenvectors of Z^T Z (shape: p x k).
    :param Lambda_inv_sqrt: Diagonal matrix of 1/sqrt(eigenvalues) (shape: k x k).
    :return: Partial U_k matrix (shape: n_part x k).
    """
    return Z_part @ V_k @ Lambda_inv_sqrt

# =====================
# Main Functions
# =====================

def qr_decomposition_simulated_map_reduce(A, num_partitions=2):
    """
    Perform QR decomposition using Map-Reduce for distributed computation.
    :param A: Input matrix (shape: m x n).
    :param num_partitions: Number of row-wise partitions for A and Q.
    :return: Q (orthogonal matrix, shape: m x n), R (upper triangular matrix, shape: n x n).
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        ai = A[:, i]
        for j in range(i):
            qj = Q[:, j]
            partial_projections = [
                map_compute_projections(qj_part, ai_part)
                for qj_part, ai_part in zip(np.array_split(qj, num_partitions), np.array_split(ai, num_partitions))
            ]
            proj = reduce_aggregate_projections(partial_projections)
            R[j, i] = proj  # Store the projection in R
            # You can optionally keep a list of projections if needed for subsequent steps.
        # After collecting all R[j,i] for j < i, proceed with orthogonalization:
        qi = ai.copy()
        for j in range(i):
            qi -= R[j, i] * Q[:, j]
        partial_norms = [map_compute_norm_part(qi_part) for qi_part in np.array_split(qi, num_partitions)]
        norm = reduce_norm(partial_norms)
        qi /= norm
        R[i, i] = norm
        Q[:, i] = qi

    return Q, R

def qr_eigenvalues_map_reduce(A, max_iterations=1000, tolerance=1e-6, num_partitions=2):
    """
    Compute eigenvalues and eigenvectors using QR decomposition with Map-Reduce.
    :param A: Input matrix (shape: n x n).
    :param max_iterations: Maximum number of QR iterations.
    :param tolerance: Convergence tolerance.
    :param num_partitions: Number of row-wise partitions.
    :return: eigenvalues (shape: n), eigenvectors (shape: n x n).
    """
    Ak = A.copy()
    eigenvectors = np.eye(A.shape[0])

    for _ in range(max_iterations):
        Q, R = qr_decomposition_simulated_map_reduce(Ak, num_partitions)
        partial_results = [map_matrix_multiply(R_part, Q) for R_part in np.array_split(R, num_partitions)]
        Ak = reduce_matrix_multiply(partial_results)
        eigenvectors_parts = [map_matrix_multiply(part, Q) for part in np.array_split(eigenvectors, num_partitions)]
        eigenvectors = reduce_matrix_multiply(eigenvectors_parts)

        if np.linalg.norm(np.tril(Ak, -1)) < tolerance:
            break

    eigenvalues = np.diag(Ak)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices for sorting
    eigenvalues = eigenvalues[sorted_indices]      # Sorted eigenvalues
    eigenvectors = eigenvectors[:, sorted_indices] # Sorted eigenvectors
    return eigenvalues, eigenvectors

def compute_evd_map_reduce(Z, k, max_iterations=1000, tolerance=1e-6, num_partitions=2):
    """
    Perform EVD using Map-Reduce and reconstruct the top-k eigenvectors.
    :param Z: Anchor graph matrix (shape: N x p).
    :param k: Number of top eigenvectors to retain.
    :param max_iterations: Maximum iterations for QR decomposition.
    :param tolerance: Convergence tolerance for QR decomposition.
    :param num_partitions: Number of row-wise partitions.
    :return: U_k (top-k eigenvectors of W, shape: N x k).
    """
    Z = Z.toarray()
    ZTZ_parts = [map_compute_ZTZ_part(part) for part in np.array_split(Z, num_partitions)]
    ZTZ = reduce_sum_ZTZ(ZTZ_parts)

    eigenvalues, eigenvectors = qr_eigenvalues_map_reduce(ZTZ, max_iterations, tolerance, num_partitions)
    V_k = eigenvectors[:, :k]
    Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues[:k]))

    U_parts = [map_compute_Uk_part(part, V_k, Lambda_inv_sqrt) for part in np.array_split(Z, num_partitions)]
    return np.vstack(U_parts)
