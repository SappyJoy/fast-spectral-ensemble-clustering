import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD

def compute_svd(W, n_components):
    """
    Computes the top n_components singular vectors of Z = W * Lambda^{-1/2}.

    Parameters:
    - W: scipy.sparse.csr_matrix of shape (N, p), similarity matrix
    - n_components: Number of singular vectors to compute

    Returns:
    - U_normalized: numpy.ndarray of shape (N, n_components), left singular vectors normalized
    """
    # Compute the degree matrix Lambda for anchors
    # Lambda is diagonal with entries Lambda_jj = sum_i W_ij
    Lambda = np.array(W.sum(axis=0)).flatten()
    
    # Compute Lambda^{-1/2}
    Lambda_inv_sqrt = 1.0 / np.sqrt(Lambda + 1e-8)  # Add epsilon to prevent division by zero
    
    # Compute Z = W * Lambda^{-1/2}
    # Since Lambda_inv_sqrt is a diagonal matrix, we can multiply each column of W by Lambda_inv_sqrt[j]
    Z = W.multiply(Lambda_inv_sqrt)
    
    # Convert Z to dense format for SVD
    Z_dense = Z.toarray()
    
    # Perform SVD on Z
    U, Sigma, VT = svds(Z_dense, k=n_components)
    
    # svds returns singular values in ascending order, so reverse them
    idx = np.argsort(-Sigma)
    Sigma = Sigma[idx]
    U = U[:, idx]
    
    # Normalize U
    U_normalized = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
    
    return U_normalized

# This implementation works with cover type dataset
# def compute_svd(W, n_components):
#     """
#     Computes the top n_components singular vectors of Z = W * Lambda^{-1/2}.
#
#     Parameters:
#     - W: scipy.sparse.csr_matrix of shape (N, p), similarity matrix
#     - n_components: Number of singular vectors to compute
#
#     Returns:
#     - U_normalized: numpy.ndarray of shape (N, n_components), left singular vectors normalized
#     """
#     # Compute the degree matrix Lambda for anchors
#     Lambda = np.array(W.sum(axis=0)).flatten()
#
#     # Compute Lambda^{-1/2}
#     Lambda_inv_sqrt = 1.0 / np.sqrt(Lambda + 1e-8)  # Add epsilon to prevent division by zero
#
#     # Compute Z = W * Lambda^{-1/2}
#     Z = W.multiply(Lambda_inv_sqrt)
#
#     # Use TruncatedSVD instead of svds for better convergence
#     svd = TruncatedSVD(n_components=n_components, n_iter=100, tol=1e-4, random_state=42)
#     U = svd.fit_transform(Z)
#
#     # Normalize U
#     U_normalized = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
#
#     return U_normalized


