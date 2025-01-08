import numpy as np

def compute_projection(qj, ai):
    """
    Compute the projection of vector ai onto vector qj.
    :param qj: Orthogonalized vector from Q.
    :param ai: Current column vector of A.
    :return: Scalar projection value.
    """
    return np.dot(qj, ai)

def normalize_vector(qi):
    """
    Normalize a vector.
    :param qi: A vector to normalize.
    :return: Normalized vector and its norm.
    """
    norm = np.linalg.norm(qi)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector (dependent columns in A).")
    return qi / norm, norm

def qr_decomposition_simulated(A):
    """
    Perform QR decomposition of matrix A with a structure for distributed computation.
    :param A: Input matrix (m x n).
    :return: Q (orthogonal matrix), R (upper triangular matrix).
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for i in range(n):
        # Extract the i-th column of A
        ai = A[:, i]
        
        # Orthogonalize ai against all previous qj
        qi = ai.copy()
        projections = []
        
        # Simulated parallel section for projections
        for j in range(i):  # Placeholder for parallelization
            qj = Q[:, j]
            proj = compute_projection(qj, ai)
            projections.append((j, proj))
        
        # Sync Point: Apply projections to orthogonalize qi
        for j, proj in projections:
            R[j, i] = proj
            qi -= proj * Q[:, j]
        
        # Normalize qi
        qi, norm = normalize_vector(qi)
        R[i, i] = norm
        Q[:, i] = qi  # Sync Point: qi is fully computed
    
    return Q, R

def qr_eigenvalues(A, max_iterations=1000, tolerance=1e-6):
    """
    Нахождение собственных значений с помощью итерационного QR-разложения.
    """
    n = A.shape[0]
    Ak = A.copy()
    eigenvectors = np.eye(A.shape[0])
    for i in range(max_iterations):
        # Q, R = np.linalg.qr(Ak)  # QR-разложение текущей матрицы
        Q, R = qr_decomposition_simulated(Ak)
        Ak = np.dot(R, Q)        # Обновляем матрицу
        eigenvectors=np.dot(eigenvectors, Q)
        
        # Проверка на треугольность (если вне диагонали элементы близки к 0)
        off_diagonal_norm = np.linalg.norm(np.tril(Ak, -1))
        if off_diagonal_norm < tolerance:
            print(f"early stop {i}")
            break

    eigenvalues = np.diag(Ak)  # Собственные значения на диагонали
    return eigenvalues, eigenvectors

def qr_sorted_eigenvectors(A, max_iterations=1000, tolerance=1e-6):
    eigenvalues, eigenvectors = qr_eigenvalues(A, max_iterations, tolerance)
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Индексы для сортировки по убыванию
    eigenvalues = eigenvalues[sorted_indices]      # Упорядоченные собственные значения
    eigenvectors = eigenvectors[:, sorted_indices] # Упорядоченные собственные векторы
    
    return eigenvalues, eigenvectors

def compute_evd(Z, k, max_iterations=1000, tolerance=1e-6):
    """
    Perform EVD on the matrix Z^T Z and reconstruct the top-k eigenvectors of W.
    
    Parameters:
        Z (numpy.ndarray): Normalized anchor graph of size (N, p).
        k (int): Number of top eigenvectors to retain.
        max_iterations (int): Maximum iterations for the QR algorithm.
        tolerance (float): Convergence tolerance for the QR algorithm.
    
    Returns:
        U_k (numpy.ndarray): Top-k eigenvectors of W of size (N, k).
    """
    # Compute Z^T Z (size p x p)
    ZTZ = Z.T @ Z
    ZTZ = ZTZ.toarray()
    # Perform EVD on Z^T Z
    eigenvalues, eigenvectors = qr_sorted_eigenvectors(ZTZ, max_iterations, tolerance)
    
    # Take top-k eigenvalues and eigenvectors
    top_eigenvalues = eigenvalues[:k]
    top_eigenvectors = eigenvectors[:, :k]
    
    # Reconstruct eigenvectors of W
    Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(top_eigenvalues))
    U_k = Z @ top_eigenvectors @ Lambda_inv_sqrt
    
    return U_k