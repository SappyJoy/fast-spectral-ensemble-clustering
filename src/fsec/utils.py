import numpy as np
from sklearn import datasets
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_circles, make_moons, make_blobs

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy using the Hungarian algorithm.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) / y_pred.size

def generate_2d_datasets(n_samples=3000, random_state=170):
    noisy_circles = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state
    )
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
    blobs = datasets.make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=random_state)
    rng = np.random.RandomState(random_state)
    no_structure = rng.rand(n_samples, 2), None
    
    # Anisotropicly distributed data
    X, y = datasets.make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    
    # Blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )
    
    datasets_list = [
        ("Noisy Circles", noisy_circles, {"n_clusters": 2}),
        ("Noisy Moons", noisy_moons, {"n_clusters": 2}),
        ("Varied Blobs", varied, {"n_clusters": 3}),
        ("Anisotropic Blobs", aniso, {"n_clusters": 3}),
        ("Blobs", blobs, {"n_clusters": 3}),
        ("No Structure", no_structure, {"n_clusters": 5}),  # Assuming 5 clusters for illustration
    ]
    
    return datasets_list
