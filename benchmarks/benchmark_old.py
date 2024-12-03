from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from fsec.clustering import FSEC
from fsec.utils import clustering_accuracy

def benchmark_algorithms(X, y, algorithms):
    results = {}
    for name, algorithm in algorithms.items():
        print(f"Running {name}...")
        labels = algorithm.fit_predict(X)
        nmi = normalized_mutual_info_score(y, labels)
        ari = adjusted_rand_score(y, labels)
        acc = clustering_accuracy(y, labels)
        results[name] = {'NMI': nmi, 'ARI': ari, 'Accuracy': acc}
        print(f"{name}: NMI={nmi:.4f}, ARI={ari:.4f}, Accuracy={acc:.4f}\n")
    return results

if __name__ == "__main__":
    datasets = {
        'Digits': load_digits(return_X_y=True),
        'Iris': load_iris(return_X_y=True),
        'Wine': load_wine(return_X_y=True)
    }

    algorithms = {
        'FSEC': FSEC(final_n_clusters=10),
        'KMeans': KMeans(n_clusters=10),
        'SpectralClustering': SpectralClustering(n_clusters=10),
        'DBSCAN': DBSCAN()
    }

    for dataset_name, (X, y) in datasets.items():
        print(f"Benchmarking on {dataset_name} dataset")
        X = StandardScaler().fit_transform(X)
        benchmark_algorithms(X, y, algorithms)

