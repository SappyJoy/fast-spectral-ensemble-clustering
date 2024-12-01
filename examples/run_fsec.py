from data.loaders import get_dataset
from fsec.clustering import FSEC
from fsec.utils import clustering_accuracy
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def run_fsec_on_dataset(dataset_name, params):
    X, y = get_dataset(dataset_name)
    fsec = FSEC(**params)
    predicted_labels = fsec.fit_predict(X)
    
    nmi = normalized_mutual_info_score(y, predicted_labels)
    ari = adjusted_rand_score(y, predicted_labels)
    acc = clustering_accuracy(y, predicted_labels)
    
    print(f"Dataset: {dataset_name}")
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}")
    
    return {
        'dataset': dataset_name,
        'nmi': nmi,
        'ari': ari,
        'acc': acc
    }

if __name__ == "__main__":
    datasets = [
        # 'Covertype', # TODO: SVD error
        'PenDigits',
        'Letters',
        'MNIST',
        'USPS',
        'FashionMNIST',
        'CIFAR10',
        # 'KannadaMNIST' # TODO: Need to implement loader
    ]
    
    params = {
        'num_anchors': 50,
        'K_prime': 50,  # Например, 10*K=50
        'K': 5,
        'n_components': 2,
        'num_clusters_list': [2, 3, 4, 5],
        'final_n_clusters': 10,  # В зависимости от датасета
        'n_jobs': -1
    }
    
    results = {}
    for dataset in datasets:
        results[dataset] = run_fsec_on_dataset(dataset, params)

