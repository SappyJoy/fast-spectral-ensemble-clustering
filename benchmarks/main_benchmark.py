import json
import random

import lightning.pytorch as lp
import numpy as np
import optuna
import torch
from benchmark import benchmark_clustering_algorithm, save_optuna_plots_final

from clustering_algorithms.agglomerative_clustering import \
    AgglomerativeClusteringClustering
from clustering_algorithms.dbscan_clustering import DBSCANClustering
from clustering_algorithms.fsec_clustering import FSECClustering
from clustering_algorithms.gmm_clustering import GMMClustering
from clustering_algorithms.kmeans_clustering import KMeansClustering
from clustering_algorithms.spectral_clustering import \
    SpectralClusteringClustering
from data.loaders import get_dataset


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    lp.seed_everything(seed)


def optimize_and_benchmark(dataset_name, algorithm_class, n_trials=50):
    """
    Optimize hyperparameters for a given algorithm and dataset, then benchmark it.
    """
    # Define the study
    study = optuna.create_study(direction='maximize', study_name=f"{algorithm_class.__name__}_{dataset_name}_study")
    X, _, final_n_clusters = get_dataset(dataset_name)
    num_samples = X.shape[0]

    # Define the objective function for Optuna
    def objective(trial):
        # Initialize algorithm with current trial's parameters
        params = {}
        if algorithm_class == KMeansClustering:
            params['n_clusters'] = final_n_clusters
            params['init'] = trial.suggest_categorical('init', ['k-means++', 'random'])
            params['n_init'] = trial.suggest_int('n_init', 10, 50)
        elif algorithm_class == SpectralClusteringClustering:
            params['n_clusters'] = final_n_clusters
            # params['affinity'] = trial.suggest_categorical('affinity', ['nearest_neighbors', 'rbf', 'precomputed'])
            params['affinity'] = trial.suggest_categorical('affinity', ['nearest_neighbors'])
            params['gamma'] = trial.suggest_float('gamma', 0.01, 10.0)
        elif algorithm_class == AgglomerativeClusteringClustering:
            params['n_clusters'] = final_n_clusters
            params['linkage'] = trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'])
        elif algorithm_class == GMMClustering:
            params['n_components'] = final_n_clusters
            params['covariance_type'] = trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical'])
            params['init_params'] = trial.suggest_categorical('init_params', ['kmeans', 'random'])
        elif algorithm_class == DBSCANClustering:
            params['eps'] = trial.suggest_float('eps', 0.1, 10.0)
            params['min_samples'] = trial.suggest_int('min_samples', 1, 20)
            params['metric'] = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine'])
        elif algorithm_class == FSECClustering:
            # FSEC-specific hyperparameters
            n_components = trial.suggest_int('n_components', 2, 100)
            num_anchors = trial.suggest_int('num_anchors', n_components + 1, min(n_components + 2, num_samples))
            K = trial.suggest_int('K', 1, n_components)  # Ensure K <= n_components
            K_prime = trial.suggest_int('K_prime', K + 1, min(num_samples, 100 * K))  # Ensure K_prime > K

            num_clusters_list_size = trial.suggest_int('num_clusters_list_size', 1, 10)

            # Define the range around final_n_clusters
            cluster_min = max(2, int(final_n_clusters * 0.75))  # Ensure minimum value is 2
            cluster_max = int(final_n_clusters * 1.25)

            # Initialize the list
            num_clusters_list = []

            # Generate each tuple based on the size
            for i in range(num_clusters_list_size):
                # Suggest cluster numbers around final_n_clusters
                cluster_1 = trial.suggest_int(f'num_clusters_list_{i}', cluster_min, cluster_max)
                
                # Create a tuple and append to the list
                num_clusters_list.append(cluster_1)

            params = {
                'num_anchors': num_anchors,
                'K_prime': K_prime,
                'K': K,
                'n_components': n_components,
                'num_clusters_list': tuple(num_clusters_list),
                'final_n_clusters': final_n_clusters,
                'n_jobs': -1  # Fixed as per your configuration
            }
        else:
            raise ValueError(f"Unsupported algorithm class: {algorithm_class}")

        # Perform benchmarking
        nmi = benchmark_clustering_algorithm(trial, dataset_name, algorithm_class, params=params, study=study)

        return nmi

    study.optimize(objective, n_trials=n_trials)

    print(f"\nBest trial for {dataset_name} using {algorithm_class.__name__}:")
    trial = study.best_trial
    print(f"  NMI: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Define fixed parameters based on the algorithm
    fixed_params = {}
    if algorithm_class in [KMeansClustering, SpectralClusteringClustering, AgglomerativeClusteringClustering]:
        fixed_params['n_clusters'] = final_n_clusters
    elif algorithm_class == GMMClustering:
        fixed_params['n_components'] = final_n_clusters
    elif algorithm_class == FSECClustering:
        # Handle FSEC-specific fixed parameters
        num_clusters_list_size = trial.params.get('num_clusters_list_size', 1)
        num_clusters_list = []
        for i in range(num_clusters_list_size):
            cluster_key = f'num_clusters_list_{i}'
            cluster_value = trial.params.get(cluster_key, 2)
            num_clusters_list.append(cluster_value)
        fixed_params.update({
            'num_clusters_list': tuple(num_clusters_list),
            'final_n_clusters': final_n_clusters,
            'n_jobs': -1
        })

    # Run final clustering with best parameters
    best_params = {**fixed_params, **trial.params}

    if algorithm_class == FSECClustering:
        # For FSEC, reconstruct num_clusters_list from trial params
        num_clusters_list_size = best_params.get('num_clusters_list_size', 1)
        num_clusters_list = []
        for i in range(num_clusters_list_size):
            cluster_key = f'num_clusters_list_{i}'
            cluster_value = best_params.get(cluster_key, 2)
            num_clusters_list.append(cluster_value)
        best_params['num_clusters_list'] = tuple(num_clusters_list)

        fsec_expected_params = {
            'num_anchors',
            'K_prime',
            'K',
            'n_components',
            'final_n_clusters',
            'num_clusters_list',
            'n_jobs'
        }

        # Create a new params dictionary containing only expected parameters
        filtered_params = {k: v for k, v in best_params.items() if k in fsec_expected_params}
        filtered_params['num_clusters_list'] = num_clusters_list  # Add the reconstructed list
        filtered_params['final_n_clusters'] = final_n_clusters
        best_params = filtered_params

    # Benchmark the final run
    final_metrics = benchmark_clustering_algorithm(trial=trial, dataset_name=dataset_name, algorithm_class=algorithm_class, params=best_params, hyperparameter_tuning=False, study=study)
    result = {
        'algorithm': algorithm_class.__name__,
        'dataset': dataset_name,
        'best_params': best_params,
        'nmi': final_metrics
    }

    return result

def main():
    # Set global seed for overall reproducibility
    set_seed(123)

    datasets = [
        'PenDigits',
        'Letters',
        'USPS',
        'FashionMNIST',
        'CIFAR10',
        # Add more datasets as needed
    ]

    algorithms = [
        KMeansClustering,
        SpectralClusteringClustering,
        AgglomerativeClusteringClustering,
        GMMClustering,
        DBSCANClustering,
        FSECClustering
    ]

    n_trials = 50  # Number of Optuna trials per algorithm per dataset
    results = []

    for dataset in datasets:
        for algorithm_class in algorithms:
            print(f"\n=== Optimizing and Benchmarking {algorithm_class.__name__} on dataset: {dataset} ===\n")
            result = optimize_and_benchmark(dataset, algorithm_class, n_trials=n_trials)
            results.append(result)

    # Save the results to a JSON file
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\n=== Benchmarking Completed ===\n")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()

