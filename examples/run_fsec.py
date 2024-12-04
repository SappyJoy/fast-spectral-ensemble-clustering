import json
import os
import random
import traceback
import warnings

import click
import lightning.pytorch as lp
import mlflow
import numpy as np
import optuna
import optuna.visualization as vis
import torch
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from clustering_algorithms.fsec_clustering import FSECClustering
from data.loaders import get_dataset
from src.fsec.utils import clustering_accuracy


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    lp.seed_everything(seed)


def objective(trial, dataset_name, anchor_method):
    # Suppress specific warnings within the objective
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Choices for a categorical distribution should be a tuple",
    )
    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # From sklearn if needed
    warnings.filterwarnings("ignore", category=FutureWarning)  # Other future warnings

    # Set seed for reproducibility
    set_seed(trial.number + 42)  # Unique seed per trial

    # Load dataset to determine num_samples
    X, y, final_n_clusters = get_dataset(dataset_name)
    num_samples = X.shape[0]

    # Suggest hyperparameters with dynamic constraints
    # First, suggest num_anchors
    num_anchors = trial.suggest_int(
        "num_anchors", 2, min(num_samples, max(50, np.sqrt(num_samples)))
    )

    # Then, suggest n_components based on num_anchors and X.shape[1]
    n_components = trial.suggest_int(
        "n_components", 1, min(X.shape[1], num_anchors - 1)
    )

    if n_components >= num_anchors or n_components > X.shape[1]:
        raise optuna.exceptions.TrialPruned()

    K = trial.suggest_int("K", 1, n_components)  # Ensure K <= n_components
    K_prime = trial.suggest_int(
        "K_prime", K + 1, min(num_samples, 100 * K)
    )  # Ensure K_prime > K

    # Suggest the size of num_clusters_list
    num_clusters_list_size = trial.suggest_int("num_clusters_list_size", 1, 10)

    # Define the range around final_n_clusters
    cluster_min = max(2, int(final_n_clusters * 0.75))  # Ensure minimum value is 2
    cluster_max = max(cluster_min + 1, int(final_n_clusters * 1.25))

    # Initialize the list
    num_clusters_list = []

    # Generate each cluster number based on the size
    for i in range(num_clusters_list_size):
        # Suggest cluster numbers around final_n_clusters
        cluster_1 = trial.suggest_int(
            f"num_clusters_list_{i}", cluster_min, cluster_max
        )

        # Append to the list
        num_clusters_list.append(cluster_1)

    params = {
        "num_anchors": num_anchors,
        "K_prime": K_prime,
        "K": K,
        "n_components": n_components,
        "num_clusters_list": tuple(num_clusters_list),
        "final_n_clusters": final_n_clusters,
        "anchor_method": anchor_method,  # Include anchor_method in parameters
        "n_jobs": -1,  # Fixed as per your configuration
    }

    if anchor_method == "DBSCAN":
        dbscan_eps = trial.suggest_float("dbscan_eps", 0.1, 2.0, log=True)
        dbscan_min_samples = trial.suggest_int("dbscan_min_samples", 3, 20)
        params["dbscan_eps"] = dbscan_eps
        params["dbscan_min_samples"] = dbscan_min_samples

    # Initialize MLflow Logger for Tuning
    logger = MLFlowLogger(
        experiment_name=f"FSEC_Optuna_{dataset_name}_{anchor_method}",
        run_name=f"{dataset_name}_trial_{anchor_method}_{trial.number}",
        tracking_uri="http://localhost:5000",  # Replace with your MLflow server URI if different
    )

    # Initialize FSECClustering
    algorithm_instance = FSECClustering(logger, trial, dataset_name, params)

    # Log hyperparameters
    logger.log_hyperparams(algorithm_instance.get_params())

    # Retrieve cluster labels
    predicted_labels = algorithm_instance.fit_predict(X)

    # Compute metrics
    nmi = normalized_mutual_info_score(y, predicted_labels)
    ari = adjusted_rand_score(y, predicted_labels)
    acc = clustering_accuracy(y, predicted_labels)

    # Log metrics
    logger.log_metrics({"NMI": nmi, "ARI": ari, "ACC": acc})

    print(
        f"Trial {trial.number} completed for dataset: {dataset_name} with {anchor_method} anchoring."
    )
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}")

    # Report intermediate objective value to Optuna
    trial.report(nmi, step=0)

    # Handle pruning based on intermediate value
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return nmi  # Optuna will try to maximize NMI


def save_optuna_plots_final(study, dataset_name, anchor_method, logger):
    """
    Generate Optuna optimization plots and log them as MLflow artifacts in the FSEC_Final_{Dataset} experiment.
    """
    try:
        # Optimization History Plot
        fig1 = vis.plot_optimization_history(study)
        fig1.write_image("optimization_history_final.png")

        # Parameter Importances Plot
        try:
            fig2 = vis.plot_param_importances(study)
            fig2.write_image("param_importances.png")
        except RuntimeError as re:
            # Handle specific Optuna errors
            print(
                f"RuntimeError while generating parameter importances: {re}"
            )
            # Skip the rest of the plotting
            fig2 = None

        # Optionally, add more plots if desired
        # Contour Plot
        fig3 = vis.plot_contour(study)
        fig3.write_image("contour_plot_final.png")

        # Parallel Coordinate Plot
        fig4 = vis.plot_parallel_coordinate(study)
        fig4.write_image("parallel_coordinate_plot_final.png")

        # Log plots as artifacts
        logger.experiment.log_artifact(
            logger.run_id, "optimization_history_final.png"
        )
        if fig2:
            logger.experiment.log_artifact(logger.run_id, "param_importances.png")
        logger.experiment.log_artifact(
            logger.run_id, "contour_plot_final.png"
        )
        logger.experiment.log_artifact(
            logger.run_id, "parallel_coordinate_plot_final.png"
        )

        # Remove local plot files to keep the workspace clean
        os.remove("optimization_history_final.png")
        if fig2:
            os.remove("param_importances.png")
        os.remove("contour_plot_final.png")
        os.remove("parallel_coordinate_plot_final.png")

        print(
            f"Optuna plots for {dataset_name} with {anchor_method} anchoring have been logged to MLflow Final Experiment."
        )

    except Exception as e:
        # Handle any exceptions during plot generation or logging
        error_message = f"Failed to generate or log Optuna plots for dataset {dataset_name} with {anchor_method} anchoring. Error: {e}"
        traceback_str = traceback.format_exc()

        print(error_message)
        print(traceback_str)


def optimize_hyperparameters(dataset_name, n_trials, anchor_method):
    # Define the study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"FSEC_{dataset_name}_{anchor_method}_study",
    )

    # Optimize the objective function
    study.optimize(
        lambda trial: objective(trial, dataset_name, anchor_method),
        n_trials=n_trials,
    )

    print(f"\nBest trial for {dataset_name} with {anchor_method} anchoring:")
    trial = study.best_trial

    print(f"  NMI: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study.best_trial.params, trial.value, study


def run_fsec_on_dataset(
    dataset_name, params, nmi_score, study, anchor_method
):
    # Set seed for reproducibility
    set_seed(study.best_trial.number + 42)  # Fixed seed for final run

    # Load data
    X, y, final_n_clusters = get_dataset(dataset_name)

    # Initialize MLflow Logger for Final Run
    logger = MLFlowLogger(
        experiment_name=f"FSEC_Final_{dataset_name}_{anchor_method}",
        run_name=f"{dataset_name}_final_{anchor_method}",
        tracking_uri="http://localhost:5000",  # Replace with your MLflow server URI if different
    )

    # Ensure 'final_n_clusters' is correctly set
    params["final_n_clusters"] = final_n_clusters

    # Initialize FSECClustering with the best parameters
    algorithm_instance = FSECClustering(logger, None, dataset_name, params)

    # Log parameters
    logger.log_hyperparams(algorithm_instance.get_params())

    # Retrieve cluster labels
    predicted_labels = algorithm_instance.fit_predict(X)

    # Compute metrics
    nmi = normalized_mutual_info_score(y, predicted_labels)
    ari = adjusted_rand_score(y, predicted_labels)
    acc = clustering_accuracy(y, predicted_labels)

    # Log metrics
    logger.log_metrics({"NMI": nmi, "ARI": ari, "ACC": acc})

    print(
        f"Final run for dataset: {dataset_name} with {anchor_method} anchoring"
    )
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}")

    save_optuna_plots_final(study, dataset_name, anchor_method, logger)

    return {
        "dataset": dataset_name,
        "anchor_method": anchor_method,
        "nmi": nmi,
        "ari": ari,
        "acc": acc,
    }


def save_study_summary(study, dataset_name, anchor_method):
    """Save the study summary as a JSON artifact."""
    summary = {
        "best_trial": {
            "value": study.best_trial.value,
            "params": study.best_trial.params,
        },
        "trials": [],
    }

    for trial in study.trials:
        summary["trials"].append(
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            }
        )

    summary_filename = f"study_summary_{dataset_name}_{anchor_method}.json"
    with open(summary_filename, "w") as f:
        json.dump(summary, f, indent=4)

    # Log the summary to MLflow
    mlflow.log_artifact(summary_filename)

    # Remove the local file
    os.remove(summary_filename)


@click.command()
@click.option(
    "--datasets",
    multiple=True,
    default=[
        "PenDigits",
        "Letters",
        "MNIST",
        "USPS",
        "FashionMNIST",
        "CIFAR10",
    ],
    help="List of dataset names to run FSEC on.",
)
@click.option(
    "--n-trials",
    default=50,
    show_default=True,
    help="Number of Optuna trials per dataset and anchor method.",
)
@click.option(
    "--anchor-methods",
    multiple=True,
    type=click.Choice(["BKHK", "DBSCAN"], case_sensitive=True),
    default=["BKHK", "DBSCAN"],
    help="Anchor selection methods to compare.",
)
def main(datasets, n_trials, anchor_methods):
    """
    Optimize and run FSEC clustering with different anchor selection methods.
    Compare BKHK and DBSCAN anchor selection.
    """
    # Set global seed for overall reproducibility
    set_seed(123)

    results = {}

    for dataset in datasets:
        for anchor_method in anchor_methods:
            print(
                f"\n=== Optimizing hyperparameters for dataset: {dataset} using {anchor_method.upper()} anchoring ===\n"
            )
            try:
                best_params, best_nmi, study = optimize_hyperparameters(
                    dataset, n_trials, anchor_method
                )
                print(
                    f"\nBest parameters for {dataset} with {anchor_method.upper()} anchoring: {best_params} with NMI: {best_nmi:.4f}\n"
                )

                # Reconstruct num_clusters_list from individual entries
                num_clusters_list_size = best_params.get(
                    "num_clusters_list_size", 1
                )  # Default to 1 if not found
                num_clusters_list = []
                for i in range(num_clusters_list_size):
                    cluster_key = f"num_clusters_list_{i}"
                    cluster_value = best_params.get(cluster_key, 2)  # Default to 2 if not found
                    num_clusters_list.append(cluster_value)

                num_clusters_list = tuple(
                    num_clusters_list
                )  # Convert to tuple if required by FSEC

                # Filter out tuning hyperparameters before passing to FSEC
                fsec_expected_params = {
                    "num_anchors",
                    "K_prime",
                    "K",
                    "n_components",
                    "final_n_clusters",
                    "num_clusters_list",
                    "anchor_method",
                    "n_jobs",
                }

                # Create a new params dictionary containing only expected parameters
                filtered_params = {
                    k: v for k, v in best_params.items() if k in fsec_expected_params
                }
                filtered_params["num_clusters_list"] = (
                    num_clusters_list
                )  # Add the reconstructed list

                print(
                    f"=== Running FSEC with best parameters on dataset: {dataset} using {anchor_method.upper()} anchoring ===\n"
                )
                final_metrics = run_fsec_on_dataset(
                    dataset, filtered_params, best_nmi, study, anchor_method
                )
                results[f"{dataset}_{anchor_method}"] = final_metrics

                # Optionally, save the study summary
                save_study_summary(study, dataset, anchor_method)

            except Exception as e:
                error_message = f"An error occurred while processing dataset {dataset} with {anchor_method.upper()} anchoring. Error: {e}"
                traceback_str = traceback.format_exc()
                print(error_message)
                print(traceback_str)
                results[f"{dataset}_{anchor_method}"] = {
                    "error": error_message,
                    "traceback": traceback_str,
                }

    # Optionally, save the results to a file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n=== Hyperparameter Optimization and Final Runs Completed ===\n")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()

