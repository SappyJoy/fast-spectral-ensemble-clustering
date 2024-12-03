import os
import random
import traceback
import warnings

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def benchmark_clustering_algorithm(trial, dataset_name, algorithm_class, params=None, hyperparameter_tuning=True, study=None):
    """
    Perform clustering, compute metrics, and log results to MLflow.
    """
    try:
        # Suppress specific warnings within the objective
        warnings.filterwarnings("ignore", category=UserWarning, message="Choices for a categorical distribution should be a tuple")
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Set seed for reproducibility
        if trial:
            trial_number = trial.number
            set_seed(trial_number + 42)  # Unique seed per trial
        else:
            trial_number = 'final'
            set_seed(999)  # Fixed seed for final run

        # Load dataset
        X, y, final_n_clusters = get_dataset(dataset_name)

        # Determine Experiment Name
        if hyperparameter_tuning and trial:
            experiment_name = f'Tuning_{algorithm_class.__name__}_{dataset_name}'
            run_name = f'{dataset_name}_{algorithm_class.__name__}_Trial_{trial_number}'
        else:
            experiment_name = f'Final_{algorithm_class.__name__}_{dataset_name}'
            run_name = f'{dataset_name}_{algorithm_class.__name__}_Final_Run'

        # Initialize MLflow Logger
        logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri='http://localhost:5000'  # Replace with your MLflow server URI if different
        )

        # Initialize algorithm instance
        if algorithm_class == FSECClustering:
            algorithm_instance = FSECClustering(logger, trial, dataset_name, params)
        else:
            algorithm_instance = algorithm_class(params=params)

        # Log hyperparameters if in tuning mode
        # if hyperparameter_tuning and trial:
        logger.log_hyperparams(algorithm_instance.get_params())

        # Fit the model and predict labels
        predicted_labels = algorithm_instance.fit_predict(X)

        # Compute metrics
        nmi = normalized_mutual_info_score(y, predicted_labels)
        ari = adjusted_rand_score(y, predicted_labels)
        acc = clustering_accuracy(y, predicted_labels)

        # Log metrics
        logger.log_metrics({"NMI": nmi, "ARI": ari, "ACC": acc})

        # Print run completion message
        print(f"Run '{run_name}' completed for dataset: {dataset_name} using {algorithm_instance.__class__.__name__}")
        print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}")

        # Generate and log Optuna plots if in tuning mode
        if hyperparameter_tuning and study:
            save_optuna_plots(study, dataset_name, logger)
        elif study:
            save_optuna_plots_final(study, dataset_name, logger)

        # Report intermediate objective value to Optuna if in tuning mode
        if hyperparameter_tuning and trial:
            trial.report(nmi, step=0)
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()

        return nmi  # Optuna will try to maximize NMI

    except Exception as e:
        # Log the exception details using MLflowLogger
        error_message = f"Run '{run_name if 'run_name' in locals() else 'Unknown'}' failed for dataset {dataset_name} using {algorithm_class.__name__} with error: {e}"
        traceback_str = traceback.format_exc()

        try:
            # Attempt to log error within the current MLflow run
            logger.log_param("error", error_message)
            logger.log_text(traceback_str, "error_traceback.txt")
        except Exception as log_err:
            # If logging fails, print the error
            print(f"Failed to log error to MLflow: {log_err}")

        print(error_message)
        print(traceback_str)

        # Re-raise the exception to notify failure
        raise
    finally:
        if logger is not None:
            logger.finalize()


def save_optuna_plots(study, dataset_name, logger):
    """Generate and log Optuna optimization plots as MLflow artifacts."""
    try:
        if study is not None:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed_trials) < 2:
                print(f"Not enough trials to generate parameter importances plot for dataset {dataset_name}. Skipping plot generation.")
                return

            # Generate Optimization History Plot
            fig1 = vis.plot_optimization_history(study)
            fig1.write_image("optimization_history.png")

            # Generate Parameter Importances Plot
            fig2 = vis.plot_param_importances(study)
            fig2.write_image("param_importances.png")

            # Contour Plot (optional)
            fig3 = vis.plot_contour(study)
            fig3.write_image("contour_plot.png")

            # Parallel Coordinate Plot (optional)
            fig4 = vis.plot_parallel_coordinate(study)
            fig4.write_image("parallel_coordinate_plot.png")

            # Log plots as artifacts within the current MLflow run
            logger.experiment.log_artifact(logger.run_id, "optimization_history.png")
            logger.experiment.log_artifact(logger.run_id, "param_importances.png")
            logger.experiment.log_artifact(logger.run_id, "contour_plot.png")
            logger.experiment.log_artifact(logger.run_id, "parallel_coordinate_plot.png")

            # Remove local plot files to keep the workspace clean
            os.remove("optimization_history.png")
            os.remove("param_importances.png")
            os.remove("contour_plot.png")
            os.remove("parallel_coordinate_plot.png")

            print(f"Optuna plots for {dataset_name} have been logged to MLflow.")
        else:
            print("Study object is None. Skipping plot generation.")

    except Exception as e:
        # Handle any exceptions during plot generation or logging
        error_message = f"Failed to generate or log Optuna plots for dataset {dataset_name} with error: {e}"
        traceback_str = traceback.format_exc()
        mlflow.log_param("plot_error", error_message)
        mlflow.log_text(traceback_str, "plot_error_traceback.txt")

        print(error_message)
        print(traceback_str)


def save_optuna_plots_final(study, dataset_name, logger):
    """
    Generate Optuna optimization plots and log them as MLflow artifacts in the Final experiment.
    """
    try:
        if study is not None:
            # Generate Optimization History Plot
            fig1 = vis.plot_optimization_history(study)
            fig1.write_image("optimization_history_final.png")
        
            # Generate Parameter Importances Plot
            fig2 = vis.plot_param_importances(study)
            fig2.write_image("param_importances_final.png")
        
            # Contour Plot (optional)
            fig3 = vis.plot_contour(study)
            fig3.write_image("contour_plot_final.png")
        
            # Parallel Coordinate Plot (optional)
            fig4 = vis.plot_parallel_coordinate(study)
            fig4.write_image("parallel_coordinate_plot_final.png")
        
            # Log plots as artifacts
            logger.experiment.log_artifact(logger.run_id, "optimization_history_final.png")
            logger.experiment.log_artifact(logger.run_id, "param_importances_final.png")
            logger.experiment.log_artifact(logger.run_id, "contour_plot_final.png")
            logger.experiment.log_artifact(logger.run_id, "parallel_coordinate_plot_final.png")
        
            # Remove local plot files to keep the workspace clean
            os.remove("optimization_history_final.png")
            os.remove("param_importances_final.png")
            os.remove("contour_plot_final.png")
            os.remove("parallel_coordinate_plot_final.png")
        
            print(f"Optuna plots for {dataset_name} have been logged to MLflow Final Experiment.")
        else:
            print("Study object is None. Skipping plot generation.")

    except Exception as e:
        # Handle any exceptions during plot generation or logging
        error_message = f"Failed to generate or log Optuna plots for dataset {dataset_name} with error: {e}"
        traceback_str = traceback.format_exc()
        mlflow.log_param("plot_error", error_message)
        mlflow.log_text(traceback_str, "plot_error_traceback_final.txt")

        print(error_message)
        print(traceback_str)

