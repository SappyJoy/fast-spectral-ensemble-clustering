# examples/run_fsec.py
import os
import traceback
import lightning.pytorch as lp
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from src.fsec.fsec_module import FSECModule
from src.fsec.fsec_datamodule import FSECDataModule
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.fsec.utils import clustering_accuracy
import mlflow
import torch
from data.loaders import get_dataset
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.exceptions import ConvergenceWarning
import json
import random
import numpy as np
import warnings
import optuna.visualization as vis
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    lp.seed_everything(seed)

def objective(trial, dataset_name):
    # Suppress specific warnings within the objective
    warnings.filterwarnings("ignore", category=UserWarning, message="Choices for a categorical distribution should be a tuple")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # From sklearn if needed
    warnings.filterwarnings("ignore", category=FutureWarning)  # Other future warnings


    # Set seed for reproducibility
    set_seed(trial.number + 42)  # Unique seed per trial

    # Load dataset to determine num_samples
    X, y = get_dataset(dataset_name)
    num_samples = X.shape[0]

    # Suggest hyperparameters with dynamic constraints
    # 'n_components' is suggested first to set the lower bound for 'num_anchors'
    n_components = trial.suggest_int('n_components', 2, 10)
    num_anchors = trial.suggest_int('num_anchors', n_components + 1, min(100, num_samples))
    K = trial.suggest_int('K', 3, 10)
    K_prime = trial.suggest_int('K_prime', K + 1, min(100, num_samples))  # Ensure K_prime > K
    num_clusters_list = trial.suggest_categorical('num_clusters_list', [(2,3), (3,4), (4,5)])  # Use tuples
    final_n_clusters = trial.suggest_int('final_n_clusters', 2, 10)


    params = {
        'num_anchors': num_anchors,
        'K_prime': K_prime,
        'K': K,
        'n_components': n_components,
        'num_clusters_list': num_clusters_list,
        'final_n_clusters': final_n_clusters,
        'n_jobs': -1  # Fixed as per your configuration
    }

    # Initialize DataModule
    datamodule = FSECDataModule(dataset_name, batch_size=len(X))
    datamodule.setup()

    # Initialize MLflow Logger for Tuning
    logger = MLFlowLogger(
        experiment_name=f'FSEC_Optuna_{dataset_name}',
        run_name=f"{dataset_name}_trial_{trial.number}",
        tracking_uri='http://localhost:5000'  # Replace with your MLflow server URI if different
    )

    # Initialize FSECModule
    model = FSECModule(params)

    # Initialize Trainer with Optuna Pruning Callback
    trainer = Trainer(
        logger=logger,
        max_epochs=1,  # Only one epoch since clustering is a one-time process
        enable_checkpointing=False,
        accelerator='auto',  # Utilize GPU if available
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="NMI")],
        log_every_n_steps=1,  # Adjust logging frequency
        deterministic=True,  # Ensure deterministic behavior
        enable_progress_bar=False,
        enable_model_summary=False
    )

    # Log hyperparameters
    logger.log_hyperparams(params)

    # Fit the model (perform clustering)
    trainer.fit(model, datamodule=datamodule)

    # Retrieve cluster labels
    predicted_labels = model.labels_pred

    # Compute metrics
    nmi = normalized_mutual_info_score(y, predicted_labels)
    ari = adjusted_rand_score(y, predicted_labels)
    acc = clustering_accuracy(y, predicted_labels)

    # Log metrics
    logger.log_metrics({"NMI": nmi, "ARI": ari, "ACC": acc})

    # Optionally, log the model
    # mlflow.sklearn.log_model(model.fsec, "fsec_model")

    print(f"Trial {trial.number} completed for dataset: {dataset_name}")
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}")

    # Report intermediate objective value to Optuna
    trial.report(nmi, step=0)

    # Handle pruning based on intermediate value
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return nmi  # Optuna will try to maximize NMI

def save_optuna_plots_final(study, dataset_name, logger):
    """
    Generate Optuna optimization plots and log them as MLflow artifacts in the FSEC_Final_{Dataset} experiment.
    """
    try:
        # Optimization History Plot
        fig1 = vis.plot_optimization_history(study)
        fig1.write_image("optimization_history_final.png")
        
        # Parameter Importances Plot
        fig2 = vis.plot_param_importances(study)
        fig2.write_image("param_importances_final.png")
        
        # Optionally, add more plots if desired
        # Contour Plot
        fig3 = vis.plot_contour(study)
        fig3.write_image("contour_plot_final.png")
        
        # Parallel Coordinate Plot
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
    
    except Exception as e:
        # Handle any exceptions during plot generation or logging
        error_message = f"Failed to generate or log Optuna plots for dataset {dataset_name} with error: {e}"
        traceback_str = traceback.format_exc()
        mlflow.log_param("plot_error", error_message)
        mlflow.log_text(traceback_str, "plot_error_traceback.txt")
        
        print(error_message)
        print(traceback_str)

def optimize_hyperparameters(dataset_name, n_trials=50):
    # Define the study
    study = optuna.create_study(direction='maximize', study_name=f"FSEC_{dataset_name}_study")

    # Optimize the objective function
    study.optimize(lambda trial: objective(trial, dataset_name), n_trials=n_trials)

    print(f"\nBest trial for {dataset_name}:")
    trial = study.best_trial

    print(f"  NMI: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study.best_trial.params, trial.value, study

def run_fsec_on_dataset(dataset_name, params, nmi_score, study):
    # Set seed for reproducibility
    set_seed(999)  # Fixed seed for final run

    # Initialize DataModule
    datamodule = FSECDataModule(dataset_name, batch_size=len(get_dataset(dataset_name)[0]))
    datamodule.setup()

    # Initialize MLflow Logger for Final Run
    logger = MLFlowLogger(
        experiment_name=f'FSEC_Final_{dataset_name}',
        run_name=f"{dataset_name}_final",
        tracking_uri='http://localhost:5000'  # Replace with your MLflow server URI if different
    )

    # Initialize FSECModule with the best parameters
    model = FSECModule(params)

    # Initialize Trainer
    trainer = Trainer(
        logger=logger,
        max_epochs=1,
        enable_checkpointing=False,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=1,  # Adjust logging frequency
        deterministic=True,  # Ensure deterministic behavior
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    # Log parameters
    logger.log_hyperparams(params)

    # Fit the model (perform clustering)
    trainer.fit(model, datamodule=datamodule)

    # Retrieve cluster labels
    predicted_labels = model.labels_pred

    # Retrieve true labels
    _, y = get_dataset(dataset_name)

    # Compute metrics
    nmi = normalized_mutual_info_score(y, predicted_labels)
    ari = adjusted_rand_score(y, predicted_labels)
    acc = clustering_accuracy(y, predicted_labels)

    # Log metrics
    logger.log_metrics({"NMI": nmi, "ARI": ari, "ACC": acc})


    print(f"Final run for dataset: {dataset_name}")
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}")

    save_optuna_plots_final(study, dataset_name, logger)

    return {
        'dataset': dataset_name,
        'nmi': nmi,
        'ari': ari,
        'acc': acc
    }

def save_study_summary(study, dataset_name):
    """Save the study summary as a JSON artifact."""
    summary = {
        "best_trial": {
            "value": study.best_trial.value,
            "params": study.best_trial.params
        },
        "trials": []
    }

    for trial in study.trials:
        summary["trials"].append({
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": trial.state.name
        })

    with open("study_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # Log the summary to MLflow
    mlflow.log_artifact("study_summary.json")

    # Remove the local file
    import os
    os.remove("study_summary.json")

if __name__ == "__main__":
    # Set global seed for overall reproducibility
    set_seed(123)

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

    n_trials = 50  # Number of Optuna trials per dataset
    results = {}

    for dataset in datasets:
        print(f"\n=== Optimizing hyperparameters for dataset: {dataset} ===\n")
        best_params, best_nmi, study = optimize_hyperparameters(dataset, n_trials=n_trials)
        print(f"\nBest parameters for {dataset}: {best_params} with NMI: {best_nmi:.4f}\n")

        print(f"=== Running FSEC with best parameters on dataset: {dataset} ===\n")
        final_metrics = run_fsec_on_dataset(dataset, best_params, best_nmi, study)
        results[dataset] = final_metrics

    # Optionally, save the results to a file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\n=== Hyperparameter Optimization and Final Runs Completed ===\n")
    print(json.dumps(results, indent=4))

