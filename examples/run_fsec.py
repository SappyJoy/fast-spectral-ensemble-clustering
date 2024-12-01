# examples/run_fsec.py
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

def objective(trial, dataset_name):
    # Suggest hyperparameters
    params = {
        'num_anchors': trial.suggest_int('num_anchors', 30, 100),
        'K_prime': trial.suggest_int('K_prime', 30, 100),
        'K': trial.suggest_int('K', 3, 10),
        'n_components': trial.suggest_int('n_components', 2, 10),
        'num_clusters_list': trial.suggest_categorical('num_clusters_list', [[2,3], [3,4], [4,5]]),
        'final_n_clusters': trial.suggest_int('final_n_clusters', 2, 10),
        'n_jobs': -1  # Fixed as per your configuration
    }

    # Initialize DataModule
    datamodule = FSECDataModule(dataset_name, batch_size=len(get_dataset(dataset_name)[0]))
    datamodule.setup()

    # Initialize MLflow Logger
    logger = MLFlowLogger(
        experiment_name='FSEC_Optuna_Experiments',
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
    )

    # Start MLflow run for this trial
    with mlflow.start_run(run_name=f"{dataset_name}_trial"):
        # Log hyperparameters
        mlflow.log_params(params)

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
        mlflow.log_metric("NMI", nmi)
        mlflow.log_metric("ARI", ari)
        mlflow.log_metric("ACC", acc)

        # Optionally, log the model
        # mlflow.sklearn.log_model(model.fsec, "fsec_model")

        print(f"Trial completed for dataset: {dataset_name}")
        print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}")

        # Report intermediate objective value to Optuna
        trial.report(nmi, step=0)

        # Handle pruning based on intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return nmi  # Optuna will try to maximize NMI

def optimize_hyperparameters(dataset_name, n_trials=50):
    # Define the study
    study = optuna.create_study(direction='maximize', study_name=f"FSEC_{dataset_name}_study", storage=None)

    # Optimize the objective function
    study.optimize(lambda trial: objective(trial, dataset_name), n_trials=n_trials)

    print(f"Best trial for {dataset_name}:")
    trial = study.best_trial

    print(f"  NMI: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study.best_trial.params, trial.value

def run_fsec_on_dataset(dataset_name, params, nmi_score):
    # Initialize DataModule
    datamodule = FSECDataModule(dataset_name, batch_size=len(get_dataset(dataset_name)[0]))
    datamodule.setup()

    # Initialize MLflow Logger
    logger = MLFlowLogger(
        experiment_name='FSEC_Final_Experiments',
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
    )

    # Start MLflow run
    with mlflow.start_run(run_name=f"{dataset_name}_final"):
        # Log parameters
        mlflow.log_params(params)

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
        mlflow.log_metric("NMI", nmi)
        mlflow.log_metric("ARI", ari)
        mlflow.log_metric("ACC", acc)

        # Optionally, log the model
        # mlflow.sklearn.log_model(model.fsec, "fsec_model")

        print(f"Final run for dataset: {dataset_name}")
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

    n_trials = 50  # Number of Optuna trials per dataset
    results = {}

    for dataset in datasets:
        print(f"\n=== Optimizing hyperparameters for dataset: {dataset} ===\n")
        best_params, best_nmi = optimize_hyperparameters(dataset, n_trials=n_trials)
        print(f"\nBest parameters for {dataset}: {best_params} with NMI: {best_nmi:.4f}\n")

        print(f"=== Running FSEC with best parameters on dataset: {dataset} ===\n")
        final_metrics = run_fsec_on_dataset(dataset, best_params, best_nmi)
        results[dataset] = final_metrics

    # Optionally, save the results to a file
    import json
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\n=== Hyperparameter Optimization and Final Runs Completed ===\n")
    print(json.dumps(results, indent=4))

