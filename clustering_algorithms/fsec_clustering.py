import optuna
from lightning.pytorch import Trainer
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from fsec.clustering import FSEC
from fsec.fsec_datamodule import FSECDataModule
from fsec.utils import clustering_accuracy

from .base_clustering import BaseClustering  # Ensure correct import path


class FSECClustering(BaseClustering):
    def __init__(
        self,
        logger,
        trial,
        dataset_name,
        encoding_dim=64,
        hidden_dims=[128, 64],
        autoencoder_max_epochs=50,
        params=None
    ):
        """
        Initializes the FSECClustering class with the necessary parameters.

        Parameters:
        - logger: Logger instance for logging metrics.
        - trial: Optuna trial for hyperparameter optimization.
        - dataset_name: Name of the dataset to be used.
        - encoding_dim: Dimensionality of the encoded space.
        - hidden_dims: List of hidden layer dimensions for the Autoencoder.
        - autoencoder_max_epochs: Maximum number of epochs to train the Autoencoder.
        - params: Dictionary of additional parameters for FSEC.
        """
        super().__init__(params)
        self.logger = logger
        self.trial = trial
        self.dataset_name = dataset_name
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.autoencoder_max_epochs = autoencoder_max_epochs
        self.params = params if params is not None else {}
        self.labels_pred = None

    def fit_predict(self, X=None, y=None):
        """
        Fit the Autoencoder, perform dimensionality reduction, then fit FSEC and predict cluster labels.

        Parameters:
        - X: Feature matrix (optional if using dataset_name).
        - y: True labels (optional).

        Returns:
        - labels_pred: Predicted cluster labels.
        """
        # Initialize DataModule
        datamodule = FSECDataModule(
            dataset_name=self.dataset_name,
            batch_size=256,
            encoding_dim=self.encoding_dim,
            hidden_dims=self.hidden_dims,
            max_epochs=self.autoencoder_max_epochs
        )
        datamodule.setup()

        # Train Autoencoder
        print("Training Autoencoder...")
        datamodule.train_autoencoder()
        print("Autoencoder training completed.")

        # Get encoded data
        encoded_X, y = datamodule.get_encoded_data()

        # Initialize FSEC
        fsec = FSEC(
            num_anchors=self.params.get('num_anchors', 50),
            K_prime=self.params.get('K_prime', 50),
            K=self.params.get('K', 5),
            n_components=self.params.get('n_components', 2),
            num_clusters_list=self.params.get('num_clusters_list', [10, 15]),
            final_n_clusters=self.params.get('final_n_clusters', 10),
            n_jobs=self.params.get('n_jobs', -1)
        )

        # Fit FSEC
        print("Performing FSEC clustering...")
        fsec.fit(encoded_X)
        labels = fsec.labels_
        print("Clustering completed.")

        # Optionally, compute metrics
        if y is not None:
            nmi = normalized_mutual_info_score(y, labels)
            ari = adjusted_rand_score(y, labels)
            acc = clustering_accuracy(y, labels)
            print(f"NMI: {nmi:.4f}")
            print(f"ARI: {ari:.4f}")
            print(f"Accuracy: {acc:.4f}")

            # Log metrics if logger is provided
            if self.logger is not None:
                self.logger.log_metrics({'NMI': nmi, 'ARI': ari, 'ACC': acc})

            # If using Optuna trial, report metrics for pruning
            if self.trial is not None:
                self.trial.report(nmi, step=0)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        self.labels_pred = labels
        return self.labels_pred

