import warnings

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import \
    CSVLogger  # Use CSVLogger to avoid MLflow conflicts
from optuna.integration import PyTorchLightningPruningCallback

from fsec.fsec_datamodule import FSECDataModule
from fsec.fsec_module import FSECModule

from .base_clustering import BaseClustering


class FSECClustering(BaseClustering):
    def __init__(self, logger, trial, dataset_name, params=None):
        super().__init__(params)
        self.logger = logger
        self.trial = trial
        self.dataset_name = dataset_name
        self.model = None  # Will be initialized during fit_predict

    def fit_predict(self, X, y=None):
        """
        Fit the FSEC model and predict cluster labels.
        """
        # Initialize DataModule
        datamodule = FSECDataModule(dataset_name=self.dataset_name, batch_size=len(X))
        datamodule.setup()

        # Initialize FSECModule
        self.model = FSECModule(self.params)

        # Initialize Trainer
        trainer = Trainer(
            logger=self.logger,
            max_epochs=1,
            enable_checkpointing=False,
            accelerator='auto',
            devices=1 if torch.cuda.is_available() else None,
            callbacks=[PyTorchLightningPruningCallback(self.trial, monitor="NMI")] if self.trial is not None else None,
            log_every_n_steps=1,
            deterministic=True,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # Fit the model
        trainer.fit(self.model, datamodule=datamodule)

        # Retrieve cluster labels
        self.labels_pred = self.model.labels_pred

        return self.labels_pred

