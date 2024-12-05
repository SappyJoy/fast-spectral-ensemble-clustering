import lightning.pytorch as lp
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.loaders import get_dataset
from fsec.autoencoder import Autoencoder

class FSECDataModule(lp.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        batch_size=256,
        encoding_dim=64,
        hidden_dims=[128, 64],
        max_epochs=50,
        num_workers=4
    ):
        """
        Initializes the FSECDataModule with the necessary parameters.

        Parameters:
        - dataset_name: Name of the dataset to be loaded.
        - batch_size: Batch size for training the Autoencoder.
        - encoding_dim: Dimensionality of the encoded space.
        - hidden_dims: List of hidden layer dimensions for the Autoencoder.
        - max_epochs: Maximum number of epochs to train the Autoencoder.
        - num_workers: Number of worker threads for data loading.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.max_epochs = max_epochs
        self.num_workers = num_workers

        self.X = None
        self.y = None
        self.autoencoder = None
        self.encoded_X = None

    def prepare_data(self):
        # Download or preprocess data if needed
        pass

    def setup(self, stage=None):
        # Load data
        self.X, self.y, _ = get_dataset(self.dataset_name)
        # Normalize data
        self.X = (self.X - self.X.mean(axis=0)) / (self.X.std(axis=0) + 1e-8)
        # Convert to torch tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
        
        # Initialize Autoencoder
        input_dim = self.X.shape[1]
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            encoding_dim=self.encoding_dim,
            hidden_dims=self.hidden_dims
        )
        
        # Create DataLoader for Autoencoder training
        dataset = TensorDataset(self.X, self.y)  # Labels are not used in Autoencoder
        self.train_dataloader_ae = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def train_autoencoder(self):
        """
        Trains the Autoencoder using PyTorch Lightning's Trainer.
        """
        trainer = lp.Trainer(
            max_epochs=self.max_epochs,
            devices=1 if torch.cuda.is_available() else 0,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(self.autoencoder, self.train_dataloader_ae)
        
        # Encode the data
        self.encode_data()

    def encode_data(self):
        """
        Encodes the data using the trained Autoencoder.
        """
        self.autoencoder.eval()
        with torch.no_grad():
            self.encoded_X = self.autoencoder.encoder(self.X).cpu().numpy()

    def get_encoded_data(self):
        """
        Retrieves the encoded data and corresponding labels.

        Returns:
        - encoded_X: Encoded feature matrix.
        - y: True labels.
        """
        if self.encoded_X is None:
            raise ValueError("Autoencoder has not been trained. Call train_autoencoder() first.")
        return self.encoded_X, self.y.numpy()

    def train_dataloader(self):
        # Required method but not used for Autoencoder training
        return self.train_dataloader_ae

