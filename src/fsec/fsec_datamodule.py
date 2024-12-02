import lightning.pytorch as lp
from torch.utils.data import DataLoader, TensorDataset
import torch
from data.loaders import get_dataset

class FSECDataModule(lp.LightningDataModule):
    def __init__(self, dataset_name, batch_size=1):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.X = None
        self.y = None

    def prepare_data(self):
        # This method can be used to download or preprocess data if needed
        pass

    def setup(self, stage=None):
        # Load data
        self.X, self.y, _ = get_dataset(self.dataset_name)
        # Convert to torch tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def train_dataloader(self):
        # Since clustering doesn't require batches, we can return the entire dataset as one batch
        dataset = TensorDataset(self.X, self.y)
        return DataLoader(dataset, batch_size=self.batch_size)

