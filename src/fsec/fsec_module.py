import lightning.pytorch as lp
from src.fsec.clustering import FSEC
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from src.fsec.utils import clustering_accuracy

class FSECModule(lp.LightningModule):
    def __init__(self, params):
        super(FSECModule, self).__init__()
        self.params = params
        self.fsec = FSEC(**params)
        self.nmi = 0.0
        self.ari = 0.0
        self.acc = 0.0
        self.labels_pred = None

        # Disable automatic optimization since we don't need it
        self.automatic_optimization = False

    def forward(self, X):
        # Forward pass isn't used for clustering, but must be defined
        pass

    def training_step(self, batch, batch_idx):
        # Perform clustering on the entire dataset
        X, y = batch  # Assuming batch_size = len(X)
        
        # Move tensors to CPU before converting to NumPy
        X_cpu = X.cpu().numpy()
        y_cpu = y.cpu().numpy()
        
        # Fit the FSEC model
        self.fsec.fit(X_cpu)
        self.labels_pred = self.fsec.labels_

        # Compute metrics
        self.nmi = normalized_mutual_info_score(y_cpu, self.labels_pred)
        self.ari = adjusted_rand_score(y_cpu, self.labels_pred)
        self.acc = clustering_accuracy(y_cpu, self.labels_pred)

        # Log metrics
        self.log('NMI', self.nmi)
        self.log('ARI', self.ari)
        self.log('ACC', self.acc)

        return {}
    
    def configure_optimizers(self):
        # No optimizers are needed for clustering
        return None

