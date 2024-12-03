from .base_clustering import BaseClustering
from sklearn.cluster import KMeans

class KMeansClustering(BaseClustering):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = KMeans(**self.params)

    def fit_predict(self, X, y=None):
        """
        Fit the K-Means model and predict cluster labels.
        """
        self.labels_pred = self.model.fit_predict(X)
        return self.labels_pred

