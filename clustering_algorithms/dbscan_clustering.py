from .base_clustering import BaseClustering
from sklearn.cluster import DBSCAN

class DBSCANClustering(BaseClustering):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = DBSCAN(**self.params)

    def fit_predict(self, X, y=None):
        """
        Fit the DBSCAN model and predict cluster labels.
        """
        self.labels_pred = self.model.fit_predict(X)
        return self.labels_pred

