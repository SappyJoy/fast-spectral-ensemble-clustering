from .base_clustering import BaseClustering
from sklearn.cluster import AgglomerativeClustering

class AgglomerativeClusteringClustering(BaseClustering):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = AgglomerativeClustering(**self.params)

    def fit_predict(self, X, y=None):
        """
        Fit the Agglomerative Clustering model and predict cluster labels.
        """
        self.labels_pred = self.model.fit_predict(X)
        return self.labels_pred

