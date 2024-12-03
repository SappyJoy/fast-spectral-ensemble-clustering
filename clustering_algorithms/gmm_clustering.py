from .base_clustering import BaseClustering
from sklearn.mixture import GaussianMixture

class GMMClustering(BaseClustering):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = GaussianMixture(**self.params)

    def fit_predict(self, X, y=None):
        """
        Fit the GMM model and predict cluster labels.
        """
        self.model.fit(X)
        self.labels_pred = self.model.predict(X)
        return self.labels_pred

