from abc import ABC, abstractmethod
import numpy as np

class BaseClustering(ABC):
    def __init__(self, params=None):
        """
        Initialize the clustering algorithm with given parameters.
        """
        self.params = params or {}
        self.model = None
        self.labels_pred = None

    @abstractmethod
    def fit_predict(self, X):
        """
        Fit the model to the data and predict cluster labels.
        """
        pass

    def get_labels(self):
        """
        Retrieve the predicted cluster labels.
        """
        if self.labels_pred is None:
            raise ValueError("Model has not been fitted yet.")
        return self.labels_pred

    def get_params(self):
        """
        Retrieve the parameters of the clustering algorithm.
        """
        return self.params

