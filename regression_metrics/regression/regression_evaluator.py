from abc import ABC, abstractmethod

from utils import validation


class RegressionEvaluator(ABC):
    def __init__(self, y_true, y_pred):
        """
        Parameters:
            - y_true: array-like, shape (n_samples,)
            Ground truth (correct) target values.
            - y_pred: array-like, shape (n_samples,)
            Estimated target values.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        validation.check_consistent_length(y_true, y_pred)

    @abstractmethod
    def calculate(self):
        pass
