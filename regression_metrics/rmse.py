import numpy as np
from sklearn.metrics import mean_squared_error

from regression_metrics.regression.regression_evaluator import RegressionEvaluator


class RMSE(RegressionEvaluator):
    """
    Root Mean Squared Error
    """
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)

    def calculate(self):
        """
        Calculate Root Mean Squared Error

        return:
            rmse (float): Root Mean Squared Error
        """
        try:
            mse = mean_squared_error(self.y_true, self.y_pred)
            return np.sqrt(mse)
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
