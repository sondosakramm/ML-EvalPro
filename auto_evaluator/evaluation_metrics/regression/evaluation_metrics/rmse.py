import numpy as np
from sklearn.metrics import mean_squared_error

from auto_evaluator.evaluation_metrics.regression.regression_evaluator import RegressionEvaluator


class RMSE(RegressionEvaluator):
    """
    Root Mean Squared Error
    """
    def measure(self):
        """
        Calculate Root Mean Squared Error

        return:
            rmse (float): Root Mean Squared Error
        """
        try:
            mse = mean_squared_error(self.target, self.prediction)
            return np.sqrt(mse)
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
