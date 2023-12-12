import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from regression_metrics.regression.regression_evaluator import RegressionEvaluator


class MAPE(RegressionEvaluator):
    """
    Mean Absolute Percentage Error
    """
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)

    def calculate(self):
        """
        Calculate Mean Absolute Percentage Error

        return:
            mape (float): Mean Absolute Percentage Error
        """
        try:
            return mean_absolute_percentage_error(self.y_true, self.y_pred) * 100
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
