import numpy as np

from regression_metrics.regression.regression_evaluator import RegressionEvaluator


class MeanBiasDeviation(RegressionEvaluator):
    """
    Mean Bias Deviation.
    """
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)

    def calculate(self):
        """
        Calculate Mean Bias Deviation.

        return:
            mean_bias_deviation (float): Mean Bias Deviation.
        """
        try:
            return np.mean(self.y_pred - self.y_true)
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
