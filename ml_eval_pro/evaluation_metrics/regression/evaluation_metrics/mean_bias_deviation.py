import numpy as np

from ml_eval_pro.evaluation_metrics.regression.regression_evaluator import RegressionEvaluator


class MeanBiasDeviation(RegressionEvaluator):
    """
    Mean Bias Deviation.
    """
    def measure(self):
        """
        Calculate Mean Bias Deviation.

        return:
            mean_bias_deviation (float): Mean Bias Deviation.
        """
        try:
            return np.mean(self.target - self.prediction)
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
