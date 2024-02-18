from sklearn.metrics import median_absolute_error

from ml_eval_pro.evaluation_metrics.regression.regression_evaluator import RegressionEvaluator


class MEDAE(RegressionEvaluator):
    """
    Median Absolute Error.
    """

    def measure(self):
        """
        Calculate Median Absolute Error.

        return:
            medae (float): Median Absolute Error.
        """
        try:
            return median_absolute_error(self.target, self.prediction)
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
