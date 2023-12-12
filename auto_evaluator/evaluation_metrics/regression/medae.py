from sklearn.metrics import median_absolute_error

from auto_evaluator.evaluation_metrics.regression.regression_evaluator.regression_evaluator import RegressionEvaluator


class MEDAE(RegressionEvaluator):
    """
    Median Absolute Error.
    """
    def __init__(self, target, prediction):
        super().__init__(target, prediction)

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
