from sklearn.metrics import median_absolute_error

from regression_metrics.regression.regression_evaluator import RegressionEvaluator


class MEDAE(RegressionEvaluator):
    """
    Median Absolute Error.
    """
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)

    def calculate(self):
        """
        Calculate Median Absolute Error.

        return:
            medae (float): Median Absolute Error.
        """
        try:
            return median_absolute_error(self.y_true, self.y_pred)
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
