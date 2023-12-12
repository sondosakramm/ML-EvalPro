from sklearn.metrics import mean_absolute_error

from regression_metrics.regression.regression_evaluator import RegressionEvaluator


class MAE(RegressionEvaluator):
    """
    Mean Absolute Error
    """
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)

    def calculate(self):
        """
        Calculate Mean Absolute Error

        return:
            mae (float): Mean Absolute Error
        """
        try:
            return mean_absolute_error(self.y_true, self.y_pred)
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
