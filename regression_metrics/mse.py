from sklearn.metrics import mean_squared_error

from regression_metrics.regression.regression_evaluator import RegressionEvaluator


class MSE(RegressionEvaluator):
    """
    Mean Squared Error
    """
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)

    def calculate(self):
        """
        Calculate Mean Squared Error

        return:
            mse (float): Mean Squared Error
        """
        try:
            return mean_squared_error(self.y_true, self.y_pred)
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
