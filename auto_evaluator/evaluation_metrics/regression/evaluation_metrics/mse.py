from sklearn.metrics import mean_squared_error

from auto_evaluator.evaluation_metrics.regression.regression_evaluator import RegressionEvaluator


class MSE(RegressionEvaluator):
    """
    Mean Squared Error
    """

    def measure(self):
        """
        Calculate Mean Squared Error

        return:
            mse (float): Mean Squared Error
        """
        try:
            return mean_squared_error(self.target, self.prediction)
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
