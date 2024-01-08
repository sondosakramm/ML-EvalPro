from sklearn.metrics import mean_absolute_percentage_error

from auto_evaluator.evaluation_metrics.regression.regression_evaluator import RegressionEvaluator


class MAPE(RegressionEvaluator):
    """
    Mean Absolute Percentage Error
    """
    def measure(self):
        """
        Calculate Mean Absolute Percentage Error

        return:
            mape (float): Mean Absolute Percentage Error
        """
        try:
            return mean_absolute_percentage_error(self.target, self.prediction) * 100
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
