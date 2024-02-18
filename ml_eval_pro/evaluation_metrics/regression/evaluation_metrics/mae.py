from sklearn.metrics import mean_absolute_error

from ml_eval_pro.evaluation_metrics.regression.regression_evaluator import RegressionEvaluator


class MAE(RegressionEvaluator):
    """
    Mean Absolute Error
    """
    def measure(self):
        """
        Calculate Mean Absolute Error

        return:
            mae (float): Mean Absolute Error
        """
        try:
            return mean_absolute_error(self.target, self.prediction)
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
