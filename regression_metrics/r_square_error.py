from sklearn.metrics import r2_score

from regression_metrics.regression.regression_evaluator import RegressionEvaluator


class RSquare(RegressionEvaluator):
    """
     R-Squared.
    """
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)

    def calculate(self):
        """
        Calculate  R-Squared value.

        return:
            r_squared (float):  R-Squared value.
        """
        try:
            return r2_score(self.y_true, self.y_pred)
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
