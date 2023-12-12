from sklearn.metrics import r2_score

from auto_evaluator.evaluation_metrics.regression.regression_evaluator.regression_evaluator import RegressionEvaluator


class RSquare(RegressionEvaluator):
    """
     R-Squared.
    """
    def __init__(self, target, prediction):
        super().__init__(target, prediction)

    def measure(self):
        """
        Calculate  R-Squared value.

        return:
            r_squared (float):  R-Squared value.
        """
        try:
            return r2_score(self.target, self.prediction)
        except ValueError as ve:
            raise ValueError(f'Error calculating mean absolute error: {ve}')
