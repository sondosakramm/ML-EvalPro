from sklearn.metrics import r2_score

from ml_eval_pro.evaluation_metrics.regression.regression_evaluator import RegressionEvaluator


class RSquare(RegressionEvaluator):
    """
     R-Squared.
    """

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
