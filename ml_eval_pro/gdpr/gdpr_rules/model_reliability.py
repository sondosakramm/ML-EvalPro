from ml_eval_pro.evaluation_metrics.evaluators_factory import EvaluatorsFactory
from ml_eval_pro.gdpr.gdpr_compliance import GdprCompliance


class ModelReliability(GdprCompliance):

    def __init__(self, model=None, X_test=None, y_test=None, problem_type='classification', X_train=None, y_train=None,
                 features_description: dict = None, num_of_classes: int = 2, n_bins: int = 5):
        super().__init__(model, X_test, y_test, problem_type, X_train, y_train, features_description, num_of_classes,
                         n_bins)
        self.prediction = None

    def get_metric(self):
        self.prediction = self.model.predict(self.X_test, predict_class=False)
        evaluator = EvaluatorsFactory.get_evaluator("expected calibration error",
                                                    self.y_test, self.prediction,
                                                    self.num_of_classes, self.n_bins)
        ece = evaluator.measure()
        return ece
