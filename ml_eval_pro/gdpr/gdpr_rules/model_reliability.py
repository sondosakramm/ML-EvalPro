from ml_eval_pro.evaluation_metrics.evaluators_factory import EvaluatorsFactory
from ml_eval_pro.gdpr.gdpr_compliance import GdprCompliance


class ModelReliability(GdprCompliance):

    def __init__(self, model=None, X_test=None, y_test=None, problem_type='classification', X_train=None, y_train=None,
                 features_description: dict = None, num_of_classes: int = 2, n_bins: int = 5):
        super().__init__(model=model, X_test=X_test, y_test=y_test, problem_type=problem_type,
                         X_train=X_train, y_train=y_train, features_description=features_description,
                         num_of_classes=num_of_classes,
                         n_bins=n_bins)
        self.prediction = None

    def get_metric(self):
        self.prediction = self.model.predict(self.X_test, predict_class=False)
        if self.problem_type == 'regression':
            return EvaluatorsFactory.create(f"{self.problem_type} reliability evaluation",
                                            self.X_test,
                                            self.prediction,
                                            n_bins=self.n_bins).measure()

        elif self.problem_type == 'classification':
            evaluator = EvaluatorsFactory.create("Expected Calibration Error",
                                                 self.y_test, self.prediction,
                                                 self.num_of_classes, self.n_bins)
            ece = evaluator.measure()
            return ece
