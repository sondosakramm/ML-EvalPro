from ml_eval_pro.evaluation_metrics.evaluators_factory import EvaluatorsFactory
from ml_eval_pro.evaluator.base_evaluator import BaseEvaluator
from ml_eval_pro.evaluator.interface_evaluator import InterfaceEvaluator


class MetricsEvaluator(BaseEvaluator):
    """
    A class of the metrics evaluator.
    """

    def __init__(self, metrics: [str], evaluator: InterfaceEvaluator):
        """
        Initializing the metrics evaluator.
        :param metrics: a list of the metrics to be evaluated.
        :param evaluator: an instance of the evaluator used to initialize the main parameters and evaluate it.
        """
        super().__init__(evaluator)

        self.metrics = metrics

        self.__test_metrics_values = None
        self.__train_metrics_values = None

    def evaluate(self, **kwargs):
        """
        Evaluate the metrics from the different evaluation analysis provided.
        :param kwargs: the keys needed to evaluate the metrics.
        """
        super().evaluate(**kwargs)

        print("Evaluating the model using the input evaluation metrics ...")
        bins = kwargs['bins']
        self.__test_metrics_values = self.__get_metrics_values(self.test_target, self.test_predictions,
                                                               self.test_predictions_proba, bins)
        self.__train_metrics_values = {} if self.train_target is None \
            else self.__get_metrics_values(self.train_target,
                                           self.train_predictions,
                                           self.train_predictions_proba, bins)

    @property
    def test_metrics_values(self):
        return self.__test_metrics_values

    @property
    def train_metrics_values(self):
        return self.__train_metrics_values

    def __get_metrics_values(self, target, predictions, predictions_prob, bins):
        """
        Calculating the evaluation metrics.
        :param target: the target true values.
        :param predictions: the target prediction values.
        :param predictions_prob: the target prediction probability values of each class (for classification only).
        :return: A dictionary of the values.
        """
        print("Evaluating the model using the input evaluation metrics ...")
        res = {}
        for metric in self.metrics:
            if metric == 'Expected Calibration Error' or metric == 'AUC':
                res[metric] = EvaluatorsFactory.get_evaluator(metric, target,
                                                              predictions_prob,
                                                              num_of_classes=self.num_classes,
                                                              n_bins=bins).measure()
            else:
                res[metric] = EvaluatorsFactory.get_evaluator(metric, target,
                                                              predictions,
                                                              num_of_classes=self.num_classes,
                                                              n_bins=bins).measure()
        return res
