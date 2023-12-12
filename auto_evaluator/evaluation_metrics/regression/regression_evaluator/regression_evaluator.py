from abc import abstractmethod

from auto_evaluator.evaluation_metrics.evaluation_metric import EvaluationMetric


class RegressionEvaluator(EvaluationMetric):
    def __init__(self, target, prediction):
        super().__init__(target, prediction)

    @abstractmethod
    def measure(self):
        pass
