from abc import ABC

from auto_evaluator.evaluation_metrics.evaluation_metric import EvaluationMetric


class RegressionEvaluator(EvaluationMetric, ABC):
    def __init__(self, target, prediction):
        super().__init__(target, prediction)
