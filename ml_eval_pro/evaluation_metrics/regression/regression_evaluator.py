from abc import ABC

from ml_eval_pro.evaluation_metrics.evaluation_metric import EvaluationMetric


class RegressionEvaluator(EvaluationMetric, ABC):
    def __init__(self, target, prediction):
        super().__init__(target, prediction)
