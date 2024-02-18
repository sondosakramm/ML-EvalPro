from abc import ABC

from ml_eval_pro.evaluation_metrics.evaluation_metric import EvaluationMetric


class ClassificationEvaluation(EvaluationMetric, ABC):
    """
    An abstract class for a classification evaluation metric.
    """
    def __init__(self, target, prediction, num_of_classes: int = 2):
        super().__init__(target, prediction)
        if num_of_classes < 2:
            raise ValueError(f"Number of classes should be greater than 2, num_of_classes={num_of_classes}")
        else:
            self.num_of_classes = num_of_classes