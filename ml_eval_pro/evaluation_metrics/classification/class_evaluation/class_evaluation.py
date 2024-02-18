from abc import ABC

from ml_eval_pro.evaluation_metrics.classification.classification_evaluation import ClassificationEvaluation


class ClassClassification(ClassificationEvaluation, ABC):
    """
    An abstract class for a class-based classification evaluation metric.
    """
    pass