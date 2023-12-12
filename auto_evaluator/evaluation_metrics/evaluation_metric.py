from abc import ABC, abstractmethod


class EvaluationMetric(ABC):
    """
    An abstract class for an evaluation metric.
    """
    def __init__(self, target, prediction):
        """
        Initializing the evaluation metric needed values.
        :param target: the target true values.
        :param prediction: the target prediction values.
        """
        self.target = target
        self.prediction = prediction


    @abstractmethod
    def measure(self):
        pass
