from abc import abstractmethod

from ml_eval_pro.evaluator.interface_evaluator import InterfaceEvaluator


class BaseEvaluator(InterfaceEvaluator):
    """
    An abstract class of the evaluator.
    """
    def __init__(self, evaluator: InterfaceEvaluator):
        """
        Initializing the evaluator.
        :param evaluator: an instance of the evaluator used to initialize the main parameters and evaluate it.
        """
        super().__init__(*evaluator.__dict__.values())
        self.evaluator = evaluator

    def evaluate(self, **kwargs):
        """
        Evaluate the model from the different evaluation analysis provided.
        :param kwargs: the keys needed to evaluate the model.
        """
        self.evaluator.evaluate(**kwargs)
