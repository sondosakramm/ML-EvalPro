from ml_eval_pro.evaluation_metrics.evaluators_factory import EvaluatorsFactory
from ml_eval_pro.evaluator.base_evaluator import BaseEvaluator
from ml_eval_pro.evaluator.interface_evaluator import InterfaceEvaluator


class ReliabilityEvaluator(BaseEvaluator):
    """
    A class of the reliability diagram evaluator.
    """

    def __init__(self, evaluator: InterfaceEvaluator):
        """
        Initializing the reliability diagram evaluator.
        :param evaluator: an instance of the evaluator used to initialize the main parameters and evaluate it.
        """
        super().__init__(evaluator)

        self.__reliability_diagram = None

    def evaluate(self, **kwargs):
        """
        Evaluate the reliability diagram from the different evaluation analysis provided.
        :param kwargs: the keys needed to evaluate the reliability diagram.
        """
        super().evaluate(**kwargs)

        print("Extracting the model reliability diagram ...")
        bins = kwargs['bins']
        self.__reliability_diagram = EvaluatorsFactory.get_evaluator(
            f"{self.problem_type} reliability evaluation",
            self.test_target,
            self.test_predictions if self.problem_type == 'regression' else
            self.test_predictions_proba,
            num_of_classes=self.num_classes,
            n_bins=bins).measure()

    @property
    def reliability_diagram(self):
        return self.__reliability_diagram
