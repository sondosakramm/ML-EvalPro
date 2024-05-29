from abc import abstractmethod, ABC

from ml_eval_pro.evaluation_metrics.evaluators_factory import EvaluatorsFactory


class ModelVariance(ABC):

    def __init__(self, model, model_type: str, test_dataset, target, evaluation_metric):

        self.model = model
        self.test_dataset = test_dataset
        self.target = target
        self.model_type = model_type
        self.evaluation_metric = evaluation_metric


    @abstractmethod
    def calculate_variance(self):
        """
        Abstract method for calculating the model variance.

        This method must be implemented in the derived class.
        """
        pass

    def calculate_errors(self, target, predictions):
        return EvaluatorsFactory.get_evaluator(evaluation_metric=self.evaluation_metric, target=target,
                                               prediction=predictions).measure()

    def get_diff(self):
        pass

