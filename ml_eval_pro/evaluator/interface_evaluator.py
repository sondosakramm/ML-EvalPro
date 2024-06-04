from abc import ABC, abstractmethod


class InterfaceEvaluator(ABC):
    """
    An interface of the evaluator.
    """

    def __init__(self, model_pipeline, problem_type, num_classes,
                 test_dataset, test_target, test_predictions, test_predictions_proba,
                 train_dataset, train_target, train_predictions, train_predictions_proba, *args, **kwargs):
        self.model_pipeline = model_pipeline
        self.problem_type = problem_type
        self.num_classes = num_classes

        self.test_dataset = test_dataset
        self.test_target = test_target
        self.test_predictions = test_predictions
        self.test_predictions_proba = test_predictions_proba

        self.train_dataset = train_dataset
        self.train_target = train_target
        self.train_predictions = train_predictions
        self.train_predictions_proba = train_predictions_proba

    @abstractmethod
    def evaluate(self, **kwargs):
        """
        Evaluate the model from the different evaluation analysis provided.
        :param kwargs: the keys needed to evaluate the model.
        """
        pass
