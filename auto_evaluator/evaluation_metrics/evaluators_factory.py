from auto_evaluator.evaluation_metrics.classification.classification_evaluation import ClassificationEvaluation
from auto_evaluator.evaluation_metrics.regression.regression_evaluator.regression_evaluator import RegressionEvaluator


class EvaluatorsFactory:
    @staticmethod
    def create_evaluator(evaluator_type: str, target, prediction, num_of_classes: int = 2):
        """
        Create an instance of the appropriate evaluation metric based on the task type.
        :param evaluator_type: The type of the evaluation task ("classification" or "regression").
        :param target: The target true/actual values.
        :param prediction: The target predicted values.
        :param num_of_classes: Number of classes (only for classification tasks).
        :return: An instance of the evaluation metric.
        """
        try:
            if evaluator_type == 'regression':
                return RegressionEvaluator(target, prediction)
            elif evaluator_type == 'classification':
                return ClassificationEvaluation(target, prediction, num_of_classes)
        except NotImplementedError:
            raise NotImplementedError(f'{evaluator_type} not implemented.')
