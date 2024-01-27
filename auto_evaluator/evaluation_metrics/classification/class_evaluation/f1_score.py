from sklearn.metrics import f1_score

from auto_evaluator.evaluation_metrics.classification.class_evaluation.class_evaluation import ClassClassification


class F1Score(ClassClassification):
    """
    A class for evaluating the model using F1-score evaluation metric.
    """

    def measure(self):
        """
        Evaluating the model with F1-score.
        :return: the F1-score value.
        """
        return f1_score(self.target, self.prediction, average='micro')