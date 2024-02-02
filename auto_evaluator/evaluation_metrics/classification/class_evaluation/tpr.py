from sklearn.metrics import recall_score

from auto_evaluator.evaluation_metrics.classification.class_evaluation.class_evaluation import ClassClassification


class TPR(ClassClassification):
    """
    A class for evaluating the model using True Positive Rate evaluation metric.
    """

    def measure(self):
        """
        Evaluating the model with TPR.
        :return: the TPR value.
        """
        return recall_score(self.target, self.prediction, average='micro')
