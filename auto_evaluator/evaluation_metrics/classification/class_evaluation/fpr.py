from sklearn.metrics import recall_score

from auto_evaluator.evaluation_metrics.classification.class_evaluation.class_evaluation import ClassClassification


class FPR(ClassClassification):
    """
    A class for evaluating the model using False Positive Rate evaluation metric.
    """

    def measure(self):
        """
        Evaluating the model with FPR.
        :return: the FPR value.
        """
        return 1 - recall_score(self.target, self.prediction, pos_label=0, average='micro')
