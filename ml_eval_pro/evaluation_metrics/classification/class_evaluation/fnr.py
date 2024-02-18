from sklearn.metrics import recall_score

from ml_eval_pro.evaluation_metrics.classification.class_evaluation.class_evaluation import ClassClassification


class FNR(ClassClassification):
    """
    A class for evaluating the model using False Negative Rate evaluation metric.
    """

    def measure(self):
        """
        Evaluating the model with FNR.
        :return: the FNR value.
        """
        return 1 - recall_score(self.target, self.prediction, average='micro')
