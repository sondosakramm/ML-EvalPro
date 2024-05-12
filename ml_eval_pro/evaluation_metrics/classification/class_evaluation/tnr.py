from sklearn.metrics import recall_score

from ml_eval_pro.evaluation_metrics.classification.class_evaluation.class_evaluation import ClassClassification


class TNR(ClassClassification):
    """
    A class for evaluating the model using True Negative Rate evaluation metric.
    """

    def measure(self):
        """
        Evaluating the model with TNR.
        :return: the TNR value.
        """
        return recall_score(self.target, self.prediction, pos_label=0, average='micro') * 100
