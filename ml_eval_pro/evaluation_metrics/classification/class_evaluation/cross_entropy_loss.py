import numpy as np
from scipy.special import softmax

from ml_eval_pro.evaluation_metrics.classification.class_evaluation.class_evaluation import ClassClassification


class CrossEntropyLoss(ClassClassification):
    """
    A class for evaluating the model using Cross Entropy Loss evaluation metric.
    """
    def measure(self):
        """
        Measure the cross entropy loss of the model.
        :return: the cross entropy loss of the model.
        """
        softmax_vals = softmax(self.prediction)
        return -np.sum(self.target * np.log(softmax_vals))