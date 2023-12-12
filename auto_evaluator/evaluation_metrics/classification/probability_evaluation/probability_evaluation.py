from abc import ABC

import numpy as np

from auto_evaluator.evaluation_metrics.classification.classification_evaluation import ClassificationEvaluation


class ProbabilityClassification(ClassificationEvaluation, ABC):
    """
    An abstract class for a probability-based classification evaluation metric.
    """
    def __init__(self, target, prediction_prob, number_of_classes:int =2, n_bins: int=5):
        """
        Initializing the values needed for the probability-based classification evaluation metric.
        :param target: the target true values.
        :param prediction_prob: the prediction probability values.
        :param number_of_classes: the number of classes in this model.
        :param n_bins: the number of bins for calculating the confidence in different ranges.
        """
        self.prediction_prob = prediction_prob
        super().__init__(target, self.__get_predictions_class(), number_of_classes)
        if n_bins > 0:
            self.n_bins = n_bins
        else:
            raise ValueError(f"Number of bins should be greater than zero, n_bins={n_bins}")

    def __get_predictions_class(self):
        """
        Getting the class of each probability value maximum probability value among classes.
        :return: An array containing the class index of each value.
        """
        return np.argmax(self.prediction_prob, axis=1)