import numpy as np
from sklearn.calibration import calibration_curve

from auto_evaluator.evaluation_metrics.classification.probability_evaluation.probability_evaluation import \
    ProbabilityClassification


class ReliabilityDiagram(ProbabilityClassification):
    """
    A class for evaluating the model using Reliability Diagram evaluation metric.
    """

    def __init__(self, target: np.ndarray, predictions_prob: np.ndarray, number_of_classes: int=2,
                 n_bins: int=5, display: bool=False):
        """
        Initializing reliability diagram evaluation metric.
        :param target: the target class true values.
        :param predictions_prob: the prediction probability of the positive class.
        :param number_of_classes: the number of classes in this model.
        :param n_bins: the number of bins needed.
        :param display: display the reliability diagram plot.
        """
        super().__init__(target, predictions_prob, number_of_classes, n_bins)
        self.display = display

    def measure(self):
        """
        Evaluating the model with reliability diagram.
        :return: the reliability diagram bins values.
        """
        prediction_prob = np.copy(self.prediction_prob)

        if self.num_of_classes == 2:
            prediction_prob = prediction_prob[:,1]

            return calibration_curve(self.target, prediction_prob, n_bins=self.n_bins)

        calibration_info = []
        for class_index in range(self.num_of_classes):
            calibration_info.append(calibration_curve(self.target == class_index, self.prediction_prob[:, class_index],
                                                 n_bins=self.n_bins))

        return calibration_info