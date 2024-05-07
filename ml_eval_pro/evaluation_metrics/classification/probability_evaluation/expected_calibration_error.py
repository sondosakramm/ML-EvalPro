import numpy as np

from ml_eval_pro.evaluation_metrics.classification.probability_evaluation.probability_evaluation import \
    ProbabilityClassification
from ml_eval_pro.evaluation_metrics.classification.probability_evaluation.reliability_diagram import ReliabilityDiagram


class ECEMetric(ProbabilityClassification):
    """
    A class for evaluating the model using expected (or estimated) calibration error evaluation metric.
    """

    def measure(self):
        """
        Evaluating the model with ECE.
        :return: the ECE value.
        """
        ece = np.array([])

        confidence_vals, accuracy_vals, weights = ReliabilityDiagram(self.target, self.prediction_prob,
                                                                     self.num_of_classes, self.n_bins).measure()

        for i in range(len(confidence_vals)):
            b_ece = abs(confidence_vals[i] - accuracy_vals[i]) * weights[i]
            ece = np.append(ece, [b_ece])

        return ece.sum()
