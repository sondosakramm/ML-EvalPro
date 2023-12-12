import numpy as np

from auto_evaluator.evaluation_metrics.classification.probability_evaluation.probability_evaluation import \
    ProbabilityClassification


class ECEMetric(ProbabilityClassification):
    """
    A class for evaluating the model using expected (or estimated) calibration error evaluation metric.
    """
    def __init__(self, target: np.ndarray, predictions_prob: np.ndarray, number_of_classes:int =2, n_bins: int=5):
        """
        Initializing expected (or estimated) calibration error evaluation metric.
        :param target: the target class true values.
        :param predictions_prob: the prediction probability of the positive class.
        :param number_of_classes: the number of classes in this model.
        :param n_bins: the number of bins needed.
        """
        super().__init__(target, predictions_prob, number_of_classes, n_bins)

    def measure(self):
        """
        Evaluating the model with ECE.
        :return: the ECE value.
        """
        ece = np.array([])

        confidence = self.__get_model_confidence()
        accuracy = (self.target == self.prediction) * 1

        n = confidence.shape[0]
        binning_ranges = np.linspace(0, 1, self.n_bins + 1)
        for i in range(1, self.n_bins + 1):
            b_indices = np.logical_and(confidence >= binning_ranges[i - 1], confidence < binning_ranges[i])
            b_size = b_indices.sum()

            if b_size > 0:
                b_weight = b_size / n
                b_conf = confidence[b_indices]
                b_acc = accuracy[b_indices]

                b_conf_avg = np.mean(b_conf)
                b_acc_avg = np.mean(b_acc)

                b_ece = abs(b_conf_avg - b_acc_avg) * b_weight

                ece = np.append(ece, [b_ece])
        return ece.sum()

    def __get_model_confidence(self):
        """
        Get the confidence of each value of the prediction.
        :return: the confidence values.
        """
        return np.max(self.prediction_prob, axis=1)

