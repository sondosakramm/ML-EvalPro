import numpy as np

from ml_eval_pro.evaluation_metrics.classification.probability_evaluation.probability_evaluation import \
    ProbabilityClassification


class ReliabilityDiagram(ProbabilityClassification):
    """
    A class for evaluating the model using Reliability Diagram evaluation metric.
    """

    def measure(self):
        """
        Evaluating the model with ECE.
        :return: the ECE value.
        """
        confidence_vals = []
        accuracy_vals = []
        weights = []

        confidence = self.__get_model_confidence()
        accuracy = (self.target == self.prediction) * 1

        n = confidence.shape[0]
        binning_ranges = np.linspace(0, 1, self.n_bins + 1)
        for i in range(1, self.n_bins + 1):
            b_indices = np.logical_and(confidence > binning_ranges[i - 1], confidence <= binning_ranges[i])
            b_size = b_indices.sum()

            if b_size > 0:
                b_weight = b_size / n
                b_conf = confidence[b_indices]
                b_acc = accuracy[b_indices]

                b_conf_avg = np.mean(b_conf)
                b_acc_avg = np.mean(b_acc)

                confidence_vals.append(b_conf_avg)
                accuracy_vals.append(b_acc_avg)
                weights.append(b_weight)

        return confidence_vals, accuracy_vals, weights

    def __get_model_confidence(self):
        """
        Get the confidence of each value of the prediction.
        :return: the confidence values.
        """
        return np.max(self.prediction_prob, axis=1)
