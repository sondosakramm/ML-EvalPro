import unittest
import numpy as np

from auto_evaluator.evaluation_metrics.classification.probability_evaluation.expected_calibration_error import ECEMetric


class TestECEMetric(unittest.TestCase):

    def setUp(self):
        # Test data setup
        self.target = np.array([1, 0, 1, 1, 0, 1])
        self.predictions_prob = np.array([
            [0.8, 0.2],
            [0.1, 0.9],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.3, 0.7],
            [0.9, 0.1]
        ])
        self.n_bins = 3

    def test_measure(self):
        # Create an instance of ECEMetric class
        ece_metric = ECEMetric(self.target, self.predictions_prob, self.n_bins)

        # Calculate the expected ECE using the mocked data
        confidence = np.max(ece_metric.prediction_prob, axis=1)
        accuracy = (ece_metric.target == np.argmax(ece_metric.prediction_prob, axis=1)) * 1

        ece = np.array([])
        n = confidence.shape[0]
        binning_ranges = np.linspace(0, 1, ece_metric.n_bins + 1)
        for i in range(1, ece_metric.n_bins + 1):
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

        expected_ece = ece.sum()

        # Call the measure method and check if it returns the expected ECE
        self.assertAlmostEqual(ece_metric.measure(), expected_ece, places=6)

    def test_get_model_confidence(self):
        # Create an instance of ECEMetric class
        ece_metric = ECEMetric(self.target, self.predictions_prob, self.n_bins)

        # Get the confidence using the mocked data
        confidence = np.max(ece_metric.prediction_prob, axis=1)

        # Call the internal method __get_model_confidence and compare the result
        self.assertTrue(np.array_equal(ece_metric._ECEMetric__get_model_confidence(), confidence))

if __name__ == '__main__':
    unittest.main()
