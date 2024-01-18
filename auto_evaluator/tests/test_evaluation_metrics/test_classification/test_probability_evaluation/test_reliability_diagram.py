import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from auto_evaluator.evaluation_metrics.classification.probability_evaluation.reliability_diagram import \
    ReliabilityDiagram


class TestReliabilityDiagram(unittest.TestCase):

    def setUp(self):
        # Test data setup for binary and multi-class scenarios
        self.binary_target = np.array([1, 0, 1, 1, 0, 1])
        self.binary_predictions_prob = np.array([
            [0.8, 0.2],
            [0.1, 0.9],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.3, 0.7],
            [0.9, 0.1]
        ])

        self.multi_target = np.array([0, 1, 2, 0, 1, 2])
        self.multi_predictions_prob = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.2, 0.3, 0.5],
            [0.8, 0.1, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7]
        ])

        self.n_bins = 3

    def test_measure_binary(self):
        # Create an instance of ReliabilityDiagram class for binary classification
        reliability_diagram = ReliabilityDiagram(self.binary_target, self.binary_predictions_prob, n_bins=self.n_bins)

        # Mock the calibration_curve function and its return value
        mock_calibration_curve = MagicMock(return_value=([1.0, 1.0, 0.0], [0.2, 0.4, 0.8]))
        with patch('sklearn.calibration.calibration_curve', mock_calibration_curve):
            calibration_info = reliability_diagram.measure()

            # Check if the returned calibration info matches the expected calibration curve outputs
            expected_calibration_info = ([1.0, 1.0, 0.0], [0.2, 0.4, 0.8])

            # Check each element separately due to possible numerical differences
            for i in range(len(calibration_info)):
                self.assertTrue(np.allclose(calibration_info[i], expected_calibration_info[i]))

    def test_measure_multi_class(self):
        # Create an instance of ReliabilityDiagram class for multi-class classification
        reliability_diagram = ReliabilityDiagram(self.multi_target, self.multi_predictions_prob, 3, n_bins=self.n_bins)

        # Mock the calibration_curve function and its return value
        mock_calibration_curve = MagicMock(return_value=[([0.0, 1.0], [0.15, 0.75]), ([0.0, 1.0], [0.2, 0.55]), ([0.0, 1.0, 1.0], [0.2, 0.5, 0.7])])
        with patch('sklearn.calibration.calibration_curve', mock_calibration_curve):
            calibration_info = reliability_diagram.measure()

            print(calibration_info)

            # Check if the returned calibration info matches the expected calibration curve outputs
            expected_calibration_info = [([0.0, 1.0], [0.15, 0.75]), ([0.0, 1.0], [0.2, 0.55]), ([0.0, 1.0, 1.0], [0.2, 0.5, 0.7])]

            # Check each element separately due to possible numerical differences
            for i in range(len(calibration_info)):
                for j in range (len(calibration_info[i])):
                    self.assertTrue(np.allclose(calibration_info[i][j], expected_calibration_info[i][j]))

if __name__ == '__main__':
    unittest.main()
