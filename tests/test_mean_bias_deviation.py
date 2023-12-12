import unittest

import numpy as np

from regression_metrics.mean_bias_deviation import MeanBiasDeviation


class TestMeanBiasDeviation(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([3, -0.5, 2, 7])
        self.y_pred = np.array([2.5, 0.0, 2, 8])

    def test_mean_bias_deviation_calculation(self):
        mean_bias_deviation_evaluator = MeanBiasDeviation(self.y_true, self.y_pred)
        result = mean_bias_deviation_evaluator.calculate()
        expected_result = 0.25
        self.assertAlmostEqual(result, expected_result, places=5)



if __name__ == '__main__':
    unittest.main()
