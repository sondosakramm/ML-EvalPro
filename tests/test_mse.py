import unittest

import numpy as np

from regression_metrics.mse import MSE


class TestMSE(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([3, -0.5, 2, 7])
        self.y_pred = np.array([2.5, 0.0, 2, 8])

    def test_mse_calculation(self):
        mse_evaluator = MSE(self.y_true, self.y_pred)
        result = mse_evaluator.calculate()
        expected_result = 0.375
        self.assertAlmostEqual(result, expected_result, places=5)



if __name__ == '__main__':
    unittest.main()
