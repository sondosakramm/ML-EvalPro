import unittest

import numpy as np

from auto_evaluator.evaluation_metrics.regression.mae import MAE


class TestMAE(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([3, -0.5, 2, 7])
        self.y_pred = np.array([2.5, 0.0, 2, 8])

    def test_mae_calculation(self):
        mae_evaluator = MAE(self.y_true, self.y_pred)
        result = mae_evaluator.measure()
        expected_result = 0.5
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_mae_with_invalid_input(self):
        invalid_y_true = np.array([1, 2, 3])
        invalid_y_pred = np.array([4, 5, 6, 7])
        with self.assertRaises(ValueError):
            mae_evaluator = MAE(invalid_y_true, invalid_y_pred)
            mae_evaluator.measure()


if __name__ == '__main__':
    unittest.main()
