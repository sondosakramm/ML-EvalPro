import unittest

import numpy as np

from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.r_square_error import RSquare


class TestRSquare(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([3, -0.5, 2, 7])
        self.y_pred = np.array([2.5, 0.0, 2, 8])

    def test_r_square_calculation(self):
        r_square_evaluator = RSquare(self.y_true, self.y_pred)
        result = r_square_evaluator.measure()
        expected_result = 0.9486081370449679
        self.assertAlmostEqual(result, expected_result, places=5)



if __name__ == '__main__':
    unittest.main()
