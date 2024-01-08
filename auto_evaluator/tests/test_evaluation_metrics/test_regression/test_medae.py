import unittest

import numpy as np

from auto_evaluator.evaluation_metrics.regression.evaluation_metrics.medae import MEDAE


class TestMEDAE(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([3, -0.5, 2, 7])
        self.y_pred = np.array([2.5, 0.0, 2, 8])

    def test_medae_calculation(self):
        medae_evaluator = MEDAE(self.y_true, self.y_pred)
        result = medae_evaluator.measure()
        expected_result = 0.5
        self.assertAlmostEqual(result, expected_result, places=5)



if __name__ == '__main__':
    unittest.main()
