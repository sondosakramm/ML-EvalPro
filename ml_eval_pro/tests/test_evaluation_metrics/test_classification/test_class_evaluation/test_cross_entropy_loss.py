import unittest
import numpy as np
from scipy.special import softmax

from ml_eval_pro.evaluation_metrics.classification.class_evaluation.cross_entropy_loss import CrossEntropyLoss


class TestCrossEntropyLoss(unittest.TestCase):

    def test_measure(self):
        target = np.array([1, 0, 1])
        prediction = np.array([0.9, 0.1, 0.8])

        cross_entropy_loss = CrossEntropyLoss(target, prediction)

        softmax_vals = softmax(cross_entropy_loss.prediction)
        expected_loss = -np.sum(cross_entropy_loss.target * np.log(softmax_vals))

        self.assertAlmostEqual(cross_entropy_loss.measure(), expected_loss, places=6)


if __name__ == '__main__':
    unittest.main()
