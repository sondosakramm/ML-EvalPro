import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from ml_eval_pro.evaluation_metrics.classification.probability_evaluation.auc import AUC


class TestAUC(unittest.TestCase):

    def test_measure(self):

        target = [0, 1, 1, 0, 1]
        prediction = np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.25, 0.75]])

        auc = AUC(target, prediction)

        expected_auc = roc_auc_score(target, prediction[:,1])

        self.assertAlmostEqual(auc.measure(), expected_auc, places=6)


    def test_measure_multiclass(self):

        target = [0, 1, 1, 0, 2]
        prediction = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.6, 0.2], [0.7, 0.1, 0.2], [0.25, 0.35, 0.4]])

        auc = AUC(target, prediction, 3)

        expected_auc = roc_auc_score(target, prediction, multi_class="ovo")

        self.assertAlmostEqual(auc.measure(), expected_auc, places=6)


if __name__ == '__main__':
    unittest.main()
