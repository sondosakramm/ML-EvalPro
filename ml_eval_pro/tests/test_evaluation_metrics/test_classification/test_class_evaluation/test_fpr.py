import unittest
from sklearn.metrics import recall_score

from ml_eval_pro.evaluation_metrics.classification.class_evaluation.fpr import FPR


class TestFPR(unittest.TestCase):

    def test_measure(self):
        target = [1, 0, 1, 1, 0, 1]
        prediction = [1, 0, 0, 1, 0, 1]

        fpr_instance = FPR(target, prediction)

        expected_fpr = 1 - recall_score(fpr_instance.target, fpr_instance.prediction, pos_label=0)

        self.assertAlmostEqual(fpr_instance.measure(), expected_fpr, places=6)


if __name__ == '__main__':
    unittest.main()
