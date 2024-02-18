import unittest
from sklearn.metrics import recall_score

from ml_eval_pro.evaluation_metrics.classification.class_evaluation.fpr import FPR
from ml_eval_pro.evaluation_metrics.classification.class_evaluation.tpr import TPR


class TestTPR(unittest.TestCase):

    def test_measure(self):
        target = [1, 0, 1, 1, 0, 1]
        prediction = [1, 0, 0, 1, 0, 1]

        tpr_instance = TPR(target, prediction)

        expected_tpr = recall_score(tpr_instance.target, tpr_instance.prediction)

        self.assertAlmostEqual(tpr_instance.measure(), expected_tpr, places=6)


if __name__ == '__main__':
    unittest.main()
