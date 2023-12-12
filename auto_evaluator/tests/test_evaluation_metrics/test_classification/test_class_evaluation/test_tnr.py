import unittest
from sklearn.metrics import recall_score

from auto_evaluator.evaluation_metrics.classification.class_evaluation.tnr import TNR


class TestTNR(unittest.TestCase):

    def test_measure(self):
        target = [1, 0, 1, 1, 0, 1]
        prediction = [1, 0, 0, 1, 0, 1]

        tnr_instance = TNR(target, prediction)

        expected_tnr = recall_score(tnr_instance.target, tnr_instance.prediction, pos_label=0)

        self.assertAlmostEqual(tnr_instance.measure(), expected_tnr, places=6)


if __name__ == '__main__':
    unittest.main()
