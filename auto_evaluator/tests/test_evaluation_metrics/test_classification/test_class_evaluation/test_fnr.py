import unittest
from sklearn.metrics import recall_score

from auto_evaluator.evaluation_metrics.classification.class_evaluation.fnr import FNR


class TestFNR(unittest.TestCase):

    def test_measure(self):
        target = [1, 0, 1, 1, 0, 1]
        prediction = [1, 0, 0, 1, 0, 1]

        fnr_instance = FNR(target, prediction)

        expected_fnr = 1 - recall_score(fnr_instance.target, fnr_instance.prediction)

        self.assertAlmostEqual(fnr_instance.measure(), expected_fnr, places=6)


if __name__ == '__main__':
    unittest.main()
