import unittest
from sklearn.metrics import f1_score

from auto_evaluator.evaluation_metrics.classification.class_evaluation.f1_score import F1Score


class TestF1Score(unittest.TestCase):

    def test_measure(self):
        target = [1, 0, 1, 1, 0, 1]
        prediction = [1, 0, 0, 1, 0, 1]

        f1_score_instance = F1Score(target, prediction)
        expected_f1 = f1_score(f1_score_instance.target, f1_score_instance.prediction)

        self.assertAlmostEqual(f1_score_instance.measure(), expected_f1, places=6)


if __name__ == '__main__':
    unittest.main()
