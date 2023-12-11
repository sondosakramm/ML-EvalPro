import unittest
from sklearn.metrics import accuracy_score

from auto_evaluator.evaluation_metrics.classification.class_evaluation.accuracy import Accuracy

class TestAccuracy(unittest.TestCase):

    def test_measure(self):
        target = [0, 2, 1, 3, 4]
        prediction = [0, 2, 1, 3, 4]

        accuracy = Accuracy(target, prediction)

        expected_accuracy = accuracy_score(accuracy.target, accuracy.prediction)

        self.assertEqual(accuracy.measure(), expected_accuracy)


if __name__ == '__main__':
    unittest.main()
