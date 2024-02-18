import unittest
from abc import ABC

from ml_eval_pro.evaluation_metrics.classification.class_evaluation.class_evaluation import ClassClassification
from ml_eval_pro.evaluation_metrics.classification.classification_evaluation import ClassificationEvaluation


class TestClassClassification(unittest.TestCase):

    def test_inheritance(self):

        self.assertTrue(issubclass(ClassClassification, ClassificationEvaluation))
        self.assertTrue(issubclass(ClassClassification, ABC))

    def test_instantiation(self):
        target = [0, 2, 1, 3, 4]
        prediction = [0, 2, 1, 3, 4]
        with self.assertRaises(TypeError):
            obj = ClassClassification(target, prediction)

if __name__ == '__main__':
    unittest.main()
