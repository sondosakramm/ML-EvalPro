import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from auto_evaluator.bias.feature_bias.feature_bias import FeatureBias


class TestFeatureBias(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock(predict=Mock(return_value=np.array([1, 0, 1])))
        self.target = pd.Series([1, 0, 1])
        self.feature = pd.Series([0, 1, 0])
        self.bias = FeatureBias(self.mock_model, self.target, self.feature)

    def test_initialization(self):
        self.assertEqual(self.bias.model, self.mock_model)
        self.assertTrue((self.bias.target == self.target).all())
        self.assertTrue((self.bias.features == self.feature).all())
        self.assertEqual(self.bias.performance_metric, 'accuracy')
        self.assertEqual(self.bias.significance, 0.05)

    def test_calculate_categorical_metrics(self):
        categorical_features = pd.Series(['A', 'B', 'A'])
        eval_metrics = self.bias._FeatureBias_calculate_categorical_metrics(categorical_features)
        self.assertEqual(len(eval_metrics), 2)

    def test_calculate_average_absolute_performance(self):
        eval_metrics = [0.9, 0.8, 0.7]

        avg_difference = FeatureBias._FeatureBias_calculate_average_absolute_performance(eval_metrics)

        self.assertEqual(len(avg_difference[0]), 3)  # Assuming 3 metrics provided
        self.assertIsInstance(avg_difference[1], float)
