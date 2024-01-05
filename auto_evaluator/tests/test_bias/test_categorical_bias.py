import unittest
import pandas as pd
from unittest.mock import Mock

from auto_evaluator.bias.feature_bias.categorical_bias import CategoricalBias


class TestCategoricalBias(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock(predict=Mock(return_value=['A', 'A', 'A']))
        self.target = pd.Series(['A', 'B', 'C'])
        self.feature = pd.Series(['A', 'B', 'C'])
        self.bias = CategoricalBias(self.mock_model, self.target, self.feature)

    def test_call_method(self):
        avg_abs_performance, is_biased = self.bias()

        # Perform assertions for the returned values
        self.assertIsInstance(avg_abs_performance, float)
        self.assertTrue(type(is_biased), bool)

    def test_call_method_with_bias(self):
        # Creating a biased scenario by setting a significance level higher than performance
        self.bias.significance = 1.0

        avg_abs_performance, is_biased = self.bias()
        print(avg_abs_performance)

        # Ensure the model is biased as significance is set higher than performance
        self.assertTrue(is_biased)

    def test_call_method_without_bias(self):
        # Creating a scenario where there is no bias
        self.bias.significance = 0.01  # Set a very low significance level

        avg_abs_performance, is_biased = self.bias()

        # Ensure the model is not biased as significance is set lower than performance
        self.assertFalse(is_biased)
