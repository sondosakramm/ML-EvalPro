import unittest
import pandas as pd
from unittest.mock import Mock

from auto_evaluator.bias.model_bias import ModelBias


class TestModelBias(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock(predict=Mock(return_value=[1, 0, 1]))
        self.data = pd.DataFrame({'feature_1': [1, 2, 3], 'feature_2': ['A', 'B', 'A']})
        self.target = pd.Series([1, 0, 1])
        self.bias = ModelBias(self.mock_model, self.data, self.target)

    def test_call_method(self):
        features_bias = self.bias()

        # Perform assertions for the returned values
        self.assertIsInstance(features_bias, list)
        self.assertEqual(len(features_bias), 2)  # Assuming 2 features in the test dataset

        for feature_bias in features_bias:
            self.assertIsInstance(feature_bias, tuple)
            self.assertEqual(len(feature_bias), 2)
            self.assertIsInstance(feature_bias[0], float)
            self.assertIsInstance(feature_bias[1], bool)

    def test_call_method_with_mocked_bias(self):
        with unittest.mock.patch('auto_evaluator.bias.feature_bias_factory.FeatureBiasFactory.create') as mock_create:
            mock_create.return_value.return_value = (0.8, False)

            features_bias = self.bias()

            self.assertEqual(features_bias, [(0.8, False), (0.8, False)])
