import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.cluster import KMeans

from auto_evaluator.bias.feature_bias.numerical_bias import NumericalBias


class TestNumericalBias(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock(predict=Mock(return_value=np.array([1, 0, 1])))
        self.target = pd.Series([1, 0, 1])
        self.feature = pd.Series([0, 1, 0])
        self.bias = NumericalBias(self.mock_model, self.target, self.feature)

    def test_call_method(self):
        # Mocking KMeans clustering to return fixed clusters for predictable test results
        with patch.object(KMeans, 'fit_predict') as mock_kmeans_fit_predict:
            mock_kmeans_fit_predict.return_value = np.array([0, 1, 0])  # Mocking clusters for feature

            avg_abs_performance, is_biased = self.bias()

            # Ensure the correct data is passed to KMeans clustering
            mock_kmeans_fit_predict.assert_called_once_with(self.feature.values.reshape(-1, 1))

            # Perform assertions for the returned values
            self.assertIsInstance(avg_abs_performance, float)
            self.assertIsInstance(is_biased, bool)

    def test_call_method_with_bias(self):
        # Creating a biased scenario by setting a significance level higher than performance
        self.bias.significance = 1.0

        # Mocking KMeans clustering to return clusters where there is a significant bias
        with patch.object(KMeans, 'fit_predict') as mock_kmeans_fit_predict:
            mock_kmeans_fit_predict.return_value = np.array([0, 0, 1])  # Mocking clusters for significant bias

            avg_abs_performance, is_biased = self.bias()

            # Ensure the model is biased as significance is set higher than performance
            self.assertTrue(is_biased)

    def test_call_method_without_bias(self):
        # Creating a scenario where there is no bias
        self.bias.significance = 0.01  # Set a very low significance level

        # Mocking KMeans clustering to return clusters where there is no significant bias
        with patch.object(KMeans, 'fit_predict') as mock_kmeans_fit_predict:
            mock_kmeans_fit_predict.return_value = np.array([0, 0, 0])  # Mocking clusters for no significant bias

            avg_abs_performance, is_biased = self.bias()

            # Ensure the model is not biased as significance is set lower than performance
            self.assertFalse(is_biased)