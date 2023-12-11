import unittest
import numpy as np
from unittest.mock import Mock

from auto_evaluator.predictor.normal_predictor import NormalPredictor


class TestNormalPredictor(unittest.TestCase):

    def setUp(self):
        # Create an instance of NormalPredictor and set up mock objects
        model_pipeline = Mock()
        features = np.array([[1, 2], [3, 4]])
        self.normal_predictor = NormalPredictor(model_pipeline, features)

    def test_predict_model(self):
        # Mock the model_pipeline.predict method to return a set of dummy predictions
        self.normal_predictor.model_pipeline.predict.return_value = np.array([0, 1])

        # Call the predict_model method
        predictions = self.normal_predictor.predict_model()

        # Check if the method returns the expected predictions
        self.assertTrue(np.array_equal(predictions, np.array([0, 1])))

    def test_predict_model_empty_features(self):
        # Set features to an empty array
        self.normal_predictor.features = np.array([])

        # Call the predict_model method
        predictions = self.normal_predictor.predict_model()

        # Check if the method returns an empty array when features are empty
        self.assertTrue(np.array_equal(predictions, np.array([])))

if __name__ == '__main__':
    unittest.main()
