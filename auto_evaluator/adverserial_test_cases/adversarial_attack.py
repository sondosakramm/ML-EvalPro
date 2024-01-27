from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from art.estimators.regression.scikitlearn import ScikitlearnRegressor
from art.estimators.classification import SklearnClassifier

import numpy as np

class AdversarialAttack(ABC):
    """
    An abstract class for generating an adversarial attack.
    """
    def __init__(self, model, model_type:Tuple[str, str], test_input_features, test_target_features,
                 train_input_features=None, train_target_features=None,
                 significance=0.05, num_classes=None):
        """
        Create an adversarial attack.
        :param model: the model to be tested against adversarial attacks.
        :param model_type: the model type i.e. Regression, Classification, etc.
        :param test_input_features: the input features of the test dataset.
        :param test_target_features: the target feature(s) of the test dataset.
        :param train_input_features: the input features of the train dataset.
        :param train_target_features: the target feature(s) of the train dataset.
        :param significance: the significance value of the evaluation to the defined threshold.
        :param num_classes: the number of classes of a classification model.
        :return: the created adversarial attack.
        """
        self.model_type = model_type
        self.model = model
        self.robust = None
        self.score = None

        # Transformation to numpy array if the data is pandas.
        if isinstance(train_input_features, pd.DataFrame):
            train_input_features = train_input_features.to_numpy()

        if isinstance(train_target_features, pd.DataFrame):
            train_target_features = train_target_features.to_numpy()

        if isinstance(test_input_features, pd.DataFrame):
            test_input_features = test_input_features.to_numpy()

        if isinstance(test_target_features, pd.DataFrame):
            test_target_features = test_target_features.to_numpy()

        self.test_input_features = test_input_features
        self.test_target_features = test_target_features

        self.train_input_features = train_input_features
        self.train_target_features = train_target_features

        self.significance = significance

        # Splitting the test data if there is no training data
        if not isinstance(self.train_input_features, np.ndarray):
            split_size = int(0.8 * self.test_target_features.shape[0])
            self.train_input_features = self.test_input_features[:split_size]
            self.train_target_features = self.test_target_features[:split_size]
            self.test_input_features = self.test_input_features[split_size:]
            self.test_target_features = self.test_target_features[split_size:]

        self.train_model_predictions = self.model.predict(self.train_input_features)
        self.test_model_predictions = self.model.predict(self.test_input_features)

        # Get the number of classes from the dataset given if the number of classes is not given
        self.num_classes = num_classes
        if self.model_type[0] == 'classification' and num_classes is None:
            self.num_classes = np.unique(self.train_target_features).shape[0]


    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Generate adversarial attack test cases from the test data.
        :return: the created adversarial attack test cases.
        """
        pass

    @abstractmethod
    def evaluate_robustness(self, adversarial_examples):
        """
        Evaluate the model robustness to adversarial attacks.
        :param adversarial_examples: the generated adversarial examples.
        :return: a flag indicating whether the model is robust to adversarial attacks.
        """
        pass

    def __str__(self):
        robust_str = "robust" if self.robust else "not robust"

        return (f'According to the results of evaluating the model predictions of the original examples'
                f' to the adversarial examples, the model is {robust_str} to adversarial attacks with score {self.score * 100}%')