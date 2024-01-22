from abc import ABC, abstractmethod

from art.estimators.regression.scikitlearn import ScikitlearnRegressor
from art.estimators.classification import SklearnClassifier

import numpy as np

class AdversarialAttack(ABC):
    """
    An abstract class for generating an adversarial attack.
    """
    def __init__(self, model, model_type:str, input_features:np.ndarray, target_features:np.ndarray):
        """
        Create an adversarial attack.
        :param model: the model to be tested against adversarial attacks.
        :param model_type: the model type i.e. Regression, Classification, etc.
        :param input_features: the input features of the dataset.
        :param target_features: the target feature(s) of the dataset.
        :return: the created adversarial attack.
        """
        self.model_type = model_type
        self.model = model
        self.input_features = input_features
        self.target_features = target_features


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


    @classmethod
    def _generate_adversarial_attack_model(cls, model_type:str, **kwargs):
        supported_model_types = {
            'regression': ScikitlearnRegressor,
            'classification': SklearnClassifier,
        }

        if model_type in supported_model_types:
            subclass = supported_model_types.get(model_type)
            return subclass(**kwargs)
        else:
            raise Exception(f'Cannot find "{model_type}"')


    def _get_model_type(self):
        model_type = type(self.model).__name__
        if model_type == 'LinearRegression':
            return ['regression', 'lr']
        elif model_type == 'LogisticRegression':
            return ['classification', 'lr']
        elif model_type == 'SVR':
            return ['regression', 'svm']
        elif model_type == 'SVC':
            return ['classification', 'svm']
        elif model_type == 'RandomForestRegressor':
            return ['regression', 'rf']
        elif model_type == 'RandomForestClassifier':
            return ['classifier', 'rf']
        elif model_type == 'DecisionTreeRegressor':
            return ['regression', 'dt']
        elif model_type == 'DecisionTreeClassifier':
            return ['classifier', 'dt']
        elif model_type == 'KNeighborsRegressor':
            return ['regression', 'knn']
        elif model_type == 'KNeighborsClassifier':
            return ['classifier', 'knn']
        else:
            raise ValueError(f'Model {model_type} is not supported yet.')