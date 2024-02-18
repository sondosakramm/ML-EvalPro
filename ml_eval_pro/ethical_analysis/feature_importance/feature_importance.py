from abc import ABC, abstractmethod

import pandas as pd


class FeatureImportance(ABC):
    """
    An abstract class for different feature importance methods.
    """
    def __init__(self, model, data: pd.DataFrame):
        """
        Initializing the model feature importance method needed inputs.
        :param model: the model.
        :param data: the dataset containing all the features.
        """
        self.model = model
        self.data = data

    @abstractmethod
    def calculate(self):
        """
        An abstract method for calculating the feature importance according to the method implemented.
        """
        pass
