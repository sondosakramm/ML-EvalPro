from abc import ABC, abstractmethod


class Predictor(ABC):
    """
    An abstract class for generating predictions from a set of features of a model.
    """
    def __init__(self, features, model_pipeline):
        """
        Initializing the values needed for prediction.
        :param features: the input features.
        :param model_pipeline: the model pipeline.
        """
        self.features = features
        self.model_pipeline = model_pipeline


    @abstractmethod
    def predict_model(self):
        pass