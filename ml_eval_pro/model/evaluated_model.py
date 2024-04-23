from abc import ABC, abstractmethod

import mlflow


class EvaluatedModel(ABC):
    def __init__(self, model_uri, problem_type, model_type):
        self.model = EvaluatedModel.load(model_uri)
        self.problem_type = problem_type
        self.model_type = model_type

    @classmethod
    def load(cls, model_uri):
        return mlflow.pyfunc.load_model(model_uri=model_uri)

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def predict_class(self, data):
        pass
