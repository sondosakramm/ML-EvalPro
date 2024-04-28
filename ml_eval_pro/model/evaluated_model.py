# from abc import ABC, abstractmethod

import mlflow
import numpy as np


class EvaluatedModel:
    def __init__(self, model_uri, problem_type):
        self.model_uri = model_uri
        self.model = self.load()
        self.problem_type = problem_type

    def load(self):
        print(f"Loading the model ...")
        return mlflow.pyfunc.load_model(model_uri=self.model_uri)

    def predict(self, data, predict_class=True):
        if self.problem_type == "classification" and predict_class:
            return np.argmax(mlflow.pyfunc.predict(data).to_numpy(), axis=1)

        return mlflow.pyfunc.predict(data)
