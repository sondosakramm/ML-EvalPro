# from abc import ABC, abstractmethod

import mlflow
import numpy as np

from ml_eval_pro.utils.validation import convert_dataframe_to_numpy


class EvaluatedModel:
    def __init__(self, model_uri, problem_type):
        self.model_uri = model_uri
        self.model = self.load()
        self.problem_type = problem_type

    def load(self):
        print(f"Loading the model ...")
        return mlflow.pyfunc.load_model(model_uri=self.model_uri)

    def predict(self, data, predict_class=True):
        predictions = convert_dataframe_to_numpy(self.model.predict(data))

        if self.problem_type == "classification" and predict_class:

            if predictions.shape[1] == 1:
                return ((predictions >= 0.5) * 1).reshape(-1,)

            return np.argmax(predictions, axis=1)

        return predictions.reshape(-1,)
