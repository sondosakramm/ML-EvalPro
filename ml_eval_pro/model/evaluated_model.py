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

        if self.problem_type == "classification":
            # For binary classification with the probability of the positive class ONLY
            if len(predictions.shape) == 1 or predictions.shape[1] == 1:
                predictions = np.concatenate([1-predictions, predictions], axis=1)

            if predict_class:
                return np.argmax(predictions, axis=1)

        elif self.problem_type == "regression":
            if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                predictions = predictions.reshape(-1,)

        return predictions
