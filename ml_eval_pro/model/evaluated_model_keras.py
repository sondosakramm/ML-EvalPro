# from abc import ABC, abstractmethod

import mlflow
import numpy as np

from ml_eval_pro.model.evaluated_model import EvaluatedModel
from ml_eval_pro.utils.validation import convert_dataframe_to_numpy


class EvaluatedModelKeras(EvaluatedModel):
    """
    A class for generating the evaluated model object.
    """
    def load(self):
        """
        Loading the model from the PythonFunc flavor.
        """
        print(f"Loading the model ...")
        return mlflow.keras.load_model(model_uri=self.model_uri)
