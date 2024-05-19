# from abc import ABC, abstractmethod

import mlflow
import numpy as np

from ml_eval_pro.model.evaluated_model import EvaluatedModel
from ml_eval_pro.utils.validation import convert_dataframe_to_numpy


class EvaluatedModelTensorflow(EvaluatedModel):
    """
    A class for generating the evaluated model object.
    """
    def load(self):
        """
        Loading the model from the PythonFunc flavor.
        """
        print(f"Loading the model ...")
        return mlflow.tensorflow.load_model(model_uri=self.model_uri)
