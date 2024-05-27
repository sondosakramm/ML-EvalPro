import mlflow
import numpy as np

from ml_eval_pro.model.evaluated_model import EvaluatedModel
from ml_eval_pro.utils.validation import convert_dataframe_to_numpy


class EvaluatedModelPyTorch(EvaluatedModel):
    """
    A class for generating the evaluated model object.
    """
    def load(self):
        """
        Loading the model from the PythonFunc flavor.
        """
        print(f"Loading the model ...")
        return mlflow.pytorch.load_model(model_uri=self.model_uri)
