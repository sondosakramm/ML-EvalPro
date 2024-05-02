import numpy as np

from ml_eval_pro.model.evaluated_model import EvaluatedModel
from ml_eval_pro.utils.validation import convert_dataframe_to_numpy


class EvaluatedModelH2O(EvaluatedModel):
    """
    A class for generating the evaluated H2O model object.
    """

    def predict(self, data, predict_class=True):
        """
        Initializing the evaluation metric needed values.
        :param data: the data to be predicted.
        :param predict_class: indicating whether the prediction is a class prediction (in case of classification only).
        """
        print(f"Prediction class: {predict_class}")
        predictions = convert_dataframe_to_numpy(self.model.predict(data))

        if self.problem_type == "classification":
            return predictions[:, 0].reshape(-1,) if predict_class else predictions[:, 1:]

        elif self.problem_type == "regression":
            return predictions.reshape(-1,)

        return predictions
