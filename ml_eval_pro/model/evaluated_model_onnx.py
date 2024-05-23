import numpy as np
import mlflow

from ml_eval_pro.model.evaluated_model import EvaluatedModel
from ml_eval_pro.utils.validation import convert_dataframe_to_numpy


class EvaluatedModelONNX(EvaluatedModel):

    def predict(self, data, predict_class=True):
        """
        Initializing the evaluation metric needed values.
        :param data: the data to be predicted.
        :param predict_class: indicating whether the prediction is a class prediction (in case of classification only).
        """
        predictions = convert_dataframe_to_numpy(self.model.predict(data))

        if self.problem_type == "classification":
            # For binary classification with the probability of the positive class ONLY
            if not predict_class:
                predictions_proba = [[value for value in item[1].values()] for item in predictions]
                return np.array(predictions_proba)

            if predict_class:
                return np.array([pred[0] for pred in predictions])

        elif self.problem_type == "regression":
            if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                predictions = predictions.reshape(-1,)

        return predictions
