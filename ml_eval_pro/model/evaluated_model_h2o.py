import numpy as np

from ml_eval_pro.model.evaluated_model import EvaluatedModel
from ml_eval_pro.utils.validation import convert_dataframe_to_numpy


class EvaluatedModelH2O(EvaluatedModel):

    def predict(self, data, predict_class=True):
        print(f"Prediction class: {predict_class}")
        predictions = convert_dataframe_to_numpy(self.model.predict(data))

        if self.problem_type == "classification":
            print(predictions)
            return predictions[:, 0].reshape(-1,) if predict_class else predictions[:, 1:]

        elif self.problem_type == "regression":
            print(predictions.reshape(-1,))
            return predictions.reshape(-1,)

        print(predictions)
        return predictions
