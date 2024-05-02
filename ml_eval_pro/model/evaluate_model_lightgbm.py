import numpy as np

from ml_eval_pro.model.evaluated_model import EvaluatedModel
from ml_eval_pro.utils.validation import convert_dataframe_to_numpy


class EvaluatedModelLightGBM(EvaluatedModel):

    def predict(self, data, predict_class=True):
        predictions = convert_dataframe_to_numpy(self.model.predict(data))
        if self.problem_type == "classification":
            if len(predictions.shape) == 1 or predictions.shape[1] == 1:
                predictions = np.concatenate([1 - predictions.reshape(-1, 1), predictions.reshape(-1, 1)], axis=1)

            if predict_class:
                return np.argmax(predictions, axis=1)

        return predictions