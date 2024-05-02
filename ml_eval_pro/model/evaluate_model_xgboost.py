import numpy as np

from ml_eval_pro.model.evaluated_model import EvaluatedModel
from ml_eval_pro.utils.validation import convert_dataframe_to_numpy


class EvaluatedModelXGBoost(EvaluatedModel):

    def predict(self, data, predict_class=True):
        return self.model.predict(data)