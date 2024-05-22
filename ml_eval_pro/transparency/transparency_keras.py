import keras
import numpy as np

from ml_eval_pro.transparency.transparency import Transparency


class TransparencyKeras(Transparency):
    """
    A class for generating transparency for keras models.
    """

    def get_model_algorithm(self):
        return self.model.model._model_impl.keras_model

    def get_model_score(self, model_algorithm, **kwargs):
        complexity_score = 0

        num_layers = len(model_algorithm.layers)
        complexity_score += num_layers * 10

        total_params = model_algorithm.count_params()
        complexity_score += np.log1p(total_params)

        for layer in model_algorithm.layers:
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.Conv3D)):
                complexity_score += 20
            elif isinstance(layer, keras.layers.LSTM):
                complexity_score += 30
            elif isinstance(layer, keras.layers.Dense):
                complexity_score += 5
            elif isinstance(layer, (keras.layers.BatchNormalization, keras.layers.Dropout)):
                complexity_score += 10

        return complexity_score
