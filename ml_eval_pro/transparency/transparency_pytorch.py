import numpy as np
import torch.nn as nn

from ml_eval_pro.transparency.transparency import Transparency


class TransparencyPyTorch(Transparency):
    """
    A class for generating transparency for pytorch models.
    """

    def get_model_algorithm(self):
        return self.model.model._model_impl.pytorch_model

    def get_model_score(self, model_algorithm, **kwargs):
        complexity_score = 0

        num_layers = len(list(model_algorithm.children()))
        complexity_score += num_layers * 10

        total_params = sum(p.numel() for p in model_algorithm.parameters())
        complexity_score += np.log1p(total_params)

        for layer in model_algorithm.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv3d)):
                complexity_score += 20
            elif isinstance(layer, nn.LSTM):
                complexity_score += 30
            elif isinstance(layer, nn.Linear):
                complexity_score += 5
            elif isinstance(layer, (nn.BatchNorm2d, nn.Dropout)):
                complexity_score += 10

        return complexity_score
