from typing import Tuple, List

from ml_eval_pro.transparency.transparency import Transparency


class TransparencyLightGBM(Transparency):
    """
    A class for generating transparency for lightgbm models.
    """

    def get_model_algorithm(self):
        pass

    def get_model_score(self, model_algorithm, **kwargs):
        return 110
