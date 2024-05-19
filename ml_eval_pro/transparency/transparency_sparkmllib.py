from typing import Tuple

from ml_eval_pro.transparency.transparency import Transparency


class TransparencySparkML(Transparency):
    def get_model_algorithm(self):
        return self.model.__dict__["model"]

    def get_model_score(self, model_algorithm, **kwargs):
        pass
