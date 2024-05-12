from typing import Tuple

from ml_eval_pro.model_transparency.model_transparency import ModelTransparency


class ModelTransparencyH2O(ModelTransparency):
    def get_model_algorithm(self):
        pass

    def get_model_algorithms_complexity(self) -> Tuple[list, list, list]:
        pass
