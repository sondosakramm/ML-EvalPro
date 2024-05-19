from typing import Tuple

from ml_eval_pro.transparency.transparency import Transparency


class TransparencyH2O(Transparency):
    def get_model_algorithm(self):
        pass

    def get_model_algorithms_complexity(self) -> Tuple[list, list, list]:
        pass
