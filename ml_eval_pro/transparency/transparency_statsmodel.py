from typing import Tuple, List
from ml_eval_pro.transparency.transparency import Transparency


class ModelTransparencyStats(Transparency):

    def get_model_algorithm(self):
        model_class = type(self.model.model.model).__name__
        return model_class

    def get_model_algorithms_complexity(self) -> Tuple[List[str], List[str], List[str]]:
        explainable_models = ['OLS', 'GLS', 'WLS']
        partially_explainable_models = ['GLM']
        complex_models = ['MixedLM', 'GEE']

        return explainable_models, partially_explainable_models, complex_models

    def get_model_score(self, model_algorithm, **kwargs):
        explainable_models = kwargs["explainable_models"]
        partially_explainable_models = kwargs["partially_explainable_models"]
        complex_models = kwargs["complex_models"]

        if model_algorithm in explainable_models:
            return 25
        elif model_algorithm in partially_explainable_models:
            return 65
        elif model_algorithm in complex_models:
            return 110

        return -1
