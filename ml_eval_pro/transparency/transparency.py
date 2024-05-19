from abc import ABC, abstractmethod
from typing import Tuple

from ml_eval_pro.model.evaluated_model import EvaluatedModel


class Transparency(ABC):
    def __init__(self, model: EvaluatedModel):
        self.model = model

    def get_model_transparency(self) -> str:
        model_algorithm = self.get_model_algorithm()
        explainable_models, partially_explainable_models, complex_models = self.get_model_algorithms_complexity()
        model_complexity_score = self.get_model_score(model_algorithm,
                                                      explainable_models=explainable_models,
                                                      partially_explainable_models=partially_explainable_models,
                                                      complex_models=complex_models)
        return self.__get_transparency_level(model_complexity_score)

    def get_model_algorithms_complexity(self) -> Tuple[list, list, list]:
        return [], [], []

    @abstractmethod
    def get_model_score(self, model_algorithm, **kwargs):
        pass

    @abstractmethod
    def get_model_algorithm(self):
        pass

    @classmethod
    def __get_transparency_level(cls, model_complexity_score):
        if 0 <= model_complexity_score < 50:
            return "A"
        elif 50 <= model_complexity_score < 100:
            return "B"
        elif model_complexity_score >= 100:
            return "C"

        return "I"
