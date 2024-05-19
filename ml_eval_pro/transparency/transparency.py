from abc import ABC, abstractmethod
from typing import Tuple

from ml_eval_pro.model.evaluated_model import EvaluatedModel


class Transparency(ABC):
    def __init__(self, model: EvaluatedModel):
        self.model = model

    def get_model_transparency(self) -> str:
        model_algorithm = self.get_model_algorithm()
        explainable_models, partially_explainable_models, complex_models = self.get_model_algorithms_complexity()
        return self.__get_transparency_level(model_algorithm, explainable_models, partially_explainable_models,
                                             complex_models)

    @abstractmethod
    def get_model_algorithms_complexity(self) -> Tuple[list, list, list]:
        pass

    @abstractmethod
    def get_model_algorithm(self):
        pass

    @classmethod
    def __get_transparency_level(cls, model_algorithm, explainable_models,
                                 partially_explainable_models, complex_models):
        if any(isinstance(model_algorithm, model_type) for model_type in explainable_models):
            return "A"
        elif any(isinstance(model_algorithm, model_type) for model_type in partially_explainable_models):
            return "B"
        elif any(isinstance(model_algorithm, model_type) for model_type in complex_models):
            return "C"
        else:
            return "I"
