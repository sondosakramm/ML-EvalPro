from abc import ABC, abstractmethod
from typing import Tuple

from ml_eval_pro.model.evaluated_model import EvaluatedModel


class Transparency(ABC):
    """
    An abstract base class for evaluating the transparency of a machine learning model.
    """

    def __init__(self, model: EvaluatedModel):
        """
        Initializes the Transparency class with the given evaluated model.

        Args:
            model (EvaluatedModel): The evaluated model instance.
        """
        self.model = model

    def get_model_transparency(self) -> str:
        """
        Evaluates and returns the transparency level of the model based on its complexity score.

        Returns:
            str: The transparency level of the model ("A", "B", "C", or "I").
        """
        model_algorithm = self.get_model_algorithm()
        explainable_models, partially_explainable_models, complex_models = self.get_model_algorithms_complexity()
        model_complexity_score = self.get_model_score(model_algorithm,
                                                      explainable_models=explainable_models,
                                                      partially_explainable_models=partially_explainable_models,
                                                      complex_models=complex_models)
        return self.__get_transparency_level(model_complexity_score)

    def get_model_algorithms_complexity(self) -> Tuple[list, list, list]:
        """
        Initialize lists representing explainable, partially explainable, and complex models.

        Returns:
            Tuple[list, list, list]: Three empty lists.
        """
        return [], [], []

    @abstractmethod
    def get_model_score(self, model_algorithm, **kwargs):
        """
        Abstract method to compute and return a complexity score for the model algorithm.

        Args:
            model_algorithm: The algorithm used by the model.
            **kwargs: Additional arguments for calculating the score.

        Returns:
            The complexity score of the model algorithm.
        """
        pass

    @abstractmethod
    def get_model_algorithm(self):
        """
        Abstract method to identify and return the model algorithm.

        Returns:
            The algorithm used by the model.
        """
        pass

    @classmethod
    def __get_transparency_level(cls, model_complexity_score):
        """
        Determines the transparency level based on the complexity score.

        Args:
            model_complexity_score: The complexity score of the model.

        Returns:
            str: The transparency level ("A", "B", "C", or "I").
                - "A": 0 <= model_complexity_score < 50
                - "B": 50 <= model_complexity_score < 100
                - "C": model_complexity_score >= 100
                - "I": Invalid score
        """
        if 0 <= model_complexity_score < 50:
            return "A"
        elif 50 <= model_complexity_score < 100:
            return "B"
        elif model_complexity_score >= 100:
            return "C"

        return "I"
