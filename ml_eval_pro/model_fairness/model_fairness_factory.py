from ml_eval_pro.model_fairness.model_bias import ModelBias
from ml_eval_pro.model_fairness.model_equalized_odds import ModelEqualizedOdds
from ml_eval_pro.model_fairness.model_fairness import ModelFairness


class ModelFairnessFactory:
    """
    A class for generating a model fairness object.
    """

    @classmethod
    def create(cls, fairness_type: str, *args, **kwargs) -> ModelFairness:
        """
        Create a fairness based on the type.
        :param fairness_type: the input fairness type.
        :return: the created fairness class according to its type.
        """
        _factory_supported_classes = {"bias": ModelBias,
                                      "equalized odds": ModelEqualizedOdds}

        if fairness_type in _factory_supported_classes:
            subclass = _factory_supported_classes.get(fairness_type)
            return subclass(*args, **kwargs)
        else:
            raise Exception(f'Cannot find "{fairness_type}"')

