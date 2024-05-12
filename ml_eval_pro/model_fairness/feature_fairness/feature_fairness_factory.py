from ml_eval_pro.model_fairness.feature_fairness.categorical_fairness import CategoricalFairness
from ml_eval_pro.model_fairness.feature_fairness.feature_fairness import FeatureFairness
from ml_eval_pro.model_fairness.feature_fairness.numerical_fairness import NumericalFairness
from ml_eval_pro.utils.feature_type_enum import FeatureTypeEnum


class FeatureFairnessFactory:
    """
    A class for generating a feature fairness object.
    """

    @classmethod
    def create(cls, feature_type: str, *args, **kwargs) -> FeatureFairness:
        """
        Create a bias based on the feature type.
        :param feature_type: the input feature type.
        :return: the created feature fairness class according to its type.
        """
        _factory_supported_classes = {FeatureTypeEnum.NUMERICAL.value: NumericalFairness,
                                      FeatureTypeEnum.CATEGORICAL.value: CategoricalFairness}

        if feature_type in _factory_supported_classes:
            subclass = _factory_supported_classes.get(feature_type)
            return subclass(*args, **kwargs)
        else:
            raise Exception(f'Cannot find "{feature_type}"')