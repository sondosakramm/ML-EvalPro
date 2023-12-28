from auto_evaluator.bias.feature_bias.categorical_bias import CategoricalBias
from auto_evaluator.bias.feature_bias.feature_bias import FeatureBias
from auto_evaluator.bias.feature_bias.numerical_bias import NumericalBias
from auto_evaluator.utils.feature_type_enum import FeatureTypeEnum


class FeatureBiasFactory:
    """
    A class for generating a feature bias object.
    """

    @classmethod
    def create(cls, feature_type: str, *args, **kwargs) -> FeatureBias:
        """
        Create a bias based on the feature type.
        :param feature_type: the input feature type.
        :return: the created feature bias class according to its type.
        """
        _factory_supported_classes = {FeatureTypeEnum.NUMERICAL.value: NumericalBias,
                                      FeatureTypeEnum.CATEGORICAL.value: CategoricalBias}

        if feature_type in _factory_supported_classes:
            subclass = _factory_supported_classes.get(feature_type)
            return subclass(*args ,**kwargs)
        else:
            raise Exception(f'Cannot find "{feature_type}"')