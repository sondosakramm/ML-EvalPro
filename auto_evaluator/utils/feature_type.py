import pandas as pd

from auto_evaluator.utils.feature_type_enum import FeatureTypeEnum


def check_feature_type(feature:pd.Series) -> FeatureTypeEnum:
    """
    Checks the feature type according to its type and number of unique values.
    :param feature: the input feature.
    :return: An enum indicating the feature type.
    """
    if feature.astype(object) and feature.unique().size < feature.size:
        return FeatureTypeEnum.CATEGORICAL
    return FeatureTypeEnum.NUMERICAL
