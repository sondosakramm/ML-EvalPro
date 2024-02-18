import pandas as pd
from pandas.core.dtypes.common import is_string_dtype

from ml_eval_pro.utils.feature_type_enum import FeatureTypeEnum


def check_feature_type(feature: pd.Series) -> FeatureTypeEnum:
    """
    Checks the feature type according to its type and number of unique values.
    :param feature: the input feature.
    :return: An enum indicating the feature type.
    """
    if is_string_dtype(feature) or feature.unique().size < int(feature.size / 2):
        return FeatureTypeEnum.CATEGORICAL
    return FeatureTypeEnum.NUMERICAL
