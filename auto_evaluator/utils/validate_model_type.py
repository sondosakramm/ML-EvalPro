import numpy as np
import pandas as pd
from sklearn.base import is_regressor, is_classifier

from auto_evaluator.utils.feature_type import check_feature_type
from auto_evaluator.utils.feature_type_enum import FeatureTypeEnum


def check_model_type(model_target):
    target_type = check_feature_type(model_target)
    if target_type == FeatureTypeEnum.NUMERICAL:
        return "regression"
    elif target_type == FeatureTypeEnum.CATEGORICAL:
        return "classification"
    else:
        raise ValueError("Other model are not supported yet!")


def get_num_classes(model_type: str, test_target: pd.Series):
    if model_type == 'classification':
        return test_target.unique().size
    else:
        raise ValueError("Not applicable to non-classification model!")
