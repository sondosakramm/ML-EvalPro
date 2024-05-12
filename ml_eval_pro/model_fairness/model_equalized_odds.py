import pandas as pd

from ml_eval_pro.model_fairness.model_fairness import ModelFairness
from ml_eval_pro.utils.feature_type import check_feature_type
from ml_eval_pro.utils.feature_type_enum import FeatureTypeEnum


class ModelEqualizedOdds(ModelFairness):
    """
    A class for measuring the model equalized odds.
    """
    def __init__(self, model, model_type: str, data: pd.DataFrame, target: pd.Series):
        """
        Initializing the model fairness needed inputs.
        :param model: the model.
        :param model_type: the model type.
        :param data: the dataset containing all the features.
        :param target: the target values.
        """
        if model_type == 'regression':
            raise TypeError("Cannot calculate equalized odds for regression problems!")

        super().__init__(model, model_type, data, target,
                         ["True Positive Rate", "False Positive Rate"])

    def get_features_names(self):
        features_names = self.data.columns.tolist()
        cat_features_names = []

        for feature_name in features_names:
            feature_type = check_feature_type(self.data[feature_name])

            if feature_type == FeatureTypeEnum.CATEGORICAL:
                cat_features_names.append(feature_name)

        return cat_features_names

    def execute_post_steps(self, features_abs_avg_performance: pd.DataFrame):
        avg_eval_metrics = features_abs_avg_performance.mean(axis=0)
        return avg_eval_metrics.to_dict()
