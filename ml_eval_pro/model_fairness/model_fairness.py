from abc import ABC, abstractmethod

import pandas as pd

from ml_eval_pro.model_fairness.feature_fairness.feature_fairness_factory import FeatureFairnessFactory
from ml_eval_pro.utils.feature_type import check_feature_type


class ModelFairness(ABC):
    """
    A class for measuring the model fairness.
    """

    def __init__(self, model, model_type: str, data: pd.DataFrame, target: pd.Series,
                 evaluation_metrics: [str]):
        """
        Initializing the model fairness needed inputs.
        :param model: the model.
        :param model_type: the model type.
        :param data: the dataset containing all the features.
        :param target: the target values.
        :param evaluation_metrics: the evaluation metric used to measure fairness.
        """
        self.model = model
        self.model_type = model_type
        self.data = data
        self.target = target
        self.evaluation_metrics = evaluation_metrics

    def get_model_fairness(self):
        """
        Calculating the model fairness values.
        """
        features_names = self.get_features_names()
        features_abs_avg_performance = self.__get_all_features_abs_avg_performance(features_names)
        return self.execute_post_steps(features_abs_avg_performance)

    @abstractmethod
    def get_features_names(self):
        pass

    def execute_post_steps(self, features_abs_avg_performance):
        pass

    def __get_all_features_abs_avg_performance(self, features_names) -> pd.DataFrame:
        """
        Calculating the model fairness values.
        """
        features_fairness = {}

        for feature_name in features_names:
            feature_fairness = self.__get_feature_abs_avg_performance(feature_name)
            features_fairness[feature_name] = feature_fairness

        return pd.DataFrame(features_fairness)

    def __get_feature_abs_avg_performance(self, feature_name) -> list:
        feature_fairness_res = []

        for evaluation_metric in self.evaluation_metrics:
            feature_type = check_feature_type(self.data[feature_name])

            fairness = FeatureFairnessFactory.create(feature_type.value, self.model, self.target,
                                                     self.data, feature_name, evaluation_metric)

            feature_fairness = fairness.measure_fairness()
            feature_fairness_res.append(feature_fairness)

        return feature_fairness_res
