from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd

from ml_eval_pro.evaluation_metrics.evaluators_factory import EvaluatorsFactory


class FeatureBias(ABC):
    """
    An abstract class for input feature bias.
    """

    def __init__(self, model, model_type: str, target: pd.Series, features: pd.DataFrame,
                 feature_name: str, significance: float = 0.05):
        """
        Initializing the feature bias needed inputs.
        :param model: the model.
        :param model_type: the model type.
        :param target: the target prediction values.
        :param features: the input feature values.
        :param feature_name: the input feature name.
        :param significance: the significance value to measure bias.
        """
        self.model = model
        self.model_type = model_type
        self.target = target
        self.features = features
        self.feature_name = feature_name
        self.significance = significance

    @abstractmethod
    def check_bias(self):
        """
        Calculating the bias of a single feature. :return: the average absolute performances and a boolean indicating
        if the model is biased according to that feature.
        """
        pass

    def _check_feature_bias(self, categorical_feature: pd.Series):
        """
        Calculating the bias of a single feature.
        :param categorical_feature: the categorical feature used for
        measuring the performance.
        :return: the average absolute performances and a boolean indicating if the model
        is biased according to that feature.
        """
        eval_metrics = self.__calculate_metrics(categorical_feature)
        pairwise_diff, avg_abs_performance = FeatureBias._calculate_average_absolute_performance(eval_metrics)

        return [self.feature_name, avg_abs_performance, avg_abs_performance >= self.significance]

    def __calculate_metrics(self, categorical_feature: pd.Series) -> list:
        """
        Calculating the feature performance.
        :param categorical_feature: the categorical feature used for measuring the performance.
        :return: the model performances divided by the categorical feature.
        """
        categories = categorical_feature.unique()
        eval_metrics = []

        for category in categories:
            category_data = self.features[categorical_feature == category]
            category_data_index = category_data.index.tolist()
            category_predictions = self.model.predict(category_data)

            if self.model_type == 'regression':
                eval_metrics.append(EvaluatorsFactory.get_evaluator('mape', self.target[category_data_index].tolist(),
                                                                    category_predictions).measure() / 100)
            elif self.model_type == 'classification':
                eval_metrics.append(
                    EvaluatorsFactory.get_evaluator('accuracy', self.target[category_data_index].tolist(),
                                                    category_predictions).measure())

        return eval_metrics

    @classmethod
    def _calculate_average_absolute_performance(cls, eval_metrics: list) -> Tuple[list, float]:
        """
        Calculating the average absolute performance of a feature.
        :param eval_metrics: the model performances divided by the categorical feature.
        :return: the pair-wise difference of the feature categories and the average absolute performance of the feature.
        """
        pairwise_difference = []
        eval_metrics_size = len(eval_metrics)

        for i in range(eval_metrics_size):
            pairwise_difference.extend(
                [abs(eval_metrics[i] - eval_metrics[j]) for j in range(i + 1, eval_metrics_size)])

        return pairwise_difference, np.average(pairwise_difference)
