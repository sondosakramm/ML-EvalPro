from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ml_eval_pro.evaluation_metrics.evaluators_factory import EvaluatorsFactory


class FeatureFairness(ABC):
    """
    An abstract class for input feature fairness.
    """

    def __init__(self, model, target: pd.Series, features: pd.DataFrame,
                 feature_name: str, evaluation_metric: str):
        """
        Initializing the feature bias needed inputs.
        :param model: the model.
        :param target: the target prediction values.
        :param features: the input feature values.
        :param feature_name: the input feature name.
        :param evaluation_metric: evaluation metric used for evaluating the module.
        """
        self.model = model
        self.target = target
        self.features = features
        self.feature_name = feature_name
        self.evaluation_metric = evaluation_metric

    def measure_fairness(self) -> float:
        categorical_features = self.get_categorical_features()
        metrics = self.__calculate_metrics(categorical_features)
        pairwise_differences = self.__calculate_pairwise_differences(metrics)
        avg_abs_performance = self.__calculate_average_absolute_performance(pairwise_differences)
        return avg_abs_performance

    @abstractmethod
    def get_categorical_features(self):
        pass

    @classmethod
    def __calculate_average_absolute_performance(cls, pairwise_differences: list) -> float:
        """
        Calculating the average absolute performance of a feature.
        :param pairwise_differences: the pairwise difference of the feature categories.
        :return:  and the average absolute performance of the feature.
        """
        return np.average(pairwise_differences)

    @classmethod
    def __calculate_pairwise_differences(cls, eval_metrics: list):
        """
        Calculating the average absolute performance of a feature.
        :param eval_metrics: the model performances divided by the categorical feature.
        :return: the pairwise difference of the feature categories.
        """
        pairwise_difference = []
        eval_metrics_size = len(eval_metrics)

        for i in range(eval_metrics_size):
            pairwise_difference.extend(
                [abs(eval_metrics[i] - eval_metrics[j]) for j in range(i + 1, eval_metrics_size)])
        return pairwise_difference

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

            eval_metrics.append(
                EvaluatorsFactory.get_evaluator(self.evaluation_metric,
                                                self.target[category_data_index].tolist(),
                                                category_predictions).measure())
        return eval_metrics
