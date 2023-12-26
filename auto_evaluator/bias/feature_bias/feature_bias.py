from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd

from auto_evaluator.evaluation_metrics.classification.class_evaluation.accuracy import Accuracy


class FeatureBias(ABC):
    """
    An abstract class for input feature bias.
    """
    def __init__(self, model, target:pd.Series, feature:pd.Series,
                 performance_metric: str='accuracy', significance:float=0.05):
        """
        Initializing the feature bias needed inputs.
        :param model: the model.
        :param target: the target prediction values.
        :param feature: the input feature values.
        :param performance_metric: the performance metric used for measuring the bias.
        :param significance: the significance value to measure bias.
        """
        self.model = model
        self.target = target
        self.feature = feature
        self.performance_metric = performance_metric
        self.significance = significance

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def _calculate_categorical_metrics(self, categorical_features:pd.Series) -> list:
        """
        Calculating the categorical feature performance.
        :param categorical_features: the categorical feature used for measuring the performance.
        :return: the model performances divided by the categorical feature.
        """
        categories = categorical_features.unique()
        eval_metrics = []

        for category in categories:
            category_data = self.feature[categorical_features == category]
            category_predictions = self.model.predict(category_data)
            # TODO: Replacing with a evaluation metric factory.
            eval_metrics.append(Accuracy(self.target, category_predictions).measure())

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
            pairwise_difference.extend([abs(eval_metrics[i] - eval_metrics[j]) for j in range(i+1, eval_metrics_size)])

        return pairwise_difference, np.average(pairwise_difference)
