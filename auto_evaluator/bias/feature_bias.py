from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from auto_evaluator.evaluation_metrics.classification.class_evaluation.accuracy import Accuracy


class FeatureBias(ABC):
    def __init__(self, model, target:pd.Series, feature:pd.Series,
                 performance_metric: str='accuracy', significance:float=0.05):
        self.model = model
        self.target = target
        self.feature = feature
        self.performance_metric = performance_metric
        self.significance = significance

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def __calculate_categorical_metrics(self, categorical_features):
        categories = categorical_features.unique()
        eval_metrics = []

        for category in categories:
            category_data = self.feature[categorical_features == category]
            category_predictions = self.model.predict(category_data)
            # TODO: Replacing with a evaluation metric factory.
            eval_metrics.append(Accuracy(self.target, category_predictions))

        return eval_metrics

    @classmethod
    def __calculate_average_absolute_performance(cls, eval_metrics: list) -> tuple[list, float]:
        pairwise_difference = []
        eval_metrics_size = len(eval_metrics)

        for i in range(eval_metrics_size):
            pairwise_difference.extend([abs(eval_metrics[i] - eval_metrics[j]) for j in range(i+1, eval_metrics_size)])

        return pairwise_difference, np.average(pairwise_difference)
