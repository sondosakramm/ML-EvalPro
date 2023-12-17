import pandas as pd

from auto_evaluator.bias.categorical_bias import CategoricalBias
from auto_evaluator.bias.numerical_bias import NumericalBias
from auto_evaluator.utils.feature_type import check_feature_type


class ModelBias:
    def __init__(self, model, data: pd.DataFrame, target, performance_metric='accuracy',
                 significance: float = 0.05, no_of_clusters: int = 5):
        self.model = model
        self.data = data
        self.target = target
        self.performance_metric = performance_metric
        self.significance = significance
        self.no_of_clusters = no_of_clusters


    def __call__(self, *args, **kwargs):
        features_names = self.data.columns.tolist()
        features_bias = []
        for feature_name in features_names:
            # TODO: Model feature factory
            feature_type = check_feature_type(self.data[feature_name])
            if feature_type.NUMERICAL:
                numerical_bias = NumericalBias(self.model, self.target, self.data[feature_name],
                                               self.performance_metric, self.significance, self.no_of_clusters)
                features_bias.append(numerical_bias())
            else:
                categorical_bias = CategoricalBias(self.model, self.target, self.data[feature_name],
                                                   self.performance_metric, self.significance)
                features_bias.append(categorical_bias())

        return features_bias