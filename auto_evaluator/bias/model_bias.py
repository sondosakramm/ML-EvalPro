import pandas as pd

from auto_evaluator.bias.feature_bias_factory import FeatureBiasFactory
from auto_evaluator.utils.feature_type import check_feature_type


class ModelBias:
    """
    A class for measuring the model bias.
    """
    def __init__(self, model, data: pd.DataFrame, target:pd.Series, performance_metric='accuracy',
                 significance: float = 0.05, no_of_clusters: int = 5):
        """
        Initializing the model bias needed inputs.
        :param model: the model.
        :param data: the dataset containing all the features.
        :param target: the target values.
        :param performance_metric: the performance metric used for measuring the bias.
        :param significance: the significance value to measure bias.
        :param no_of_clusters: the number of binning ranges of the numerical feature.
        """
        self.model = model
        self.data = data
        self.target = target
        self.performance_metric = performance_metric
        self.significance = significance
        self.no_of_clusters = no_of_clusters


    def __call__(self, *args, **kwargs):
        """
        Calculating the model bias.
        """
        features_names = self.data.columns.tolist()
        features_bias = []
        for feature_name in features_names:
            feature_type = check_feature_type(self.data[feature_name])

            bias = FeatureBiasFactory.create(feature_type.value, self.model, self.target, self.data[feature_name],
                                             performance_metric=self.performance_metric, significance=self.significance,
                                             no_of_clusters=self.no_of_clusters)

            features_bias.append(bias())

        return features_bias