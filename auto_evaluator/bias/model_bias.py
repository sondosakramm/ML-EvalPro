import pandas as pd

from auto_evaluator.bias.feature_bias_factory import FeatureBiasFactory
from auto_evaluator.utils.feature_type import check_feature_type


class ModelBias:
    """
    A class for measuring the model bias.
    """

    def __init__(self, model, data: pd.DataFrame, target: pd.Series, performance_metric='accuracy',
                 significance: float = 0.05):
        """
        Initializing the model bias needed inputs.
        :param model: the model.
        :param data: the dataset containing all the features.
        :param target: the target values.
        :param performance_metric: the performance metric used for measuring the bias.
        :param significance: the significance value to measure bias.
        """
        self.model = model
        self.data = data
        self.target = target
        self.performance_metric = performance_metric
        self.significance = significance

    def __call__(self, *args, **kwargs):
        """
        Calculating the model bias.
        """
        features_names = self.data.columns.tolist()
        features_bias = []
        for feature_name in features_names:
            feature_type = check_feature_type(self.data[feature_name])

            bias = FeatureBiasFactory.create(feature_type.value, self.model, self.target, self.data,
                                             feature_name, performance_metric=self.performance_metric,
                                             significance=self.significance)
            features_bias.append(bias.check_bias())

        return features_bias

    def __str__(self):
        features_bias = self.__call__()
        results_string = ("The model bias results state the following according to each feature:\n"
                          "---------------------------------------------------------------------\n")

        for bias in features_bias:
            biased = "biased"
            if not bias[2]:
                biased = "not" + biased

            results_string += f"The model is '{biased}' for the feature '{bias[0]}' with value {bias[1]}\n"
        return results_string
