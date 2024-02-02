import pandas as pd

from auto_evaluator.bias.feature_bias.feature_bias import FeatureBias


class NumericalBias(FeatureBias):
    """
    A class for the numerical bias of an input feature.
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
        super().__init__(model, model_type, target, features, feature_name, significance)

        self.no_of_bins = 10
        self.features_binned = self.__get_binning_indices()

    def check_bias(self, *args):
        """
        Calculating the numerical bias of a single feature.
        :return: the average absolute performances and a boolean indicating if the model is biased according to that feature.
        """
        return self._check_feature_bias(self.features_binned)

    def __get_binning_indices(self) -> pd.Series:
        """
        Binning the features by sorting them in ascending order and labeling them
        :return: the features after binning.
        """
        ordered_feature = self.features[self.feature_name].sort_values()
        feature_binned = self.features[self.feature_name].copy()

        bin_instances_size = int(self.features.size / self.no_of_bins)

        for i in range(self.no_of_bins):
            start_index = i * bin_instances_size
            end_index = (i + 1) * bin_instances_size

            if ordered_feature.size < end_index or i == self.no_of_bins - 1:
                end_index = ordered_feature.size

            feature_binned.replace(ordered_feature[start_index:end_index].tolist(), i, inplace=True)

        return feature_binned
