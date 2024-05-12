import pandas as pd

from ml_eval_pro.model_fairness.feature_fairness.feature_fairness import FeatureFairness
from ml_eval_pro.utils.optimal_clusters import calculate_optimal_bins


class NumericalFairness(FeatureFairness):
    """
    A class for the numerical fairness of an input feature.
    """
    def get_categorical_features(self):
        num_bins = calculate_optimal_bins(self.features[self.feature_name].to_numpy())
        return self.__get_binning_indices(num_bins)

    def __get_binning_indices(self, num_bins: int) -> pd.Series:
        """
        Binning the features by sorting them in ascending order and labeling them
        :return: the features after binning.
        """
        ordered_feature = self.features[self.feature_name].sort_values()
        feature_binned = self.features[self.feature_name].copy()

        bin_instances_size = int(self.features.size / num_bins)

        for i in range(num_bins):
            start_index = i * bin_instances_size
            end_index = (i + 1) * bin_instances_size

            if ordered_feature.size < end_index or i == num_bins - 1:
                end_index = ordered_feature.size

            feature_binned.replace(ordered_feature[start_index:end_index].tolist(), i, inplace=True)

        return feature_binned
