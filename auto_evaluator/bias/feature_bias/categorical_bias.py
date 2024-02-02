from auto_evaluator.bias.feature_bias.feature_bias import FeatureBias


class CategoricalBias(FeatureBias):
    """
    A class for the categorical bias of an input feature.
    """

    def check_bias(self):
        """
        Calculating the bias of a single feature.
        :return: the average absolute performances and a boolean indicating if the model is biased according to that feature.
        """
        return self._check_feature_bias(self.features[self.feature_name])
