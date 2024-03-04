import pandas as pd

from ml_eval_pro.bias.feature_bias_factory import FeatureBiasFactory
from ml_eval_pro.utils.feature_type import check_feature_type


class ModelBias:
    """
    A class for measuring the model bias.
    """

    def __init__(self, model, model_type: str, data: pd.DataFrame, target: pd.Series,
                 significance: float = 0.05):
        """
        Initializing the model bias needed inputs.
        :param model: the model.
        :param model_type: the model type.
        :param data: the dataset containing all the features.
        :param target: the target values.
        :param significance: the significance value to measure bias.
        """
        self.model = model
        self.model_type = model_type
        self.data = data
        self.target = target
        self.significance = significance

    def __call__(self, *args, **kwargs):
        """
        Calculating the model bias.
        """
        features_names = self.data.columns.tolist()
        features_bias = []
        for feature_name in features_names:
            feature_type = check_feature_type(self.data[feature_name])

            bias = FeatureBiasFactory.create(feature_type.value, self.model, self.model_type,
                                             self.target, self.data, feature_name,
                                             significance=self.significance)
            curr_bias = bias.check_bias()
            if curr_bias[2]:
                features_bias.append(curr_bias)

        self.features_bias = features_bias
        return features_bias
