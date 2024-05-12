from ml_eval_pro.model_fairness.feature_fairness.feature_fairness import FeatureFairness


class CategoricalFairness(FeatureFairness):
    """
    A class for the categorical bias of an input feature.
    """
    def get_categorical_features(self):
        return self.features[self.feature_name]