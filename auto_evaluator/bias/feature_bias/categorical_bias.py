from auto_evaluator.bias.feature_bias.feature_bias import FeatureBias

class CategoricalBias(FeatureBias):
    """
    A class for the categorical bias of an input feature.
    """
    def __call__(self, *args, **kwargs):
        """
        Calculating the categorical bias of a single feature.
        :return: the average absolute performances and a boolean indicating if the model is biased according to that feature.
        """
        eval_metrics = self._calculate_categorical_metrics(self.feature)
        pairwise_diff, avg_abs_performance = FeatureBias._calculate_average_absolute_performance(eval_metrics)

        return avg_abs_performance, avg_abs_performance >= self.significance