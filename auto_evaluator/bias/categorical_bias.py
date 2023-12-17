from auto_evaluator.bias.feature_bias import FeatureBias

class CategoricalBias(FeatureBias):
    def __call__(self, *args, **kwargs):
        eval_metrics = self.__calculate_categorical_metrics(self.feature)
        pairwise_diff, avg_abs_performance = CategoricalBias.__calculate_average_absolute_performance(eval_metrics)

        return pairwise_diff, avg_abs_performance, avg_abs_performance > self.significance