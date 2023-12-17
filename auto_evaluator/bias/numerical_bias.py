import pandas as pd

from auto_evaluator.bias.feature_bias import FeatureBias
from sklearn.cluster import KMeans

class NumericalBias(FeatureBias):
    def __init__(self, model, target: pd.Series, feature: pd.Series,
                 performance_metric: str = 'accuracy', significance: float = 0.05,
                 no_of_clusters: int = 5):
        super().__init__(model, target, feature, performance_metric, significance)
        self.no_of_clusters = no_of_clusters


    def __call__(self, *args, **kwargs):
        kmeans_clustering = KMeans(n_clusters=self.no_of_clusters)
        clusters = kmeans_clustering.fit_predict(self.feature)

        eval_metrics = self.__calculate_categorical_metrics(clusters)
        pairwise_diff, avg_abs_performance = NumericalBias.__calculate_average_absolute_performance(eval_metrics)

        return pairwise_diff, avg_abs_performance, avg_abs_performance > self.significance


